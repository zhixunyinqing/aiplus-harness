import os
import shutil
from pathlib import Path

from baselines.model import Model, checkpoint_digest
from baselines.predict import predict as run_predict
from contestlib.decision import build_messages, parse_decision
from contestlib.drift import compare_stats, dataset_stats
from contestlib.runner import metric_value, train_candidate
from contestlib.state import load_records, save_state

PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
METRIC = {"A": ("macro_f1", "higher"), "B": ("rmse", "lower"), "C": ("rmse", "lower")}
_CACHE = {}


def parent_value(records, metric):
    value = None
    for item in records:
        if item.get("metric") != metric:
            continue
        if value is None:
            value = item.get("before")
        if item.get("selected") == "candidate":
            value = item.get("after")
    return value


def improved(before, after, direction):
    if before is None or after is None:
        return False
    return after > before if direction == "higher" else after < before


def train(rnd):
    work = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    key = rnd.dataset_key
    train_path = rnd.task["public_projection"]["train_path"]
    validation_path = rnd.task["public_projection"]["validation_path"]
    task = rnd.task["task"]
    metric, direction = METRIC.get(key, ("macro_f1", "higher"))

    records = load_records(rnd, work)
    parent = work / "parent.npz"
    shutil.copyfile(_parent_path(rnd), parent)
    parent_sha = checkpoint_digest(parent)
    model = Model.load(parent)
    classes, seed = model.meta["classes"], int(model.meta["seed"])
    try:
        stats = dataset_stats(train_path, task, classes)
        drift = compare_stats(stats, records[-1].get("stats") if records else None, task)
    except Exception as exc:
        stats = {}
        drift = {"available": False, "note": "统计不可用：%s" % type(exc).__name__}
    messages = build_messages(key, task, metric, direction, records, drift, rnd.round_index - 1)
    reply = rnd.model(messages, max_tokens=min(400, rnd.max_output_tokens), temperature=0.2)
    decision = parse_decision(reply.text, key, "第 %d 轮：沿用父模型并小步微调。" % rnd.round_index)

    failure = None
    try:
        output, report, used, attempts = train_candidate(
            key, train_path, validation_path, parent, decision, work,
            rnd.round_index, metric)
        before = report["validation_before"].get(metric)
        after = metric_value(report, metric)
    except Exception as exc:
        failure = type(exc).__name__
        output, attempts = None, []
        before = after = parent_value(records, metric)
        used = {"epochs": 0, "learning_rate": 0.0,
                "hypothesis": decision["hypothesis"],
                "change_summary": "本轮训练异常（%s），保留父产物以完成轮次。" % failure,
                "reason": "%s 训练异常，按稳健策略不推进状态链。" % decision["reason"]}
    keep_parent = output is None or not improved(before, after, direction)
    record = {
        "round": rnd.round_index,
        "metric": metric,
        "epochs": used["epochs"],
        "learning_rate": used["learning_rate"],
        "before": before,
        "after": after,
        "attempts": attempts,
        "hypothesis": used["hypothesis"],
        "change_summary": used["change_summary"],
        "reason": used["reason"],
        "drift": drift,
        "stats": stats,
        "artifact_dir": None if output is None else Path(output).name,
        "selected": "parent" if keep_parent else "candidate",
        "failure": failure,
    }
    configuration = {"epochs": used["epochs"], "learning_rate": used["learning_rate"],
                     "seed": seed, "drift": drift, "attempts": attempts,
                     "resumed_from_parent": True, "parent_checkpoint_sha256": parent_sha,
                     "failure": failure}
    submitted = None if output is None else rnd.submit_candidate(
        output,
        hypothesis=used["hypothesis"],
        change_summary=used["change_summary"],
        configuration=configuration,
        seed=seed,
        checkpoint_paths=["checkpoint.npz"],
        code_paths=[],
        predict_argv=PREDICT_ARGV,
    )
    if submitted is None:
        keep_parent = True
        record["selected"] = "parent"
    state_checkpoint = parent if keep_parent else output / "checkpoint.npz"
    save_state(rnd, work, state_checkpoint, records + [record])
    history_ids = [item["id"] for item in rnd.history()]
    summary = "%s %s -> %s；漂移=%s" % (metric, before, after, "是" if drift.get("drift") else "否")
    if keep_parent and output is not None:
        summary += "；候选未优于父模型，保留父产物"
    elif output is None:
        summary += "；保留父产物"
    rnd.finish(
        selected_artifact_id=rnd.parent.artifact_id if keep_parent else submitted["artifact_id"],
        hypothesis=used["hypothesis"],
        change_summary=used["change_summary"],
        reason="%s 平台测量：%s。" % (used["reason"], summary),
        configuration=configuration,
        seed=seed,
        history_ids=history_ids,
        model_call_ids=list(rnd.call_ids),
    )


def predict(sample, model_dir):
    path = os.environ.get("CONTEST_ARTIFACT_PATH") or str(Path(model_dir) / "checkpoint.npz")
    key = str(Path(path).resolve())
    model = _CACHE.get(key)
    if model is None:
        model = Model.load(key)
        _CACHE.clear()
        _CACHE[key] = model
    return run_predict(model, sample.inputs)


def _parent_path(rnd):
    local = os.environ.get("CONTEST_PREVIOUS_DIR")
    if local:
        return rnd.parent.path("checkpoint.npz", local)
    return rnd.parent.path("checkpoint.npz")
