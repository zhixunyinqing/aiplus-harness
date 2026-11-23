import os
import shutil
from pathlib import Path

from baselines.model import Model, checkpoint_digest
from baselines.predict import predict as run_predict
from contestlib.decision import build_messages, parse_decision
from contestlib.drift import compare_stats, dataset_stats
from contestlib.runner import B_GRID, metric_value, safe_train, search_train
from contestlib.state import load_records, save_state

PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
METRIC = {"A": ("macro_f1", "higher"), "B": ("rmse", "lower"), "C": ("rmse", "lower")}
POLICY = {"A": {"budget": 10, "margin": 0.0, "min_epochs": 2},
          "B": {"budget": 10, "margin": 0.0, "min_epochs": 2},
          "C": {"budget": 1, "margin": 0.0, "min_epochs": 2}}
_CACHE = {}


def adopted(records):
    return sum(1 for item in records if item.get("selected") == "candidate")


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


def floor_epochs(decision, floor):
    if decision["epochs"] < floor:
        decision["epochs"] = floor
        decision["change_summary"] += "；harness 依公开网格把 epochs 下限设为 %d。" % floor
    return decision


def improved(before, after, direction, margin):
    if before is None or after is None:
        return False
    if direction == "higher":
        return after > before + margin
    return after < before * (1.0 - margin)


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
    policy = POLICY.get(key, {"budget": 10 ** 9, "margin": 0.0, "min_epochs": 2})
    exhausted = adopted(records) >= policy["budget"]

    stats = dataset_stats(train_path, task, classes)
    drift = compare_stats(stats, records[-1].get("stats") if records else None, task)
    messages = build_messages(key, task, metric, direction, records, drift, rnd.round_index - 1)
    reply = rnd.model(messages, max_tokens=min(600, rnd.max_output_tokens), temperature=0.2)
    decision = parse_decision(reply.text, key, "第 %d 轮：沿用父模型并小步微调。" % rnd.round_index)
    decision = floor_epochs(decision, policy["min_epochs"])

    if exhausted:
        output, attempts = None, []
        before = after = parent_value(records, metric)
        used = {"epochs": 0, "learning_rate": 0.0,
                "hypothesis": decision["hypothesis"],
                "change_summary": "采纳预算 %d 已用尽，本轮不训练，保留父产物。" % policy["budget"],
                "reason": "%s 预算用尽，不再推进状态链。" % decision["reason"]}
    elif key == "B" and rnd.round_index == 1:
        decision["epochs"], decision["learning_rate"] = 10, 0.001
        decision["change_summary"] += "；B 首轮固定为 R1 实测起点 10 轮 / 0.001。"
        output, report, used, attempts = safe_train(
            key, train_path, validation_path, parent, decision, work,
            rnd.round_index, metric, direction)
        before = report["validation_before"].get(metric)
        after = metric_value(report, metric)
    elif key == "B":
        output, report, used, attempts = search_train(
            key, train_path, validation_path, parent, decision, work,
            rnd.round_index, metric, direction, B_GRID)
        before = report["validation_before"].get(metric)
        after = metric_value(report, metric)
    else:
        output, report, used, attempts = safe_train(key, train_path, validation_path, parent,
                                                    decision, work, rnd.round_index, metric, direction)
        before = report["validation_before"].get(metric)
        after = metric_value(report, metric)
    keep_parent = not improved(before, after, direction, policy["margin"])
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
        "artifact_dir": None if exhausted else Path(output).name,
        "selected": "parent" if keep_parent else "candidate",
    }
    configuration = {"epochs": used["epochs"], "learning_rate": used["learning_rate"],
                     "seed": seed, "drift": drift, "attempts": attempts,
                     "resumed_from_parent": True, "parent_checkpoint_sha256": parent_sha,
                     "budget": policy["budget"], "min_epochs": policy["min_epochs"],
                     "adopted_before": adopted(records),
                     "budget_exhausted": exhausted}
    submitted = None if exhausted else rnd.submit_candidate(
        output,
        hypothesis=used["hypothesis"],
        change_summary=used["change_summary"],
        configuration=configuration,
        seed=seed,
        checkpoint_paths=["checkpoint.npz"],
        code_paths=[],
        predict_argv=PREDICT_ARGV,
    )
    state_checkpoint = parent if keep_parent else output / "checkpoint.npz"
    save_state(rnd, work, state_checkpoint, records + [record])
    history_ids = [item["id"] for item in rnd.history()]
    summary = "%s %s -> %s；漂移=%s" % (metric, before, after, "是" if drift.get("drift") else "否")
    if keep_parent:
        summary += "；候选未优于父模型，保留父产物"
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
    model = _CACHE.get("model")
    if model is None:
        path = os.environ.get("CONTEST_ARTIFACT_PATH") or str(Path(model_dir) / "checkpoint.npz")
        model = Model.load(path)
        _CACHE["model"] = model
    return run_predict(model, sample.inputs)


def _parent_path(rnd):
    local = os.environ.get("CONTEST_PREVIOUS_DIR")
    if local:
        return rnd.parent.path("checkpoint.npz", local)
    return rnd.parent.path("checkpoint.npz")
