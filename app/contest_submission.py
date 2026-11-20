import os
import shutil
from pathlib import Path

from baselines.model import Model
from baselines.predict import predict as run_predict
from contestlib.decision import build_messages, parse_decision
from contestlib.drift import compare_stats, dataset_stats
from contestlib.runner import better, metric_value, safe_train
from contestlib.state import load_records, save_state

PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
METRIC = {"A": ("macro_f1", "higher"), "B": ("rmse", "lower"), "C": ("rmse", "lower")}
_CACHE = {}


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
    classes = Model.load(parent).meta["classes"]

    stats = dataset_stats(train_path, task, classes)
    drift = compare_stats(stats, records[-1].get("stats") if records else None, task)
    messages = build_messages(key, task, metric, direction, records, drift, rnd.round_index - 1)
    reply = rnd.model(messages, max_tokens=min(600, rnd.max_output_tokens), temperature=0.2)
    decision = parse_decision(reply.text, key, "第 %d 轮：沿用父模型并小步微调。" % rnd.round_index)

    output, report, used, attempts = safe_train(key, train_path, validation_path, parent,
                                                decision, work, rnd.round_index, metric, direction)
    before = report["validation_before"].get(metric)
    after = metric_value(report, metric)
    keep_parent = not better(before, after, direction)
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
        "selected": "parent" if keep_parent else "candidate",
    }
    configuration = {"epochs": used["epochs"], "learning_rate": used["learning_rate"],
                     "seed": report["seed"], "drift": drift, "attempts": attempts}
    submitted = rnd.submit_candidate(
        output,
        hypothesis=used["hypothesis"],
        change_summary=used["change_summary"],
        configuration=configuration,
        seed=report["seed"],
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
        seed=report["seed"],
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
