import json

import numpy as np

from baselines.data import read_dataset


def dataset_stats(path, task, classes):
    data = read_dataset(path, task, classes)
    x = data["x"]
    stats = {
        "rows": int(len(data["ids"])),
        "mean": [round(float(v), 4) for v in x.mean(axis=0)],
        "scale": [round(float(v), 4) for v in np.maximum(x.std(axis=0), 1e-6)],
    }
    if task == "classification":
        counts = np.asarray([np.sum(data["y"] == i) for i in range(len(classes))], dtype=float)
        stats["class_ratio"] = [round(float(v / counts.sum()), 4) for v in counts]
    else:
        stats["target_mean"] = round(float(np.mean(data["y"])), 3)
    return stats


def compare_stats(current, previous, task):
    if not previous:
        return {"available": False, "note": "首轮没有上一轮对照"}
    mean_now = np.asarray(current["mean"])
    mean_old = np.asarray(previous["mean"])
    scale_old = np.asarray(previous["scale"])
    z = np.abs(mean_now - mean_old) / np.maximum(scale_old, 1e-6)
    feature_shift = float(np.mean(z))
    result = {
        "available": True,
        "feature_shift": round(feature_shift, 4),
        "top_features": [int(i) for i in np.argsort(z)[-3:][::-1]],
        "rows_before": previous.get("rows"),
        "rows_now": current.get("rows"),
    }
    if task == "classification" and "class_ratio" in previous:
        delta = np.abs(np.asarray(current["class_ratio"]) - np.asarray(previous["class_ratio"]))
        result["label_shift"] = round(float(delta.sum() / 2), 4)
    elif "target_mean" in previous:
        base = max(abs(previous["target_mean"]), 1e-6)
        result["target_shift"] = round(float(abs(current["target_mean"] - previous["target_mean"]) / base), 4)
    drift = feature_shift > 0.5 or result.get("label_shift", 0) > 0.1 or result.get("target_shift", 0) > 0.2
    result["drift"] = bool(drift)
    return result


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
