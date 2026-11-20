from baselines.train import train_run


def better(before, after, direction):
    if before is None or after is None:
        return False
    return after > before if direction == "higher" else after < before


def metric_value(report, metric):
    return report["validation_after"].get(metric)


def safe_train(key, train_path, validation_path, parent, decision, work, round_index, metric, direction):
    primary_dir = work / ("round-%02d" % round_index)
    primary = train_run(key, train_path, validation_path, primary_dir, resume=parent,
                        epochs=decision["epochs"], learning_rate=decision["learning_rate"],
                        stage_id="round-%d" % round_index)
    primary_value = metric_value(primary, metric)
    base_value = primary["validation_before"].get(metric)
    attempts = [{"label": "primary", "epochs": decision["epochs"],
                 "learning_rate": decision["learning_rate"], "after": primary_value}]
    if better(base_value, primary_value, direction):
        return primary_dir, primary, dict(decision), attempts
    fallback = dict(decision)
    fallback["epochs"] = max(1, int(decision["epochs"]) // 2)
    fallback["learning_rate"] = max(0.0001, round(float(decision["learning_rate"]) / 2, 6))
    fallback_dir = work / ("round-%02d-alt" % round_index)
    alternative = train_run(key, train_path, validation_path, fallback_dir, resume=parent,
                            epochs=fallback["epochs"], learning_rate=fallback["learning_rate"],
                            stage_id="round-%d-fallback" % round_index)
    fallback_value = metric_value(alternative, metric)
    attempts.append({"label": "fallback", "epochs": fallback["epochs"],
                     "learning_rate": fallback["learning_rate"], "after": fallback_value})
    if better(primary_value, fallback_value, direction):
        return fallback_dir, alternative, fallback, attempts
    return primary_dir, primary, dict(decision), attempts
