from baselines.train import train_run

B_GRID = [(2, 0.0005), (5, 0.0005), (8, 0.0005), (2, 0.001),
          (5, 0.001), (10, 0.001), (15, 0.001)]


def better(before, after, direction):
    if before is None or after is None:
        return False
    return after > before if direction == "higher" else after < before


def metric_value(report, metric):
    return report["validation_after"].get(metric)


def candidate_configs(decision, grid, limit):
    pairs, seen = [], set()
    for epochs, learning_rate in [(decision["epochs"], decision["learning_rate"])] + list(grid):
        pair = (max(1, min(10000, int(epochs))), max(0.0001, round(float(learning_rate), 6)))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
        if len(pairs) >= limit:
            break
    return pairs


def search_train(key, train_path, validation_path, parent, decision, work, round_index,
                 metric, direction, grid, limit=8):
    best, attempts = None, []
    for index, (epochs, learning_rate) in enumerate(candidate_configs(decision, grid, limit)):
        output = work / ("round-%02d-c%d" % (round_index, index))
        report = train_run(key, train_path, validation_path, output, resume=parent,
                           epochs=epochs, learning_rate=learning_rate,
                           stage_id="round-%d-c%d" % (round_index, index))
        after = metric_value(report, metric)
        attempts.append({"label": "c%d" % index, "epochs": epochs,
                         "learning_rate": learning_rate, "after": after})
        if after is not None and (best is None or _better(after, best["after"], direction)):
            best = {"output": output, "report": report, "epochs": epochs,
                    "learning_rate": learning_rate, "after": after}
    used = dict(decision)
    used["epochs"], used["learning_rate"] = best["epochs"], best["learning_rate"]
    used["change_summary"] += "；harness 在模型建议附近扫 %d 组配置，按可见指标取最优。" % len(attempts)
    return best["output"], best["report"], used, attempts


def _better(after, current, direction):
    return after > current if direction == "higher" else after < current


def fixed_train(key, train_path, validation_path, parent, decision, work, round_index, metric):
    output = work / ("round-%02d" % round_index)
    report = train_run(key, train_path, validation_path, output, resume=parent,
                       epochs=decision["epochs"], learning_rate=decision["learning_rate"],
                       stage_id="round-%d" % round_index)
    used = dict(decision)
    used["change_summary"] += "；按既定链路固定单组配置训练，不回退。"
    attempts = [{"label": "primary", "epochs": decision["epochs"],
                 "learning_rate": decision["learning_rate"],
                 "after": metric_value(report, metric)}]
    return output, report, used, attempts


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
