from baselines.train import train_run


def metric_value(report, metric):
    return report["validation_after"].get(metric)


def train_candidate(profile, train_path, validation_path, parent, decision, work, round_index, metric):
    output = work / ("round-%02d" % round_index)
    report = train_run(profile, train_path, validation_path, output, resume=parent,
                       epochs=decision["epochs"], learning_rate=decision["learning_rate"],
                       stage_id="round-%d" % round_index)
    used = dict(decision)
    attempts = [{"label": "primary", "epochs": decision["epochs"],
                 "learning_rate": decision["learning_rate"],
                 "after": metric_value(report, metric)}]
    return output, report, used, attempts
