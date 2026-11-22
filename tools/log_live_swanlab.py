import argparse
import json
from pathlib import Path

import swanlab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="experiments/live_<tag>.json")
    parser.add_argument("--project", default="aiplus-harness")
    args = parser.parse_args()

    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    project = swanlab.init(
        project=args.project,
        experiment_name="llm-eval-%s" % data["tag"],
        config={"tag": data["tag"], "base_model": data["model"], "weights": data.get("weights"),
                "rounds_per_dataset": data["rounds_per_dataset"], "mode": "live-inference"},
    )
    for key, rounds in data["datasets"].items():
        for item in rounds:
            step = int(item["round"])
            before, after = float(item["before"]), float(item["after"])
            higher = key.upper() == "A"
            delta = (after - before) if higher else (before - after)
            swanlab.log({
                "%s_before" % key: before,
                "%s_after" % key: after,
                "%s_improvement" % key: delta,
                "%s_epochs" % key: int(item["decision"]["epochs"]),
                "%s_learning_rate" % key: float(item["decision"]["learning_rate"]),
                "%s_latency_s" % key: float(item["model"]["latency"]),
                "%s_prompt_tokens" % key: int(item["model"]["prompt_tokens"]),
                "%s_output_tokens" % key: int(item["model"]["output_tokens"]),
            }, step=step)
    swanlab.finish()
    print("SwanLab 实验：llm-eval-%s，项目 %s" % (data["tag"], args.project))


if __name__ == "__main__":
    main()
