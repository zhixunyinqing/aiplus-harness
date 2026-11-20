import argparse
import json
from pathlib import Path

import swanlab


def load_records(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit("records.json 为空或格式不对")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="A / B / C")
    parser.add_argument("--records", required=True, help="state_out/records.json")
    parser.add_argument("--model", required=True, help="平台基础模型名称")
    parser.add_argument("--project", default="aiplus-harness")
    args = parser.parse_args()

    records = load_records(args.records)
    metric = records[0]["metric"]
    run = swanlab.init(
        project=args.project,
        experiment_name="dataset-%s" % args.dataset,
        config={
            "dataset": args.dataset,
            "metric": metric,
            "model_source": "base",
            "base_model": args.model,
            "harness_image": "aiplus-harness:0.1",
            "rounds": len(records),
        },
    )
    for record in records:
        step = int(record["round"])
        swanlab.log(
            {
                "validation_before": float(record["before"]),
                "validation_after": float(record["after"]),
                "epochs": int(record["epochs"]),
                "learning_rate": float(record["learning_rate"]),
                "selected_candidate": 1 if record["selected"] == "candidate" else 0,
                "drift": 1 if record.get("drift", {}).get("drift") else 0,
                "attempts": len(record.get("attempts", [])),
            },
            step=step,
        )
    swanlab.finish()
    url = getattr(getattr(run, "cloud", None), "url", None)
    print("SwanLab 实验链接：", url or "请到 swanlab.cn 项目 %s 查看" % args.project)


if __name__ == "__main__":
    main()
