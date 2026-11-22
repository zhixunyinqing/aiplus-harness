import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from local_check import run_round
from local_frame import package_dir
from pytrio_model import STATS, build_model

FIELDS = ("epochs", "learning_rate", "hypothesis", "change_summary", "reason")


def run_dataset(model, key, rounds, root):
    pkg = package_dir(ROOT, key)
    parent_dir, state_dir = pkg, None
    output = []
    for index in range(1, rounds + 1):
        work = root / key / ("work-%d" % index)
        started, mark = time.time(), len(STATS)
        if not run_round(key, pkg, work, parent_dir, index, state_dir, model):
            raise SystemExit("dataset %s round %d failed" % (key, index))
        calls = STATS[mark:]
        record = json.loads((work / "state_out" / "records.json").read_text(encoding="utf-8"))[-1]
        output.append({
            "round": index,
            "decision": {field: record[field] for field in FIELDS},
            "before": record["before"], "after": record["after"],
            "selected": record["selected"], "attempts": record["attempts"],
            "drift": record["drift"],
            "model": {"calls": len(calls),
                      "latency": round(sum(item["latency"] for item in calls), 2),
                      "prompt_tokens": sum(item["prompt_tokens"] for item in calls),
                      "output_tokens": sum(item["output_tokens"] or 0 for item in calls)},
            "wall_seconds": round(time.time() - started, 1),
        })
        parent_dir, state_dir = work / ("round-%02d" % index), work / "state_out"
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="A,B,C")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--weights", default="", help="可选：trio:// 微调权重路径")
    parser.add_argument("--tag", default="base-v1")
    args = parser.parse_args()

    model = build_model(args.model, model_path=args.weights or None)
    root = Path(tempfile.mkdtemp(prefix="live-%s-" % args.tag))
    result = {"tag": args.tag, "model": args.model, "weights": args.weights or None,
              "rounds_per_dataset": args.rounds, "datasets": {}}
    keys = [item.strip().upper() for item in args.datasets.split(",") if item.strip()]
    for key in keys:
        result["datasets"][key] = run_dataset(model, key, args.rounds, root)
    out = ROOT / "experiments" / ("live_%s.json" % args.tag)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, rounds in result["datasets"].items():
        for item in rounds:
            print("%s r%d: %s -> %s (%s) %s" % (key, item["round"], item["before"], item["after"],
                                                item["selected"], item["decision"]["epochs"]))
    print("saved", out)


if __name__ == "__main__":
    main()
