import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
from baselines.train import train_run
from contestlib.decision import build_messages
from local_frame import METRIC, package_dir

FULL_GRID = [(e, lr) for e in (1, 2, 3, 5, 8, 10) for lr in (0.0005, 0.001)]
PART_GRID = [(e, lr) for e in (1, 2, 5, 10) for lr in (0.0005, 0.001)]
SFT_DIR = ROOT / "experiments" / "sft"


def states_for(key, live_dir, prefix="", initial=True):
    pkg = package_dir(ROOT, key)
    states = []
    if initial:
        states.append({"name": prefix + "initial", "parent": pkg / "checkpoint.npz",
                       "records": [], "grid": FULL_GRID})
    base = live_dir / key
    if (base / "work-2" / "state_in" / "checkpoint.npz").exists():
        records = json.loads((base / "work-1" / "state_out" / "records.json").read_text(encoding="utf-8"))
        states.append({"name": prefix + "round1", "parent": base / "work-2" / "state_in" / "checkpoint.npz",
                       "records": records, "grid": PART_GRID})
        records2 = json.loads((base / "work-2" / "state_out" / "records.json").read_text(encoding="utf-8"))
        states.append({"name": prefix + "round2", "parent": base / "work-2" / "state_out" / "checkpoint.npz",
                       "records": records2, "grid": PART_GRID})
    return states


def improves(key, item):
    _, direction, _ = METRIC[key]
    best = item["best"]
    return best["after"] > best["before"] if direction == "higher" else best["after"] < best["before"]


def prune_examples(examples, grids):
    kept = []
    for item in examples:
        key, state = item["meta"]["dataset"], item["meta"]["state"]
        grid = next((g for g in grids.get(key, []) if g["state"] == state), None)
        if grid is None or improves(key, grid):
            kept.append(item)
        else:
            print("  剔除无提升样本", key, state)
    return kept


def run_grid(key, state, out_root):
    metric, direction, _ = METRIC[key]
    pkg = package_dir(ROOT, key)
    results = []
    for index, (epochs, rate) in enumerate(state["grid"]):
        output = out_root / ("%s-%s-%02d" % (key, state["name"], index))
        report = train_run(key, str(pkg / "public" / "train.jsonl"),
                           str(pkg / "public" / "validation.jsonl"), output,
                           resume=state["parent"], epochs=epochs, learning_rate=rate,
                           stage_id="grid-%s" % state["name"])
        results.append({"epochs": epochs, "learning_rate": rate,
                        "before": report["validation_before"].get(metric),
                        "after": report["validation_after"].get(metric)})
        print("  %s %s e%d lr%s -> %.4f" % (key, state["name"], epochs, rate, results[-1]["after"]))
    best = max(results, key=lambda item: item["after"]) if direction == "higher" \
        else min(results, key=lambda item: item["after"])
    return best, results


def make_answer(key, state, best):
    metric = METRIC[key][0]
    return {
        "epochs": int(best["epochs"]),
        "learning_rate": float(best["learning_rate"]),
        "hypothesis": "在数据集%s的当前状态下，%d 轮与学习率 %s 的网格实测最优，验证%s达到 %.4f。"
                      % (key, best["epochs"], best["learning_rate"], metric, best["after"]),
        "change_summary": "epochs=%d，learning_rate=%s，网络结构与特征不变。"
                          % (best["epochs"], best["learning_rate"]),
        "reason": "状态 %s 下按 epochs×learning_rate 网格实测，该组合相对父模型 %.4f → %.4f 提升最大。"
                  % (state["name"], best["before"], best["after"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-dir", required=True)
    parser.add_argument("--tag", default="sft-v1")
    parser.add_argument("--states", default="initial,round1,round2")
    parser.add_argument("--seed-from", default="", help="从已有 tag 的数据集开始合并，如 sft-v1")
    parser.add_argument("--prune-only", action="store_true", help="不跑网格，仅按无提升规则重写数据集")
    parser.add_argument("--more", action="append", default=[],
                        help="补充状态来源，格式 name=live目录，如 sft1=path")
    args = parser.parse_args()
    wanted = [item.strip() for item in args.states.split(",") if item.strip()]
    extra = []
    for item in args.more:
        name, _, path = item.partition("=")
        extra.append((name.strip(), Path(path.strip().strip('"'))))
    live_dir = Path(args.live_dir)
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    grid_file = SFT_DIR / ("%s_grids.json" % args.tag)
    jsonl_file = SFT_DIR / ("%s.jsonl" % args.tag)
    seed = SFT_DIR / ("%s_grids.json" % args.seed_from) if args.seed_from else None
    seed_jsonl = SFT_DIR / ("%s.jsonl" % args.seed_from) if args.seed_from else None
    if seed and seed.exists() and not grid_file.exists():
        grid_file.write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
        if seed_jsonl.exists():
            jsonl_file.write_text(seed_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
    old_grids = json.loads(grid_file.read_text(encoding="utf-8")) if grid_file.exists() else {}
    old_examples = [json.loads(line) for line in jsonl_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()] if jsonl_file.exists() else []
    if args.prune_only:
        examples = prune_examples(old_examples, old_grids)
        jsonl_file.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in examples),
                              encoding="utf-8")
        print("examples:", len(examples), "saved", jsonl_file)
        return
    out_root = Path(tempfile.mkdtemp(prefix="grid-%s-" % args.tag))
    grids, new_examples, state_order = {}, [], {}
    for key in ("A", "B", "C"):
        merged = {item["state"]: item for item in old_grids.get(key, [])}
        order_names = [item["state"] for item in old_grids.get(key, [])]
        candidates = states_for(key, live_dir)
        for name, path in extra:
            candidates += states_for(key, path, prefix=name + "-", initial=False)
        for state in candidates:
            if state["name"] not in wanted:
                continue
            if state["name"] not in order_names:
                order_names.append(state["name"])
            best, results = run_grid(key, state, out_root)
            merged[state["name"]] = {"state": state["name"], "best": best, "runs": results}
            metric, direction, task = METRIC[key]
            drift = state["records"][-1].get("drift") if state["records"] else None
            messages = build_messages(key, task, metric, direction, state["records"], drift,
                                      len(state["records"]))
            answer = make_answer(key, state, best)
            new_examples.append({"messages": messages + [{"role": "assistant",
                                                          "content": json.dumps(answer, ensure_ascii=False)}],
                                 "meta": {"dataset": key, "state": state["name"], "grid": results}})
        grids[key] = [merged[name] for name in order_names if name in merged]
        state_order[key] = order_names
    replaced = {(item["meta"]["dataset"], item["meta"]["state"]) for item in new_examples}
    examples = [item for item in old_examples
                if (item["meta"]["dataset"], item["meta"]["state"]) not in replaced] + new_examples
    examples = prune_examples(examples, grids)
    examples.sort(key=lambda item: (("A", "B", "C").index(item["meta"]["dataset"]),
                                    state_order[item["meta"]["dataset"]].index(item["meta"]["state"])))
    jsonl_file.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in examples),
                          encoding="utf-8")
    grid_file.write_text(json.dumps(grids, ensure_ascii=False, indent=2), encoding="utf-8")
    print("examples:", len(examples), "grid runs:", sum(len(g["runs"]) for v in grids.values() for g in v))
    print("saved", SFT_DIR / ("%s.jsonl" % args.tag))


if __name__ == "__main__":
    main()
