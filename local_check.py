import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from local_frame import METRIC, package_dir, start_frame, stub_model

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app"
sys.path.insert(0, str(ROOT / "sdk"))
from tests.support import Host


def pick_model():
    if os.environ.get("LOCAL_MODEL") != "pytrio":
        return stub_model
    sys.path.insert(0, str(ROOT / "tools"))
    from pytrio_model import build_model
    return build_model(os.environ.get("LOCAL_BASE_MODEL", "Qwen/Qwen3.8-27B"))


def run_round(key, pkg, work, parent_dir, round_index, state_dir, model):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(APP), CONTEST_WORK=str(work),
               CONTEST_PREVIOUS_DIR=str(parent_dir), PYTHONUTF8="1")
    process = subprocess.Popen([sys.executable, str(APP / "harness.py"), "train"],
                               cwd=str(APP), env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8")
    frame = start_frame(key, pkg, parent_dir, round_index, state_dir)
    host = Host(frame, model)
    thread = host.serve(process)
    stderr = process.stderr.read()
    code = process.wait(timeout=600)
    thread.join(timeout=10)
    print("round %d exit: %s failure: %s" % (round_index, code, host.failure))
    if code != 0 or host.failure:
        print(stderr[-2000:])
        return None
    records = json.loads((work / "state_out" / "records.json").read_text(encoding="utf-8"))
    print("  after:", records[-1]["after"], "selected:", records[-1]["selected"],
          "attempts:", len(records[-1]["attempts"]),
          "drift:", json.dumps(records[-1]["drift"], ensure_ascii=False)[:120])
    return records[-1]["selected"] == "candidate"


def main():
    key = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    pkg = Path(sys.argv[3]) if len(sys.argv) > 3 else package_dir(ROOT, key)
    metric, direction, task = METRIC[key]
    root = Path(tempfile.mkdtemp(prefix="local-%s-" % key))
    model = pick_model()
    parent_dir, state_dir = pkg, None
    for index in range(1, rounds + 1):
        work = root / ("work-%d" % index)
        selected = run_round(key, pkg, work, parent_dir, index, state_dir, model)
        if selected is None:
            return 1
        if selected:
            parent_dir = work / ("round-%02d" % index)
        state_dir = work / "state_out"
    sample = json.loads((pkg / "public" / "validation.jsonl").read_text(encoding="utf-8").splitlines()[0])
    env = os.environ.copy()
    env.update(PYTHONPATH=str(APP), PYTHONUTF8="1", CONTEST_TASK=task,
               CONTEST_DATASET_KEY=key, CONTEST_ARTIFACT_PATH=str(state_dir / "checkpoint.npz"))
    result = subprocess.run([sys.executable, str(APP / "harness.py"), "predict"],
                            cwd=str(APP), env=env, text=True, encoding="utf-8",
                            input=json.dumps({"task": task, "inputs": sample["inputs"]}) + "\n",
                            capture_output=True)
    print("predict exit:", result.returncode)
    print("predict out:", result.stdout.strip()[:400])
    if os.environ.get("LOCAL_MODEL") == "pytrio":
        from pytrio_model import summary
        print("model stats:", summary())
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
