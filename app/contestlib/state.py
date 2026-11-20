import json
import shutil
from pathlib import Path

RECORDS = "records.json"
CHECKPOINT = "checkpoint.npz"


def load_records(rnd, work):
    source = rnd.state_workspace(str(Path(work) / "state_in"))
    path = Path(source) / RECORDS
    if not path.is_file():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return records if isinstance(records, list) else []


def save_state(rnd, work, checkpoint, records):
    target = rnd.state_workspace(str(Path(work) / "state_out"))
    target = Path(target)
    payload = json.dumps(records[-50:], ensure_ascii=False, sort_keys=True)
    (target / RECORDS).write_text(payload, encoding="utf-8")
    shutil.copyfile(checkpoint, target / CHECKPOINT)
    rnd.save_state(target)
    return target
