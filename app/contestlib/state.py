import json
import shutil
import uuid
from pathlib import Path

RECORDS = "records.json"
CHECKPOINT = "checkpoint.npz"


def _fresh(work, prefix):
    root = Path(work)
    root.mkdir(parents=True, exist_ok=True)
    return root / ("%s-%s" % (prefix, uuid.uuid4().hex[:12]))


def load_records(rnd, work):
    source = rnd.state_workspace(str(_fresh(work, "state_in")))
    path = Path(source) / RECORDS
    if not path.is_file():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(records, list):
        return []
    return [item for item in records if isinstance(item, dict)]


def save_state(rnd, work, checkpoint, records):
    target = _fresh(work, "state_out")
    target.mkdir(parents=True)
    payload = json.dumps(records[-50:], ensure_ascii=False, sort_keys=True)
    (target / RECORDS).write_text(payload, encoding="utf-8")
    shutil.copyfile(checkpoint, target / CHECKPOINT)
    rnd.save_state(target)
    return target
