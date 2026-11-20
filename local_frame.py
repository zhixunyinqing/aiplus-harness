import hashlib
import json
from pathlib import Path

METRIC = {"A": ("macro_f1", "higher", "classification"),
          "B": ("rmse", "lower", "regression"),
          "C": ("rmse", "lower", "regression")}


def package_dir(root, key):
    if key == "A":
        return root.parent / "AI+" / "数据包A"
    return root.parent / "AI+" / ("数据包%s_extracted" % key)


def start_frame(key, pkg, parent_dir=None, round_index=1, state_dir=None):
    parent_dir = Path(parent_dir or pkg)
    checkpoint = parent_dir / "checkpoint.npz"
    raw = checkpoint.read_bytes()
    sha = "sha256:" + hashlib.sha256(raw).hexdigest()
    parent = {
        "artifact_id": "parent-%s-r%d" % (key.lower(), round_index - 1),
        "artifact_sha256": sha,
        "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [],
                     "predict_argv": ["python3", "/app/harness.py", "predict"]},
        "files": [{"path": "checkpoint.npz", "sha256": sha, "size_bytes": len(raw)}],
    }
    metric, direction, task = METRIC[key]
    return {
        "protocol": "contest.harness.workspace.v2",
        "run_id": "local-run",
        "dataset_key": key,
        "round_index": round_index,
        "stage_id": "local",
        "prompt": "在授权预算内自主提出假设并完成一轮续训。",
        "prompt_version": "industrial-workspace-flexible-v3",
        "rounds_per_dataset": 10,
        "parent_checkpoint_sha256": sha,
        "task_contract": {
            "dataset_key": key,
            "task": task,
            "metric": metric,
            "metric_direction": direction,
            "public_projection": {
                "train_path": str(pkg / "public" / "train.jsonl"),
                "validation_path": str(pkg / "public" / "validation.jsonl"),
                "row_fields": ["id", "inputs", "answer"],
                "source_metadata": False,
            },
        },
        "model_name": "base:local",
        "parent_artifact": parent,
        "artifacts": [parent],
        "state_sha256": "",
        "state_path": None if state_dir is None else str(state_dir),
        "max_output_tokens": 700,
        "resource_limits": {"artifact_upload_protocol": "chunked-v1"},
        "current_validation": {},
        "wall_seconds": 600,
    }


def stub_model(messages, options):
    decision = {"epochs": 2, "learning_rate": 0.001,
                "hypothesis": "本地自检：小步微调两轮。",
                "change_summary": "仅调整轮次与学习率，保持网络结构。",
                "reason": "按历史验证指标做保守更新。"}
    return {"request_id": "call-local",
            "id": "chatcmpl-local",
            "choices": [{"message": {"role": "assistant",
                                     "content": json.dumps(decision, ensure_ascii=False)}}]}
