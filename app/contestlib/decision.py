import json
import re

DEFAULTS = {
    "A": {"epochs": 5, "learning_rate": 0.001},
    "B": {"epochs": 5, "learning_rate": 0.0005},
    "C": {"epochs": 5, "learning_rate": 0.0005},
}

SYSTEM = (
    "你是工业模型自训练系统的决策模块。根据公开验证结果和历史轮次，给下一轮续训提出一个可执行的改动。"
    "只输出一个 JSON 对象，不要输出解释或代码块，字段为："
    "epochs(整数 2-20)、learning_rate(0.0001-0.005)、"
    "hypothesis(本轮假设，不超过 200 字)、change_summary(相对父模型改了什么，不超过 200 字)、"
    "reason(依据哪条历史证据做此决定，不超过 200 字)。"
)


def build_messages(dataset, task, metric, direction, records, drift, parent_epochs):
    lines = [
        "数据集：%s，任务：%s，主指标：%s（%s）。" % (dataset, task, metric, direction),
        "父模型累计轮次：%s。" % parent_epochs,
    ]
    if drift:
        lines.append("数据漂移信号：%s" % json.dumps(drift, ensure_ascii=False, sort_keys=True))
    if records:
        lines.append("历史轮次（最近 5 轮）：")
        for item in records[-5:]:
            lines.append(json.dumps({
                "round": item.get("round"),
                "epochs": item.get("epochs"),
                "learning_rate": item.get("learning_rate"),
                "before": item.get("before"),
                "after": item.get("after"),
                "hypothesis": item.get("hypothesis"),
            }, ensure_ascii=False, sort_keys=True))
    else:
        lines.append("历史轮次：无，这是本次运行的第一轮。")
    lines.append("请给出本轮唯一改动，并保证 epochs 与 learning_rate 落在允许区间内。")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_decision(text, dataset, fallback_reason):
    base = dict(DEFAULTS.get(dataset, DEFAULTS["A"]))
    data = _extract_json(text)
    result = {
        "epochs": base["epochs"],
        "learning_rate": base["learning_rate"],
        "hypothesis": "延续父模型并微调优化超参数",
        "change_summary": "使用基线网络结构，仅调整训练轮次与学习率",
        "reason": fallback_reason,
    }
    if not isinstance(data, dict):
        return result
    if _is_int(data.get("epochs")):
        result["epochs"] = max(2, min(20, int(data["epochs"])))
    rate = data.get("learning_rate")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0:
        result["learning_rate"] = max(0.0001, min(0.005, float(rate)))
    for key in ("hypothesis", "change_summary", "reason"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()[:600]
    return result


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _extract_json(text):
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"```(?:json)?", "", text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
