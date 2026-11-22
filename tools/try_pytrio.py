import json
import sys
import time
from pathlib import Path

import pytrio as trio

BASE_MODEL = "Qwen/Qwen3.8-27B"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from contestlib.decision import build_messages


def unwrap(value):
    return value.result() if hasattr(value, "result") else value


def load_case(dataset):
    path = ROOT / "experiments" / ("records-%s.json" % dataset)
    records = json.loads(path.read_text(encoding="utf-8"))
    last = records[-1]
    messages = build_messages(dataset, last["metric"] and "regression" if dataset != "A" else "classification",
                              last["metric"], "higher" if last["metric"] == "macro_f1" else "lower",
                              records[:-1], last.get("drift"), last["round"] - 1)
    return records, messages


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "B"
    records, messages = load_case(dataset)
    client = trio.ServiceClient()
    models = unwrap(client.get_supported_models())
    names = [m if isinstance(m, str) else getattr(m, "name", str(m)) for m in models]
    print("可用模型：", names)
    model = BASE_MODEL if BASE_MODEL in names else next((n for n in names if "Qwen" in n), BASE_MODEL)
    print("使用模型：", model)

    sampler = client.create_sampling_client(base_model=model)
    tokenizer = sampler.get_tokenizer()
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = trio.ModelInput.from_ints(tokenizer.encode(text, add_special_tokens=False))
    params = trio.SamplingParams(max_tokens=500, temperature=0.2)
    started = time.time()
    result = sampler.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
    elapsed = time.time() - started

    seq = result.sequences[0]
    print("耗时：%.2fs  输入token：%d  输出token：%s"
          % (elapsed, len(prompt.tokens) if hasattr(prompt, "tokens") else -1,
             getattr(seq, "tokens", None) and len(seq.tokens)))
    print("原始输出：\n" + seq.text)

    cleaned = seq.text.strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    try:
        decision = json.loads(cleaned[start:end + 1])
        required = {"epochs", "learning_rate", "hypothesis", "change_summary", "reason"}
        print("JSON 解析：成功；缺失字段：", sorted(required - set(decision)))
        print("决策：", json.dumps({k: decision.get(k) for k in ("epochs", "learning_rate")}, ensure_ascii=False))
        print("历史真实下一轮：", "before=%s after=%s" % (records[-1]["before"], records[-1]["after"]))
    except Exception as exc:
        print("JSON 解析：失败 ->", exc)


if __name__ == "__main__":
    main()
