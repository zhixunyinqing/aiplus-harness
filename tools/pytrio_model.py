import time
import uuid

import pytrio as trio

STATS = []


def build_model(base_model, temperature=0.2, model_path=None):
    client = trio.ServiceClient()
    if model_path:
        sampler = client.create_sampling_client(base_model=base_model, model_path=model_path)
    else:
        sampler = client.create_sampling_client(base_model=base_model)
    tokenizer = sampler.get_tokenizer()

    def render(messages):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def model(messages, options):
        ids = tokenizer.encode(render(messages), add_special_tokens=False)
        params = trio.SamplingParams(
            max_tokens=int(options.get("max_tokens") or 600),
            temperature=float(options.get("temperature") or temperature),
        )
        started = time.time()
        result = sampler.sample(prompt=trio.ModelInput.from_ints(ids),
                                sampling_params=params, num_samples=1).result()
        elapsed = time.time() - started
        sequence = result.sequences[0]
        output_tokens = len(sequence.tokens) if getattr(sequence, "tokens", None) is not None else None
        STATS.append({"latency": elapsed, "prompt_tokens": len(ids),
                      "output_tokens": output_tokens, "chars": len(sequence.text)})
        return {"request_id": "call-" + uuid.uuid4().hex[:8],
                "id": "chatcmpl-" + uuid.uuid4().hex[:8],
                "choices": [{"message": {"role": "assistant", "content": sequence.text}}]}

    model.base_model = base_model
    return model


def summary():
    if not STATS:
        return {}
    return {
        "calls": len(STATS),
        "avg_latency": round(sum(item["latency"] for item in STATS) / len(STATS), 2),
        "prompt_tokens": sum(item["prompt_tokens"] for item in STATS),
        "output_tokens": sum(item["output_tokens"] or 0 for item in STATS),
    }
