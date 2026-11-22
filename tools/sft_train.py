import argparse
import json
from pathlib import Path

import numpy as np
import pytrio as trio
import swanlab

ROOT = Path(__file__).resolve().parents[1]


def apply_template(tokenizer, messages, generation):
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=generation, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=generation)


def make_datum(messages, tokenizer):
    prompt_tokens = tokenizer.encode(apply_template(tokenizer, messages[:-1], True),
                                     add_special_tokens=False)
    completion_tokens = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        completion_tokens = completion_tokens + [tokenizer.eos_token_id]
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
    tokens = prompt_tokens + completion_tokens
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"weights": np.asarray(weights[1:], dtype=np.float32),
                        "target_tokens": np.asarray(tokens[1:], dtype=np.int32)},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="experiments/sft/sft-v1.jsonl")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--base-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    items = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    client = trio.ServiceClient()
    training = client.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    tokenizer = training.get_tokenizer()
    data = [make_datum(item["messages"], tokenizer) for item in items]

    swanlab.init(project="aiplus-harness", experiment_name="sft-%s" % args.tag,
                 config={"base_model": args.base_model, "rank": args.rank, "steps": args.steps,
                         "learning_rate": args.lr, "examples": len(data), "data": args.data})
    losses = []
    for step in range(1, args.steps + 1):
        result = training.forward_backward(data, "cross_entropy").result()
        training.optim_step(trio.AdamParams(learning_rate=args.lr)).result()
        logprobs = np.concatenate([out["logprobs"].tolist() for out in result.loss_fn_outputs])
        weights = np.concatenate([item.loss_fn_inputs["weights"].tolist() for item in data])
        loss = float(-np.dot(logprobs, weights) / weights.sum())
        losses.append({"step": step, "loss": loss})
        swanlab.log({"sft_loss": loss}, step=step)
        print("step %d loss %.4f" % (step, loss))

    saved = training.save_weights_for_sampler(name="aiplus-%s" % args.tag).result()
    meta = {"tag": args.tag, "base_model": args.base_model, "rank": args.rank,
            "steps": args.steps, "learning_rate": args.lr, "examples": len(data),
            "weights_path": saved.path, "losses": losses}
    out = ROOT / "experiments" / "sft" / ("%s_run.json" % args.tag)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    swanlab.finish()
    print("weights:", saved.path)
    print("saved", out)


if __name__ == "__main__":
    main()
