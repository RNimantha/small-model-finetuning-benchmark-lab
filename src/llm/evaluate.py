import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_prompt(schema: str, document: str) -> str:
    return f"""### Instruction:
You are an information extraction model.

Extract structured JSON from the document below.
Return only valid JSON.
Do not explain your answer.

JSON Schema:
{schema}

Document:
{document}

### Response:
"""


def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

            if limit and len(rows) >= limit:
                break

    return rows


def try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def flatten_keys(obj: Any, prefix: str = "") -> set[str]:
    keys = set()

    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.add(full_key)
            keys.update(flatten_keys(value, full_key))

    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            keys.update(flatten_keys(item, f"{prefix}[{index}]"))

    return keys


def key_overlap_score(pred: Any, target: Any) -> float:
    pred_keys = flatten_keys(pred)
    target_keys = flatten_keys(target)

    if not target_keys:
        return 0.0

    return len(pred_keys.intersection(target_keys)) / len(target_keys)


def load_model_and_tokenizer(config: dict, mode: str):
    base_model_name = config["model"]["base_model"]
    adapter_path = Path(config["model"]["output_dir"]) / "final_adapter"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if mode == "adapter":
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter not found at {adapter_path}. Train the model first."
            )

        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:" in decoded:
        return decoded.split("### Response:", 1)[1].strip()

    return decoded.strip()


def evaluate(config_path: str, mode: str, limit: int) -> dict:
    config = load_config(config_path)
    validation_file = config["dataset"]["validation_file"]

    rows = load_jsonl(validation_file, limit=limit)
    model, tokenizer = load_model_and_tokenizer(config, mode)

    valid_json_count = 0
    exact_match_count = 0
    key_scores = []
    latencies = []

    for idx, row in enumerate(rows):
        prompt = build_prompt(
            schema=row["schema"],
            document=row["input_text"],
        )

        target_raw = row["target_json"]
        target_json = try_parse_json(target_raw)

        start_time = time.time()
        prediction_raw = generate_response(model, tokenizer, prompt)
        latency = time.time() - start_time
        latencies.append(latency)

        prediction_json = try_parse_json(prediction_raw)

        if prediction_json is not None:
            valid_json_count += 1

        if prediction_json == target_json:
            exact_match_count += 1

        if prediction_json is not None and target_json is not None:
            key_scores.append(key_overlap_score(prediction_json, target_json))
        else:
            key_scores.append(0.0)

        print(f"Evaluated {idx + 1}/{len(rows)}")

    total = len(rows)

    results = {
        "mode": mode,
        "num_examples": total,
        "valid_json_rate": valid_json_count / total,
        "exact_match_rate": exact_match_count / total,
        "avg_key_overlap": sum(key_scores) / total,
        "avg_latency_seconds": sum(latencies) / total,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/llm_qlora.yaml",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["base", "adapter"],
        default="base",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    results = evaluate(
        config_path=args.config,
        mode=args.mode,
        limit=args.limit,
    )

    print("\nEvaluation Results")
    print("=" * 80)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()