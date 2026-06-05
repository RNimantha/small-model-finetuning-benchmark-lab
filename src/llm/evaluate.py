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

            if limit is not None and len(rows) >= limit:
                break

    return rows


def extract_json_candidate(text: str) -> str:
    """
    Some models generate extra text before/after JSON.
    This function tries to extract the JSON-looking part.
    """
    text = text.strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return text

    return text[first_brace : last_brace + 1]


def try_parse_json(text: str) -> Any | None:
    candidate = extract_json_candidate(text)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def normalize_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


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

        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    latency = time.time() - start_time

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:" in decoded:
        response = decoded.split("### Response:", 1)[1].strip()
    else:
        response = decoded.strip()

    return response, latency


def evaluate(config_path: str, mode: str, limit: int | None) -> dict:
    config = load_config(config_path)
    validation_file = config["dataset"]["validation_file"]

    rows = load_jsonl(validation_file, limit=limit)
    model, tokenizer = load_model_and_tokenizer(config, mode)

    valid_json_count = 0
    exact_match_count = 0
    key_scores = []
    latencies = []
    examples = []

    for idx, row in enumerate(rows):
        prompt = build_prompt(
            schema=row["schema"],
            document=row["input_text"],
        )

        target_raw = row["target_json"]
        target_json = try_parse_json(target_raw)

        prediction_raw, latency = generate_response(model, tokenizer, prompt)
        prediction_json = try_parse_json(prediction_raw)

        is_valid_json = prediction_json is not None

        is_exact_match = False
        if prediction_json is not None and target_json is not None:
            is_exact_match = normalize_json(prediction_json) == normalize_json(target_json)

        if is_valid_json:
            valid_json_count += 1

        if is_exact_match:
            exact_match_count += 1

        if prediction_json is not None and target_json is not None:
            key_score = key_overlap_score(prediction_json, target_json)
        else:
            key_score = 0.0

        key_scores.append(key_score)
        latencies.append(latency)

        examples.append(
            {
                "index": idx,
                "topic": row.get("topic"),
                "medium": row.get("medium"),
                "valid_json": is_valid_json,
                "exact_match": is_exact_match,
                "key_overlap": key_score,
                "latency_seconds": latency,
                "prediction_preview": prediction_raw[:1000],
                "target_preview": target_raw[:1000],
            }
        )

        print(
            f"[{idx + 1}/{len(rows)}] "
            f"valid_json={is_valid_json} "
            f"exact_match={is_exact_match} "
            f"key_overlap={key_score:.3f} "
            f"latency={latency:.2f}s"
        )

    total = len(rows)

    results = {
        "mode": mode,
        "base_model": config["model"]["base_model"],
        "num_examples": total,
        "valid_json_rate": valid_json_count / total if total else 0.0,
        "exact_match_rate": exact_match_count / total if total else 0.0,
        "avg_key_overlap": sum(key_scores) / total if total else 0.0,
        "avg_latency_seconds": sum(latencies) / total if total else 0.0,
        "examples": examples,
    }

    return results


def save_results(results: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {path}")


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

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    results = evaluate(
        config_path=args.config,
        mode=args.mode,
        limit=args.limit,
    )

    print("\nEvaluation Results")
    print("=" * 80)
    print(json.dumps({k: v for k, v in results.items() if k != "examples"}, indent=2))

    if args.output is not None:
        save_results(results, args.output)


if __name__ == "__main__":
    main()