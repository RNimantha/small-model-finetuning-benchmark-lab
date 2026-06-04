import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_instruction(schema: str, document: str) -> str:
    return f"""
You are an information extraction model.

Extract structured JSON from the document below.
Return only valid JSON.
Do not explain your answer.

JSON Schema:
{schema}

Document:
{document}
""".strip()


def build_prompt(schema: str, document: str) -> str:
    instruction = build_instruction(schema=schema, document=document)

    return f"""### Instruction:
{instruction}

### Response:
"""


def load_example(validation_file: str, index: int = 0) -> dict:
    with open(validation_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    if index >= len(lines):
        raise IndexError(f"Index {index} is outside validation file length {len(lines)}")

    return json.loads(lines[index])


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

        print(f"Loading LoRA adapter from: {adapter_path}")
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
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:" in decoded:
        return decoded.split("### Response:", 1)[1].strip()

    return decoded.strip()


def main(config_path: str, mode: str, example_index: int) -> None:
    config = load_config(config_path)

    validation_file = config["dataset"]["validation_file"]
    example = load_example(validation_file, example_index)

    prompt = build_prompt(
        schema=example["schema"],
        document=example["input_text"],
    )

    print("=" * 80)
    print(f"Mode: {mode}")
    print(f"Example index: {example_index}")
    print("=" * 80)

    model, tokenizer = load_model_and_tokenizer(config, mode)

    prediction = generate_response(model, tokenizer, prompt)

    print("\nMODEL PREDICTION:")
    print(prediction)

    print("\nEXPECTED TARGET:")
    print(example["target_json"])


if __name__ == "__main__":
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
        help="Use base model or fine-tuned adapter.",
    )

    parser.add_argument(
        "--example-index",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        mode=args.mode,
        example_index=args.example_index,
    )