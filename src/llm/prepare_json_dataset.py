import json
from datasets import load_dataset


DATASET_NAME = "paraloq/json_data_extraction"
OUTPUT_TRAIN = "data/sample/train.jsonl"
OUTPUT_VALIDATION = "data/sample/validation.jsonl"


def build_prompt(example: dict) -> str:
    """
    Convert one dataset row into instruction fine-tuning text.
    The model will learn:
    input text + schema -> target JSON output
    """

    instruction = f"""
You are an information extraction model.

Extract structured JSON from the document below.
Return only valid JSON.
Do not explain your answer.

JSON Schema:
{example["schema"]}

Document:
{example["text"]}
""".strip()

    response = example["item"]

    return f"""### Instruction:
{instruction}

### Response:
{response}"""


def main():
    dataset = load_dataset(DATASET_NAME)["train"]

    # Create train/validation split because original dataset only has train
    split = dataset.train_test_split(test_size=0.15, seed=42)

    train_dataset = split["train"]
    validation_dataset = split["test"]

    print(f"Train rows: {len(train_dataset)}")
    print(f"Validation rows: {len(validation_dataset)}")

    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for example in train_dataset:
            record = {
                "text": build_prompt(example),
                "input_text": example["text"],
                "schema": example["schema"],
                "target_json": example["item"],
                "topic": example["topic"],
                "medium": example["medium"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(OUTPUT_VALIDATION, "w", encoding="utf-8") as f:
        for example in validation_dataset:
            record = {
                "text": build_prompt(example),
                "input_text": example["text"],
                "schema": example["schema"],
                "target_json": example["item"],
                "topic": example["topic"],
                "medium": example["medium"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved train file to: {OUTPUT_TRAIN}")
    print(f"Saved validation file to: {OUTPUT_VALIDATION}")


if __name__ == "__main__":
    main()