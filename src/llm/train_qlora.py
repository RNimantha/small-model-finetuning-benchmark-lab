import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main(config_path: str) -> None:
    config = load_config(config_path)

    seed = config["training"]["seed"]
    set_seed(seed)

    model_name = config["model"]["base_model"]
    output_dir = config["model"]["output_dir"]

    train_file = config["dataset"]["train_file"]
    validation_file = config["dataset"]["validation_file"]
    text_field = config["dataset"]["text_field"]
    max_length = config["dataset"]["max_length"]

    print(f"Loading dataset from: {train_file}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": validation_file,
        },
    )

    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.config.use_cache = False

    peft_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config["lora"]["target_modules"],
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field=text_field,
        max_length=max_length,
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["epochs"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        fp16=config["training"]["fp16"],
        report_to="none",
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    final_dir = Path(output_dir) / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"Saved final adapter to: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/llm_qlora.yaml",
        help="Path to training config YAML file",
    )

    args = parser.parse_args()
    main(args.config)