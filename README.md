# Small Model Fine-Tuning & Benchmarking Lab

This project demonstrates how to fine-tune small, efficient models and benchmark them against larger baseline models.

The goal is to evaluate whether smaller fine-tuned models can achieve acceptable task performance with lower cost, lower latency, and easier deployment.

## Current Scope

### Phase 1: LLM Fine-Tuning

- Fine-tune a small instruction model using LoRA / QLoRA
- Benchmark base model vs fine-tuned model
- Compare against a larger baseline model
- Produce reproducible training configs and evaluation reports

### Future Scope

- Speech transcription fine-tuning and WER/CER benchmarking
- Video understanding benchmarking
- GCP training job templates
- Quantized deployment experiments

## Tech Stack

- Python
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- TRL
- bitsandbytes
- Accelerate
- PyTorch

## Phase 1 Task

The first task is instruction fine-tuning for structured output generation.

Example use cases:

- Text-to-SQL
- JSON extraction
- Customer ticket classification
- Domain-specific instruction following

## Project Structure

```text
configs/        Training and benchmark configs
data/           Dataset documentation and samples
docs/           Methodology and architecture notes
notebooks/      Experiment notebooks
reports/        Benchmark reports
src/            Training, inference, and evaluation code
tests/          Basic tests


# Data Documentation

## Phase 1 Dataset Candidate

Dataset: `paraloq/json_data_extraction`

Purpose:
This dataset is used for structured JSON extraction from unstructured text.

Why selected:
- Suitable for supervised fine-tuning
- Easy to evaluate using JSON validity and field-level accuracy
- Practical business use case
- Useful for comparing small fine-tuned models against larger baselines

Planned evaluation:
- Valid JSON rate
- Schema-following accuracy
- Field-level exact match
- Full-output exact match
- Latency and inference cost

Notes:
Before production use, dataset license, provenance, and redistribution terms must be reviewed carefully.