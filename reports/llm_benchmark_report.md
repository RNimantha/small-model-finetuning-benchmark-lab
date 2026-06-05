# LLM Fine-Tuning Benchmark Report

## Task

Structured JSON extraction from unstructured documents.

## Dataset

Dataset: `paraloq/json_data_extraction`

The dataset contains unstructured documents, JSON schemas, and target JSON outputs.

## Models Compared

| Model | Description |
|---|---|
| Base small model | Qwen/Qwen2.5-0.5B-Instruct |
| Fine-tuned small model | Qwen/Qwen2.5-0.5B-Instruct + LoRA/QLoRA adapter |

## Metrics

| Metric | Meaning |
|---|---|
| Valid JSON rate | Percentage of outputs that can be parsed as JSON |
| Exact match rate | Percentage of outputs exactly matching target JSON |
| Key overlap | Percentage of target JSON fields present in prediction |
| Latency | Average generation time per example |

## Results

| Model | Valid JSON Rate | Exact Match Rate | Key Overlap | Avg Latency |
|---|---:|---:|---:|---:|
| Base model | TBD | TBD | TBD | TBD |
| Fine-tuned adapter | TBD | TBD | TBD | TBD |

## Notes

The purpose is not only to maximize accuracy, but to measure whether a small fine-tuned model can provide a better cost/latency/deployment trade-off than larger general-purpose models.