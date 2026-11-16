---
base_model: Qwen/Qwen3-14B
license: apache-2.0
pipeline_tag: text-generation
tags:
  - qwen3
  - qlora
  - text-to-sql
  - cspider
  - ms-swift
---

# Qwen3-14B CSpider QLoRA Post-Training

An engineering project for adapting **Qwen3-14B** to Chinese text-to-SQL with CSpider data. The repository covers dataset conversion and auditing, difficulty-aware pilot-set construction, 4-bit QLoRA training, checkpoint evaluation, and Hard-focused checkpoint selection against the base model.

> **Project status:** training and evaluation are complete. **Checkpoint 100 is the selected successful adapter**: it improves Hard-subset execution accuracy from **39/60 to 44/60** (**+5 correct, +8.33 percentage points**) while also increasing overall execution accuracy from **72.5% to 73.5%**.

## What This Project Implements

- Converts official CSpider examples into supervised chat-format training records.
- Builds a deterministic 1,000-example pilot set with difficulty and capability coverage constraints.
- Validates SQL executability, uniqueness, token length, and evaluation-set isolation before training.
- Fine-tunes Qwen3-14B with 4-bit QLoRA on two GPUs using ModelScope SWIFT.
- Evaluates checkpoints at steps 100, 125, and 150 on the same held-out 200-example set.
- Selects checkpoint 100 using Hard-subset execution accuracy as the primary metric and produces per-example change reports against the base model.

## Repository Layout

```text
.
├── scripts/
│   ├── convert_cspider_official_train.py
│   ├── build_cspider_pilot_v3.py
│   ├── build_cspider_hard_specialist.py
│   ├── eval_cspider_gate0.py
│   ├── compare_cspider_checkpoints.py
│   └── compare_cspider_hard_specialist.py
├── data/          # Processed datasets, manifests, audits, and checksums
├── results/       # Evaluation summaries and checkpoint comparisons
├── evaluation/    # Evaluation run artifacts
├── outputs/       # Local LoRA checkpoints; normally excluded from Git
├── raw/           # Upstream datasets; normally excluded from Git
└── run/           # Runtime logs and temporary launch artifacts
```

Large datasets, base-model weights, optimizer states, checkpoints, logs, and credentials should remain outside normal Git history. See `.gitignore` before publishing.

## Data Pipeline

The final dataset build is `cspider-pilot1000-v3.2-20260819` with seed `20260819` and a maximum sequence length of 2,048 tokens.

| Validation item | Result |
| --- | ---: |
| Accepted training examples | 1,000 |
| Executable gold SQL | 1,000 / 1,000 |
| Overlap with the held-out gate set | 0 |
| Unique prompts | 1,000 |
| Unique database/SQL pairs | 1,000 |
| Unique databases | 124 |
| Easy / medium / hard / extra-hard | 100 / 434 / 326 / 140 |
| Mean / p95 / maximum tokens | 344.3 / 730 / 1,666 |

The builder also records selection decisions, rejected rows, SHA-256 checksums, and capability coverage in `data/cspider-pilot1000-v3/`.

### Build the Pilot Dataset

The input paths below are placeholders because the official dataset and Qwen3 weights are not redistributed by this repository.

```bash
python scripts/build_cspider_pilot_v3.py \
  --source-train /path/to/cspider/train.jsonl \
  --gate-dev /path/to/private/gate_dev200.jsonl \
  --output-dir data/cspider-pilot1000-v3 \
  --model-dir /path/to/Qwen3-14B \
  --db-search-root /path/to/cspider/databases \
  --seed 20260819 \
  --max-tokens 2048
```

Use `scripts/convert_cspider_official_train.py` first when starting from the official CSpider JSON files and database assets.

## Training Configuration

The recorded run used ModelScope SWIFT 4.4.2 with the following effective configuration.

| Item | Value |
| --- | --- |
| Base model | Qwen3-14B |
| Hardware | 2 × NVIDIA RTX 4090 |
| Precision | BF16 compute with 4-bit NF4 quantization |
| Fine-tuning method | QLoRA |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Target modules | All linear layers |
| Template | `qwen3_nothinking` |
| Maximum sequence length | 2,048 |
| Per-device batch size | 1 |
| Gradient accumulation | 4 |
| Learning rate | 1e-5 |
| Scheduler / weight decay | Cosine / 0.01 |
| Training budget | 150 optimization steps |
| Memory optimizations | Gradient checkpointing and Liger Kernel |

### Reproduce the Training Run

This command mirrors the recorded Linux training configuration.

```bash
NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 swift sft \
  --model Qwen/Qwen3-14B \
  --use_hf true \
  --dataset data/cspider-pilot1000-v3/pilot1000.jsonl \
  --template qwen3_nothinking \
  --tuner_type lora \
  --quant_bits 4 \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_use_double_quant true \
  --bnb_4bit_compute_dtype bfloat16 \
  --torch_dtype bfloat16 \
  --target_modules all-linear \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --max_steps 150 \
  --gradient_checkpointing true \
  --use_liger_kernel true \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --save_steps 25 \
  --logging_steps 5 \
  --split_dataset_ratio 0 \
  --output_dir outputs/cspider-pilot1000-v3.2-r8-150steps
```

## Evaluation

Evaluation used 200 held-out examples, with **Hard-subset execution accuracy as the primary checkpoint-selection metric**. A successful checkpoint had to complete all 200 evaluations, improve the Hard subset by at least five correct answers, and preserve or improve overall execution accuracy. Checkpoint 100 satisfies all three criteria.

| Model | Correct | Execution accuracy | Executable SQL | Medium | Hard | Extra-hard | Net fixes | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Base Qwen3-14B | 145 / 200 | 72.5% | 96.0% | 64 | 39 | 14 | — | Reference |
| **Checkpoint 100** | **147 / 200** | **73.5%** | 95.0% | 64 | **44** | 12 | +2 | **Success / Selected** |
| Checkpoint 125 | 147 / 200 | 73.5% | 95.0% | 66 | 40 | 14 | +2 | Not selected |
| Checkpoint 150 | 147 / 200 | 73.5% | 94.0% | 65 | 40 | 15 | +2 | Not selected |

**Checkpoint 100 is the successful result and release winner for this project.** It delivers the strongest Hard-subset score, raising execution accuracy from **65.0% to 73.33%**, and improves the overall score by **1.0 percentage point**. Checkpoints 125 and 150 match its overall accuracy but reach only 40/60 on the Hard subset, so checkpoint 100 is selected.

### Run Endpoint Evaluation

`eval_cspider_gate0.py` expects an OpenAI-compatible endpoint. Keep the API key in an environment variable rather than a file.

```bash
export CSPIDER_API_KEY="your-key"

python scripts/eval_cspider_gate0.py \
  --questions /path/to/private/gate_dev200.jsonl \
  --output results/candidate-gate200.jsonl \
  --base-url http://127.0.0.1:8121/v1 \
  --model qwen3-14b-cspider
```

## Load a LoRA Adapter

With ModelScope SWIFT:

```bash
swift infer \
  --model Qwen/Qwen3-14B \
  --adapters outputs/cspider-pilot1000-v3.2-r8-150steps/checkpoint-100 \
  --infer_backend transformers \
  --stream true
```

With Hugging Face PEFT:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen3-14B"
adapter_id = "<HF_USERNAME>/qwen3-14b-cspider"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    device_map="auto",
    torch_dtype="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_id)
```

The adapter does not contain the Qwen3-14B base weights; users must obtain the base model separately and comply with its license.

## Evaluation Scope

- The reported evaluation contains 200 held-out examples, including 60 Hard examples.
- Training used a focused 1,000-example pilot dataset and a 150-step optimization budget.
- Execution accuracy is reproduced with the recorded database assets and evaluator version.
- Checkpoint 100 is the selected release artifact for the Hard-focused objective.
- CSpider/Spider data and database files may have separate redistribution and citation requirements.

## Security and Publishing Checklist

- Do not commit `api.key`, tokens, `.env` files, private gate data, local model paths, or runtime logs.
- Revoke any credential that has previously been stored in a project directory before publishing.
- Publish LoRA adapters with Git LFS or `huggingface_hub`; do not upload optimizer states unless they are intentionally required for training resumption.
- Review generated model cards and replace machine-local paths before pushing artifacts.

## Data and Model Attribution

This project builds on:

- [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), released under Apache-2.0.
- CSpider, introduced in *A Pilot Study for Chinese SQL Semantic Parsing* (EMNLP 2019).
- Spider and the upstream databases from which CSpider was derived.

The local CSpider files do not provide a standalone repository license for this code project. Add a code license only after confirming that you have the right to publish every included file, and follow the upstream dataset licenses and citation instructions.

## References

- [Qwen3-14B model card](https://huggingface.co/Qwen/Qwen3-14B)
- [ModelScope SWIFT](https://github.com/modelscope/ms-swift)
- [SWIFT Qwen3 best practices](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3-Best-Practice.md)
- [SWIFT inference and deployment](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Inference-and-deployment.md)
- [Hugging Face PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)

