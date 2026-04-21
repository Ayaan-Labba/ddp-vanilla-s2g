# Vanilla S2G — Automatic Knowledge Graph Generation from Unstructured Text

A seq2seq approach to joint entity and relation extraction using a linearised **Structured Extraction Language (SEL)** and a **Structural Schema Instructor (SSI)** prefix. Built on **Flan-T5 Large** (~780M parameters) with pre-training on [REBEL](https://huggingface.co/datasets/Babelscape/rebel-dataset) and fine-tuning on standard relation-extraction benchmarks.

## Architecture Overview

The model treats knowledge-graph extraction as a text-to-text problem. The encoder receives a sentence prefixed by an SSI that enumerates the relation types in scope. The decoder produces a flat SEL string that encodes all entities, their pairwise relations, and explicit rejections of schema types absent from the text. At inference time, a finite-state machine (FSM) constrains decoding to produce only valid SEL expressions.

## Project Structure

```
vanilla_s2g/
├── configs/                    # YAML configuration files
│   ├── pretrain.yaml           #   Pre-training hyperparameters
│   ├── finetune.yaml           #   Fine-tuning hyperparameters (placeholder)
│   └── sweep_pretrain.yaml     #   W&B sweep definition
│
├── linearisation/              # Core linearisation module
│   ├── special_tokens.py       #   Token registry (6 structural markers)
│   ├── ssi.py                  #   SSI prefix construction
│   └── sel.py                  #   SEL construction and parsing
│
├── data/                       # Data loading and collation
│   ├── preprocess_rebel.py     #   REBEL dataset preprocessing script
│   ├── dataset.py              #   PyTorch Dataset wrapper
│   └── collator.py             #   Dynamic SSI collator with type sampling
│
├── model/                      # Model management
│   ├── model.py                #   S2GModel wrapper
│   └── constraint_decoder.py   #   FSM-based constrained decoding
│
├── evaluation/                 # Metrics and callbacks
│   ├── metrics.py              #   P/R/F1 at boundary and strict levels
│   └── callbacks.py            #   Step tracking, sample generation, checkpointing
│
├── scripts/                    # Entry-point scripts
│   ├── config_utils.py         #   YAML + CLI config loader
│   ├── pretrain.py             #   Pre-training on REBEL
│   ├── evaluate.py             #   Evaluation with constrained generation
│   ├── inference.py            #   Interactive and batch triplet extraction
│   └── finetune.py             #   Fine-tuning (placeholder)
│
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone and install dependencies.
pip install -r requirements.txt

# Download NLTK tokeniser data.
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Quick Start

### 1. Preprocess the REBEL Dataset

```bash
python -m vanilla_s2g.data.preprocess_rebel \
    --output_dir data/rebel \
    --top_k 220
```

This downloads the REBEL dataset from HuggingFace, filters to the top 220 relation types, and writes `train.jsonl`, `val.jsonl`, `test.jsonl`, and `relation.schema` to `data/rebel/`.

### 2. Pre-train

```bash
python -m vanilla_s2g.scripts.pretrain \
    --config configs/pretrain.yaml \
    --gpu_ids 0,1,2,3
```

Key configuration options (override via CLI):

| Flag | Default | Description |
|------|---------|-------------|
| `--lr` | `5e-5` | Learning rate |
| `--train_batch_size` | `4` | Per-device batch size |
| `--gradient_acc_steps` | `8` | Gradient accumulation (effective batch = 32) |
| `--max_steps` | `1000000` | Total training steps |
| `--gpu_ids` | `null` | GPU indices, e.g. `0,1,7` |
| `--positive_rate` | `0.9` | Bernoulli rate for positive type inclusion |
| `--negative_rate` | `0.1` | Bernoulli rate for negative type sampling |

### 3. Resume After Interruption

If training is interrupted (e.g., SSH disconnection), resume from the last checkpoint:

```bash
python -m vanilla_s2g.scripts.pretrain \
    --config configs/pretrain.yaml \
    --resume_from outputs/pretrain/checkpoint-last
```

This restores the full training state: model weights, optimiser, scheduler, global step, RNG seeds, and W&B run continuity. Safety-net checkpoints are saved every 5,000 steps by default (configurable via `--checkpoint_every_n_steps`).

### 4. Evaluate

```bash
python -m vanilla_s2g.scripts.evaluate \
    --checkpoint outputs/pretrain/best_model \
    --data_dir data/rebel \
    --schema_file data/rebel/relation.schema \
    --split test \
    --output_dir outputs/pretrain/eval \
    --constraint_decoding true
```

This produces four output files: `test_out.jsonl` (generated SEL), `test_preds.jsonl` (parsed entities and relations), `test_results.jsonl` (gold vs predicted), and `test_metrics.json` (all metrics).

### 5. Interactive Inference

```bash
python -m vanilla_s2g.scripts.inference \
    --checkpoint outputs/pretrain/best_model \
    --schema_file data/rebel/relation.schema
```

Type sentences at the prompt to see extracted triplets in real time.

## Hyperparameter Tuning with W&B

```bash
# Create a sweep.
wandb sweep configs/sweep_pretrain.yaml

# Launch an agent (repeat on multiple machines for parallel search).
wandb agent <sweep_id>
```

The sweep configuration defines a Bayesian search over learning rate, batch size, gradient accumulation, sampling rates, and other structural choices. Hyperband early termination stops unpromising runs after 50,000 steps.

## Monitoring

Training logs the following to Weights & Biases:

| Metric | Frequency |
|--------|-----------|
| Training loss | Every 100 steps |
| Learning rate | Every 100 steps |
| Validation loss, P, R, F1 | Every 10,000 steps |
| Sample predictions table | Every 50,000 steps |
| Checkpoint save | Every 5,000 steps (safety net) |