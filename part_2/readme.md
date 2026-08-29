# Part 2 — Tiny GPT From Scratch

This directory contains the implementation for **Part 2**, building a small GPT-style language model from scratch using PyTorch.

## Repository Layout

```text
part_2/
├── orchestrator.py              # Runs quick smoke train + eval + sample
├── tokenizer.py                 # 2.1 Byte-level tokenizer (0–255)
├── dataset.py                   # 2.2 Dataset, batching, and label shifting
├── utils.py                     # 2.3 Sampling helpers (top-k / top-p)
├── model_gpt.py                 # Tiny GPT: token/position embeddings + blocks + head
├── train.py                     # 2.3/2.4 Training loop with validation and checkpoints
├── sample.py                    # 2.5 Text generation from a checkpoint
├── eval_loss.py                 # 2.6 Evaluate loss on a file/checkpoint
├── tests/
│   ├── test_tokenizer.py        # Tokenizer encode/decode round-trip tests
│   └── test_dataset_shift.py    # Label-shift sanity tests
└── runs/                        # Created at runtime: checkpoints and logs
```

## Note on Imports

All imports are **local**.

Run commands from inside the `part_2/` directory:

```bash
cd part_2
```

## Quick Start

The following example is **CPU-compatible**.

### 1. Train the Model

```bash
python train.py --data tiny.txt --steps 300 --sample_every 100
```

This trains the model for 300 steps and generates a sample every 100 steps.

Checkpoints and logs are written to:

```text
runs/min-gpt/
```

### 2. Generate Text

Once training is complete, generate text from the best checkpoint:

```bash
python sample.py \
    --ckpt runs/min-gpt/model_best.pt \
    --tokens 200 \
    --prompt "Once upon a time"
```

## Tests

Run the test suite from inside `part_2/`:

```bash
pytest tests/
```

The tests cover:

* Tokenizer encode/decode round trips
* Dataset input/label shifting

## Runtime Output

The `runs/` directory is created automatically during training and may contain:

```text
runs/
└── min-gpt/
    ├── model_best.pt
    └── ...
```

The `runs/` directory contains generated checkpoints and logs and does not need to exist before starting training.
