# Repository Layout (Part 4)

```text
part_4/
├── orchestrator.py             # Run unit tests + optional smoke train & sample
├── tokenizer_bpe.py            # 4.1 BPE tokenization (train/save/load)
├── dataset_bpe.py              # Streaming dataset + batching & label shift
├── lr_scheduler.py             # 4.3 Warmup + cosine decay scheduler
├── amp_accum.py                # 4.2 AMP (autocast + GradScaler) + grad accumulation helpers
├── checkpointing.py            # 4.4 Save/resume (model/opt/scaler/scheduler/tokenizer)
├── logger.py                   # 4.5 Logging backends (W&B / TensorBoard / noop)
├── train.py                    # Core training loop (no Trainer API)
├── sample.py                   # Load checkpoint & generate text
└── tests/
    ├── test_tokenizer_bpe.py
    ├── test_scheduler.py
    └── test_resume_shapes.py
```

## Running Part 4

Run the following commands from inside the `part_4/` directory:

```bash
cd part_4
```

### Run the Smoke Demo

Runs a tiny smoke-training run on `../tiny.txt`:

```bash
python orchestrator.py --demo
```

### Run the Test Suite

Run all unit tests with:

```bash
pytest -q
```

### Launch TensorBoard

To view the training logs:

```bash
tensorboard --logdir=runs/part4-demo
```
