```markdown
## Repository Layout (Part 6)

```text
part_6/
├── orchestrator.py         # run unit tests + optional tiny SFT demo
├── formatters.py           # 6.1 prompt/response templates
├── dataset_sft.py          # HF dataset loaded (+tiny fallback) -> (prompt, response)
├── collator_sft.py         # 6.2 causal LM labels with masking
├── curriculum.py           # 6.3 length-based curriculum sampler
├── evaluate.py             # 6.4 simple exact/F1 metrics
├── train_sft.py            # minimal one-GPU SFT loop (few steps)
├── sample_sft.py           # load ckpt & generate from instructions
└── tests/
    ├── test_formatter.py
    └── test_masking.py

```

## Running the Code

Run from inside `part_6/`:

```bash
cd part_6
python orchestrator.py --demo
pytest -q

```

```

```