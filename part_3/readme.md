## Repository Layout — Part 3

```text
part_3/
├── orchestrator.py        # Runs tests + a small generation demo
├── tokenizer.py           # Local byte-level tokenizer (self-contained)
├── rmsnorm.py             # 3.1 RMSNorm
├── rope.py                # 3.2 RoPE cache + apply
├── swiglu.py              # 3.3 SwiGLU FFN
├── kv_cache.py            # 3.4/3.6 KV Cache + rolling buffer
├── attn_modern.py         # Attention with RoPE, sliding window, sink, optional KV cache
├── block_modern.py        # Block = (RMSNorm/LN) + modern attention + (SwiGLU/GELU)
├── model_modern.py        # GPTModern wrapper with feature flags
├── demo_generate.py       # Simple generation demo (KV cache + sliding window)
└── tests/
    ├── test_rmsnorm.py
    ├── test_rope_apply.py
    └── test_kvcache_shapes.py
```

### Running Part 3

Run the following commands from inside the `part_3/` directory:

```bash
cd part_3
```

Run the smoke tests and generation demo:

```bash
python orchestrator.py --demo
```

Run the test suite:

```bash
pytest -q
```

### Quick Start

```bash
cd part_3

# Run tests + generation demo
python orchestrator.py --demo

# Run all tests
pytest -q
```