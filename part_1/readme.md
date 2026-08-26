
---

# Transformer From Scratch (`part_1`)

A clean, modular, and educational implementation of Transformer components built from the ground up using NumPy and PyTorch. This codebase is designed to demystify the inner mechanics of transformers—from positional encodings and attention mechanisms to full transformer blocks—complete with shape tracing, visualizations, and tests.

---

## 📂 Repository Layout

```text
part_1/
│
├── orchestrator.py             # Runs demos, tests, and visualizations for Part 1
├── pos_encoding.py             # 1.1 Positional encodings (learned + sinusoidal)
├── attn_numpy_demo.py          # 1.2 Self-attention match with tiny numbers (NumPy)
├── single_head.py              # 1.3 Single attention head (PyTorch)
├── multi_head.py               # 1.4 Multi-head attention (with shape tracing)
├── ffn.py                      # 1.5 Feed-forward network (GELU, width = mult * d_model)
├── block.py                    # 1.6 Transformer block (residuals + LayerNorm)
├── attn_mask.py                # Causal mask helpers
├── vis_utils.py                # Plotting helpers (matrices & attention maps)
├── demo_mha_shapes.py          # Prints explicit matrix multiplications & shapes step-by-step
├── demo_visualize_multi_head.py # Saves attention heatmaps per head (grid)
│
├── out/                        # Created at runtime - images and logs live here
│
└── tests/                      
    ├── test_attn_math.py       # Correctness: tiny example vs. PyTorch single head
    └── test_causal_math.py     # Verifies causal masking behavior

```

---

## ⚡ Quick Start

The codebase uses local imports. To run the scripts correctly, execute commands from inside the `part_1/` directory.

### Prerequisites

* Python 3.8+
* PyTorch
* NumPy
* Matplotlib (for visualizations)

### Running Demos

You can run the full suite of demos, tests, and visualizations using the orchestrator:

```bash
cd part_1
python orchestrator.py --visualize

```

---

## 📝 Note on Imports

All imports within the scripts are relative/local to `part_1/`. Always ensure your terminal working directory is inside `part_1/` before running any module or test directly.

---
