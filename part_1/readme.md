Repository Layout

part_1/
orchestrator.py     #runs demos/test/visualizations for part 1
pos_encoding.py     #1.1 positional encodings (learned + sinusoidals)
attn_numpy_demo.py  #1.2 self attention match with tiny numbers(NumPy)
single_head.py      #1.3 single attention head (PyTorch)
multi_head.py       #1.4 multi_head attention  (with shape tracing)
ffn.py              #1.5 feed-forward network (GELU, width =  mult*d_model)
block.py            #1.6 Transformer block (residuals + LayerNorm)
attn_mask.py        # causal mask helpers
vis_utils.py        # plotting helpers (matrices & attnetion maps)
demo_mha_shapes.py  # prints explicit matrix multiplications & shapes step-by-step
demo_visualize_multi_head.py    #saves attention heatmaps per head (grid)
out/                # created at runtime - images & logs live here
tests/              # 
    test_attn_math.py           # correctness: tiny example vs PyTorch single head
    test_causal_math.py         # verifies masking behaviour

Note on Imports
All imports are local. RUn from inside part_1/
Example quick start cpu ok
cd  part_1
python orchestrator.py --visualize