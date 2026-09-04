Repository layout (Part5)

    part_5/
        orchestrator.py # run unit tests + optional MoE demo
        about.md #5.1/ 5.3 concept notes (compact MD)
        gatting.py  #router/gating (top-k) + load-balancing aux loss
        experts.py # MLP experts (SwiGLU or GELU)
        moe.py #Mixture-of-experts layer (dispatch/combine)
        block_hybrid.py # Hybrid dense+MoE block examples
        demo_moe.py #small forward pass demo + routing histogram
        tests/
            test_gate_shapes.py
            test_moe_forward.py
            test_hybrid_block.py

RUn from inside part_5:
    cd part_5
    python orchestrator.py --demo
    pytest - q