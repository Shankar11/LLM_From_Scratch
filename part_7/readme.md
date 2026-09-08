Repository Layout (Part 7)

    part_7/
        orchestrator.py         #run unit tests + optional tiny RM demo
        data_prefs.py           #7.1 HF preference loader (+tiny fallback)
        collator_rm.py          #pair wise tokenization -> (pos, neg) tensors
        model_reward.py         #7.2 reward model (Transformer encoder  -> scalar)
        loss_reward.py          #7.3 Bradley-Terry & margin-ranking losses
        train_rm.py             #minimal one-GPU training on tiny slice
        eval_rm.py              #7.4 sanity checks & simple accuracy on val
        tests/
            test_bt_loss.py
            test_reward_forward.py

Run from inside part_7/ :
    cd part_7
    python orchestrator.py --demo
    pytest -q