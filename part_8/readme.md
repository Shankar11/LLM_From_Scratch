Repository layout (part 8 - RLHF with PPO)

    part_8/
    orchestrator.py     #run unit tests + optional tiny ppo demo
    policy.py           #policy = SFT LM + value head(toy head on logits)
    rollout.py          # prompt formatting, sampling, logprobs/KL utilities
    ppo_loss.py         #PPO clipped objective + value +entropy + KL penalty
    train_ppo.py        # single-GPU RLHF loop (tiny, on-policy)
    eval_ppo.py         #compare reward vs. reference on a small set
    tests/
        test_ppo_loss.py
        test_policy_forward.py

Run from inside 'part_8/':
    cd part_8
    python orchestrator.py --demo
    pytest -q
