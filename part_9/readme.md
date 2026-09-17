Repository layout (Part 9 - RLHF with GRPO)

    part_9/
        orchestrator.py         #run unit tests + optional tiny PPO demo
        policy.py               #policy = SFT LM + value head(toy head on logits)
        rollout.py              #prompt formatting, sampling, logprobs/KL utilitis
        grpo_loss.py            #PPO clipped objective + value + entropy + KL penalty
        train_ppo.py            #singlr-GPU RLHF loop (tiny, on-policy)
        eval_ppo.py             #compare reward vs. reference on a small set
        tests/
            test_ppo_loss.py
            test_policy_forward.py

Run from inside `part_9`:
    cd part_8
    python orchestrator.py --demo
    pytest -q