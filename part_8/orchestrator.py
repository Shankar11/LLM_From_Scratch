import argparse , pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd: str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(cmd.split(), cwd=ROOT)
    if res.returncode !=0:
        sys.exit(res.returncode)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action='store_true', help="tiny PPO demo")
    args = p.parse_args()

    #1) unit tests
    run("python -m pytest -q tests/test_ppo_poss.py")
    run("python -m pytest -q tests/test_policy_forward.py")

    #2) optional demo(required SFT+RM checkpoints from parts 6 & 7)
    if args.demo:
        # run("python train_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 10 --batch_size 4 --resp_len 128 --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        # run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

        # run("python train_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 50 --batch_size 4 --resp_len 128 --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        # run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

        run("python train_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 100 --batch_size 4 --resp_len 128 --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        run("python eval_ppo.py --policy_ckpt runs/ppo-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

    print("\n Part 8 checks complete.")