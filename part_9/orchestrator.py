import argparse, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd: str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(cmd.split(), cwd=ROOT)
    if res.returncode !=0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="tiney GRPO demo")
    args = p.parse_args()

    #1 unit tests
    run("python -m pytest -q tests/test_grpo_loss.py")

    #2 optional demo (requires SFT+RM checkpoints from Parts 6 & 7)
    if args.demo:
        run("python train_grpo.py --group_size 4 --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 200" \
        " --batch_prompts 4 --resp_len 128 --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        run("python eval_ppo.py --policy_ckpt runs/grpo-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4-demo/tokenizer")

    print("\nPart 9 checks complete")