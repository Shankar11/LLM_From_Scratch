import argparse , pathlib, subprocess, sys, shlex

ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd:str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode !=0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action='store_true', help= " run a tiny MoW demo")
    args = p.parse_args()

    # 1. unit tests
    run("python -m pytest -q tests/test_gate_shapes.py")
    run("python -m pytest -q tests/test_moe_forwar.py")
    run("python -m pytest -q tests/test_hybrid_block.py")

    # 2. optional demo
    if args.demo:
        run("python demo_moe.py --tokens 6 --hidden 128 --experts 4 --top_k 1")

    print("\n part5 checks complete")