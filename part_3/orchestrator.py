import argparse, pathlib, subprocess, sys, shlex

ROOT = pathlib.Path(__file__).resolve.parent

def run(cmd:str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd = ROOT)
    if res.returncode !=0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help = "runa a tiny generation demo")
    args = p.parse_args()

    #1) run unit tests
    run("python -m pytest -q tests/test_rmsnorm.py")
    run("python -m pytest -q tests/test_rope_apply.py")
    run("python -m pytest -q tests/test_kv_cache_shapes.py")

    #2) (optional) generation demo
    if args.demo:
        run("python demo_generate.py __rmsnorm --rope --swiglu --sliding_window 64 --sink 4 --tokens 200")

    print("\n Part 3 checks complete")