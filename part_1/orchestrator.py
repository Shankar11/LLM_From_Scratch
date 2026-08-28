import subprocess, pathlib, argparse, shlex

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "out"

def run(cmd:str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode !=0:
        SystemExit(res.returncode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visualize", action="store_true", help = "run visualization scripts and save PNGs to ./out")
    args =p.parse_args()

    OUT.mkdir(exist_ok=True)

    #1.2 sanity check: NumPy tiny example
    run("python attn_numpy_demo.py")

    #1.3/1.4 unit tests
    run("python -m pytest -q tests/test_attn_math.py")
    run("python -m pytest -q tests/test_causal_mask.py")

    #Matrix math walkthrough for MHA 
    run("python demo_mha_shapes.py")

    if args.visualize:
        run("python demo_visualize_multi_head.py")
        print(f"\n Visualization images saved to: {OUT} ")

    print("\nAll Part1 demos/test completed")


if __name__ == "__main__":
    main()

    