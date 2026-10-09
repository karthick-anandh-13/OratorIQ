import subprocess
import sys
import os
import pathlib

def run_make_target(target: str):
    cwd = pathlib.Path("d:/OratorIQ")
    result = subprocess.run(["make", target], cwd=cwd, capture_output=True, text=True)
    print(f"Running make {target}...")
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"make {target} failed")

if __name__ == "__main__":
    # Install dependencies first via poetry
    subprocess.check_call([sys.executable, "-m", "pip", "install", "poetry==1.8.2"], stdout=subprocess.DEVNULL)
    subprocess.check_call(["poetry", "install", "--no-interaction", "--no-ansi"], cwd="d:/OratorIQ")
    # Run lint and test
    run_make_target("lint")
    run_make_target("test")
