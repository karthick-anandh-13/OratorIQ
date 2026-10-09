import subprocess
import sys
import os

# Simple script to run after repo setup to ensure environment works

def run_poetry_install():
    print("Installing dependencies via Poetry...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "poetry==1.8.2"], stdout=subprocess.DEVNULL)
    subprocess.check_call([sys.executable, "-m", "poetry", "install", "--no-interaction", "--no-ansi"], cwd="d:/OratorIQ")

if __name__ == "__main__":
    run_poetry_install()
