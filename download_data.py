"""
Clones jjpn2/eval_awareness from HuggingFace, decrypts the dataset, and copies
eval_awareness_val.json into the project root.

Prerequisites (one-time):
  sudo apt install git-lfs        # macOS: brew install git-lfs
  git lfs install
  pip install --upgrade huggingface_hub
  huggingface-cli login           # paste your HF access token

Usage:
  uv run download_data.py [--clone-dir DIR]

The dataset repo is cloned to a sibling directory (default: ../eval_awareness_data),
then scripts/decrypt.sh is run inside it to produce dataset.json.
The validation split is then copied here as eval_awareness_val.json.
"""
import argparse
import os
import shutil
import subprocess
import sys


REPO_URL = "https://huggingface.co/datasets/jjpn2/eval_awareness"
DEFAULT_CLONE_DIR = "../eval_awareness_data"
OUT_FILE = "eval_awareness_val.json"


def run(cmd: list[str], cwd: str | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clone-dir", default=DEFAULT_CLONE_DIR, help="Where to clone the dataset repo")
    args = parser.parse_args()

    clone_dir = os.path.abspath(args.clone_dir)

    # Clone (or pull if already exists)
    if not os.path.exists(clone_dir):
        print(f"Cloning {REPO_URL} -> {clone_dir}")
        run(["git", "clone", REPO_URL, clone_dir])
    else:
        print(f"Repo already exists at {clone_dir}, pulling latest ...")
        run(["git", "pull"], cwd=clone_dir)

    # Run the upstream decrypt script
    decrypt_script = os.path.join(clone_dir, "scripts", "decrypt.sh")
    if not os.path.exists(decrypt_script):
        print(f"Error: decrypt script not found at {decrypt_script}")
        print("Check that git-lfs pulled all files: run 'git lfs pull' inside the cloned repo.")
        sys.exit(1)

    print("Decrypting dataset ...")
    run(["bash", "scripts/decrypt.sh"], cwd=clone_dir)

    # Find the decrypted file — upstream produces dataset.json; val split may be separate
    candidates = ["eval_awareness_val.json", "dataset.json"]
    src = None
    for name in candidates:
        path = os.path.join(clone_dir, name)
        if os.path.exists(path):
            src = path
            break

    if src is None:
        print(f"Error: could not find decrypted file in {clone_dir}.")
        print(f"Expected one of: {candidates}")
        print("Check the output of scripts/decrypt.sh manually.")
        sys.exit(1)

    shutil.copy(src, OUT_FILE)
    print(f"Copied {src} -> {OUT_FILE}")


if __name__ == "__main__":
    main()
