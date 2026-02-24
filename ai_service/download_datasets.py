"""
Download skin disease datasets for training.
Run: python download_datasets.py [--ham] [--mendeley] [--out-dir data]
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def run_cmd(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    return r.returncode == 0


def download_ham(out_dir: Path):
    """Download HAM10000 from Kaggle (requires kaggle CLI + API key)."""
    out = out_dir / "HAM10000"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "HAM10000_metadata.csv").exists():
        print("HAM10000 already present.")
        return True
    if run_cmd("kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p " + str(out)):
        for z in out.glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
            z.unlink()
        print("HAM10000 downloaded.")
        return True
    print(
        "Kaggle CLI not configured. Install: pip install kaggle\n"
        "Add ~/.kaggle/kaggle.json with your API key.\n"
        "Or download manually: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000"
    )
    return False


def download_mendeley(out_dir: Path):
    """Instructions for Mendeley (acne, psoriasis, hyperpigmentation)."""
    out = out_dir / "mendeley_skin"
    out.mkdir(parents=True, exist_ok=True)
    readme = out / "README.txt"
    readme.write_text(
        "Mendeley Skin Disease Dataset (acne, vitiligo, hyperpigmentation, nail-psoriasis, SJS-TEN)\n"
        "Download: https://data.mendeley.com/datasets/3hckgznc67/1\n"
        "Click 'Download All', extract here. Folders: acne, vitiligo, hyperpigmentation, nail_psoriasis, SJS-TEN\n"
        "Then run: python train_full.py --mendeley-dir " + str(out),
        encoding="utf-8",
    )
    print(f"See {readme} for Mendeley download instructions.")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ham", action="store_true", help="Download HAM10000")
    p.add_argument("--mendeley", action="store_true", help="Print Mendeley instructions")
    p.add_argument("--out-dir", default="data", help="Output directory")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.ham:
        download_ham(out)
    if args.mendeley:
        download_mendeley(out)
    if not args.ham and not args.mendeley:
        print("Use --ham and/or --mendeley. Example: python download_datasets.py --ham --out-dir data")


if __name__ == "__main__":
    main()
