"""One-shot: train (short) + evaluate + plot."""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).parent
    py = sys.executable

    print("=== 1/3 Training (15 epochs, CPU-friendly) ===")
    subprocess.check_call([py, str(root / "train.py"), "--epochs", "15", "--device", "cpu"], cwd=root)

    ckpt = root / "checkpoints" / "ddpm_prototype.pt"
    print("\n=== 2/3 Evaluation ===")
    subprocess.check_call([py, str(root / "detect.py"), "--ckpt", str(ckpt)], cwd=root)

    print("\n=== 3/3 Visualization ===")
    subprocess.check_call([py, str(root / "visualize.py"), "--ckpt", str(ckpt)], cwd=root)
    print("\nDone. Open prototype/outputs/example.png")


if __name__ == "__main__":
    main()
