"""Train both models, evaluate, plot DAM."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main():
    exp = ROOT / "experiments"
    for script, extra in [
        ("train_ddpm.py", ["--epochs", "25"]),
        ("train_lstm_ae.py", ["--epochs", "25"]),
    ]:
        print(f"\n>>> {script}")
        subprocess.check_call([PY, str(exp / script)] + extra, cwd=ROOT)

    print("\n>>> evaluate.py")
    subprocess.check_call([PY, str(exp / "evaluate.py")], cwd=ROOT)

    print("\n>>> plot_dam.py")
    subprocess.check_call([PY, str(exp / "plot_dam.py")], cwd=ROOT)
    print("\nDone. See results/metrics.json and outputs/dam_example.png")


if __name__ == "__main__":
    main()
