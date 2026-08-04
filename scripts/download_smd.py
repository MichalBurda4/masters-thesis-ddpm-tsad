"""Download one SMD machine for local experiments."""

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
MACHINE = "machine-1-1"
FILES = [
    f"train/{MACHINE}.txt",
    f"test/{MACHINE}.txt",
    f"test_label/{MACHINE}.txt",
]


def download(root: Path, machine: str = MACHINE) -> None:
    root = Path(root)
    for rel in FILES:
        rel = rel.replace(MACHINE, machine)
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE}/{rel}"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)
    print(f"Done: {root}")


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    out = project / "data" / "SMD"
    machine = sys.argv[1] if len(sys.argv) > 1 else MACHINE
    download(out, machine)
