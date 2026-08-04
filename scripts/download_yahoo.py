"""Download Yahoo S5 A1 benchmark (univariate, labeled).

Official source: Yahoo Webscope (requires agreement):
  https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70

This script downloads a public mirror of one A1 series (real_1.csv) for local
benchmarking. Format: timestamp, value, is_anomaly.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

# Fallback mirrors (first success wins)
URLS = [
    (
        "harris0704/nbaData16-17",
        "https://raw.githubusercontent.com/harris0704/nbaData16-17/master/Yahoo_S5_Data/A1Benchmark/real_1.csv",
    ),
    (
        "aadarshquant/DeepAnT",
        "https://raw.githubusercontent.com/aadarshquant/DeepAnT/master/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv",
    ),
]


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "tsad-magisterka/1.0 (research dataset download)"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())


def _validate(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    head = text.splitlines()[0].lower()
    if "value" not in head or "anomaly" not in head:
        raise ValueError(f"Unexpected CSV header in {path}: {head!r}")
    if len(text.splitlines()) < 10:
        raise ValueError(f"Downloaded file too small: {path}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dest = root / "data" / "yahoo" / "A1Benchmark.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for name, url in URLS:
        print(f"Trying {name}: {url}")
        try:
            _download(url, dest)
            _validate(dest)
            print(f"Saved: {dest} ({dest.stat().st_size // 1024} KB)")
            return
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as e:
            print(f"  Failed: {e}")
            last_err = e

    raise SystemExit(
        "Could not download Yahoo A1 benchmark from any mirror.\n"
        "Request the official Yahoo S5 dataset via Webscope and place a CSV at:\n"
        f"  {dest}\n"
        f"Last error: {last_err}"
    )


if __name__ == "__main__":
    main()
