"""Download Jena Weather CSV (Autoformer / Informer layout).

Official mirrors often move; this script tries direct URLs first, then extracts
`Weather/Y_df.csv` from the public Autoformer dataset bundle (Dropbox).
"""

from __future__ import annotations

import io
import zipfile
import urllib.error
import urllib.request
from pathlib import Path

# Direct mirrors (fast, if available)
URLS = [
    (
        "Time-Series-Library",
        "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/data/weather/weather.csv",
    ),
    (
        "Autoformer (legacy path)",
        "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/weather/weather.csv",
    ),
]

# Full benchmark bundle (~54 MB) — contains Weather/Y_df.csv
DROPBOX_ZIP = "https://www.dropbox.com/s/rlc1qmprpvuqrsv/all_six_datasets.zip?dl=1"
ZIP_MEMBER = "all_six_datasets/Weather/Y_df.csv"


def _download_bytes(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "tsad-magisterka/1.0 (research dataset download)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _validate(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if len(lines) < 100:
        raise ValueError(f"Downloaded file too small: {path}")
    head = lines[0].lower()
    if "date" not in head:
        raise ValueError(f"Missing date column in {path}: {head!r}")


def _extract_from_dropbox(dest: Path) -> None:
    print(f"Fetching dataset bundle (this may take ~1 min): {DROPBOX_ZIP}")
    data = _download_bytes(DROPBOX_ZIP, timeout=300)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        if ZIP_MEMBER not in zf.namelist():
            raise ValueError(f"{ZIP_MEMBER} not found in archive")
        dest.write_bytes(zf.read(ZIP_MEMBER))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dest = root / "data" / "weather" / "weather.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for name, url in URLS:
        print(f"Trying {name}: {url}")
        try:
            dest.write_bytes(_download_bytes(url, timeout=120))
            _validate(dest)
            print(f"Saved: {dest} ({dest.stat().st_size // 1024} KB)")
            return
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as e:
            print(f"  Failed: {e}")
            last_err = e

    try:
        _extract_from_dropbox(dest)
        _validate(dest)
        print(f"Saved: {dest} ({dest.stat().st_size // 1024} KB)")
    except Exception as e:
        raise SystemExit(
            "Could not download Weather dataset.\n"
            f"Place Autoformer weather.csv manually at: {dest}\n"
            f"Last error: {last_err or e}"
        ) from e


if __name__ == "__main__":
    main()
