"""Download UCI Gas Sensor Array Drift dataset."""

import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/270/gas+sensor+array+drift+dataset+at+different+concentrations.zip"


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "sensor_drift"
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "sensor_drift.zip"
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out)
    print(f"Extracted to: {out}")


if __name__ == "__main__":
    main()
