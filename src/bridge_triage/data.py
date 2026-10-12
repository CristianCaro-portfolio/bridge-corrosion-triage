"""Downloads the public sources and builds the small sample shipped with the repo.

Sources (no account or key needed):
- Corrosion Condition State Semantic Segmentation Dataset, Bianchi and Hebdon, Virginia Tech,
  CC0, figshare article 16624663
- NOAA NCEI Local Climatological Data, hourly, public domain
- FHWA guidance PDFs from the USDOT National Transportation Library, public domain
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

CORROSION_ZIP = "https://ndownloader.figshare.com/files/31729733"

FHWA_DOCS = {
    "FHWA-HIF-16-002": "https://rosap.ntl.bts.gov/view/dot/42715/dot_42715_DS1.pdf",
    "FHWA-BPG-2018": "https://rosap.ntl.bts.gov/view/dot/41105/dot_41105_DS1.pdf",
    "FHWA-HIF-22-046": "https://www.fhwa.dot.gov/bridge/preservation/docs/hif22046.pdf",
}

LCD_URL = (
    "https://www.ncei.noaa.gov/access/services/data/v1?dataset=local-climatological-data"
    "&stations={station}&startDate={year}-01-01T00:00:00&endDate={year}-12-31T23:59:59"
    "&dataTypes=HourlyDryBulbTemperature,HourlyRelativeHumidity,HourlyPrecipitation"
    "&format=csv&units=metric&includeStationName=true"
)

WEATHER_COLUMNS = [
    "STATION", "NAME", "DATE", "REPORT_TYPE",
    "HourlyDryBulbTemperature", "HourlyRelativeHumidity", "HourlyPrecipitation",
]  # fmt: skip

USER_AGENT = "bridge-corrosion-triage/0.1 (research use)"


def _get(url: str, dest: Path, timeout: int = 600) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.replace(dest)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_corrosion(raw_dir: Path) -> Path:
    """Downloads and unpacks the dataset. Returns the folder with Train and Test at 512 px."""
    target = raw_dir / "vt_corrosion"
    split_root = target / "512x512"
    if (split_root / "Test" / "images_512").exists():
        return split_root
    zip_path = _get(CORROSION_ZIP, raw_dir / "corrosion_condition_state.zip")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if "/512x512/" not in member or member.endswith("/"):
                continue
            rel = member.split("/512x512/", 1)[1]
            out = split_root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return split_root


def fetch_fhwa(raw_dir: Path) -> dict[str, Path]:
    out = {}
    for doc, url in FHWA_DOCS.items():
        dest = raw_dir / "fhwa" / f"{doc}.pdf"
        if not dest.exists():
            _get(url, dest)
        out[doc] = dest
    return out


def fetch_weather(station: str, year: int, dest: Path) -> Path:
    """Downloads one station-year and keeps only the routine hourly rows and needed columns."""
    url = LCD_URL.format(station=station, year=year)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=900) as resp:
        text = resp.read().decode("utf-8")
    return compact_weather(io.StringIO(text), dest)


def compact_weather(src, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    reader = csv.DictReader(src)
    with gzip.open(dest, "wt", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=WEATHER_COLUMNS)
        writer.writeheader()
        for row in reader:
            if row.get("REPORT_TYPE", "").strip() == "FM-15":
                writer.writerow({k: row.get(k, "") for k in WEATHER_COLUMNS})
    return dest


def build_image_sample(split_root: Path, out_dir: Path, names: list[str], size: int = 256) -> None:
    """Copies a few test photos and masks, downsized, so the demo and CI run offline."""
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    for name in names:
        img = Image.open(split_root / "Test" / "images_512" / f"{name}.jpeg").convert("RGB")
        img.resize((size, size), Image.BILINEAR).save(out_dir / "images" / f"{name}.jpeg", quality=85)
        mask = Image.open(split_root / "Test" / "mask_512" / f"{name}.png").convert("RGB")
        mask.resize((size, size), Image.NEAREST).save(out_dir / "masks" / f"{name}.png", optimize=True)
