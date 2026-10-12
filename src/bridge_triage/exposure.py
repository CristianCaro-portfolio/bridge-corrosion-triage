"""Climate exposure of a bridge site from NOAA hourly observations.

Time of wetness (TOW) is the number of hours a year with relative humidity of 80% or more and
air temperature above 0 C, the definition used by ISO 9223. Freezing precipitation days are a
proxy for deicing salt use, which the FHWA handbook names as the main corrosion driver on
inland highway bridges.
"""

from __future__ import annotations

import csv
import gzip
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# ISO 9223 (1992) time of wetness bands, upper bounds in hours per year
TOW_BANDS = ((10, "t1"), (250, "t2"), (2500, "t3"), (5500, "t4"), (float("inf"), "t5"))
_NUM = re.compile(r"^-?\d+(\.\d+)?")


class ExposureDataError(ValueError):
    """Raised when the weather record is too incomplete to trust."""


@dataclass(frozen=True)
class Exposure:
    station: str
    station_name: str
    year: int
    hours_observed: int
    coverage: float
    tow_hours: int
    tow_band: str
    freeze_thaw_days: int
    freezing_precip_days: int
    precip_mm: float
    exposure_class: str  # mild, moderate, severe

    def as_dict(self) -> dict:
        return asdict(self)


def _num(value: str) -> float | None:
    """NOAA values can carry flags such as '12.2s' or 'T' for trace."""
    value = (value or "").strip()
    if value == "T":
        return 0.0
    m = _NUM.match(value)
    return float(m.group(0)) if m else None


def tow_band(hours: float) -> str:
    for upper, name in TOW_BANDS:
        if hours <= upper:
            return name
    return "t5"


def classify(tow: str, freezing_precip_days: int, marine: bool) -> str:
    """Rule of thumb that mirrors the handbook's mild / moderate / severe split."""
    if marine or tow == "t5" or freezing_precip_days >= 20:
        return "severe"
    if tow == "t4" or freezing_precip_days >= 5:
        return "moderate"
    return "mild"


def from_lcd_csv(path: Path, marine: bool = False, min_coverage: float = 0.8) -> Exposure:
    """Reads a NOAA Local Climatological Data CSV (hourly columns) for one station-year."""
    hourly: dict[datetime, tuple[float, float]] = {}
    precip: dict[datetime, float] = {}
    station = name = ""
    years: set[int] = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("REPORT_TYPE", "").strip() != "FM-15":  # routine hourly METAR only
                continue
            station, name = row["STATION"], row.get("NAME", "")
            ts = datetime.fromisoformat(row["DATE"]).replace(minute=0, second=0)
            years.add(ts.year)
            t = _num(row.get("HourlyDryBulbTemperature", ""))
            rh = _num(row.get("HourlyRelativeHumidity", ""))
            if t is not None and rh is not None and 0 <= rh <= 100 and -60 < t < 60:
                hourly.setdefault(ts, (t, rh))
            p = _num(row.get("HourlyPrecipitation", ""))
            if p is not None and p >= 0:
                precip[ts] = max(precip.get(ts, 0.0), p)
    if len(years) != 1:
        raise ExposureDataError(f"{path.name}: expected one year, found {sorted(years)}")
    year = years.pop()
    hours_in_year = 8784 if year % 4 == 0 else 8760
    coverage = len(hourly) / hours_in_year
    if coverage < min_coverage:
        raise ExposureDataError(f"{path.name}: only {coverage:.0%} of hours observed")

    tow_observed = sum(1 for t, rh in hourly.values() if rh >= 80 and t > 0)
    tow = round(tow_observed / coverage)  # scale missing hours at the observed rate

    days: dict[str, list[float]] = {}
    for ts, (t, _) in hourly.items():
        days.setdefault(ts.date().isoformat(), []).append(t)
    wet_days: dict[str, float] = {}
    for ts, p in precip.items():
        key = ts.date().isoformat()
        wet_days[key] = wet_days.get(key, 0.0) + p
    freeze_thaw = sum(1 for ts in days.values() if min(ts) < 0 < max(ts))
    freezing_precip = sum(1 for d, ts in days.items() if min(ts) <= 0 and wet_days.get(d, 0) > 0)
    band = tow_band(tow)
    return Exposure(
        station=station,
        station_name=name,
        year=year,
        hours_observed=len(hourly),
        coverage=round(coverage, 3),
        tow_hours=tow,
        tow_band=band,
        freeze_thaw_days=freeze_thaw,
        freezing_precip_days=freezing_precip,
        precip_mm=round(sum(precip.values()), 1),
        exposure_class=classify(band, freezing_precip, marine),
    )
