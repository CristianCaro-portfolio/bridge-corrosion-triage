import gzip

import pytest

from bridge_triage.exposure import ExposureDataError, classify, from_lcd_csv, tow_band

HEADER = "STATION,NAME,DATE,REPORT_TYPE,HourlyDryBulbTemperature,HourlyRelativeHumidity,HourlyPrecipitation\n"


def _year_csv(tmp_path, hours, temp, rh, precip="0.00", year=2023, report="FM-15"):
    lines = [HEADER]
    for h in range(hours):
        day, hour = divmod(h, 24)
        ts = f"{year}-{1 + day // 28:02d}-{1 + day % 28:02d}T{hour:02d}:54:00"
        lines.append(f"X1,TEST,{ts},{report},{temp(h)},{rh(h)},{precip}\n")
    path = tmp_path / "lcd.csv.gz"
    with gzip.open(path, "wt") as fh:
        fh.writelines(lines)
    return path


@pytest.mark.parametrize(
    "hours,band", [(0, "t1"), (10, "t1"), (11, "t2"), (2500, "t3"), (2501, "t4"), (6000, "t5")]
)
def test_tow_bands(hours, band):
    assert tow_band(hours) == band


def test_classify_rules():
    assert classify("t3", 0, marine=False) == "mild"
    assert classify("t4", 0, marine=False) == "moderate"
    assert classify("t3", 6, marine=False) == "moderate"
    assert classify("t3", 25, marine=False) == "severe"
    assert classify("t2", 0, marine=True) == "severe"


def test_wet_hours_need_humidity_and_above_freezing(tmp_path):
    # 28 days x 12 months x 24 h = 8064 hours, 92% of a year
    path = _year_csv(tmp_path, 8064, temp=lambda h: "5.0s" if h % 2 else "-2.0", rh=lambda h: "85")
    exp = from_lcd_csv(path)
    assert exp.coverage == pytest.approx(8064 / 8760, abs=1e-3)
    # half the hours are wet, scaled to the full year
    assert exp.tow_hours == round(4032 * 8760 / 8064)
    assert exp.freeze_thaw_days == 336


def test_trace_precipitation_counts_as_zero(tmp_path):
    path = _year_csv(tmp_path, 8064, temp=lambda h: "10", rh=lambda h: "50", precip="T")
    exp = from_lcd_csv(path)
    assert exp.precip_mm == 0.0 and exp.tow_hours == 0 and exp.exposure_class == "mild"


def test_incomplete_record_fails_closed(tmp_path):
    path = _year_csv(tmp_path, 2000, temp=lambda h: "10", rh=lambda h: "90")
    with pytest.raises(ExposureDataError, match="observed"):
        from_lcd_csv(path)


def test_bundled_richmond_2024(root):
    exp = from_lcd_csv(root / "sample_data/weather/lcd_72401013740_2024.csv.gz")
    assert exp.year == 2024 and exp.coverage > 0.99
    assert exp.tow_band == "t4" and exp.exposure_class == "moderate"
