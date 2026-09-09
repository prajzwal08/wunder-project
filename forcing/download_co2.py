"""Fetch Mauna Loa CO2 from NOAA GML and put it on the 30-minute forcing grid.

Replaces ~/data/mauna_loa_csv_30mins_197405_202509.csv, which stops 2025-09-14.

Two things this handles rather than leaves to be discovered later:

  * The daily series has real gaps. The Nov 2022 Mauna Loa eruption cut power to
    the observatory and observations moved to Mauna Kea, so 2022-11 to 2023-07 is
    sparse or substituted. Gaps are interpolated across explicitly and counted, so
    the extent of the filling is visible rather than silent.
  * It is a Northern-Hemisphere background record, not a local measurement. That
    goes in the output header and in the forcing file's attributes.

The local alternatives both stop too early to be useful for a 2023-2026 run:
~/data/cams/cams_europe_2003_2020.nc ends 2020, and the C3S merged XCO2 product
ends 2022. Mauna Loa is the only source that reaches the present.

Usage:
    python forcing/download_co2.py [--out PATH] [--start YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_daily_mlo.txt"
OUT = Path.home() / "data" / "wunder" / "co2_maunaloa_30min.csv"
DEFAULT_START = "2023-01-01"


def download(url: str = URL, timeout: int = 120) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def parse_daily(text: str) -> pd.Series:
    """NOAA daily means: '# ' comment header, then  year month day decimal co2."""
    body = "\n".join(
        line for line in text.splitlines() if line.strip() and not line.startswith("#")
    )
    frame = pd.read_csv(
        io.StringIO(body),
        sep=r"\s+",
        header=None,
        names=["year", "month", "day", "decimal", "co2"],
    )
    frame = frame[frame["co2"] > 0]  # NOAA marks absent days -999.99
    index = pd.to_datetime(frame[["year", "month", "day"]])
    return pd.Series(frame["co2"].to_numpy(), index=index, name="co2").sort_index()


def to_30min(daily: pd.Series, start: str, end: str | None = None) -> pd.Series:
    """Daily means -> 30-minute series, linearly interpolated across gaps."""
    end = end or str(daily.index.max().date())
    grid = pd.date_range(start, f"{end} 23:30", freq="30min")

    # Reindex onto the union first so interpolation runs on real spacing, then
    # take the grid: a naive .resample().interpolate() would treat a 6-month gap
    # the same as a 1-day one.
    union = daily.index.union(grid)
    series = daily.reindex(union).interpolate(method="time", limit_area="inside")
    series = series.reindex(grid).rename("co2")

    # Each daily mean stands for its whole day, but interpolation is anchored at
    # 00:00, so the final day is left empty. Carry the last value across it --
    # bounded to one day, so a genuinely short record still raises rather than
    # being quietly extended.
    return series.ffill(limit=47)


def gap_report(daily: pd.Series, start: str) -> pd.DataFrame:
    """Runs of missing days after `start`, longest first."""
    window = daily[daily.index >= pd.Timestamp(start)]
    if window.empty:
        return pd.DataFrame(columns=["from", "to", "days"])
    full = pd.date_range(window.index.min(), window.index.max(), freq="D")
    missing = full.difference(window.index)
    if missing.empty:
        return pd.DataFrame(columns=["start", "end", "days"])
    breaks = (missing.to_series().diff() != pd.Timedelta("1D")).cumsum()
    runs = missing.to_series().groupby(breaks).agg(["min", "max", "size"])
    runs.columns = ["start", "end", "days"]
    return runs.sort_values("days", ascending=False).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--start", default=DEFAULT_START)
    args = parser.parse_args()

    print(f"fetching {URL}")
    daily = parse_daily(download())
    print(f"  {len(daily)} daily means, {daily.index.min().date()} .. "
          f"{daily.index.max().date()}, last value {daily.iloc[-1]:.2f} ppm")

    gaps = gap_report(daily, args.start)
    if len(gaps):
        print(f"  {len(gaps)} gap(s) after {args.start}, "
              f"{int(gaps['days'].sum())} missing days, longest "
              f"{int(gaps['days'].max())} d — interpolated across:")
        for row in gaps.head(5).itertuples():
            print(f"    {row.start.date()} .. {row.end.date()}  ({row.days} d)")
    else:
        print(f"  no gaps after {args.start}")

    series = to_30min(daily, args.start)
    missing = int(series.isna().sum())
    if missing:
        raise SystemExit(
            f"{missing} NaN in the 30-min series — the requested start "
            f"({args.start}) is outside the daily record, or it ends early."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        handle.write(
            "# Mauna Loa daily mean CO2, NOAA GML, interpolated to 30 minutes.\n"
            f"# source: {URL}\n"
            f"# daily record: {daily.index.min().date()} .. {daily.index.max().date()}\n"
            "# NOTE: Northern-Hemisphere background site, not a local measurement.\n"
        )
        series.to_csv(handle, index_label="time")

    print(f"wrote {args.out}  ({len(series)} rows, "
          f"{series.index.min()} .. {series.index.max()}, "
          f"{series.iloc[0]:.2f} -> {series.iloc[-1]:.2f} ppm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
