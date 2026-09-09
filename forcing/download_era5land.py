"""Pull ERA5-Land at each site's own point, from Google Earth Engine.

Point extraction, not a regional box. GEE does the spatial subsetting server-side
-- `getRegion` on a point returns rows of numbers, so nothing is queued, no
NetCDF is written and nothing is sampled afterwards. The CDS route this replaces
charges by *field* (one variable, one time step, the whole area, regardless of
how little of it you want) and runs one job at a time per user, which made the
same data a multi-hour queue.

It is also the better answer for sites that do not exist yet. A downloaded box
has an edge that a future station can fall outside of; a point extraction takes a
latitude and longitude and nothing else. Adding a site is a row in sites.yaml and
a re-run.

Two groups of bands, at the resolution each is actually used at:

  forcing (8)   hourly. Everything a complete FLX_*.nc needs with no logger
                present: Tair, SWdown, LWdown, Psurf, Precip, Wind, and -- via
                temperature_2m + dewpoint_temperature_2m -- the vapour terms.
  initial (11)  daily at 00 UTC. The nine soil/skin scalars the model starts
                from, plus the two LAI bands as a last-resort fallback. Runs
                begin on a day boundary, so hourly resolution would buy nothing.

The `*_hourly` radiation and precipitation bands are used deliberately: GEE
publishes both the raw ECMWF accumulations (cumulative from 00 UTC, resetting
each day) and per-hour increments. Taking the increments removes the
de-accumulation step, and with it the daily sawtooth that getting it wrong
produces. Verified at NL-Gla on 2025-06-15: the raw band runs 0 -> 12,882,156
J/m2 over the day while `_hourly` gives a 480 W/m2 afternoon peak and 0 at night.

Usage:
    python forcing/download_era5land.py [--site NL-Gla] [--end 2026-09-03]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path.home() / "data" / "wunder" / "era5land"

COLLECTION = "ECMWF/ERA5_LAND/HOURLY"

#: ERA5-Land's native grid spacing. Sampling at the native scale returns the
#: value of the cell containing the point rather than a resampled blend.
SCALE_M = 11132

FORCING_BANDS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "surface_pressure",
    "surface_solar_radiation_downwards_hourly",
    "surface_thermal_radiation_downwards_hourly",
    "total_precipitation_hourly",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
]

INITIAL_BANDS = [
    "skin_temperature",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
]


def _initialise():
    import ee

    try:
        ee.Initialize()
    except Exception:  # noqa: BLE001 - fall back to the project in the credentials
        import json

        path = Path.home() / ".config" / "earthengine" / "credentials"
        project = json.loads(path.read_text()).get("project")
        if not project:
            raise
        ee.Initialize(project=project)
    return ee


def _fetch(ee, point, bands: list[str], start: str, end: str,
           hour: int | None = None) -> pd.DataFrame:
    """One `getRegion` call, returned as a UTC-indexed frame.

    `hour` keeps only that UTC hour, which is how the daily bands stay small
    enough to fetch a whole run window at once.
    """
    collection = (
        ee.ImageCollection(COLLECTION).filterDate(start, end).select(bands)
    )
    if hour is not None:
        collection = collection.filter(ee.Filter.calendarRange(hour, hour, "hour"))

    rows = collection.getRegion(point, SCALE_M).getInfo()
    if len(rows) <= 1:
        return pd.DataFrame(columns=bands)

    frame = pd.DataFrame(rows[1:], columns=rows[0])
    frame.index = pd.to_datetime(frame["time"], unit="ms", utc=True)
    frame.index.name = "time"
    return frame[bands].sort_index().astype("float64")


def _chunks(start: pd.Timestamp, end: pd.Timestamp, months: int):
    """Month-aligned [from, to) spans, so no request outgrows the getInfo limit."""
    edge = start
    while edge < end:
        nxt = min(edge + pd.DateOffset(months=months), end)
        yield edge, nxt
        edge = nxt


def _fetch_chunked(ee, point, bands: list[str], start: str, end: str, *,
                   label: str, months: int, hour: int | None = None) -> pd.DataFrame:
    """Fetch a whole window in `months`-sized pieces, halving on refusal.

    Both groups need this, for the same underlying reason but at different sizes.
    An hourly request is limited by the size of the reply; the daily one is
    limited by the work behind it, because `calendarRange` makes Earth Engine
    walk every hourly image in the span to keep one per day. Asking for 3.3 years
    at once returns "User memory limit exceeded" rather than a partial answer, so
    the span is cut until it fits.
    """
    frames: list[pd.DataFrame] = []
    for lo, hi in _chunks(pd.Timestamp(start), pd.Timestamp(end), months):
        size = months
        while True:
            try:
                parts = [
                    _fetch(ee, point, bands, str(a.date()), str(b.date()), hour=hour)
                    for a, b in _chunks(lo, hi, size)
                ]
                frames.extend(parts)
                break
            except Exception as exc:  # noqa: BLE001 - shrink and retry, or give up
                if size == 1:
                    raise
                size = max(1, size // 2)
                print(f"    {label} {lo.date()}..{hi.date()}: retrying at "
                      f"{size}-month chunks ({type(exc).__name__})")

    out = pd.concat(frames).sort_index() if frames else pd.DataFrame(columns=bands)
    out = out[~out.index.duplicated(keep="last")]
    print(f"    {label}: {len(out)} rows")
    return out


def fetch_site(ee, lat: float, lon: float,
               start: str, end: str, chunk_months: int = 2) -> dict[str, pd.DataFrame]:
    """Hourly forcing and daily initial-condition frames for one point."""
    point = ee.Geometry.Point(lon, lat)
    return {
        "hourly": _fetch_chunked(ee, point, FORCING_BANDS, start, end,
                                 label="hourly", months=chunk_months),
        "daily": _fetch_chunked(ee, point, INITIAL_BANDS, start, end,
                                label="daily 00 UTC", months=6, hour=0),
    }


def era5land_end(ee) -> pd.Timestamp:
    """Last timestamp the collection currently offers, read rather than assumed."""
    collection = ee.ImageCollection(COLLECTION)
    millis = collection.aggregate_max("system:time_start").getInfo()
    return pd.Timestamp(millis, unit="ms", tz="UTC")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", action="append", dest="sites",
                        help="site code; repeatable. Default: all in sites.yaml")
    parser.add_argument("--start", default=None,
                        help="override the per-site run_start")
    parser.add_argument("--end", default=None,
                        help="last day to fetch (default: ERA5-Land's own end)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from wunder.metadata import forcing_sites

    ee = _initialise()
    available = era5land_end(ee)
    end = pd.Timestamp(args.end, tz="UTC") if args.end else available
    print(f"ERA5-Land currently ends {available:%Y-%m-%d %H:%M} UTC; "
          f"fetching to {end:%Y-%m-%d}")

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = args.sites or list(forcing_sites())

    for code in wanted:
        cfg = forcing_sites()[code]
        start = args.start or str(cfg["run_start"])
        print(f"\n{code} ({cfg['latitude']:.5f}, {cfg['longitude']:.5f})  "
              f"{start} -> {end.date()}")
        frames = fetch_site(ee, cfg["latitude"], cfg["longitude"],
                            start, str(end.date() + pd.Timedelta(days=1)))
        for kind, frame in frames.items():
            path = args.out / f"era5land_{code}_{kind}.parquet"
            frame.to_parquet(path)
            print(f"  wrote {path.name}  {len(frame)} rows  "
                  f"{frame.index.min()} .. {frame.index.max()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
