"""Build each station's LAI series from MODIS, via Google Earth Engine.

Same pixels, quality filter and processing as
`trial/Get_MODIS_LAI_for_all_sites.R` plus `laiprocessing/main.py`: the 3x3 block
of 500 m pixels around the mast, quality-filtered, averaged, smoothed, then
carried to the 30-minute forcing grid. The one deliberate departure is that the
surviving pixels are averaged plainly rather than by inverse variance -- see
`composite` for why that rule misfires on this product.

Earth Engine rather than R because MODISTools would not install here and the rest
of the pipeline already reads its ancillary data from GEE -- one authentication,
one set of conventions, and the neighbourhood reduction happens server-side.

**MCD15A3H, not MCD15A2H.** Earth Engine does not carry the 8-day combined
product the R script asks for. MCD15A3H is the same Terra+Aqua sensors and the
same retrieval algorithm on a 4-day compositing window, so it gives twice as many
observations rather than fewer, and runs to 2026-08-29. If exact continuity with
the existing ORNL subsets in ~/data/fieldsitesNLLAI ever matters, MOD15A2H and
MYD15A2H are both in GEE and can be combined to approximate MCD15A2H instead.

500 m is the right scale for this: STEMMUS_SCOPE is a one-dimensional ecosystem
column standing for a homogeneous patch at roughly flux-footprint scale, so the
pixel matches what the model represents, and it is the same source PLUMBER2 uses.
It does mean the value describes the surrounding landscape rather than an
individual plot, which matters when field measurements arrive to replace it.

The script reports what each number is made of rather than only writing a file,
because `OPEN_ISSUES.md` #9/#10 traces NL-Loo's implausible Bowen ratio to this
variable being wrong.

Usage:
    python forcing/download_lai.py [--site NL-Gl1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path.home() / "data" / "wunder" / "lai"
COLLECTION = "MODIS/061/MCD15A3H"
SCALE_M = 500

#: Radius around the mast. 750 m takes the 3x3 block of 500 m pixels that
#: `laiprocessing/utils.py` selects as `pixel_no = [7,8,9,12,13,14,17,18,19]`.
RADIUS_M = 750

#: Acceptable `FparLai_QC` values, carried over from `laiprocessing`. These are
#: the combinations with MODLAND_QC good and the main radiative-transfer
#: algorithm used, with or without saturation.
GOOD_QC = {0, 2, 24, 26, 32, 34, 56, 58}

#: MCD15A3H stores LAI and its standard deviation as tenths.
LAI_SCALE = 0.1

#: Rejected beyond this; the product uses high values as fill.
LAI_MAX = 10.0

#: Below this the reported standard deviation is not credible; a pixel claiming
#: zero uncertainty is a retrieval artefact rather than a perfect measurement.
SD_FLOOR = 0.1


def _initialise():
    import ee

    try:
        ee.Initialize()
    except Exception:  # noqa: BLE001 - fall back to the project in the credentials
        import json

        project = json.loads(
            (Path.home() / ".config" / "earthengine" / "credentials").read_text()
        ).get("project")
        if not project:
            raise
        ee.Initialize(project=project)
    return ee


def fetch_raw(ee, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Every 500 m pixel in the 3x3 block, every 4-day composite, unfiltered."""
    point = ee.Geometry.Point(lon, lat)
    bands = ["Lai", "LaiStdDev", "FparLai_QC"]
    collection = (
        ee.ImageCollection(COLLECTION).filterDate(start, end).select(bands)
    )
    rows = collection.getRegion(point.buffer(RADIUS_M), SCALE_M).getInfo()
    frame = pd.DataFrame(rows[1:], columns=rows[0])
    frame["date"] = pd.to_datetime(frame["time"], unit="ms", utc=True)
    return frame.dropna(subset=["Lai"])


def composite(raw: pd.DataFrame) -> tuple[pd.Series, dict]:
    """One LAI value per composite date, averaged over the good pixels."""
    df = raw.copy()
    before = len(df)

    df = df[df["FparLai_QC"].isin(GOOD_QC)]
    after_qc = len(df)

    df["lai"] = df["Lai"] * LAI_SCALE
    df["sd"] = df["LaiStdDev"] * LAI_SCALE
    df = df[(df["lai"] <= LAI_MAX) & (df["sd"] >= SD_FLOOR)]
    after_range = len(df)

    # Plain mean over the pixels that pass quality control.
    #
    # NOT the inverse-variance weighting `laiprocessing/main.py` uses. That rule
    # is right when `sd` is an absolute uncertainty, and MCD15A3H's is not: pixel
    # LAI and its reported sd correlate at 0.913, so the uncertainty is close to
    # proportional. LAI 0.4 with sd 0.1 and LAI 5.0 with sd 1.2 are both about
    # 25% uncertain -- equally trustworthy -- yet 1/sd^2 gives the first 144
    # times the weight of the second. The result is a weighting by smallness
    # rather than by confidence, which pulled the NL-Gl1 summer mean down to
    # 1.32 from 1.86. Over a mixed block of farmland and vegetation that is a
    # systematic low bias, and LAI is the variable OPEN_ISSUES.md #9/#10 blames
    # for NL-Loo's implausible Bowen ratio.
    grouped = df.groupby("date")
    series = grouped["lai"].mean()
    stats = {
        "pixel_rows": before,
        "passed_qc": after_qc,
        "passed_range": after_range,
        "dates": len(series),
        "pixels_per_date": round(after_range / max(len(series), 1), 1),
    }
    return series.sort_index(), stats


#: Smoothing window, in days. `laiprocessing/utils.py` uses 13 points on 8-day
#: composites for both its rolling mean and its Savitzky-Golay filter, which is
#: this span. Expressed in days rather than points so it means the same thing on
#: the 4-day product used here -- 13 points would be half the intended window.
SMOOTH_WINDOW_DAYS = 13 * 8


def smooth(series: pd.Series) -> pd.Series:
    """Gap-fill and smooth the composite series, then return it daily.

    Follows `laiprocessing/utils.py` (`interpolate_NA_LAI` then `smoothing_LAI`):

      1. put the composites on a regular grid and fill the ones quality control
         removed, by cubic interpolation over composite number -- MODIS drops
         most of a Dutch winter, so this is a large share of the record;
      2. take a climatology, the mean of each composite-of-year across years;
      3. subtract it, smooth the remaining anomaly with a centred rolling mean,
         and add the climatology back. Smoothing the anomaly rather than the
         series keeps the seasonal cycle sharp while removing retrieval noise,
         which a plain filter of that width would flatten;
      4. a Savitzky-Golay pass over the same span for the residual jitter;
      5. clip at zero.

    One deliberate omission: `interpolate_NA_LAI` ends with `filled_lai[-1] = 0`,
    forcing the final composite to zero. That is an artefact of terminating the
    interpolation, and copying it would put LAI = 0 at the end of every record --
    including the present-day end of a rolling build.
    """
    from scipy.signal import savgol_filter

    step_days = max(int(round(series.index.to_series().diff().dt.days.median())), 1)
    grid = pd.date_range(series.index.min(), series.index.max(),
                         freq=f"{step_days}D", tz="UTC")
    composites = series.reindex(series.index.union(grid)).reindex(grid)

    # Climatology first, from whatever survived quality control, then use it to
    # fill the gaps.
    #
    # `interpolate_NA_LAI` interpolates the gaps cubically before taking the
    # climatology. That works on a record with short gaps; it does not work here.
    # MODIS rejects most of a Dutch winter, and a cubic through a two-month hole
    # overshoots wildly -- doing it that way gave a maximum LAI of 15.2 and
    # erased the seasonal cycle entirely (summer 4.72 against winter 4.44).
    # Filling a missing composite with the climatology for that time of year is
    # both stabler and the more defensible guess: the best estimate of an
    # unobserved January is what January usually looks like.
    slot = (composites.index.dayofyear - 1) // step_days
    climatology = composites.groupby(slot).transform("mean")
    climatology = climatology.interpolate().bfill().ffill()
    composites = composites.fillna(climatology).clip(0.0, LAI_MAX)

    window = max(int(round(SMOOTH_WINDOW_DAYS / step_days)), 3)
    if window % 2 == 0:
        window += 1

    anomaly = composites - climatology
    smoothed = climatology + anomaly.rolling(
        window, center=True, min_periods=1
    ).mean()

    if len(smoothed) > window >= 5:
        smoothed[:] = savgol_filter(
            smoothed.to_numpy(), window_length=window, polyorder=3
        )

    daily_index = pd.date_range(series.index.min(), series.index.max(),
                                freq="D", tz="UTC")
    return (
        smoothed.reindex(smoothed.index.union(daily_index))
        .interpolate(method="time")
        .reindex(daily_index)
        .clip(0.0, LAI_MAX)
    )


def to_grid(daily: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Daily LAI onto the 30-minute forcing grid, held across each day."""
    union = daily.index.union(index)
    return (
        daily.reindex(union)
        .interpolate(method="time", limit_area="inside")
        .reindex(index)
        .ffill()
        .bfill()
        .clip(lower=0.0)
        .rename("LAI")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", action="append", dest="sites")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from wunder.metadata import forcing_sites

    ee = _initialise()
    args.out.mkdir(parents=True, exist_ok=True)

    for code in args.sites or list(forcing_sites()):
        cfg = forcing_sites()[code]
        start = str(cfg["run_start"])
        end = str(pd.Timestamp.now(tz="UTC").date())
        raw = fetch_raw(ee, cfg["latitude"], cfg["longitude"], start, end)
        series, stats = composite(raw)
        daily = smooth(series)

        path = args.out / f"lai_{code}_daily.parquet"
        daily.rename("LAI").to_frame().to_parquet(path)

        summer = daily[daily.index.month.isin([6, 7, 8])]
        winter = daily[daily.index.month.isin([12, 1, 2])]
        print(f"\n{code}  {cfg.get('label','')}")
        print(f"  {stats['pixel_rows']} pixel-observations -> {stats['passed_qc']} pass QC "
              f"-> {stats['passed_range']} in range, over {stats['dates']} composites "
              f"({stats['pixels_per_date']} pixels/date)")
        print(f"  LAI  min {daily.min():.2f}  summer mean {summer.mean():.2f}  "
              f"winter mean {winter.mean():.2f}  max {daily.max():.2f}")
        print(f"  wrote {path.name}  ({len(daily)} days, "
              f"{daily.index.min().date()} .. {daily.index.max().date()})")

    print("\nCheck the seasonal shape against soil-moisture drawdown before "
          "trusting these: a 500 m pixel over a small plot is mostly its surroundings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
