"""The ERA5-Land baseline that underlies every forcing file.

`wunder.gapfill` builds a forcing file in three passes -- ERA5-Land everywhere,
then a sister station, then the site's own logger -- and this module is the first
pass. Because it fills *every* variable at *every* timestamp, there is never an
unfilled hole for the later passes to leave behind, and a site with no logger at
all still produces a complete, runnable file. That is the whole reason the
baseline comes first rather than being used to patch gaps.

The store is one Parquet pair per site under ~/data/wunder/era5land, extracted at
the site's own point from Google Earth Engine by `forcing/download_era5land.py`.
A new site is a lat/lon and a re-run -- there is no box with an edge to fall
outside of.

Two conventions this module exists to get right, neither of which fails loudly:

  * **Stamps are period ends.** The `*_hourly` value at 01:00 covers 00:00-01:00,
    whereas the 30-minute forcing grid labels each step by its start. Getting
    this wrong shifts radiation by an hour, which then looks like a timezone bug.
  * **Wind is at 10 m**, the ATMOS-41 at 2 m. At a site with a logger the bias
    fit absorbs the difference; a logger-free site has no such safety net, so the
    conversion is explicit.

De-accumulation is deliberately *not* here: GEE publishes both the raw ECMWF
accumulations (cumulative from 00 UTC, resetting daily) and per-hour increments,
and the downloader takes the increments.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .export import FREQ, saturation_vapour_pressure, specific_humidity

STORE = Path.home() / "data" / "wunder" / "era5land"

#: GEE band -> the role it plays here. The `_hourly` radiation and precipitation
#: bands are already per-hour increments, not accumulations.
T_AIR = "temperature_2m"
T_DEW = "dewpoint_temperature_2m"
PRESSURE = "surface_pressure"
SW = "surface_solar_radiation_downwards_hourly"
LW = "surface_thermal_radiation_downwards_hourly"
PRECIP = "total_precipitation_hourly"
U10 = "u_component_of_wind_10m"
V10 = "v_component_of_wind_10m"

#: Instantaneous at the stamp, so interpolated; versus accumulated over the hour
#: *ending* at the stamp, so shifted back and held.
INSTANT_BANDS = [T_AIR, T_DEW, PRESSURE, U10, V10]
PERIOD_BANDS = [SW, LW, PRECIP]

#: The nine scalars STEMMUS_SCOPE initialises its soil column from. Both engines
#: read exactly these names, so they are renamed to match on the way out.
IC_RENAME = {
    "skin_temperature": "skt",
    "soil_temperature_level_1": "stl1",
    "soil_temperature_level_2": "stl2",
    "soil_temperature_level_3": "stl3",
    "soil_temperature_level_4": "stl4",
    "volumetric_soil_water_layer_1": "swvl1",
    "volumetric_soil_water_layer_2": "swvl2",
    "volumetric_soil_water_layer_3": "swvl3",
    "volumetric_soil_water_layer_4": "swvl4",
}

LAI_BANDS = ["leaf_area_index_high_vegetation", "leaf_area_index_low_vegetation"]


def path_for(code: str, kind: str, store: Path | None = None) -> Path:
    return (store or STORE) / f"era5land_{code}_{kind}.parquet"


@lru_cache(maxsize=8)
def _read(code: str, kind: str, store: str | None = None) -> pd.DataFrame:
    path = path_for(code, kind, Path(store) if store else None)
    if not path.exists():
        raise FileNotFoundError(
            f"no ERA5-Land store for {code!r} at {path}. "
            f"Run: python forcing/download_era5land.py --site {code}"
        )
    frame = pd.read_parquet(path)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    return frame


def wind_10m_to_2m(speed, roughness_m: float = 0.01476):
    """Scale a 10 m wind speed to 2 m with a logarithmic profile.

    The default roughness is the FAO-56 reference-grass value, for which the
    factor is `ln(2/z0)/ln(10/z0)` = 0.748 -- the standard conversion, and the
    right neutral choice because ERA5-Land's 10 m wind is itself diagnosed over
    its own tile roughness rather than over the actual canopy.

    At a site with a logger the bias fit in `wunder.gapfill` absorbs whatever
    this leaves over; at a logger-free site it is the only correction there is.
    """
    return speed * (np.log(2.0 / roughness_m) / np.log(10.0 / roughness_m))


def _to_grid(series: pd.Series, index: pd.DatetimeIndex, *, kind: str) -> pd.Series:
    """Put an hourly ERA5-Land series onto the 30-minute forcing grid.

    `instant` -- the stamp is the value at that instant, so interpolate to the
    midpoint of each half-hour bin, which is what a bin labelled by its start
    actually represents.

    `period` -- the stamp is the *end* of the hour it covers, so shift back one
    hour to label by period start, then hold the rate across both halves. A rate,
    unlike a total, is not halved when the interval is.
    """
    if kind == "instant":
        midpoints = index + pd.Timedelta(FREQ) / 2
        union = series.index.union(midpoints)
        filled = series.reindex(union).interpolate(method="time", limit_area="inside")
        out = filled.reindex(midpoints)
        out.index = index
        return out

    if kind == "period":
        shifted = series.copy()
        shifted.index = shifted.index - pd.Timedelta(hours=1)
        return shifted.reindex(index, method="ffill", limit=1)

    raise ValueError(f"unknown kind {kind!r}")


def baseline(code: str, index: pd.DatetimeIndex, *, store: Path | None = None
             ) -> pd.DataFrame:
    """Every forcing primitive, on `index`, from ERA5-Land alone.

    This is pass 1. Returned in model units and in the primitives
    `wunder.export.PRIMITIVES`, ready to be overlaid by in-situ data -- or used
    as-is for a site that has none. `VPD`, `RH` and `Qair` are deliberately not
    produced here; they are derived once from the merged primitives.

    `index` must be a tz-aware UTC DatetimeIndex at 30-minute spacing.
    """
    if index.tz is None:
        raise ValueError("index must be tz-aware UTC")

    raw = _read(code, "hourly", str(store) if store else None)

    # ERA5-Land drops the occasional hour (7 in 28,896 at NL-Gla). Interpolating
    # on the hourly series first keeps a missing stamp from becoming two missing
    # half-hours downstream.
    hourly = raw.reindex(
        pd.date_range(raw.index.min(), raw.index.max(), freq="h", tz="UTC")
    ).interpolate(method="time", limit=3)

    out = pd.DataFrame(index=index)

    t2m = _to_grid(hourly[T_AIR], index, kind="instant")
    d2m = _to_grid(hourly[T_DEW], index, kind="instant")
    sp = _to_grid(hourly[PRESSURE], index, kind="instant")

    out["Tair"] = t2m
    out["Psurf"] = sp

    # Vapour pressure from the dewpoint, on the same saturation curve the logger
    # path uses, so the two sources cannot disagree by convention.
    ea = saturation_vapour_pressure(d2m - 273.15)                     # kPa
    out["ea_kPa"] = ea.clip(upper=saturation_vapour_pressure(t2m - 273.15))

    out["SWdown"] = (_to_grid(hourly[SW], index, kind="period") / 3600.0).clip(lower=0.0)
    out["LWdown"] = (_to_grid(hourly[LW], index, kind="period") / 3600.0).clip(lower=0.0)
    # m of water per hour -> kg m-2 s-1
    out["Precip"] = (
        _to_grid(hourly[PRECIP], index, kind="period") * 1000.0 / 3600.0
    ).clip(lower=0.0)

    u10 = _to_grid(hourly[U10], index, kind="instant")
    v10 = _to_grid(hourly[V10], index, kind="instant")
    out["Wind"] = wind_10m_to_2m(np.hypot(u10, v10))

    return out


def lai(code: str, index: pd.DatetimeIndex, *, store: Path | None = None) -> pd.Series:
    """ERA5-Land's own LAI, high plus low vegetation, on the forcing grid.

    The last-resort fallback for a site with no MODIS subset. ERA5-Land's LAI is
    a climatology rather than an observation of the year in question, so prefer
    MODIS wherever it exists.
    """
    daily = _read(code, "daily", str(store) if store else None)
    total = daily[LAI_BANDS].sum(axis=1)
    union = total.index.union(index)
    return (
        total.reindex(union)
        .interpolate(method="time", limit_area="inside")
        .reindex(index)
        .clip(lower=0.0)
        .rename("LAI")
    )


def initial_condition(code: str, when: pd.Timestamp, *, store: Path | None = None
                      ) -> dict[str, float]:
    """The nine soil/skin scalars STEMMUS_SCOPE starts from, at `when`.

    Always ERA5-Land, at every site: `skt` is a skin temperature nothing in the
    network measures, and `stl4`/`swvl4` span 100-289 cm while the deepest WUNDER
    probe is 80 cm -- and those map to the model's bottom boundary condition. The
    store holds these daily at 00 UTC, which is why runs start on a day boundary.
    """
    if when.tz is None:
        when = when.tz_localize("UTC")

    daily = _read(code, "daily", str(store) if store else None)
    position = daily.index.get_indexer([when], method="nearest")[0]
    stamp = daily.index[position]
    if abs(stamp - when) > pd.Timedelta(days=1):
        raise ValueError(
            f"nearest ERA5-Land soil state for {code} is {stamp}, "
            f"{abs(stamp - when)} from the requested {when}; "
            "the store does not cover this run start"
        )

    row = daily.iloc[position]
    values = {short: float(row[band]) for band, short in IC_RENAME.items()}
    missing = [k for k, v in values.items() if not np.isfinite(v)]
    if missing:
        raise ValueError(f"ERA5-Land initial condition for {code} is NaN at {missing}")
    return values
