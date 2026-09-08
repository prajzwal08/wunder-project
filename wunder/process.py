"""Derived quantities and resampling.

The root-zone weighting comes from trial/Ketelbroek_DataReport.ipynb cell 8, vectorised here
because the full record is ~350k rows per logger and `DataFrame.apply` over that is slow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metadata import DEPTH_ORDER, Logger, measures
from .metadata import logger as _logger

# How each variable must be aggregated when resampling. Three kinds, not two:
#
#   SUM       a quantity accumulated *over* the interval. Averaging rainfall would divide
#             an hourly total by twelve.
#   CIRCULAR  a compass bearing. The arithmetic mean of 350 deg and 10 deg is 180 deg --
#             due south for a wind that blew due north. On this record, 17% of hours differ
#             from the correct circular mean by more than 20 deg, with errors up to 180.
#             Resolved as a vector mean, weighted by wind speed when it is available, which
#             is what "mean wind direction" means meteorologically.
#   MEAN      everything else: states and flux densities that simply *are* at each instant
#             -- temperatures, moisture, potentials, pressures, radiation [W m-2], wind
#             speed. Averaging is correct for all of these.
ACCUMULATING = {"Precipitation observed"}
CIRCULAR = {"Wind direction observation"}
WIND_SPEED = "Wind speed observation"


def depth_columns(df: pd.DataFrame, measure: str = "moisture") -> list[str]:
    """Columns for `measure` that carry data, shallowest first.

    Presence in `df.columns` is not enough -- the API returns entirely empty columns.
    """
    prefix = measures()[measure]
    out = []
    for d in DEPTH_ORDER:
        c = f"{prefix} {d}cm"
        if c in df.columns and df[c].notna().any():
            out.append(c)
    return out


def depths_of(columns: list[str]) -> list[float]:
    """['Soil moisture 5cm', ...] -> [5.0, ...]"""
    return [float(c.split()[-1].removesuffix("cm")) for c in columns]


def thicknesses(depths: list[float]) -> list[float]:
    """Layer thickness for each sensor depth, boundaries at the midpoints between sensors.

    The top layer starts at the surface; the bottom extends half its own depth below the
    deepest sensor. Verbatim from the notebook's `getThicknesses`.
    """
    out = []
    prev = 0.0
    for i in range(len(depths) - 1):
        boundary = (depths[i] + depths[i + 1]) / 2.0
        out.append(boundary - prev)
        prev = boundary
    out.append((depths[-1] + depths[-1] / 2.0) - prev)
    return out


# Depth-weighted averaging is only meaningful for *extensive* quantities — ones that add up
# over a profile. Water content does: thickness-weighting gives the real water stored.
# Matric potential and electrical conductivity are *intensive* state variables, like
# temperature or pressure. Averaging potential by thickness is a category error: it spans
# orders of magnitude (-10 kPa wet to -1500 kPa at wilting), so a linear mean is dominated
# by the wettest layer and understates stress; and a plant extracts from the least-negative
# layer rather than experiencing the profile mean. A defensible aggregate would need
# root-density and hydraulic-conductance weighting, which this network cannot supply.
EXTENSIVE = {"moisture"}
# Temperature is intensive too, but a thickness-weighted mean profile temperature is a
# long-standing convention in soil physics and is what the notebook computes, so it stays.
AVERAGEABLE = EXTENSIVE | {"temperature", "temperature_2"}


def root_zone(
    df: pd.DataFrame,
    measure: str = "moisture",
    *,
    method: str = "trapezoid",
    columns: list[str] | None = None,
    min_coverage: float = 0.9,
) -> pd.Series:
    """Depth-weighted root-zone average over all reporting depths.

    method="trapezoid" is the equation written up in the notebook's action point 3:

        RZSM = (2*t1*L1 + (t1+t2)*L2 + ... + (t_{i-1}+t_i)*Li) / (2*(L1+...+Li))

    i.e. the profile varies linearly between sensors.

    method="weighted" is the simple thickness-weighted mean (`get_rzsm2`), which treats each
    sensor as representative of its whole layer. **This is what the notebook actually plots**,
    so use it to reproduce those figures. The two differ most when the profile has a strong
    gradient near the surface.
    """
    if measure not in AVERAGEABLE:
        raise ValueError(
            f"{measure!r} is an intensive state variable — a depth-weighted mean of it is "
            "not physically meaningful. See the note above EXTENSIVE in this module. "
            f"Averageable measures: {sorted(AVERAGEABLE)}."
        )
    cols = columns if columns is not None else depth_columns(df, measure)

    # A depth that died mid-record would otherwise poison every later row with NaN and wipe
    # out the whole series -- z6-21178 lost 20 cm in Feb 2024 and has data for three years
    # after. Drop sparse depths so the profile definition stays constant through time, which
    # is also what makes the result comparable year to year.
    if columns is None and cols and min_coverage:
        keep = [c for c in cols if df[c].notna().mean() >= min_coverage]
        cols = keep or cols

    if not cols:
        return pd.Series(index=df.index, dtype="float64")

    L = np.asarray(thicknesses(depths_of(cols)), dtype="float64")
    values = df[cols].to_numpy(dtype="float64")

    if method == "weighted":
        num = values @ L
        den = L.sum()
    elif method == "trapezoid":
        num = 2.0 * values[:, 0] * L[0]
        if len(cols) > 1:
            num = num + (values[:, :-1] + values[:, 1:]) @ L[1:]
        den = 2.0 * L.sum()
    else:
        raise ValueError(f"method must be 'trapezoid' or 'weighted', got {method!r}")

    return pd.Series(num / den, index=df.index, name=f"root_zone_{measure}")


def rzsm(df: pd.DataFrame, **kw) -> pd.Series:
    """Root-zone soil moisture [m3/m3]."""
    return root_zone(df, "moisture", **kw).rename("rzsm")


def rzst(df: pd.DataFrame, **kw) -> pd.Series:
    """Root-zone soil temperature [oC]."""
    return root_zone(df, "temperature", **kw).rename("rzst")


def circular_mean(
    direction: pd.Series,
    interval: str,
    speed: pd.Series | None = None,
) -> pd.Series:
    """Resample a compass bearing as a vector mean, in degrees 0-360.

    Weighted by `speed` when given, so a strong gust counts for more than a calm drift --
    the meteorological definition of a mean wind direction. Directions are "blowing from",
    which is what this network reports.
    """
    d = pd.to_numeric(direction, errors="coerce")
    w = pd.to_numeric(speed, errors="coerce") if speed is not None else 1.0
    rad = np.deg2rad(d)
    # Meteorological convention: decompose into the vector the wind blows *towards*.
    u = (-w * np.sin(rad)).resample(interval).mean()
    v = (-w * np.cos(rad)).resample(interval).mean()
    out = (np.degrees(np.arctan2(-u, -v)) % 360.0)
    # A resultant of zero length has no defined direction (opposing winds cancel).
    return out.where(np.hypot(u, v) > 1e-9)


def resample(df: pd.DataFrame, interval: str = "30min") -> pd.DataFrame:
    """Resample to `interval`, aggregating each column by its kind.

    Precipitation is summed, wind direction is resolved as a vector mean, everything else is
    averaged. See the ACCUMULATING / CIRCULAR notes above for why all three are needed.
    """
    if df.empty:
        return df
    acc = [c for c in df.columns if c in ACCUMULATING]
    circ = [c for c in df.columns if c in CIRCULAR]
    rest = [c for c in df.columns if c not in ACCUMULATING and c not in CIRCULAR]

    r = df.resample(interval)
    parts = []
    if rest:
        parts.append(r[rest].mean())
    if acc:
        parts.append(r[acc].sum(min_count=1))
    for c in circ:
        speed = df[WIND_SPEED] if WIND_SPEED in df.columns else None
        parts.append(circular_mean(df[c], interval, speed).rename(c))

    out = pd.concat(parts, axis=1)
    return out[[c for c in df.columns if c in out.columns]]


def wind_rose_table(
    df: pd.DataFrame,
    *,
    sectors: int = 16,
    speed_bins: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 6.0, np.inf),
    calm_threshold: float = 0.5,
) -> tuple[pd.DataFrame, float, int]:
    """Bin wind into direction sectors x speed classes.

    Returns (table, calm_fraction, n_used). The table is percent-of-total per
    sector/speed-bin, so roses from different loggers or periods are comparable.

    Records below `calm_threshold` are excluded: at those speeds the vane direction is noise,
    and ~30% of the record is that slow. They are reported separately rather than smeared
    across sectors.
    """
    ws_col, wd_col = "Wind speed observation", "Wind direction observation"
    if ws_col not in df.columns or wd_col not in df.columns:
        raise ValueError("no wind data on this logger")

    d = df[[ws_col, wd_col]].dropna()
    if d.empty:
        raise ValueError("wind columns are empty")

    n_total = len(d)
    calm = d[ws_col] < calm_threshold
    d = d[~calm]
    calm_fraction = float(calm.sum()) / n_total
    if d.empty:
        return pd.DataFrame(), calm_fraction, 0

    width = 360.0 / sectors
    # Sector 0 is centred on north, so shift by half a sector before binning.
    idx = (((d[wd_col] % 360.0) + width / 2.0) // width % sectors).astype(int)
    labels = _sector_labels(sectors)

    speeds = pd.cut(d[ws_col], bins=list(speed_bins), right=False)
    table = (
        pd.crosstab(pd.Categorical(idx, categories=range(sectors)), speeds)
        .reindex(range(sectors), fill_value=0)
        .astype("float64")
    )
    table.index = labels
    table = 100.0 * table / n_total  # % of all records, so calm shows as missing area
    table.columns = [_speed_label(iv) for iv in table.columns]
    return table, calm_fraction, len(d)


def _sector_labels(sectors: int) -> list[str]:
    if sectors == 16:
        return ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    if sectors == 8:
        return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return [f"{i * 360 / sectors:.0f}" for i in range(sectors)]


def _speed_label(interval) -> str:
    lo, hi = interval.left, interval.right
    return f"{lo:g}+ m/s" if np.isinf(hi) else f"{lo:g}-{hi:g} m/s"


def coverage(df: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """Fraction of expected samples present per column per period.

    Shows when a sensor went dark -- e.g. z6-21178's 20 cm probe.
    """
    if df.empty:
        return pd.DataFrame()
    expected = pd.Timedelta(freq) / pd.Timedelta(minutes=5)
    return (df.resample(freq).count() / expected).clip(upper=1.0)


def cumulative_year(s: pd.Series, *, freq: str = "1D", how: str = "sum") -> pd.Series:
    """Running total within each calendar year, restarting on 1 January.

    Makes "how much rain have we had so far this year" comparable with the same date in
    another year. `how="sum"` for rainfall (a total), `"mean"` for a state like VPD whose
    daily means are then accumulated into a demand index.
    """
    if s.empty:
        return s
    daily = s.resample(freq).agg(how).fillna(0)
    return daily.groupby(daily.index.year).cumsum()


def field_series(
    frames: dict[str, pd.DataFrame],
    measure: str = "moisture",
    *,
    depth: str | None = None,
    freq: str = "1h",
) -> pd.Series:
    """One series representing a field, averaged across its loggers.

    A field holds several loggers a few tens of metres apart; the field-level value is their
    mean. Loggers are resampled onto a common grid first, because they do not share
    timestamps exactly. Loggers lacking the measure are skipped rather than counted as zero.
    """
    parts = []
    for df in frames.values():
        if df.empty:
            continue
        if depth:
            col = f"{measures()[measure]} {depth}cm"
            if col not in df.columns or not df[col].notna().any():
                continue
            s = df[col]
        else:
            if measure not in AVERAGEABLE or not depth_columns(df, measure):
                continue
            s = root_zone(df, measure)
        s = s.dropna()
        if not s.empty:
            parts.append(s.resample(freq).mean())
    if not parts:
        return pd.Series(dtype="float64")
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True).dropna()


def sensor_status(df: pd.DataFrame, *, within_days: int = 30) -> pd.DataFrame:
    """Per-column lifetime: coverage, first and last reading, and whether it is still live.

    Sensors fail mid-record rather than being absent -- z6-21178's 20 cm probe reported for
    nine months before dying in Feb 2024, and z6-08820's 5 cm probe ran at 98.8% until it
    stopped in Aug 2026. "Live" therefore means *recent relative to this logger's own last
    record*, not merely "has some data somewhere".
    """
    if df.empty:
        return pd.DataFrame()
    latest = df.index.max()
    cutoff = latest - pd.Timedelta(days=within_days)
    rows = []
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "column": c,
                "coverage": len(s) / len(df),
                "first": s.index.min(),
                "last": s.index.max(),
                "days_silent": (latest - s.index.max()).days,
                "live": s.index.max() >= cutoff,
            }
        )
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def active_measures(df: pd.DataFrame, *, within_days: int = 30) -> dict[str, list[str]]:
    """Measure -> depths still reporting near the end of the record.

    Use this to decide which panels to show. `depth_columns` answers a different question --
    "what does this window contain" -- and is the right test when plotting a chosen window.
    """
    if df.empty:
        return {}
    cutoff = df.index.max() - pd.Timedelta(days=within_days)
    out: dict[str, list[str]] = {}
    for measure, prefix in measures().items():
        depths = []
        for d in DEPTH_ORDER:
            c = f"{prefix} {d}cm"
            if c in df.columns:
                s = df[c].dropna()
                if not s.empty and s.index.max() >= cutoff:
                    depths.append(d)
        if depths:
            out[measure] = depths
    return out


def summary(ref: str | Logger, df: pd.DataFrame) -> dict:
    """Headline numbers for a logger, for the status view."""
    lg = ref if isinstance(ref, Logger) else _logger(ref)
    if df.empty:
        return {"logger": lg.label, "status": "offline", "rows": 0}
    last = df.index.max()
    return {
        "logger": lg.label,
        "status": "ok",
        "rows": len(df),
        "from": df.index.min(),
        "to": last,
        "age_hours": round((pd.Timestamp.now() - last).total_seconds() / 3600, 1),
        "reporting": lg.reporting(df),
    }
