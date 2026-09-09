"""Merge ERA5-Land and one weather station into a complete forcing record.

Two passes over a single 30-minute UTC grid:

    1. ERA5-Land     fill every variable, everywhere         qc = 2
    2. this station  overwrite where it has data             qc = 0

**One station per file.** Sensors from different masts are never combined, even
within a site: Glanerbeek F1 and F2 build separate files, and so do Ketelbroek
K1 and K2. Patching one station's gap from its neighbour reads well until the
neighbour turns out to be a different microclimate -- K1 and K2 are 175 m apart
and disagree by a factor of four in wind speed. Anything the station does not
supply comes from ERA5-Land, a documented and uniform source, rather than from
another mast whose differences would be invisible in the result.

Consequences of putting the baseline first rather than using it to patch:

  * **There is never an unfilled hole.** Filling gaps *from* ERA5-Land would
    leave NaN wherever the station had none, and neither engine has a NaN guard.
  * **A location with no logger works unchanged** -- pass 1 alone is a complete,
    runnable file, needing only a code and a lat/lon.
  * **Excluding a bad sensor is configuration, not a code path.** Both Ketelbroek
    rain gauges read a fraction of the true total, so those stations list
    `Precipitation observed` in `exclude_columns` and the baseline stays.
  * **A dead sensor or a station that stops mid-record needs no handling** --
    the overlay simply does not cover those timestamps.

Because the baseline underlies the whole record rather than a few gaps, it is
bias-corrected against the station over the full overlap *before* being overlaid,
so the stretches it supplies sit continuously with the measured ones rather than
stepping at the joins.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import era5land, export
from .metadata import Logger, forcing_site
from .metadata import logger as _logger
from .metadata import loggers as _all_loggers
from .process import resample

#: How each primitive's bias correction is fitted. The three kinds exist for
#: reasons, not tidiness:
#:
#:   "linear"  ordinary least squares with an intercept. For variables whose
#:             offset is physical -- Psurf differs from the site by a constant
#:             set by the elevation of ERA5-Land's 9 km cell.
#:   "scale"   least squares through the origin. For variables that must not be
#:             able to go negative, and whose error is proportional: SWdown, and
#:             Wind, where a 9 km cell of open farmland reads about 47% windier
#:             than a sheltered food forest.
#:   "volume"  slope = sum(obs)/sum(era5), no intercept. For precipitation,
#:             where reanalysis gets the total roughly right and the timing
#:             poorly (hourly r = 0.45 at NL-Gla). Preserving total water is
#:             worth more to a soil model than fitting individual half-hours.
FIT_KIND = {
    "Tair": "linear",
    "Psurf": "linear",
    "ea_kPa": "linear",
    "SWdown": "scale",
    "Wind": "scale",
    "Precip": "volume",
    # LWdown gets none: nothing in the network measures downwelling longwave, so
    # there is nothing to fit against. It stays raw ERA5-Land, flagged qc = 2.
}


@dataclass
class BiasFit:
    """One variable's ERA5-Land correction, and how well it is determined."""

    variable: str
    kind: str
    slope: float
    intercept: float
    r: float
    rmse_before: float
    rmse_after: float
    n: int

    def apply(self, series: pd.Series) -> pd.Series:
        return series * self.slope + self.intercept

    def describe(self) -> str:
        return (
            f"{self.variable}: {self.kind} slope={self.slope:.4f} "
            f"intercept={self.intercept:.4g} r={self.r:.4f} "
            f"rmse {self.rmse_before:.4g} -> {self.rmse_after:.4g} (n={self.n})"
        )


@dataclass
class BuildReport:
    """What the build actually did, for the file header and for the operator."""

    code: str
    start: pd.Timestamp
    end: pd.Timestamp
    steps: int
    source: str = "unknown"
    fits: dict[str, BiasFit] = field(default_factory=dict)
    qc_fractions: dict[str, dict[int, float]] = field(default_factory=dict)
    screened: dict[str, dict[str, int]] = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [f"{self.code}  {self.start} .. {self.end}  ({self.steps} steps)",
               f"station data from: {self.source}"]
        out.append("bias fits (in-situ vs ERA5-Land):")
        out += [f"  {f.describe()}" for f in self.fits.values()]
        if any(self.screened.values()):
            out.append("screened as out of physical range:")
            out += [f"  {src}: {counts}" for src, counts in self.screened.items() if counts]
        out.append("provenance (fraction of steps):")
        for name, fractions in self.qc_fractions.items():
            parts = " ".join(
                f"qc{level}={fractions.get(level, 0.0):.4f}" for level in sorted(fractions)
            )
            out.append(f"  {name:8s} {parts}")
        return out


def _logger_or_none(ref: str):
    """Resolve a logger reference, or None for "no station at all".

    Refuses a logger with no weather station rather than silently producing a
    reanalysis-only file: asking for soil-only hardware to supply meteorology is
    a mistake worth hearing about.
    """
    if not ref:
        return None
    lg = _logger(ref)
    if not lg.has_weather_station:
        raise ValueError(
            f"{lg.label} has no weather station, so it cannot supply met forcing. "
            "Stations with one: "
            + ", ".join(l.label for l in _all_loggers() if l.has_weather_station)
        )
    return lg


def _utc(value) -> pd.Timestamp:
    """A UTC timestamp, whether the caller passed a date, a string or a stamp.

    `pd.Timestamp(x, tz="UTC")` raises if `x` is already tz-aware, which it is
    when the driver has resolved dates before calling in.
    """
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")


def fit_bias(observed: pd.Series, modelled: pd.Series, kind: str,
             variable: str) -> BiasFit | None:
    """Fit the ERA5-Land correction for one variable over their overlap.

    Returns None when there is no usable overlap -- which is the normal case for
    a site with no logger, and must not be an error.
    """
    both = pd.concat([observed.rename("obs"), modelled.rename("era5")], axis=1).dropna()
    if len(both) < 100:
        return None

    x = both["era5"].to_numpy()
    y = both["obs"].to_numpy()

    if kind == "linear":
        slope, intercept = np.polyfit(x, y, 1)
    elif kind == "scale":
        denominator = float(x @ x)
        slope = float(x @ y) / denominator if denominator else 1.0
        intercept = 0.0
    elif kind == "volume":
        total = float(x.sum())
        slope = float(y.sum()) / total if total else 1.0
        intercept = 0.0
    else:
        raise ValueError(f"unknown fit kind {kind!r}")

    corrected = x * slope + intercept
    correlation = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
    return BiasFit(
        variable=variable,
        kind=kind,
        slope=float(slope),
        intercept=float(intercept),
        r=correlation,
        rmse_before=float(np.sqrt(np.mean((x - y) ** 2))),
        rmse_after=float(np.sqrt(np.mean((corrected - y) ** 2))),
        n=len(both),
    )


def station_record(lg: Logger, *, cache_dir=None, verbose: bool = True
                   ) -> tuple[pd.DataFrame, str]:
    """The station's own measurements, from wherever they can be had.

    A fresh clone has no `data/raw/` cache, so this tries three sources in order
    of cost and returns which one answered:

      raw        the 5-minute Parquet cache, if a previous run built it.
      published  `data/published/`, the 30-minute snapshot committed to the repo.
                 Present in any clone and instant, at the cost of sub-half-hour
                 detail the forcing does not use anyway.
      api        the MajiSys server. Correct but slow -- a logger's full history
                 is ~130 MB of CSV -- and it needs the network.

    Returns an empty frame rather than raising if none of them answer. That is
    not a fudge: pass 1 has already filled every variable from ERA5-Land, so the
    build still produces a complete and valid file, just one made entirely of
    reanalysis. The caller reports which happened.
    """
    from . import publish
    from .fetch import FetchError, _read_cache, fetch

    cached, _ = _read_cache(lg.serial, cache_dir)
    if not cached.empty:
        return cached, "raw cache"

    snapshot = publish.read(lg)
    if not snapshot.empty:
        if verbose:
            print(f"    {lg.name}: using the published 30-minute snapshot "
                  "(no raw cache in this clone)")
        return snapshot, "published snapshot"

    try:
        if verbose:
            print(f"    {lg.name}: downloading from the MajiSys API — "
                  "this takes a minute and is cached afterwards")
        return fetch(lg.serial, cache_dir=cache_dir), "api"
    except (FetchError, OSError) as exc:
        if verbose:
            print(f"    {lg.name}: UNAVAILABLE ({type(exc).__name__}: {exc}). "
                  "Falling back to the ERA5-Land baseline for every variable.")
        return pd.DataFrame(), "unavailable"


def logger_primitives(lg: Logger, index: pd.DatetimeIndex, *,
                      exclude: list[str] | None = None,
                      cache_dir=None,
                      record: pd.DataFrame | None = None,
                      ) -> tuple[pd.DataFrame, dict[str, int]]:
    """One logger's contribution to the forcing, on the UTC grid.

    Resampled by `wunder.process.resample`, which already sums precipitation,
    takes a vector mean of wind direction and averages the rest; converted to
    UTC by `wunder.export.to_utc`, which handles the daylight-saving shift; then
    screened, so a spike leaves the baseline standing rather than replacing it.
    """
    raw = record if record is not None else station_record(lg, cache_dir=cache_dir)[0]
    if raw.empty:
        return pd.DataFrame(index=index), {}
    columns = [c for c in export.MET_COLUMNS if c in raw.columns]
    for name in exclude or []:
        if name in columns:
            columns.remove(name)
    if not columns:
        return pd.DataFrame(index=index), {}

    # Only columns that carry data: a met column can be present in the API header
    # and be entirely empty (z6-21179 returns five such columns, 0 non-null in
    # 28,513 rows).
    live = [c for c in columns if raw[c].notna().any()]
    if not live:
        return pd.DataFrame(index=index), {}

    on_grid = resample(export.to_utc(raw[live]), export.FREQ).reindex(index)
    primitives = export.from_logger(on_grid)
    screened, dropped = export.screen(primitives)
    return screened, dropped


def build(code: str, start=None, end=None, *, cache_dir=None, store=None,
          met_logger: str | None = None,
          ) -> tuple[pd.DataFrame, pd.DataFrame, BuildReport]:
    """Build one site's forcing record.

    Returns `(forcing, qc, report)`: the model variables on a 30-minute UTC grid,
    an integer provenance flag per variable and step, and what the build did.

    `met_logger` overrides which station supplies the measurements -- a serial
    like "z6-21176" or a device name like "F1_1_ATMOS_SMST1". Pass `""` to build
    from the ERA5-Land baseline alone. Without it the station named in
    sites.yaml for this code is used, which is the one whose field the code
    refers to.
    """
    cfg = dict(forcing_site(code))
    if met_logger is not None:
        cfg["met"] = _logger_or_none(met_logger)

    first = _utc(start if start is not None else cfg["run_start"])
    if end is None:
        available = era5land._read(code, "hourly", str(store) if store else None)
        end = available.index.max().floor("D") - pd.Timedelta(export.FREQ)
    index = pd.date_range(first, _utc(end), freq=export.FREQ, tz="UTC")

    report = BuildReport(code=code, start=index[0], end=index[-1], steps=len(index))

    # --- pass 1: ERA5-Land everywhere ------------------------------------
    merged = era5land.baseline(code, index, store=store)
    qc = pd.DataFrame(export.QC_ERA5, index=index, columns=merged.columns, dtype="int8")

    if cfg["met"] is None:
        raw, report.source = pd.DataFrame(), "ERA5-Land only (no station)"
    else:
        raw, report.source = station_record(cfg["met"], cache_dir=cache_dir)
    if cfg["met"] is None:
        observed, dropped = pd.DataFrame(index=index), {}
    else:
        observed, dropped = logger_primitives(
            cfg["met"], index, exclude=cfg["exclude_columns"],
            cache_dir=cache_dir, record=raw,
        )
    if dropped:
        report.screened[cfg["met"].name] = dropped

    # --- bias-correct the baseline, against the PRISTINE baseline ---------
    # Before the overlay, not after. Fitting afterwards would compare the station
    # against values already written into `merged`, which flatters the fit and
    # hides the real reanalysis error.
    for variable, kind in FIT_KIND.items():
        if variable not in observed.columns or variable not in merged.columns:
            continue
        fitted = fit_bias(observed[variable], merged[variable], kind, variable)
        if fitted is not None:
            report.fits[variable] = fitted
            merged[variable] = fitted.apply(merged[variable])

    # --- pass 2: overlay this station, and only this station --------------
    for variable in observed.columns:
        if variable not in merged.columns:
            continue
        usable = observed[variable].notna()
        merged.loc[usable, variable] = observed.loc[usable, variable]
        qc.loc[usable, variable] = export.QC_MEASURED

    # --- derive the humidity trio once, from the merged primitives --------
    forcing = export.derive_humidity(merged)

    # VPD, RH and Qair inherit the weaker provenance of the primitives they came
    # from, so a value is never advertised as better measured than its inputs.
    thermo = qc[["Tair", "ea_kPa", "Psurf"]].max(axis=1)
    for derived in ("VPD", "RH", "Qair"):
        qc[derived] = thermo

    forcing = forcing.drop(columns=["ea_kPa"])
    qc = qc.drop(columns=["ea_kPa"])

    for name in forcing.columns:
        counts = qc[name].value_counts(normalize=True)
        report.qc_fractions[name] = {int(k): float(v) for k, v in counts.items()}

    return forcing, qc, report
