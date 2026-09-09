"""Makkink reference evapotranspiration.

ET0 is the evaporative *demand* the atmosphere puts on a well-watered short grass
sward: what would evaporate if water were never limiting. It is not what this
canopy actually transpires — these are food forests, not grass — so read it as a
demand index to set the soil-moisture record against, and as an independent check
on whatever the model produces for the same days.

**Why Makkink and not Penman-Monteith.** Makkink is the Dutch standard: it is what
KNMI publishes as `EV24` for every station in the country, so a number computed
here is directly comparable with the national record. It needs only global
radiation and air temperature — the two things every ATMOS-41 in this network
measures well — where Penman-Monteith also needs wind at a defined height and a
vapour-pressure deficit whose errors it is sensitive to. At Dutch sites the two
agree closely for grass; the radiation term dominates in this climate.

    ET0 = C * s/(s + g) * Rs / (lambda * rho_w)          [mm d-1]

with `C = 0.65`, `s` the slope of the saturation vapour pressure curve at the
day's mean temperature, `g` the psychrometric constant, `Rs` the day's global
radiation and `lambda` the latent heat of vaporisation.

**Daily, deliberately.** The coefficient 0.65 is fitted to *daily* totals; applied
to a half-hour it is meaningless (it would carry the day's radiation-to-ET ratio
into a single sunlit interval, and go to zero at night where the real canopy is
still losing water). So everything here resamples to whole days first, and a day
missing more than `1 - MIN_COVERAGE` of its readings is dropped rather than
averaged from whatever hours happen to be present — a window that starts at noon
would otherwise report a day of nothing but afternoon sun.

    import wunder as w
    df  = w.fetch("F1_1_ATMOS_SMST1")
    et0 = w.et.reference_et(df)              # mm/day, one value per day
    et0 = w.et.from_forcing("model_input/forcing/FLX_NL-Gl1_...nc")   # same, from the
                                                                     # model's own input
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .export import LOGGER_TZ, saturation_vapour_pressure

#: The Makkink coefficient. 0.65 is the KNMI/Dutch value (De Bruin 1987), which is
#: what makes the result comparable with KNMI's published `EV24`.
MAKKINK_C = 0.65

#: Specific heat of moist air at constant pressure [MJ kg-1 degC-1].
CP = 1.013e-3
#: Ratio of the molecular weights of water vapour and dry air.
EPSILON = 0.622
#: Density of water [kg m-3]; divides out to turn kg m-2 into mm.
RHO_W = 1000.0
#: Fallback air pressure [kPa] when a station reports none. Every site here is
#: below 45 m, so the true value is within ~0.5% of this and the effect on ET0 is
#: smaller still — g enters only through s/(s+g).
STANDARD_PRESSURE = 101.3

#: A day needs at least this fraction of its expected readings to yield an ET0.
MIN_COVERAGE = 0.9

# Logger column names, repeated here rather than imported from plots so this module
# stays free of Plotly.
AIR_T = "Air Temperature observation"
RADIATION = "Radiation observation"
AIR_P = "Air Pressure"
PRECIP = "Precipitation observed"

#: What `reference_et` needs on the frame. `AIR_P` is optional.
REQUIRED = (AIR_T, RADIATION)


def slope_saturation_vapour_pressure(temp_c):
    """Slope of the saturation vapour pressure curve, `s` or `delta` [kPa degC-1].

    The analytic derivative of the same Tetens/Magnus form
    `wunder.export.saturation_vapour_pressure` uses, so the two can never disagree
    about what saturation means.
    """
    return 4098.0 * saturation_vapour_pressure(temp_c) / (temp_c + 237.3) ** 2


def latent_heat(temp_c):
    """Latent heat of vaporisation of water [MJ kg-1].

    Temperature-dependent, as KNMI's own Makkink implementation has it. The
    constant 2.45 that FAO-56 uses instead is the value at 20 degC and differs by
    up to 1.5% over the range these sites see — small, but free to do properly.
    """
    return 2.501 - 2.361e-3 * temp_c


def psychrometric_constant(pressure_kpa, temp_c):
    """Psychrometric constant `g` [kPa degC-1], from station pressure.

    Computed with the same temperature-dependent `latent_heat` as the main
    equation, rather than the tabulated `0.665e-3 * P`, which bakes in
    lambda = 2.45.
    """
    return CP * pressure_kpa / (EPSILON * latent_heat(temp_c))


def makkink(
    temperature_c,
    radiation_mj,
    pressure_kpa=None,
    *,
    coefficient: float = MAKKINK_C,
):
    """Makkink ET0 [mm d-1] from **daily** mean temperature and **daily** total radiation.

    `temperature_c` is the day's mean air temperature [degC], `radiation_mj` the
    day's global radiation total [MJ m-2 d-1], `pressure_kpa` the day's mean air
    pressure [kPa] (or None for `STANDARD_PRESSURE`).

    This is the bare equation on values already reduced to days; `reference_et`
    and `from_forcing` are the entry points that do that reduction correctly.
    """
    if pressure_kpa is None:
        pressure_kpa = STANDARD_PRESSURE
    lam = latent_heat(temperature_c)
    s = slope_saturation_vapour_pressure(temperature_c)
    g = psychrometric_constant(pressure_kpa, temperature_c)
    et0 = coefficient * (s / (s + g)) * radiation_mj / lam * (1000.0 / RHO_W)
    # Radiation is clipped at zero upstream, but a night-biased partial day could
    # still land slightly negative through a sensor offset; ET0 is not negative.
    return et0.clip(lower=0.0) if hasattr(et0, "clip") else max(et0, 0.0)


def _step_seconds(index: pd.DatetimeIndex) -> float:
    """The frame's own sampling interval, in seconds.

    Read from the data rather than assumed: the local cache is 5-minute, the
    published snapshot 30-minute, and a forcing file 30-minute. The expected
    samples per day follow from it, and the coverage test depends on getting it
    right.
    """
    if len(index) < 2:
        return 300.0
    step = pd.Series(index).diff().median()
    return float(step.total_seconds()) if pd.notna(step) else 300.0


def _daily(frame: pd.DataFrame, freq: str, min_coverage: float) -> pd.DataFrame:
    """Daily means of every column, blanked on days that are too incomplete.

    Averaging a half-empty day is the one mistake that matters here: global
    radiation over the daylight hours alone is roughly twice the 24-hour mean, so
    a partial day does not produce a slightly wrong ET0, it produces a doubled
    one. Coverage is judged on radiation, the variable the answer is most
    sensitive to.
    """
    expected = pd.Timedelta(freq).total_seconds() / _step_seconds(frame.index)
    grouped = frame.resample(freq)
    daily = grouped.mean()
    complete = grouped[RADIATION].count() / max(expected, 1) >= min_coverage
    return daily[complete.reindex(daily.index, fill_value=False)]


def reference_et(
    df: pd.DataFrame,
    *,
    freq: str = "1D",
    coefficient: float = MAKKINK_C,
    min_coverage: float = MIN_COVERAGE,
) -> pd.Series:
    """Daily Makkink ET0 [mm d-1] from a logger frame, indexed by day.

    Needs `Air Temperature observation` and `Radiation observation`; uses
    `Air Pressure` when the logger reports it. Returns an empty Series for a
    logger that carries no weather station, which is the common case in this
    network — only four of the fourteen report met variables.

    The index is the logger's own local wall-clock time, so the days are local
    days. That is the right boundary: it is the one KNMI's `EV24` uses, and a
    UTC day would split each Dutch summer afternoon across two totals.
    """
    if df.empty or not all(c in df.columns and df[c].notna().any() for c in REQUIRED):
        return pd.Series(dtype="float64", name="ET0")

    cols = [c for c in (AIR_T, RADIATION, AIR_P) if c in df.columns]
    frame = df[cols].copy()
    frame[RADIATION] = frame[RADIATION].clip(lower=0.0)
    daily = _daily(frame, freq, min_coverage)
    if daily.empty:
        return pd.Series(dtype="float64", name="ET0")

    # Mean W m-2 over 24 h -> MJ m-2 d-1.
    seconds = pd.Timedelta(freq).total_seconds()
    radiation_mj = daily[RADIATION] * seconds / 1e6
    pressure = daily[AIR_P] if AIR_P in daily and daily[AIR_P].notna().any() else None
    et0 = makkink(daily[AIR_T], radiation_mj, pressure, coefficient=coefficient)
    return et0.dropna().rename("ET0")


def from_forcing(
    path,
    *,
    tz: str | None = LOGGER_TZ,
    coefficient: float = MAKKINK_C,
    min_coverage: float = MIN_COVERAGE,
) -> pd.Series:
    """Daily Makkink ET0 [mm d-1] from a STEMMUS_SCOPE forcing NetCDF.

    The same quantity as `reference_et`, computed from exactly what the model was
    fed — gap-filled, complete, and therefore defined on every day of the run.
    Use it to set a modelled evaporation flux against the demand its own input
    implies.

    The file is on UTC; `tz` converts to local time first so the day boundaries
    match `reference_et` and KNMI. Pass `tz=None` to keep UTC days.
    """
    import xarray as xr

    with xr.open_dataset(path) as ds:
        frame = pd.DataFrame(
            {
                AIR_T: ds["Tair"].squeeze(("x", "y")).to_series() - 273.15,
                RADIATION: ds["SWdown"].squeeze(("x", "y")).to_series(),
                AIR_P: ds["Psurf"].squeeze(("x", "y")).to_series() / 1000.0,
            }
        )
    if tz:
        frame.index = frame.index.tz_localize("UTC").tz_convert(tz).tz_localize(None)
        frame = frame.sort_index()
    return reference_et(frame, coefficient=coefficient, min_coverage=min_coverage)


def water_balance(df: pd.DataFrame, *, freq: str = "1D", **kw) -> pd.DataFrame:
    """Daily precipitation, ET0 and their difference [mm], on the days both exist.

    `P - ET0` is the climatic water balance: positive days recharge the profile,
    negative days draw it down. Accumulated over a season it is the standard
    first-order drought index, and it is the natural thing to hold the soil
    moisture record against.
    """
    et0 = reference_et(df, freq=freq, **kw)
    if et0.empty or PRECIP not in df.columns or not df[PRECIP].notna().any():
        return pd.DataFrame(columns=["precip", "et0", "balance"])
    precip = df[PRECIP].clip(lower=0.0).resample(freq).sum(min_count=1)
    out = pd.DataFrame({"precip": precip, "et0": et0}).dropna()
    out["balance"] = out["precip"] - out["et0"]
    return out


def _main(argv=None) -> int:
    """`python -m wunder.et NL-Gl1` — daily ET0 for a site's forcing file or a logger.

    The forcing file is the one the model was actually fed, so this is the demand
    to hold a run's evaporation flux against.
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Makkink reference ET (mm/day).")
    parser.add_argument("target", help="site code (NL-Gl1), logger name or serial, "
                                       "or a path to a forcing NetCDF")
    parser.add_argument("--csv", type=Path, help="write the daily series here")
    parser.add_argument("--coefficient", type=float, default=MAKKINK_C)
    args = parser.parse_args(argv)

    forcing_dir = Path(__file__).resolve().parent.parent / "model_input" / "forcing"
    candidates = sorted(forcing_dir.glob(f"FLX_{args.target}_*.nc"))
    if Path(args.target).suffix == ".nc":
        source, series = args.target, from_forcing(args.target,
                                                   coefficient=args.coefficient)
    elif candidates:
        source, series = candidates[0].name, from_forcing(
            candidates[0], coefficient=args.coefficient)
    else:
        from .fetch import fetch
        from .metadata import logger as _logger

        lg = _logger(args.target)
        source = f"{lg.name} ({lg.serial})"
        series = reference_et(fetch(lg.serial), coefficient=args.coefficient)

    if series.empty:
        print(f"{source}: no complete days with radiation and air temperature.")
        return 1

    monthly = series.groupby([series.index.year, series.index.month]).mean()
    annual = series.groupby(series.index.year).agg(["sum", "count"])
    print(f"{source}  ·  Makkink C={args.coefficient}  ·  {len(series):,} days, "
          f"{series.index.min():%Y-%m-%d} to {series.index.max():%Y-%m-%d}")
    print(f"\nmean {series.mean():.2f} mm/day   "
          f"max {series.max():.2f} mm/day on {series.idxmax():%Y-%m-%d}")
    print("\nannual total (mm), and days contributing:")
    for year, row in annual.iterrows():
        print(f"  {year}  {row['sum']:6.0f}   {int(row['count']):4d} days")
    print("\nmonthly mean (mm/day):")
    for (year, month), value in monthly.items():
        print(f"  {year}-{month:02d}  {value:5.2f}")
    if args.csv:
        series.to_csv(args.csv, header=True)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
