"""Turn the logger record into STEMMUS_SCOPE forcing variables.

This module owns three things the rest of the forcing pipeline builds on:

  * **the UTC grid** — the loggers stamp local wall-clock time *with* daylight
    saving, so getting to UTC is a real conversion, not an offset (see `to_utc`);
  * **unit conversion and derived variables** — the logger's units are not the
    model's, and `RH`, `Qair` and `LAI` are not measured at all;
  * **the NetCDF writers** — `FLX_<SITE>_FLUXNET2015_FULLSET_<y0>-<y1>.nc` and
    `<SITE>_<date>_InitialCondition.nc`, in the shape both engines read.

The forcing file is built in two passes (see `wunder.gapfill`): an ERA5-Land
baseline everywhere, then the one weather station the file belongs to. This
module supplies the station pass and the writers; `wunder.era5land` supplies the
baseline. That ordering is what lets a location with no logger at all still
produce a complete, runnable file.

Nothing here imports Streamlit, cdsapi or xarray at module level beyond what the
writers need, so `import wunder` stays cheap for the viewer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The loggers report local wall-clock time including the DST shift. See the note
#: in sites.yaml `meta.timezone` for the evidence.
LOGGER_TZ = "Europe/Amsterdam"

#: Model time step. STEMMUS_SCOPE is hard-wired to 1800 s (`dt` in the Python
#: port's run config); the MATLAB path derives it from the file's own spacing.
FREQ = "30min"
STEP_SECONDS = 1800

# --- logger column -> forcing variable -------------------------------------
AIR_T = "Air Temperature observation"
AIR_P = "Air Pressure"
PRECIP = "Precipitation observed"
RADIATION = "Radiation observation"
VAPOR_PRESSURE = "Vapor Pressure"
VPD = "Vapor Pressure Deficit"
WIND_SPEED = "Wind speed observation"

#: Logger columns that carry into the forcing file, in the order they are needed.
MET_COLUMNS = [AIR_T, AIR_P, PRECIP, RADIATION, VAPOR_PRESSURE, VPD, WIND_SPEED]

#: Forcing variables that get a companion `<name>_qc` flag. `RH`, `Qair` and
#: `LAI` do not, matching the reference PLUMBER2 file: the first two would only
#: inherit `VPD_qc` and `Tair_qc`, and LAI has a single source.
QC_VARIABLES = ["Tair", "SWdown", "LWdown", "VPD", "Psurf", "Precip", "Wind", "CO2air"]

#: Provenance of every value, written to the file's global attributes.
QC_MEANING = {
    0: "measured by this file's own weather station",
    2: "ERA5-Land baseline, bias-corrected against that station",
    3: "parameterised from measured Tair/ea/SWdown",
}

QC_MEASURED = 0
#: Reserved, never emitted. It meant "measured by a neighbouring station", which
#: the design no longer permits: one file describes one mast, and a gap is filled
#: from ERA5-Land rather than from a neighbour whose microclimate may differ
#: silently. Kept so the numbering matches FLUXNET's and 0/2/3 keep their meaning.
QC_SISTER = 1
QC_ERA5 = 2
QC_PARAMETERISED = 3

UNITS = {
    "Tair": "K",
    "SWdown": "W/m2",
    "LWdown": "W/m2",
    "VPD": "hPa",
    "Psurf": "Pa",
    "Precip": "kg/m2/s",
    "Wind": "m/s",
    "RH": "%",
    "Qair": "kg/kg",
    "CO2air": "ppm",
    "LAI": "m2/m2",
}

#: Physically impossible values, used to screen inputs and to check outputs. Wide
#: enough to admit any real Dutch half-hour, narrow enough to catch a unit error.
RANGES = {
    "Tair": (250.0, 320.0),
    "SWdown": (0.0, 1400.0),
    "LWdown": (100.0, 500.0),
    "VPD": (0.0, 80.0),
    "Psurf": (90000.0, 110000.0),
    "Precip": (0.0, 0.02),
    "Wind": (0.0, 40.0),
    "RH": (0.0, 100.0),
    "Qair": (0.0, 0.05),
    "CO2air": (350.0, 500.0),
    "LAI": (0.0, 10.0),
}


def to_utc(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Reindex a logger frame from local wall-clock time onto UTC.

    The loggers stamp Europe/Amsterdam wall-clock time, so a naive `+2h` is an
    hour wrong for the whole winter half of every year.

    Two DST artefacts, both confirmed in the record rather than assumed:

    * Spring forward — 02:00-02:55 simply does not exist (the file steps 01:55 ->
      03:00). `nonexistent="NaT"` marks any such stamp and it is then dropped;
      there are none in practice, which is the point of checking.
    * Autumn fall back — the repeated hour appears **once**, not twice. The whole
      cache holds zero duplicate timestamps, so the server evidently keeps the
      first write and drops the second. `ambiguous=True` reads that single copy as
      the first (summer-time) occurrence, which is the one that was written. It is
      a one-hour uncertainty over 12 rows a year.
    """
    index = frame.index
    if index.tz is not None:
        return frame.tz_convert("UTC")

    localised = index.tz_localize(LOGGER_TZ, ambiguous=True, nonexistent="NaT")
    out = frame.copy()
    out.index = localised
    if localised.isna().any():
        out = out[localised.notna()]
    return out.tz_convert("UTC")


def saturation_vapour_pressure(temp_c: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Saturation vapour pressure over water [kPa] from temperature [degC].

    Tetens/Magnus. This is the form the ATMOS-41 itself uses: across 347k records
    at z6-21176, the logger's own `Vapor Pressure Deficit` equals
    `es(this formula) - Vapor Pressure` to a mean absolute difference of
    0.0010 kPa. One helper serves both the logger and ERA5-Land paths so the two
    can never drift apart.
    """
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def relative_humidity(vapour_pressure_kpa, vpd_kpa):
    """RH [%] from measured vapour pressure and VPD, both kPa.

    `es = ea + VPD` holds exactly for this logger (see
    `saturation_vapour_pressure`), so RH needs no saturation formula at all and
    no temperature — only two measured quantities. That is one fewer place for a
    coefficient choice to matter.

    Clipped to 100: the derivation overshoots in 9.7% of records, to a maximum of
    102.8%, which is ordinary saturation noise rather than a fault.
    """
    saturation = vapour_pressure_kpa + vpd_kpa
    rh = 100.0 * vapour_pressure_kpa / saturation
    return rh.clip(0.0, 100.0) if hasattr(rh, "clip") else np.clip(rh, 0.0, 100.0)


def specific_humidity(vapour_pressure_kpa, pressure_kpa):
    """Specific humidity [kg/kg] from vapour pressure and air pressure, both kPa.

    Standard form, `q = 0.622 ea / (p - 0.378 ea)`. Note that
    `preprocessICOSdata`'s `calculate_specific_humidity_mixing_ratio` returns
    `0.622 ea / (p - ea)` — that is the mixing ratio, and its two return values
    are named the wrong way round; it is not reused here.
    """
    return 0.622 * vapour_pressure_kpa / (pressure_kpa - 0.378 * vapour_pressure_kpa)


#: The state actually carried through the three merge passes. `VPD`, `RH` and
#: `Qair` are NOT among them: all three are functions of `Tair`, `ea_kPa` and
#: `Psurf`, so they are derived once at the end by `derive_humidity` from
#: whichever pass supplied each primitive.
#:
#: Correcting or merging the derived trio independently would let them drift out
#: of mutual consistency -- and the MATLAB path recomputes `ea` from our `RH`
#: (`forcing_io.py`: `ea = calculate_ea(t_air_celcius, rh)`), so an inconsistent
#: trio would put two different vapour pressures into the same run.
PRIMITIVES = ["Tair", "ea_kPa", "Psurf", "SWdown", "LWdown", "Precip", "Wind"]


def from_logger(met: pd.DataFrame) -> pd.DataFrame:
    """Convert a 30-minute logger frame into forcing primitives and model units.

    Expects the columns in `MET_COLUMNS`, already resampled to 30 minutes by
    `wunder.process.resample` (which sums precipitation and averages the rest)
    and already on a UTC index. Columns absent from `met` are absent from the
    result rather than filled -- the caller overlays this onto a complete
    baseline, so a missing column means "leave the baseline showing", which is
    exactly the intended behaviour for a dead sensor.

    No `LWdown`: nothing in the network measures downwelling longwave.
    """
    out = pd.DataFrame(index=met.index)

    if AIR_T in met:
        out["Tair"] = met[AIR_T] + 273.15
    if RADIATION in met:
        out["SWdown"] = met[RADIATION].clip(lower=0.0)
    if AIR_P in met:
        out["Psurf"] = met[AIR_P] * 1000.0
    if PRECIP in met:
        # mm accumulated over the half-hour -> kg m-2 s-1
        out["Precip"] = met[PRECIP].clip(lower=0.0) / STEP_SECONDS
    if WIND_SPEED in met:
        out["Wind"] = met[WIND_SPEED].clip(lower=0.0)
    if VAPOR_PRESSURE in met:
        out["ea_kPa"] = met[VAPOR_PRESSURE].clip(lower=0.0)

    return out


def derive_humidity(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `VPD`, `RH` and `Qair` from the merged `Tair`, `ea_kPa` and `Psurf`.

    Run once, after all three passes, so the trio is internally consistent no
    matter which source supplied each primitive.

    Using `es(Tair)` rather than the logger's own reported VPD costs nothing:
    across 347k records the two agree to a mean absolute 0.0010 kPa, because this
    is the saturation curve the ATMOS-41 itself uses.
    """
    out = frame.copy()
    es = saturation_vapour_pressure(out["Tair"] - 273.15)      # kPa
    ea = out["ea_kPa"].clip(upper=es)                          # supersaturation
    out["VPD"] = ((es - ea) * 10.0).clip(lower=0.0)            # hPa
    out["RH"] = (100.0 * ea / es).clip(0.0, 100.0)
    out["Qair"] = specific_humidity(ea, out["Psurf"] / 1000.0)
    return out


def screen(frame: pd.DataFrame, ranges: dict[str, tuple[float, float]] | None = None
           ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Blank values outside physical range, and report how many per variable.

    Applied to the logger passes before they override the baseline, so a spike
    leaves the reanalysis value standing rather than replacing it with nonsense.
    `z6-21176` reports a 26.9 m/s gust at a sheltered forest site whose 99th
    percentile is 5.2 m/s, which is the kind of thing this is for.
    """
    limits = ranges or RANGES
    out = frame.copy()
    dropped: dict[str, int] = {}
    for name, (low, high) in limits.items():
        if name not in out:
            continue
        bad = (out[name] < low) | (out[name] > high)
        count = int(bad.sum())
        if count:
            out.loc[bad, name] = np.nan
            dropped[name] = count
    return out, dropped


# --- NetCDF writers --------------------------------------------------------

#: Order the reference PLUMBER2 file uses, so a diff against it is readable.
FORCING_ORDER = [
    "Tair", "SWdown", "LWdown", "VPD", "Psurf", "Precip", "Wind", "RH",
    "CO2air", "Qair", "LAI",
]

#: Statics both engines read. PyStemmusScope indexes all six with no guard
#: (`forcing_io.py` lines 96-103), so a missing one is a KeyError when the MATLAB
#: path builds forcing_globals.mat -- not a warning.
STATIC_FLOATS = ["latitude", "longitude", "reference_height", "canopy_height",
                 "elevation"]
STATIC_STRINGS = ["IGBP_veg_short", "IGBP_veg_long"]


def _point(value: float) -> "np.ndarray":
    return np.array([[value]], dtype="float32")


def build_dataset(forcing: pd.DataFrame, qc: pd.DataFrame, statics: dict,
                  attrs: dict | None = None):
    """Assemble the forcing Dataset in the shape both engines read.

    Dims `(x=1, y=1, time)`, float32 throughout, `time` as minutes since the
    first step. Every variable is written explicitly rather than looped over the
    frame, so a missing one fails here instead of as a bare `KeyError` inside
    whichever engine loads it.
    """
    import xarray as xr

    missing = [v for v in FORCING_ORDER if v not in forcing.columns]
    if missing:
        raise ValueError(f"forcing is missing {missing}; both engines index these by name")
    for name in STATIC_FLOATS + STATIC_STRINGS:
        if statics.get(name) is None:
            raise ValueError(
                f"static {name!r} is not set. PyStemmusScope reads it unguarded; "
                "fill it in sites.yaml (see forcing/download_statics.py)."
            )

    index = forcing.index
    data = {}
    for name in FORCING_ORDER:
        values = forcing[name].to_numpy(dtype="float32").reshape(1, 1, -1)
        data[name] = xr.DataArray(values, dims=("x", "y", "time"),
                                  attrs={"units": UNITS[name]})
    for name in QC_VARIABLES:
        if name not in qc.columns:
            continue
        values = qc[name].to_numpy(dtype="float32").reshape(1, 1, -1)
        data[f"{name}_qc"] = xr.DataArray(values, dims=("x", "y", "time"))

    for name in STATIC_FLOATS:
        data[name] = xr.DataArray(_point(float(statics[name])), dims=("x", "y"))
    for name in STATIC_STRINGS:
        data[name] = xr.DataArray(
            np.array([[str(statics[name])]], dtype="S200"), dims=("x", "y")
        )

    ds = xr.Dataset(
        data,
        coords={
            "x": np.array([1.0]),
            "y": np.array([2.0]),
            "time": index.tz_convert(None) if index.tz else index,
        },
    )
    ds.attrs.update(attrs or {})
    return ds


def write_forcing(ds, path, *, start: pd.Timestamp | None = None):
    """Write the forcing Dataset, with the time encoding the engines expect."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    origin = (start or pd.Timestamp(ds["time"].values[0])).strftime("%Y-%m-%d")
    encoding = {
        "time": {"units": f"minutes since {origin}", "calendar": "proleptic_gregorian",
                 "dtype": "int64"},
    }
    for name in ds.data_vars:
        if ds[name].dtype.kind == "f":
            encoding[name] = {"_FillValue": np.float32(np.nan), "dtype": "float32"}
    ds.to_netcdf(path, encoding=encoding, engine="netcdf4")
    return path


def write_initial_condition(values: dict, latitude: float, longitude: float,
                            when: pd.Timestamp, path, attrs: dict | None = None):
    """Write the nine ERA5-Land scalars, in the layout both engines read.

    `skt`, `stl1-4`, `swvl1-4` as scalar floats with `latitude`, `longitude` and
    `time` as coordinates. Always ERA5-Land: nothing in the network measures skin
    temperature, and `stl4`/`swvl4` span 100-289 cm while the deepest WUNDER probe
    is 80 cm -- and those map to the model's bottom boundary condition.
    """
    import xarray as xr
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = when.tz_convert(None) if when.tz else when

    ds = xr.Dataset({k: xr.DataArray(np.float32(v)) for k, v in values.items()})
    ds = ds.assign_coords(
        latitude=np.float32(latitude),
        longitude=np.float32(longitude),
        time=stamp,
    )
    ds.attrs.update(attrs or {})
    ds.to_netcdf(
        path,
        encoding={"time": {"units": f"days since {stamp:%Y-%m-%d} 00:00:00",
                           "calendar": "proleptic_gregorian", "dtype": "int64"}},
        engine="netcdf4",
    )
    return path
