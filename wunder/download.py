"""CSV files a reader can take away, each carrying its own units and method.

Plain functions returning text: the app's sidebar offers them, and a notebook can call
them. Comment lines start with `#`, so `pandas.read_csv(path, comment="#")` reads a file
back as it is.
"""

from __future__ import annotations

import pandas as pd

from . import flux as _flux
from .et import reference_et
from .metadata import housekeeping_columns
from .process import resample, root_zone

#: Units by column prefix, as the API reports them. Not read from the local cache, which a
#: deployment does not have. Longest match wins, and case matters: "Soil temperature" and
#: "Soil Temperature second measurement" are different sensors.
UNITS = {
    "Air Pressure": "kPa", "Air Temperature observation": "degC",
    "Precipitation observed": "mm", "Radiation observation": "W/m2",
    "Vapor Pressure": "kPa", "Vapor Pressure Deficit": "kPa",
    "Wind direction observation": "deg, from", "Wind speed observation": "m/s",
    "Soil moisture": "m3/m3", "Soil temperature": "degC", "Matric Potential": "kPa",
    "Electric conductivity": "mS/cm", "Soil Temperature second measurement": "degC",
}


def _unit(column: str) -> str:
    hits = [k for k in UNITS if column.startswith(k)]
    return UNITS[max(hits, key=len)] if hits else ""


def _between(df, start, end):
    """Rows from `start` to `end`, both days included."""
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index < pd.Timestamp(end) + pd.Timedelta(days=1)]
    return df


def _csv(frame: pd.DataFrame, notes: list[str], index: str, decimals: int = 4) -> str:
    frame = frame.round(decimals)
    frame.index.name = index
    head = "".join(f"# {n}\n" for n in notes)
    return head + frame.to_csv(date_format="%Y-%m-%d %H:%M" if index == "time" else "%Y-%m-%d")


def _about(lg) -> list[str]:
    return ["WUNDER soil moisture network",
            f"logger: {lg.name} ({lg.serial}), {lg.site_name}, {lg.field_name}",
            f"location: {lg.latitude:.4f} N, {lg.longitude:.4f} E",
            "time: local time, Europe/Amsterdam"]


def measurements(lg, df: pd.DataFrame, start=None, end=None) -> str:
    """Everything the logger measures, at 30 minutes."""
    d = _between(df, start, end)
    d = d[[c for c in d.columns if c not in set(housekeeping_columns())]]
    d = resample(d, "30min").dropna(axis=1, how="all").dropna(how="all")
    notes = _about(lg) + [
        "30-minute values: rainfall is summed, wind direction is a vector mean, all other "
        "variables are averaged. The data are not quality-checked."]
    out = d.rename(columns={c: f"{c} [{_unit(c)}]" for c in d.columns if _unit(c)})
    return _csv(out, notes, "time")


def daily(lg, soil: pd.DataFrame | None, met: pd.DataFrame | None, *,
          soil_lg=None, met_lg=None, own: pd.DataFrame | None = None,
          start=None, end=None) -> str:
    """Daily rain, ETo, WSF, Makkink ETa, balances and root-zone state."""
    parts = {}
    notes = _about(lg) + ["Calculated from the measurements, which are not quality-checked."]
    if met is not None:
        notes.append(f"weather from {met_lg.name}")
        if "Precipitation observed" in met:
            r = met["Precipitation observed"].clip(lower=0).resample("1D")
            parts["rain [mm]"] = r.sum(min_count=1)
        parts["ETo [mm]"] = reference_et(met)
        notes.append("ETo: Makkink reference evapotranspiration, coefficient 0.65; days "
                     "with less than 90% of radiation readings are left out")
    if soil is not None:
        notes.append(f"soil moisture from {soil_lg.name}")
        parts["root-zone soil moisture [m3/m3]"] = root_zone(soil).resample("1D").mean()
        from .stress import profile_stress
        try:
            wsf, _, _ = profile_stress(soil, ref=soil_lg.serial)
        except (FileNotFoundError, ValueError):
            wsf = pd.Series(dtype="float64")
        if not wsf.empty:
            parts["WSF [-]"] = wsf.resample("1D").mean()
            notes.append("WSF: water stress factor, averaged over layers by thickness; "
                         "soil properties from SoilGrids")
            if "ETo [mm]" in parts:
                parts["Makkink ETa [mm]"] = parts["ETo [mm]"] * parts["WSF [-]"]
                notes.append("Makkink ETa = ETo x WSF, with crop coefficient 1")
    if own is not None:
        t = root_zone(own, "temperature")
        if t.notna().any():
            parts["root-zone soil temperature [degC]"] = t.resample("1D").mean()
    out = pd.DataFrame(parts)
    if "rain [mm]" in out:
        for col in ("ETo [mm]", "Makkink ETa [mm]"):
            if col in out:
                out[f"rain - {col}"] = out["rain [mm]"] - out[col]
    out = _between(out.dropna(how="all"), start, end)
    return _csv(out, notes, "date", 3)


def fluxes() -> str:
    """Daily ET from the eddy tower and from STEMMUS-SCOPE."""
    f = _flux.read().rename(columns={
        "eddy": "eddy covariance ET [mm]", "model": "STEMMUS-SCOPE ET [mm]",
        "eddy_measured": "measured half-hours [of 48]"})
    return _csv(f, [
        "Glanerbeek, Field 1: daily evapotranspiration, June to August 2025",
        "time: local days, Europe/Amsterdam",
        "eddy covariance ET = LE / 2.45 MJ kg-1; blank when fewer than 24 of 48 "
        "half-hours are measured or more than 2 are neither measured nor best-class "
        "gap-filled",
        "STEMMUS-SCOPE: simulated ET, forced with the Field 1 weather station",
    ], "date", 3)
