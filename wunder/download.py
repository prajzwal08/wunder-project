"""CSV files a reader can take away, each carrying its own units and method.

Plain functions returning text: the app's sidebar offers them, and a notebook can call
them. Comment lines start with `#`, so `pandas.read_csv(path, comment="#")` reads a file
back as it is.
"""

from __future__ import annotations

import pandas as pd

from . import flux as _flux
from .et import reference_et
from .metadata import forcing_sites, housekeeping_columns
from .process import CIRCULAR, WIND_SPEED, resample, root_zone

PRECIP = "Precipitation observed"
RAIN, ETO = "Rain [mm/day]", "Makkink ETo [mm/day]"
WSF, ETA = "Water stress factor, WSF [-]", "Makkink ETa (ETo x WSF) [mm/day]"
EDDY, MODEL = "Eddy covariance ET [mm/day]", "STEMMUS-SCOPE ET [mm/day]"
NET_ETO, NET_ETA = "Rain - Makkink ETo [mm/day]", "Rain - Makkink ETa [mm/day]"
RZSM, RZST = "Root-zone soil moisture [m3/m3]", "Root-zone soil temperature [degC]"

#: What the sidebar offers: category -> (time step, what it holds). Daily categories list
#: `daily` labels; 30-minute ones list column prefixes.
CATEGORIES = {
    "Evapotranspiration, daily": ("daily", (EDDY, MODEL, ETO, ETA)),
    "Rain and water balance, daily": ("daily", (RAIN, NET_ETO, NET_ETA)),
    "Water stress and root zone, daily": ("daily", (WSF, RZSM, RZST)),
    "Soil moisture by depth, 30 min": ("30min", ("Soil moisture",)),
    "Soil temperature by depth, 30 min": (
        "30min", ("Soil temperature", "Soil Temperature second measurement")),
    "Matric potential by depth, 30 min": ("30min", ("Matric Potential",)),
    "Electrical conductivity by depth, 30 min": ("30min", ("Electric conductivity",)),
    "Weather, 30 min": ("30min", (
        "Precipitation", "Air ", "Radiation", "Vapor", "Wind")),
}

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


def measurement_columns(df: pd.DataFrame) -> dict[str, str]:
    """Label with unit -> column, for every variable this logger reports."""
    hk = set(housekeeping_columns())
    return {f"{c} [{_unit(c)}]" if _unit(c) else c: c
            # Not merely non-empty: F2_2 carries one day of weather from 2023 and nothing
            # since, which is not a weather record to offer anyone.
            for c in df.columns if c not in hk and df[c].notna().mean() >= 0.01}


def offer(df: pd.DataFrame, daily_labels) -> dict[str, list[str]]:
    """Category -> the exact variables this logger has in it; empty ones are left out."""
    raw = measurement_columns(df)
    out = {}
    for name, (step, members) in CATEGORIES.items():
        have = ([m for m in members if m in daily_labels] if step == "daily" else
                [label for label in raw if label.startswith(members)])
        if have:
            out[name] = have
    return out


def measurements(lg, df: pd.DataFrame, start=None, end=None, variables=None) -> str:
    """The logger's measurements at 30 minutes; `variables` are labels, None for all."""
    names = measurement_columns(df)
    picked = [v for v in names if not variables or v in variables]
    d = _between(df, start, end)
    # Wind direction is averaged as a vector weighted by speed, so speed has to be present
    # for the averaging whether or not the reader asked for it.
    cols = [names[v] for v in picked]
    extra = [WIND_SPEED] if any(c in CIRCULAR for c in cols) and WIND_SPEED in d else []
    d = resample(d[list(dict.fromkeys(cols + extra))], "30min")[cols]
    d.columns = picked
    notes = _about(lg) + [
        "30-minute values: rainfall is summed, wind direction is a vector mean, all other "
        "variables are averaged. The data are not quality-checked."]
    return _csv(d.dropna(how="all"), notes, "time")


def daily(lg, soil: pd.DataFrame | None, met: pd.DataFrame | None, *,
          soil_lg=None, met_lg=None, own: pd.DataFrame | None = None
          ) -> tuple[pd.DataFrame, dict[str, str]]:
    """`(frame, notes)`: every daily series for this logger, and how each was made.

    Columns are labels with units. The eddy covariance tower and STEMMUS-SCOPE are there
    for Field 1 loggers, blank outside the period they cover.
    """
    parts: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}

    def add(label, series, note):
        if series is not None and series.notna().any():
            parts[label], notes[label] = series, note

    if met is not None:
        where = f"weather from {met_lg.name}"
        if PRECIP in met:
            add(RAIN, met[PRECIP].clip(lower=0).resample("1D").sum(min_count=1),
                f"daily rainfall; {where}")
        add(ETO, reference_et(met), "Makkink reference evapotranspiration, coefficient "
            f"0.65; days with less than 90% of radiation readings are left out; {where}")
    if soil is not None:
        from .stress import profile_stress
        where = f"soil moisture from {soil_lg.name}"
        try:
            wsf, _, _ = profile_stress(soil, ref=soil_lg.serial)
        except (FileNotFoundError, ValueError):
            wsf = pd.Series(dtype="float64")
        if not wsf.empty:
            add(WSF, wsf.resample("1D").mean(), "water stress factor, averaged over layers "
                f"by thickness; soil properties from SoilGrids; {where}")
            if ETO in parts:
                add(ETA, parts[ETO] * parts[WSF], "ETo x WSF, with crop coefficient 1")
    if RAIN in parts:
        for name, label in ((ETO, NET_ETO), (ETA, NET_ETA)):
            if name in parts:
                add(label, parts[RAIN] - parts[name],
                    "daily rainfall minus evapotranspiration; drainage and runoff are "
                    "not included")
    if soil is not None:
        add(RZSM, root_zone(soil).resample("1D").mean(),
            f"average over depth, weighted by layer thickness; {where}")
    if own is not None:
        add(RZST,
            root_zone(own, "temperature").resample("1D").mean(),
            "average over depth, weighted by layer thickness")
    if lg.serial in forcing_sites().get(_flux.SITE, {}).get("soil_loggers", []):
        f = _flux.read()
        if not f.empty:
            add(EDDY, f["eddy"],
                "LE / 2.45 MJ kg-1, June to August 2025; blank when fewer than 24 of 48 "
                "half-hours are measured or more than 2 are neither measured nor "
                "best-class gap-filled")
            add(MODEL, f["model"],
                "simulated ET, June to August 2025, forced with the Field 1 weather "
                "station")
    return pd.DataFrame(parts), notes


def daily_csv(lg, frame: pd.DataFrame, notes: dict[str, str], start=None, end=None,
              variables=None) -> str:
    """The chosen columns of `daily` as text, each explained in the header."""
    cols = [c for c in frame.columns if not variables or c in variables]
    out = _between(frame[cols], start, end).dropna(how="all")
    head = _about(lg) + [
        "One value per day: totals for rain and evapotranspiration, means for the rest. "
        "Calculated from measurements that are not quality-checked."]
    return _csv(out, head + [f"{c}: {notes[c]}" for c in cols], "date", 3)
