"""Plotly figure builders — publication style, white background, serif type.

Every builder takes a DataFrame and returns a Figure. No Streamlit, no global state, so
they work equally well in a notebook or exported to PNG/SVG for a paper.

Conventions:

* **Serif type (Times New Roman)** and a white ground, so a figure dropped into a
  manuscript matches the body text.
* **Depth carries a distinct hue**, in a fixed order shallow -> deep. The hue set is
  validated for colour-vision deficiency (OKLab dE on adjacent pairs) rather than picked
  by eye; three of the six sit below 3:1 against white, which the always-present legend
  and the app's table/CSV views cover.
* **Colour follows the depth value, not list position** — 20 cm is the same colour on
  every logger, whether or not that logger has a 2.5 cm probe.
* Rainfall is drawn as **bars** (a total over an interval), on a reversed right-hand axis
  above the soil-moisture traces, which is the convention in the soil-physics literature.

Don't change a hex without re-running the palette validator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .et import reference_et as _reference_et
from .et import water_balance as _water_balance
from .metadata import DEPTH_ORDER, Logger
from .metadata import logger as _logger
from .process import depth_columns, depths_of, resample, root_zone, wind_rose_table

# -- palette ----------------------------------------------------------------

SURFACE = "#ffffff"
INK = "#000000"
INK_2 = "#333333"
MUTED = "#666666"
GRID = "#e8e8e4"
AXIS = "#999999"

# Depth hues, shallow -> deep. Validated: adjacent-pair CVD dE 9.1, normal-vision 19.6.
DEPTH_HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# Comparing entities (sites, fields, years). Capped at three overlaid.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MAX_SERIES = len(SERIES)

RAIN = "#7fa9d8"
AIR = "#d03b3b"
# Soil water limits drawn behind a moisture curve -- field capacity and wilting point,
# with the stress threshold between them in a lighter tint because it is derived from a
# chosen depletion fraction, not read off the retention curve.
LIMIT = "#c0392b"
LIMIT_SOFT = "#e08a80"
# The current year in a climatology panel, and its axis when a second axis shares the
# panel with it.
YEAR_INK = "#0d366b"
RAD = "#eda100"
VPD_C = "#1baf7a"
RZ = "#6a3d9a"
# ET0 is the radiation-driven term, so it takes the radiation hue; the running total
# takes a darker shade of it, exactly as the rainfall figure does with RAIN.
ET0_C = RAD
ET0_CUM = "#a86f00"
RAIN_CUM = "#1a5aa8"

FONT = '"Times New Roman", Times, Georgia, serif'

FS_TITLE = 21
FS_LEGEND = 17
FS_AXIS_TITLE = 18
FS_TICK = 16
FS_NOTE = 14

PRECIP = "Precipitation observed"
AIR_T = "Air Temperature observation"
RADIATION = "Radiation observation"
VPD = "Vapor Pressure Deficit"
VP = "Vapor Pressure"

MAX_POINTS = 4000


def depth_color(depth: str) -> str:
    """Hue for a depth. Fixed by depth value, so it is stable across loggers."""
    try:
        return DEPTH_HUES[DEPTH_ORDER.index(str(depth))]
    except ValueError:
        return MUTED


def _col_color(col: str) -> str:
    return depth_color(col.split()[-1].removesuffix("cm"))


def _depth_label(col: str) -> str:
    return col.split()[-1].replace("cm", " cm")


# -- layout -----------------------------------------------------------------


def _style(fig: go.Figure, title: str | None, height: int, *, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="none",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=FS_TICK, color=INK_2),
        title=dict(text=title, font=dict(family=FONT, size=FS_TITLE, color=INK),
                   x=0, xanchor="left", y=0.97, yanchor="top") if title else None,
        margin=dict(l=94, r=94, t=76 if title else 40, b=64),
        height=height,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS,
                        font=dict(family=FONT, size=FS_NOTE, color=INK)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0,
                    font=dict(family=FONT, size=FS_LEGEND, color=INK_2),
                    bgcolor="rgba(0,0,0,0)", itemsizing="constant", tracegroupgap=4),
        bargap=0.02,
    )
    axis_common = dict(
        showline=True, linecolor=INK, linewidth=1, mirror=False,
        ticks="outside", tickcolor=INK, ticklen=5, tickwidth=1,
        tickfont=dict(family=FONT, size=FS_TICK, color=INK_2),
        title_font=dict(family=FONT, size=FS_AXIS_TITLE, color=INK),
        zeroline=False,
    )
    fig.update_xaxes(showgrid=False, **axis_common)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, **axis_common)
    return fig


def _panel(title: str | None = None, height: int = 400, *, secondary: bool = False,
           legend: bool = True) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]]) if secondary else go.Figure()
    return _style(fig, title, height, legend=legend)


def _title(logger: Logger | str | None, text: str) -> str:
    if logger is None:
        return text
    lg = logger if isinstance(logger, Logger) else _logger(logger)
    return f"{text} — {lg.name}"


# -- decimation -------------------------------------------------------------


def _ref_name(ref) -> str:
    """Logger name from a Logger, name or serial -- what wunder.stress looks sites up by."""
    return ref.serial if isinstance(ref, Logger) else str(ref)


def _span_freq(df: pd.DataFrame, max_points: int) -> str | None:
    if df.empty or len(df) <= max_points:
        return None
    seconds = (df.index.max() - df.index.min()).total_seconds() / max_points
    for freq, s in [("5min", 300), ("15min", 900), ("30min", 1800), ("1h", 3600),
                    ("3h", 10800), ("6h", 21600), ("12h", 43200), ("1D", 86400),
                    ("2D", 172800), ("1W", 604800)]:
        if seconds <= s:
            return freq
    return "1W"


def _decimate(df: pd.DataFrame, max_points: int | None) -> pd.DataFrame:
    if max_points is None or df.empty:
        return df
    freq = _span_freq(df, max_points)
    return resample(df, freq) if freq else df


def _precip_freq(df: pd.DataFrame) -> str:
    if df.empty:
        return "1D"
    days = (df.index.max() - df.index.min()).total_seconds() / 86400
    for limit, freq in [(400, "1W"), (120, "1D"), (30, "6h"), (7, "1h")]:
        if days > limit:
            return freq
    return "30min"


def _rain_bars(df: pd.DataFrame, freq: str | None = None) -> tuple[go.Bar | None, str]:
    if PRECIP not in df.columns or not df[PRECIP].notna().any():
        return None, ""
    freq = freq or _precip_freq(df)
    s = df[PRECIP].resample(freq).sum(min_count=1).dropna()
    if s.empty:
        return None, freq
    return go.Bar(x=s.index, y=s.values, name="Precipitation", marker_color=RAIN,
                  marker_line_width=0,
                  hovertemplate="%{y:.1f} mm<extra>Precipitation</extra>"), freq


def _line(x, y, name, color, width=1.5, dash=None, fmt=".3f"):
    return go.Scatter(x=x, y=y, name=name, mode="lines",
                      line=dict(color=color, width=width, dash=dash),
                      hovertemplate="%{y:" + fmt + "}<extra>" + name + "</extra>")


# -- soil -------------------------------------------------------------------


def soil_moisture(
    df: pd.DataFrame,
    *,
    precip: bool = True,
    logger: Logger | str | None = None,
    ylim: tuple[float, float] | None = (0.0, 0.6),
    max_points: int | None = MAX_POINTS,
) -> go.Figure:
    """Soil moisture by depth, with rainfall on a reversed right-hand axis."""
    cols = depth_columns(df, "moisture")
    d = _decimate(df, max_points)
    bar, freq = _rain_bars(df) if precip else (None, "")
    fig = _panel(_title(logger, "Soil moisture"), 460, secondary=bar is not None)

    if bar is not None:
        fig.add_trace(bar, secondary_y=True)
        fig.update_yaxes(title_text=f"Precipitation (mm {freq}<sup>-1</sup>)",
                         secondary_y=True, autorange="reversed", showgrid=False)
    for c in cols:
        tr = _line(d.index, d[c], _depth_label(c), _col_color(c))
        fig.add_trace(tr, secondary_y=False) if bar is not None else fig.add_trace(tr)

    kw = dict(secondary_y=False) if bar is not None else {}
    fig.update_yaxes(title_text="Soil moisture (m<sup>3</sup> m<sup>-3</sup>)",
                     range=list(ylim) if ylim else None, **kw)
    return fig


def soil_temperature(
    df: pd.DataFrame,
    *,
    air: bool = True,
    zero_line: bool = True,
    logger: Logger | str | None = None,
    max_points: int | None = MAX_POINTS,
) -> go.Figure:
    """Soil temperature by depth, with air temperature — same unit, one axis."""
    cols = depth_columns(df, "temperature")
    d = _decimate(df, max_points)
    fig = _panel(_title(logger, "Soil temperature"), 460)

    if air and AIR_T in d.columns and d[AIR_T].notna().any():
        fig.add_trace(_line(d.index, d[AIR_T], "Air", MUTED, width=1.0, fmt=".1f"))
    for c in cols:
        fig.add_trace(_line(d.index, d[c], _depth_label(c), _col_color(c), fmt=".1f"))
    if zero_line:
        fig.add_hline(y=0, line=dict(color=AXIS, width=1, dash="dot"))
    fig.update_yaxes(title_text="Temperature (°C)")
    return fig


def root_zone_moisture(
    df: pd.DataFrame,
    *,
    precip: bool = True,
    method: str = "trapezoid",
    logger: Logger | str | None = None,
    ylim: tuple[float, float] | None = (0.0, 0.6),
    max_points: int | None = MAX_POINTS,
) -> go.Figure:
    """Depth-weighted root-zone soil moisture, with rainfall."""
    rz = root_zone(df, "moisture", method=method)
    d = _decimate(df.assign(_rz=rz), max_points)
    depths = depths_of(depth_columns(df, "moisture"))
    label = f"RZSM ({depths[0]:g}–{depths[-1]:g} cm)" if depths else "RZSM"
    bar, freq = _rain_bars(df) if precip else (None, "")
    fig = _panel(_title(logger, "Root-zone soil moisture"), 460, secondary=bar is not None)

    if bar is not None:
        fig.add_trace(bar, secondary_y=True)
        fig.update_yaxes(title_text=f"Precipitation (mm {freq}<sup>-1</sup>)",
                         secondary_y=True, autorange="reversed", showgrid=False)
    tr = _line(d.index, d["_rz"], label, RZ, width=1.8)
    fig.add_trace(tr, secondary_y=False) if bar is not None else fig.add_trace(tr)
    kw = dict(secondary_y=False) if bar is not None else {}
    fig.update_yaxes(title_text="RZSM (m<sup>3</sup> m<sup>-3</sup>)",
                     range=list(ylim) if ylim else None, **kw)
    return fig


def matric_potential(
    df: pd.DataFrame,
    *,
    logger: Logger | str | None = None,
    max_points: int | None = MAX_POINTS,
) -> go.Figure:
    """Soil water potential by depth (TEROS21). More negative = drier."""
    cols = depth_columns(df, "matric_potential")
    d = _decimate(df, max_points)
    fig = _panel(_title(logger, "Matric potential"), 460)
    for c in cols:
        fig.add_trace(_line(d.index, d[c], _depth_label(c), _col_color(c), fmt=".1f"))
    fig.update_yaxes(title_text="Matric potential (kPa)")
    return fig


def matric_potential_vpd(
    df: pd.DataFrame,
    met: pd.DataFrame | None = None,
    *,
    depths: list[str] | None = None,
    logger: Logger | str | None = None,
    met_logger: Logger | str | None = None,
    max_points: int | None = MAX_POINTS,
) -> go.Figure:
    """Soil matric potential against vapour pressure deficit.

    Soil water supply below, atmospheric demand above — the pair that decides whether a
    plant can keep its stomata open.

    The TEROS21 loggers carry no weather station, so `met` is the frame holding VPD, taken
    from the field's ATMOS-41 logger (see `metadata.met_source`). If `met` is None, VPD is
    looked for in `df` itself.
    """
    source = met if met is not None else df
    col = VPD if (VPD in source.columns and source[VPD].notna().any()) else VP
    cols = depth_columns(df, "matric_potential")
    if depths:
        cols = [c for c in cols if c.split()[-1].removesuffix("cm") in set(depths)]

    d = _decimate(df, max_points)
    fig = _panel(_title(logger, "Matric potential and vapour pressure deficit"), 480,
                 secondary=True)

    if col in source.columns and source[col].notna().any():
        m = _decimate(source[[col]], max_points)
        fig.add_trace(
            go.Scatter(x=m.index, y=m[col], name=("VPD" if col == VPD else "Vapour pressure"),
                       mode="lines", line=dict(color=VPD_C, width=1.2),
                       fill="tozeroy", fillcolor="rgba(27,175,122,0.13)",
                       hovertemplate="%{y:.2f} kPa<extra>" +
                                     ("VPD" if col == VPD else "VP") + "</extra>"),
            secondary_y=True,
        )
        label = "VPD (kPa)" if col == VPD else "Vapour pressure (kPa)"
        if met_logger is not None:
            lgm = met_logger if isinstance(met_logger, Logger) else _logger(met_logger)
            label += f"<br><span style='font-size:11px'>from {lgm.name}</span>"
        fig.update_yaxes(title_text=label, secondary_y=True, showgrid=False)

    for c in cols:
        fig.add_trace(_line(d.index, d[c], _depth_label(c), _col_color(c), fmt=".1f"),
                      secondary_y=False)
    fig.update_yaxes(title_text="Matric potential (kPa)", secondary_y=False)
    return fig


def conductivity(
    df: pd.DataFrame,
    *,
    logger: Logger | str | None = None,
    max_points: int | None = MAX_POINTS,
) -> go.Figure:
    """Bulk electrical conductivity by depth (TEROS12)."""
    cols = depth_columns(df, "conductivity")
    d = _decimate(df, max_points)
    fig = _panel(_title(logger, "Electrical conductivity"), 400)
    for c in cols:
        fig.add_trace(_line(d.index, d[c], _depth_label(c), _col_color(c)))
    fig.update_yaxes(title_text="EC (mS cm<sup>-1</sup>)")
    return fig


# -- meteorology ------------------------------------------------------------


def precipitation(
    df: pd.DataFrame,
    *,
    freq: str | None = None,
    cumulative: bool = False,
    logger: Logger | str | None = None,
) -> go.Figure:
    """Rainfall totals as bars, with an optional cumulative curve."""
    bar, freq = _rain_bars(df, freq)
    fig = _panel(_title(logger, "Precipitation"), 440, secondary=cumulative)
    if bar is None:
        return fig
    bar.marker.color = RAIN
    fig.add_trace(bar, secondary_y=False) if cumulative else fig.add_trace(bar)
    kw = dict(secondary_y=False) if cumulative else {}
    fig.update_yaxes(title_text=f"Precipitation (mm {freq}<sup>-1</sup>)", **kw)
    if cumulative:
        s = df[PRECIP].resample(freq).sum(min_count=1).fillna(0).cumsum()
        fig.add_trace(_line(s.index, s.values, "Cumulative", "#1a5aa8", width=1.8, fmt=".0f"),
                      secondary_y=True)
        fig.update_yaxes(title_text="Cumulative (mm)", secondary_y=True, showgrid=False)
    return fig


def temperature_radiation(
    df: pd.DataFrame,
    *,
    logger: Logger | str | None = None,
    max_points: int | None = MAX_POINTS,
) -> go.Figure:
    """Air and root-zone soil temperature with solar radiation on the right axis."""
    d = _decimate(df.assign(_rzst=root_zone(df, "temperature")), max_points)
    fig = _panel(_title(logger, "Temperature and radiation"), 460, secondary=True)

    if RADIATION in d.columns and d[RADIATION].notna().any():
        fig.add_trace(
            go.Scatter(x=d.index, y=d[RADIATION], name="Solar radiation", mode="lines",
                       fill="tozeroy", line=dict(color=RAD, width=0.7),
                       fillcolor="rgba(237,161,0,0.22)",
                       hovertemplate="%{y:.0f} W m<sup>-2</sup><extra>Radiation</extra>"),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Solar radiation (W m<sup>-2</sup>)",
                         secondary_y=True, showgrid=False)
    if AIR_T in d.columns and d[AIR_T].notna().any():
        fig.add_trace(_line(d.index, d[AIR_T], "Air temperature", AIR, fmt=".1f"),
                      secondary_y=False)
    if d["_rzst"].notna().any():
        fig.add_trace(_line(d.index, d["_rzst"], "Root-zone soil temperature", RZ,
                            width=1.8, fmt=".1f"), secondary_y=False)
    fig.add_hline(y=0, line=dict(color=AXIS, width=1, dash="dot"))
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    return fig


def vpd_temperature(
    df: pd.DataFrame,
    *,
    logger: Logger | str | None = None,
    max_points: int | None = MAX_POINTS,
) -> go.Figure:
    """Vapour pressure deficit with air temperature on the right axis."""
    d = _decimate(df, max_points)
    col = VPD if (VPD in d.columns and d[VPD].notna().any()) else VP
    fig = _panel(_title(logger, "Vapour pressure deficit and air temperature"), 420,
                 secondary=True)
    if col in d.columns and d[col].notna().any():
        fig.add_trace(_line(d.index, d[col], col, VPD_C, width=1.4, fmt=".2f"),
                      secondary_y=False)
    if AIR_T in d.columns and d[AIR_T].notna().any():
        fig.add_trace(_line(d.index, d[AIR_T], "Air temperature", AIR, width=1.2, fmt=".1f"),
                      secondary_y=True)
        fig.update_yaxes(title_text="Air temperature (°C)", secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text=f"{col} (kPa)", secondary_y=False)
    return fig


def reference_et(
    df: pd.DataFrame,
    *,
    precip: bool = True,
    cumulative: bool = True,
    logger: Logger | str | None = None,
) -> go.Figure:
    """Daily Makkink reference ET, against rainfall, with the running balance.

    Rain and ET0 are both mm over the same day, so they share one axis and one bar
    width -- side by side, at the same scale, supply against demand. That is the whole
    point of the figure and a second y-scale would destroy it.

    The line on the right axis is the running total of `P - ET0` over the window shown.
    It rises while rain outpaces demand and falls through a dry spell, so the depth of
    its trough is the accumulated deficit the soil has had to cover. It restarts at the
    left edge of the window, not on 1 January -- see `climatology(kind="et0_cumulative")`
    for the year-to-date view.
    """
    et0 = _reference_et(df)
    fig = _panel(_title(logger, "Makkink reference evapotranspiration"), 460,
                 secondary=cumulative)
    if et0.empty:
        return _panel(_title(logger, "Reference ET — no weather station on this logger"),
                      320)

    rain = None
    if precip and PRECIP in df.columns and df[PRECIP].notna().any():
        rain = df[PRECIP].clip(lower=0.0).resample("1D").sum(min_count=1)
        rain = rain.reindex(et0.index)

    kw = dict(secondary_y=False) if cumulative else {}
    if rain is not None:
        fig.add_trace(go.Bar(
            x=rain.index, y=rain.values, name="Precipitation", marker_color=RAIN,
            hovertemplate="%{y:.1f} mm<extra>Rain</extra>"), **kw)
    fig.add_trace(go.Bar(
        x=et0.index, y=et0.values, name="Reference ET (Makkink)", marker_color=ET0_C,
        hovertemplate="%{y:.2f} mm<extra>ET<sub>0</sub></extra>"), **kw)
    fig.update_layout(barmode="group", bargap=0.15, bargroupgap=0.0)
    fig.update_yaxes(title_text="Water (mm d<sup>-1</sup>)", **kw)

    if cumulative:
        if rain is not None:
            balance = (rain.fillna(0.0) - et0).cumsum()
            fig.add_trace(_line(balance.index, balance.values,
                                "Running P − ET<sub>0</sub>", RZ, width=1.8, fmt=".0f"),
                          secondary_y=True)
            fig.update_yaxes(title_text="Cumulative P − ET<sub>0</sub> (mm)",
                             secondary_y=True, showgrid=False)
            fig.add_hline(y=0, line=dict(color=AXIS, width=1, dash="dot"),
                          secondary_y=True)
        else:
            run = et0.cumsum()
            fig.add_trace(_line(run.index, run.values, "Cumulative ET<sub>0</sub>",
                                ET0_CUM, width=1.8, fmt=".0f"), secondary_y=True)
            fig.update_yaxes(title_text="Cumulative ET<sub>0</sub> (mm)",
                             secondary_y=True, showgrid=False)
    return fig


def water_limited_et(
    df: pd.DataFrame,
    *,
    met: pd.DataFrame | None = None,
    ref: str | None = None,
    p: float = 0.5,
    crop_coefficient: float = 1.0,
    logger: Logger | str | None = None,
) -> go.Figure:
    """Actual ET against reference ET, with the water stress factor behind it.

    Two bars a day: what the atmosphere asked (ET0, pale) and what the soil could
    supply (WSF * Kc * ET0, solid). The gap between them *is* the water stress, so it
    is drawn as a gap rather than as a third series. The line on the right axis is the
    water stress factor itself -- FAO-56 calls it Ks, which the code keeps as a column
    name; every label the reader sees says WSF -- from 1 (the profile can meet any demand) to 0 (wilting point).
    """
    from .stress import actual_et

    ref = ref if ref is not None else logger
    try:
        out = actual_et(df, met, ref=_ref_name(ref) if ref is not None else None,
                        p=p, crop_coefficient=crop_coefficient)
    except (FileNotFoundError, ValueError) as exc:
        return _panel(_title(logger, f"Actual ET — {exc}".split(".")[0]), 320)
    if out.empty:
        return _panel(_title(logger, "Actual ET — needs a weather station and soil "
                                     "parameters"), 320)

    fig = _panel(_title(logger, "Actual evapotranspiration and water stress"), 470,
                 secondary=True)
    fig.add_trace(go.Bar(
        x=out.index, y=out["et0"], name="Reference ET (demand)",
        marker_color="rgba(237,161,0,0.32)", marker_line_width=0,
        hovertemplate="%{y:.2f} mm<extra>ET<sub>0</sub></extra>"), secondary_y=False)
    fig.add_trace(go.Bar(
        x=out.index, y=out["et"], name="Actual ET (supplied)", marker_color=ET0_CUM,
        marker_line_width=0,
        hovertemplate="%{y:.2f} mm<extra>ET</extra>"), secondary_y=False)
    # Overlaid, not grouped: actual ET is a *part* of the demand, so it belongs inside
    # the same bar rather than beside it.
    fig.update_layout(barmode="overlay", bargap=0.15)
    fig.update_yaxes(title_text="ET (mm d<sup>-1</sup>)", secondary_y=False)

    fig.add_trace(_line(out.index, out["ks"], "WSF (water stress factor)", RZ,
                        width=1.6, fmt=".2f"), secondary_y=True)
    fig.update_yaxes(title_text="WSF (—)", secondary_y=True, range=[0, 1.05],
                     showgrid=False)
    threshold = 1.0 - p
    fig.add_hline(y=threshold, line=dict(color=AXIS, width=1, dash="dot"),
                  secondary_y=True,
                  annotation_text=f"stress begins (p = {p:g})",
                  annotation_position="right",
                  annotation_font=dict(family=FONT, size=FS_NOTE, color=MUTED))
    return fig


def weekly_balance(
    df: pd.DataFrame,
    met: pd.DataFrame | None = None,
    *,
    ref: str | None = None,
    weeks: int = 52,
    crop_coefficient: float = 1.0,
    p: float = 0.5,
    logger: Logger | str | None = None,
) -> go.Figure:
    """Week by week: rain, the demand, what evaporated, and what was left over.

    Top row, three upright bars a week: rain, reference ET (what the atmosphere
    asked) and actual ET (what the soil could supply). Reading them side by side is
    the whole point -- the gap between the two ET bars is the water stress, and
    whether rain clears them is whether the week paid for itself.

    Bottom row, one bar a week: P - ET, the net the profile gained or lost. Above
    zero the store filled, below zero it paid the difference. Its own row because a
    signed residual and the fluxes it comes from do not belong on one scale.
    """
    from .stress import actual_et

    ref = ref if ref is not None else logger
    try:
        out = actual_et(df, met, ref=_ref_name(ref) if ref is not None else None,
                        p=p, crop_coefficient=crop_coefficient)
    except (FileNotFoundError, ValueError) as exc:
        return _panel(_title(logger, f"Weekly water balance — {exc}".split(".")[0]), 320)
    if out.empty:
        return _panel(_title(logger, "Weekly water balance — needs a weather station "
                                     "and soil parameters"), 320)

    # Rain comes from whichever frame carries the gauge -- the station's, when the soil
    # logger has none of its own.
    wx = met if met is not None and _has_data(met, PRECIP) else df
    if not _has_data(wx, PRECIP):
        return _panel(_title(logger, "Weekly water balance — no rain gauge"), 320)

    daily = pd.DataFrame({
        "precip": wx[PRECIP].resample("1D").sum(min_count=1),
        "et0": out["et0"],
        "et": out["et"],
    }).dropna()
    if daily.empty:
        return _panel(_title(logger, "Weekly water balance — no overlapping record"), 320)

    totals = {c: _weekly_total(daily[c]) for c in daily.columns}
    weekly = pd.DataFrame(totals).dropna().tail(weeks)
    # Widths carried by index, not by position: a week missing from one of the three
    # series drops out of the middle of the frame, and a positional slice would then
    # hand every later bar the wrong width.
    spans = pd.Series(totals["precip"].attrs["width_days"], index=totals["precip"].index)
    width = spans.reindex(weekly.index).fillna(7).to_numpy() * 0.88 * 86_400_000
    balance = weekly["precip"] - weekly["et"]

    # Two rows, one x axis: three fluxes above, their residual below. A signed residual
    # on the same axis as the fluxes would spend half the height on nothing.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.38], vertical_spacing=0.05)
    _style(fig, _title(logger, "Weekly water balance"), 540)

    # Three bars abreast, the triple centred on the week they belong to. Explicit
    # offsets rather than Plotly's grouping, because the widths vary: a part-finished
    # week is drawn narrow, and grouped bars would then sit off its centre.
    third = width / 3.0
    for i, (col, name, colour, opacity) in enumerate((
            ("precip", "rain", RAIN, 1.0),
            ("et0", "reference ET (demand)", ET0_C, 0.45),
            ("et", "actual ET (supplied)", ET0_CUM, 1.0))):
        fig.add_trace(go.Bar(
            x=weekly.index, y=weekly[col], width=third,
            offset=(i - 1.5) * third, name=name,
            marker_color=colour, opacity=opacity, marker_line_width=0,
            hovertemplate="%{y:.1f} mm<extra>" + name + "</extra>"), row=1, col=1)

    # One bar, signed: blue where the week put water in, brown where the store paid.
    # Out of the legend -- a two-colour series has no single swatch, and its one
    # swatch would collide with the ET bars it is not.
    fig.add_trace(go.Bar(
        x=weekly.index, y=balance, width=width, offset=-width / 2, name="P − ET",
        marker_color=[RAIN_CUM if v >= 0 else ET0_CUM for v in balance],
        marker_line_width=0, showlegend=False,
        hovertemplate="%{y:+.1f} mm<extra>P − ET</extra>"), row=2, col=1)

    fig.update_layout(barmode="overlay", bargap=0.15)
    for row in (1, 2):
        fig.add_hline(y=0, line=dict(color=INK, width=1), row=row, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="mm per week", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="P − ET<br>(mm per week)", row=2, col=1)
    return fig


def wind_rose(
    df: pd.DataFrame,
    *,
    sectors: int = 16,
    calm_threshold: float = 0.5,
    logger: Logger | str | None = None,
) -> go.Figure:
    """Wind rose. Bars point in the direction the wind blows *from*.

    Calm records are excluded and reported separately — about 30% of this network's record
    is below 0.5 m/s, where the vane direction is noise.
    """
    table, calm, n = wind_rose_table(df, sectors=sectors, calm_threshold=calm_threshold)
    fig = go.Figure()
    if table.empty:
        return _style(fig, _title(logger, "Wind rose — no data"), 460)

    # Speed is ordered, so the ramp is ordinal: light = slow, dark = fast.
    ramp = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
    idx = np.linspace(0, len(ramp) - 1, len(table.columns)).round().astype(int)
    for i, col in enumerate(table.columns):
        fig.add_trace(go.Barpolar(r=table[col].values, theta=table.index.tolist(),
                                  name=col, marker_color=ramp[idx[i]],
                                  marker_line_color=SURFACE, marker_line_width=0.5,
                                  hovertemplate="%{theta} · %{r:.1f}%<extra>" + col + "</extra>"))
    fig.update_layout(
        template="none", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=FS_TICK, color=INK_2),
        title=dict(text=_title(logger, "Wind rose"),
                   font=dict(family=FONT, size=FS_TITLE, color=INK), x=0, xanchor="left"),
        barmode="stack", height=560, margin=dict(l=50, r=160, t=76, b=72),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02,
                    font=dict(family=FONT, size=FS_LEGEND, color=INK_2),
                    title=dict(text="Wind speed", font=dict(family=FONT, size=FS_NOTE))),
        polar=dict(
            bgcolor=SURFACE,
            angularaxis=dict(direction="clockwise", rotation=90, gridcolor=GRID,
                             linecolor=AXIS,
                             tickfont=dict(family=FONT, size=FS_TICK, color=INK_2)),
            radialaxis=dict(ticksuffix="%", angle=45, tickangle=45, gridcolor=GRID,
                            linecolor=GRID,
                            tickfont=dict(family=FONT, size=FS_NOTE, color=MUTED)),
        ),
        annotations=[dict(text=f"calm &lt; {calm_threshold:g} m s<sup>-1</sup>: "
                               f"{100 * calm:.0f}%  ·  n = {n:,}",
                          showarrow=False, xref="paper", yref="paper", x=0, y=-0.07,
                          align="left",
                          font=dict(family=FONT, size=FS_NOTE, color=MUTED))],
    )
    return fig


# -- comparison -------------------------------------------------------------

_UNITS = {
    "moisture": "m<sup>3</sup> m<sup>-3</sup>",
    "temperature": "°C",
    "matric_potential": "kPa",
    "conductivity": "mS cm<sup>-1</sup>",
}


def _series_for(df: pd.DataFrame, measure: str, depth: str | None,
                max_points: int | None) -> pd.Series | None:
    if df.empty:
        return None
    if depth:
        from .metadata import measures as _m
        col = f"{_m()[measure]} {depth}cm"
        if col not in df.columns or not df[col].notna().any():
            return None
        return _decimate(df[[col]], max_points)[col]
    rz = root_zone(df, measure, method="trapezoid")
    return _decimate(rz.to_frame("v"), max_points)["v"]


def compare(
    frames: dict[str, pd.DataFrame],
    *,
    measure: str = "moisture",
    depth: str | None = None,
    title: str | None = None,
    max_points: int | None = MAX_POINTS,
) -> go.Figure:
    """Compare one quantity across loggers, fields or sites.

    `frames` maps a label to that logger's data. Pass `depth` for a single depth, or leave
    it None for the depth-weighted root-zone average — the only fair comparison when
    loggers carry different depth sets. Up to three overlaid; beyond that use
    `compare_facets`.
    """
    if len(frames) > MAX_SERIES:
        raise ValueError(
            f"{len(frames)} series overlaid; max {MAX_SERIES} stay distinguishable. "
            "Use compare_facets() for more."
        )
    what = (f"{measure.replace('_', ' ')} at {depth} cm" if depth
            else f"root-zone {measure.replace('_', ' ')}")
    fig = _panel(title or f"Comparison — {what}", 420)
    for i, (label, df) in enumerate(frames.items()):
        s = _series_for(df, measure, depth, max_points)
        if s is None:
            continue
        fig.add_trace(_line(s.index, s.values, label, SERIES[i], width=1.5))
    fig.update_yaxes(title_text=f"{what[0].upper()}{what[1:]} ({_UNITS.get(measure, '')})")
    if measure == "moisture":
        fig.update_yaxes(range=[0, 0.6])
    return fig


def compare_series(
    series: dict[str, pd.Series],
    *,
    ylabel: str = "",
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
    fmt: str = ".3f",
    max_points: int | None = MAX_POINTS,
) -> go.Figure:
    """Overlay prepared series — for comparisons the caller has already aggregated.

    Field-level comparison uses this: a field is several loggers, so the caller averages
    them into one series per field and passes those. Capped at three, like `compare`.
    """
    if len(series) > MAX_SERIES:
        raise ValueError(f"{len(series)} series; max {MAX_SERIES}. Facet instead.")
    fig = _panel(title, 430)
    for i, (label, s) in enumerate(series.items()):
        if s is None or s.empty:
            continue
        s = _decimate(s.to_frame("v"), max_points)["v"]
        fig.add_trace(_line(s.index, s.values, label, SERIES[i], width=1.6, fmt=fmt))
    fig.update_yaxes(title_text=ylabel, range=list(ylim) if ylim else None)
    return fig


def compare_facets(
    frames: dict[str, pd.DataFrame],
    *,
    measure: str = "moisture",
    depth: str | None = None,
    title: str | None = None,
    max_points: int | None = MAX_POINTS,
) -> go.Figure:
    """Small multiples — one panel per entity, shared axes. Use past three entities."""
    n = len(frames)
    what = (f"{measure.replace('_', ' ')} at {depth} cm" if depth
            else f"root-zone {measure.replace('_', ' ')}")
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.045, subplot_titles=list(frames))
    _style(fig, title or f"Comparison — {what}", max(320, 165 * n), legend=False)
    for i, (label, df) in enumerate(frames.items(), start=1):
        s = _series_for(df, measure, depth, max_points)
        if s is None:
            continue
        fig.add_trace(_line(s.index, s.values, label, SERIES[0]), row=i, col=1)
        if measure == "moisture":
            fig.update_yaxes(range=[0, 0.6], row=i, col=1)
    for ann in fig.layout.annotations:
        ann.font = dict(family=FONT, size=FS_LEGEND, color=INK)
        ann.x, ann.xanchor = 0, "left"
    fig.update_yaxes(title_text=_UNITS.get(measure, ""))
    return fig


def compare_variables(
    df: pd.DataFrame,
    columns: list[str],
    *,
    units: dict[str, str] | None = None,
    logger: Logger | str | None = None,
    normalise: bool = False,
    max_points: int | None = MAX_POINTS,
) -> go.Figure:
    """Several variables from one logger, as stacked panels sharing a time axis.

    Different variables carry different units, so each gets its own panel rather than
    being forced onto a shared scale. `normalise=True` instead indexes everything to
    0–1 and overlays it, which is only useful for comparing *shape*, never magnitude.
    """
    columns = [c for c in columns if c in df.columns and df[c].notna().any()]
    if not columns:
        return _panel(_title(logger, "No variables selected"), 300)
    units = units or {}
    d = _decimate(df[columns], max_points)

    if normalise:
        fig = _panel(_title(logger, "Variables (scaled 0–1)"), 440)
        for i, c in enumerate(columns):
            s = d[c]
            rng = s.max() - s.min()
            fig.add_trace(_line(s.index, (s - s.min()) / rng if rng else s * 0,
                                c, DEPTH_HUES[i % len(DEPTH_HUES)]))
        fig.update_yaxes(title_text="Scaled to range (–)")
        return fig

    n = len(columns)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    _style(fig, _title(logger, "Variables"), max(300, 175 * n), legend=False)
    for i, c in enumerate(columns, start=1):
        color = DEPTH_HUES[(i - 1) % len(DEPTH_HUES)]
        if c == PRECIP:
            bar, freq = _rain_bars(df)
            if bar is not None:
                fig.add_trace(bar, row=i, col=1)
                fig.update_yaxes(title_text=f"mm {freq}<sup>-1</sup>", row=i, col=1)
                continue
        fig.add_trace(_line(d.index, d[c], c, color), row=i, col=1)
        u = units.get(c, "")
        short = c.replace(" observation", "").replace("Soil ", "")
        fig.update_yaxes(title_text=f"{short}<br>({u})" if u else short, row=i, col=1,
                         title_font=dict(family=FONT, size=FS_NOTE, color=INK))
    return fig


# -- year on year -----------------------------------------------------------


def _weekly_total(step: pd.Series, freq: str = "1W") -> pd.Series:
    """Sum a daily series into weeks, indexed at the middle of the days each covers."""
    step = step.dropna()
    if step.empty:
        return step

    total = step.resample(freq).sum(min_count=1)
    when = pd.Series(step.index, index=step.index).resample(freq)
    first, last = when.min(), when.max()
    keep = total.notna() & first.notna()
    out = pd.Series(total[keep].to_numpy(),
                    index=pd.DatetimeIndex(first[keep] + (last[keep] - first[keep]) / 2))
    out.attrs["width_days"] = ((last[keep] - first[keep]).dt.days + 1).to_numpy()
    return out


def _common_calendar(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map any year onto 2000 (a leap year) so seasons overlay by date."""
    return pd.to_datetime(
        {"year": 2000, "month": idx.month, "day": idx.day}
    ) + pd.to_timedelta(idx.hour, unit="h")


def seasonal(
    df: pd.DataFrame,
    column: str | None = None,
    *,
    measure: str = "moisture",
    years: list[int] | None = None,
    freq: str = "1D",
    logger: Logger | str | None = None,
) -> go.Figure:
    """One line per year on a common calendar — where this year sits against last year.

    Pass `years` to compare specific years; otherwise all are drawn, with the most recent
    emphasised. The dotted vertical line is today's date.
    """
    if df.empty:
        return _panel(_title(logger, "Year on year — no data"), 400)

    if column:
        if column not in df.columns or not df[column].notna().any():
            return _panel(_title(logger, f"{column} not available"), 300)
        s, label, unit = df[column].dropna(), column, ""
    else:
        s = root_zone(df, measure, method="trapezoid").dropna()
        label = f"Root-zone {measure.replace('_', ' ')}"
        unit = _UNITS.get(measure, "")
    if s.empty:
        return _panel(_title(logger, "Year on year — no data"), 400)

    s = s.resample(freq).agg("sum" if column == PRECIP else "mean").dropna()
    available = sorted(s.index.year.unique())
    show = [y for y in (years or available) if y in available]
    if not show:
        return _panel(_title(logger, "Year on year — no matching years"), 300)

    fig = _panel(_title(logger, f"{label} — year on year"), 460)
    newest = show[-1]
    # Years are ordered, so use a light->dark blue ramp; the current year is thicker.
    ramp = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
    idx = np.linspace(0, len(ramp) - 1, max(len(show), 2)).round().astype(int)
    for i, y in enumerate(show):
        part = s[s.index.year == y]
        if part.empty:
            continue
        is_now = y == newest and not years
        fig.add_trace(go.Scatter(
            x=_common_calendar(part.index), y=part.values, name=str(y), mode="lines",
            line=dict(color=SERIES[i] if years and len(show) <= MAX_SERIES
                      else ramp[idx[i]], width=2.0 if is_now else 1.4),
            hovertemplate="%{x|%d %b} " + str(y) + ": %{y:.3f}<extra></extra>",
        ))
    today = pd.Timestamp.now()
    fig.add_vline(x=pd.Timestamp(2000, today.month, today.day).timestamp() * 1000,
                  line=dict(color=AXIS, width=1, dash="dot"),
                  annotation_text="today", annotation_position="top",
                  annotation_font=dict(family=FONT, size=FS_NOTE, color=MUTED))
    fig.update_xaxes(tickformat="%b", dtick="M1")
    fig.update_yaxes(title_text=f"{label} ({unit})" if unit else label)
    if measure == "moisture" and not column:
        fig.update_yaxes(range=[0, 0.6])
    return fig


#: What `climatology` can plot. Cumulative series restart each 1 January, so a reading is
#: "this much so far this year" and is directly comparable with the same date in past years.
#: Cumulative kinds come first: they answer "where does this year stand today", which is the
#: summary question. The instantaneous state follows.
CLIMATOLOGY_KINDS = {
    "precip_cumulative": "Cumulative precipitation",
    "et0_cumulative": "Cumulative reference ET",
    "rzsm": "Root-zone soil moisture",
    "ks": "Water stress factor (WSF)",
    "et_cumulative": "Cumulative actual ET",
    "balance_cumulative": "Cumulative P − ET₀",
    "vpd_cumulative": "Cumulative vapour pressure deficit",
}

#: Kinds that need this logger's own weather station. Only four of the fourteen loggers
#: report met variables, so a caller listing kinds for a soil logger must drop these.
MET_KINDS = {"precip_cumulative", "et0_cumulative", "balance_cumulative",
             "et_cumulative", "ks", "vpd_cumulative"}

#: Kinds that also need the site's soil parameters, so they need a `ref` to look the
#: site up and are skipped when it is missing or the site has not been extracted.
SOIL_KINDS = {"et_cumulative", "ks"}

#: A column has to be genuinely instrumented, not merely non-empty, to drive a plot.
#: z6-21179 carries 92 weather records from a sensor attached for one day in 2023 and
#: nothing since -- 0.03% coverage, which must not unlock a rainfall comparison.
MIN_COVERAGE = 0.5


def _has_data(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().mean() >= MIN_COVERAGE


def _climatology_series(df: pd.DataFrame, kind: str, freq: str,
                        ref: str | None = None, met: pd.DataFrame | None = None,
                        soil: pd.DataFrame | None = None
                        ) -> tuple[pd.Series, str, bool]:
    """(series, y-axis label, is_cumulative) for a climatology plot."""
    # A quantity comes from whichever instrument actually measures it: weather from the
    # field's ATMOS-41, soil water from the nearest logger with working probes. Falling
    # back to `df` keeps every existing single-frame call working unchanged.
    wx = met if met is not None else df
    sm = soil if soil is not None else df

    if kind == "rzsm":
        s = root_zone(sm, "moisture", method="trapezoid").dropna()
        return (s.resample(freq).mean().dropna(),
                f"Root-zone soil moisture ({_UNITS['moisture']})", False)

    if kind == "precip_cumulative":
        if not _has_data(wx, PRECIP):
            return pd.Series(dtype="float64"), "", True
        daily = wx[PRECIP].resample(freq).sum(min_count=1).fillna(0)
        return daily.groupby(daily.index.year).cumsum(), "Cumulative precipitation (mm)", True

    if kind in ("et0_cumulative", "balance_cumulative"):
        # Both start from daily Makkink ET0, which needs radiation and air temperature.
        if not (_has_data(wx, RADIATION) and _has_data(wx, AIR_T)):
            return pd.Series(dtype="float64"), "", True
        if kind == "et0_cumulative":
            daily = _reference_et(wx, freq=freq)
            ylab = "Cumulative reference ET (mm)"
        else:
            # P - ET0: the climatic water balance. Unlike the other cumulative kinds it
            # goes both ways, so the year-to-date value reads directly as a surplus or a
            # deficit rather than only as a distance from the median.
            if not _has_data(wx, PRECIP):
                return pd.Series(dtype="float64"), "", True
            daily = _water_balance(wx, freq=freq)["balance"]
            ylab = "Cumulative P − ET₀ (mm)"
        if daily.empty:
            return pd.Series(dtype="float64"), "", True
        daily = daily.reindex(
            pd.date_range(daily.index.min(), daily.index.max(), freq=freq)
        ).fillna(0.0)
        return daily.groupby(daily.index.year).cumsum(), ylab, True

    if kind in ("et_cumulative", "ks"):
        # Both come from the same pair -- the station's ET0 and the soil's WSF -- so
        # they are computed together and the kind only picks which one to return.
        # Needs the site's soil parameters, hence `ref`.
        if ref is None or not (_has_data(wx, RADIATION) and _has_data(wx, AIR_T)):
            return pd.Series(dtype="float64"), "", kind == "et_cumulative"
        from .stress import root_zone_stress

        et0 = _reference_et(wx, freq=freq)
        if et0.empty:
            return pd.Series(dtype="float64"), "", kind == "et_cumulative"
        try:
            ks, _ = root_zone_stress(sm, ref=_ref_name(ref))
        except (FileNotFoundError, ValueError):
            return pd.Series(dtype="float64"), "", kind == "et_cumulative"
        if ks.empty:
            return pd.Series(dtype="float64"), "", kind == "et_cumulative"
        daily_ks = ks.resample(freq).mean()

        if kind == "ks":
            # A state, not an accumulation: 1 means the profile can meet whatever the
            # atmosphere asks, 0 means it is at wilting point.
            return daily_ks.dropna(), "Water stress factor, WSF (—)", False

        actual = (et0 * daily_ks).dropna()
        if actual.empty:
            return pd.Series(dtype="float64"), "", True
        actual = actual.reindex(
            pd.date_range(actual.index.min(), actual.index.max(), freq=freq)
        ).fillna(0.0)
        return (actual.groupby(actual.index.year).cumsum(),
                "Cumulative actual ET (mm)", True)

    if kind == "vpd_cumulative":
        col = VPD if _has_data(wx, VPD) else VP
        if not _has_data(wx, col):
            return pd.Series(dtype="float64"), "", True
        # Daily mean VPD accumulated: an index of how much atmospheric demand the year has
        # delivered so far. Units are kPa-days.
        daily = wx[col].resample(freq).mean().fillna(0)
        return (daily.groupby(daily.index.year).cumsum(),
                f"Cumulative {'VPD' if col == VPD else 'vapour pressure'} (kPa d)", True)

    raise ValueError(f"unknown kind {kind!r}; choose from {sorted(CLIMATOLOGY_KINDS)}")


def climatology_standing(df: pd.DataFrame, kind: str, *, freq: str = "1D",
                         ref: str | None = None, met: pd.DataFrame | None = None,
                         soil: pd.DataFrame | None = None) -> dict | None:
    """Where this year stands today against the same date in previous complete years.

    Returns the current value, the median and range of earlier years on this calendar day,
    and the difference — the numbers behind the climatology chart, for a headline readout.
    None when there is nothing comparable.
    """
    if df.empty:
        return None
    try:
        s, ylab, cumulative = _climatology_series(df, kind, freq, ref, met, soil)
    except ValueError:
        return None
    if s.empty:
        return None

    newest = s.index.year.max()
    partial = {
        y for y, part in s.groupby(s.index.year)
        if y != newest
        and (part.index.min().dayofyear > 15 or part.index.max().dayofyear < 350)
    }
    s = s[~s.index.year.isin(partial)]
    past, now = s[s.index.year < newest], s[s.index.year == newest]
    if past.empty or now.empty:
        return None

    doy = now.index[-1].dayofyear
    # Compare like with like: the same calendar day in earlier years, with a few days'
    # tolerance so a missing day doesn't blank the comparison.
    same_day = past[(past.index.dayofyear - doy).map(abs) <= 3]
    if same_day.empty:
        return None

    current, median = float(now.iloc[-1]), float(same_day.median())
    return {
        "kind": kind,
        "label": CLIMATOLOGY_KINDS.get(kind, kind),
        "unit": ylab,
        "cumulative": cumulative,
        "year": int(newest),
        "as_of": now.index[-1],
        "current": current,
        "median": median,
        "min": float(same_day.min()),
        "max": float(same_day.max()),
        "delta": current - median,
        # A percentage of a quantity that changes sign is nonsense: a normal year at
        # -50 mm and this year at -20 mm is 30 mm *less* dry, but reads as "-60%".
        "pct": ((current - median) / median * 100
                if median and kind != "balance_cumulative" else None),
        "years": sorted(int(y) for y in past.index.year.unique()),
        "excluded": sorted(int(y) for y in partial),
    }


def _add_soil_limits(fig: go.Figure, df: pd.DataFrame,
                     ref: Logger | str | None) -> dict:
    """Dotted field capacity, stress onset and wilting point behind a moisture chart.

    Weighted over the same depths, by the same method, as the moisture series itself
    (`stress.root_zone_limits`), so the curve and the lines are commensurable -- a
    curve touching the lowest line really is this profile at wilting point.

    Three lines, not two: the pair of soil bounds says what the soil can hold, and the
    one between them says where the plants start to feel it, which is the reading the
    curve is usually being interrogated for. It is drawn paler because it is derived
    from a chosen depletion fraction rather than measured off the retention curve.

    Silent when the site's soil parameters are missing: the moisture curve is still
    worth showing without them. Returns the limits it drew, or `{}`.
    """
    if ref is None:
        return {}
    from .stress import root_zone_limits

    try:
        info = root_zone_limits(df, ref=_ref_name(ref))
    except (FileNotFoundError, ValueError):
        return {}
    if not info:
        return {}
    lines = (("theta_fc", "field capacity", LIMIT),
             ("threshold", f"stress begins (p = {info['p']:g})", LIMIT_SOFT),
             ("theta_wp", "wilting point", LIMIT))
    for key, label, colour in lines:
        # Labelled at the right, inside the plot: the curve's own year runs out at
        # "today", so the right-hand months are the empty part of the panel. Outside
        # the axis the text would be clipped by the margin.
        fig.add_hline(y=info[key], line=dict(color=colour, width=1, dash="dot"),
                      annotation_text=f"{label} {info[key]:.3f}",
                      annotation_position="top right",
                      annotation_font=dict(family=FONT, size=FS_NOTE, color=colour))
    return info


def climatology(
    df: pd.DataFrame,
    *,
    kind: str = "rzsm",
    freq: str = "1D",
    logger: Logger | str | None = None,
    measure: str | None = None,
    ref: str | None = None,
    met: pd.DataFrame | None = None,
    soil: pd.DataFrame | None = None,
) -> go.Figure:
    """This year against the spread of previous years, on a common calendar.

    One line, one axis, one quantity. The weekly totals that used to ride a second
    axis here live in `weekly_balance`, where rain and evaporation share a single mm
    scale instead of a running total being read against a per-week one.

    The shaded band is the min-max envelope of every earlier year and the pale line their
    median, so the current year can be read as wetter or drier, ahead or behind, relative to
    what is normal for the date. A dotted vertical marks today.

    `kind` is one of CLIMATOLOGY_KINDS. Cumulative kinds restart on 1 January, which is what
    makes "we are 90 mm behind by this date" a meaningful statement.
    """
    if measure is not None and kind == "rzsm" and measure != "moisture":
        # Only moisture is depth-averageable; see process.EXTENSIVE.
        raise ValueError("climatology(kind='rzsm') applies to soil moisture only")
    title = CLIMATOLOGY_KINDS.get(kind, kind)
    if df.empty:
        return _panel(_title(logger, f"{title} — no data"), 400)

    s, ylab, cumulative = _climatology_series(df, kind, freq,
                                              ref if ref is not None else logger,
                                              met, soil)
    if s.empty:
        return _panel(_title(logger, f"{title} — not available on this logger"), 320)

    # Only complete past years go into the comparison. A year whose record starts in May
    # accumulates from zero in May and reads as a freakishly dry year, dragging the band
    # down; one that stops in August leaves a phantom gap. The current year is exempt —
    # being partial is the whole point of it.
    newest = s.index.year.max()
    by_year = s.groupby(s.index.year)
    partial = {
        y for y, part in by_year
        if y != newest
        and (part.index.min().dayofyear > 15 or part.index.max().dayofyear < 350)
    }
    dropped = sorted(partial)
    s = s[~s.index.year.isin(partial)]
    if s.empty or s.index.year.nunique() < 2:
        return _panel(_title(logger, f"{title} — no complete earlier year to compare"), 320)

    past, now = s[s.index.year < newest], s[s.index.year == newest]
    fig = _panel(_title(logger, f"{title} — {newest} vs previous years"), 470)
    if dropped:
        fig.add_annotation(
            text=f"{', '.join(str(y) for y in dropped)} excluded — incomplete year",
            showarrow=False, xref="paper", yref="paper", x=1, y=-0.16, xanchor="right",
            font=dict(family=FONT, size=FS_NOTE, color=MUTED))
        fig.update_layout(margin=dict(l=94, r=94, t=76, b=82))

    if len(past):
        g = past.groupby(past.index.dayofyear)
        lo, hi, med = g.min(), g.max(), g.median()
        x = pd.to_datetime(lo.index - 1, unit="D", origin=pd.Timestamp("2000-01-01"))
        fig.add_trace(go.Scatter(x=x, y=hi.values, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=x, y=lo.values, mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(42,120,214,0.13)",
            name=f"{past.index.year.min()}–{newest - 1}", hoverinfo="skip"))
        fig.add_trace(_line(x, med.values, "median", "#86b6ef",
                            width=1.3, fmt=".1f" if cumulative else ".3f"))
    if len(now):
        x = pd.to_datetime(now.index.dayofyear - 1, unit="D",
                           origin=pd.Timestamp("2000-01-01"))
        fig.add_trace(_line(x, now.values, str(newest), YEAR_INK, width=2.2,
                            fmt=".1f" if cumulative else ".3f"))
        # Where the year currently stands, called out rather than left to the axis.
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[now.values[-1]], mode="markers", showlegend=False,
            marker=dict(color=YEAR_INK, size=7, line=dict(color=SURFACE, width=1.5)),
            hovertemplate="%{y:.1f}<extra>latest</extra>" if cumulative
            else "%{y:.3f}<extra>latest</extra>"))

    today = pd.Timestamp.now()
    fig.add_vline(x=pd.Timestamp(2000, today.month, today.day).timestamp() * 1000,
                  line=dict(color=AXIS, width=1, dash="dot"),
                  annotation_text="today", annotation_position="top",
                  annotation_font=dict(family=FONT, size=FS_NOTE, color=MUTED))
    fig.update_xaxes(tickformat="%b", dtick="M1")
    fig.update_yaxes(title_text=ylab,
                     range=[0, 0.6] if kind == "rzsm"
                     else [0, 1.02] if kind == "ks" else None)
    if kind == "rzsm":
        # The two lines that turn a moisture curve into a statement about the plants:
        # water is held between them, and nothing below the lower one is available.
        _add_soil_limits(fig, soil if soil is not None else df,
                         ref if ref is not None else logger)
    if kind == "balance_cumulative":
        # This one crosses zero, and which side of it the year sits on is the reading.
        fig.add_hline(y=0, line=dict(color=AXIS, width=1, dash="dot"))
    return fig


# -- generic / diagnostics --------------------------------------------------


def timeseries(
    df: pd.DataFrame,
    column: str,
    *,
    unit: str = "",
    logger: Logger | str | None = None,
    max_points: int | None = MAX_POINTS,
) -> go.Figure:
    """One variable, one panel."""
    if column == PRECIP:
        return precipitation(df, logger=logger)
    d = _decimate(df, max_points)
    fig = _panel(_title(logger, column), 380, legend=False)
    if column in d.columns:
        fig.add_trace(_line(d.index, d[column], column, SERIES[0]))
    fig.update_yaxes(title_text=f"{column} ({unit})" if unit else column)
    return fig


def coverage_heatmap(
    df: pd.DataFrame,
    *,
    freq: str = "1D",
    logger: Logger | str | None = None,
) -> go.Figure:
    """When each sensor was reporting — shows exactly when a probe went dark."""
    from .process import coverage

    cov = coverage(df, freq)
    if cov.empty:
        return _panel(_title(logger, "Coverage — no data"), 300)
    fig = go.Figure(go.Heatmap(
        z=cov.T.values, x=cov.index, y=cov.columns,
        colorscale=[[0, "#f5f5f3"], [0.5, "#9ec5f4"], [1, "#1c5cab"]],
        zmin=0, zmax=1, ygap=1,
        colorbar=dict(title=dict(text="reporting",
                                 font=dict(family=FONT, size=FS_NOTE)),
                      thickness=12, outlinewidth=0,
                      tickfont=dict(family=FONT, size=FS_NOTE, color=MUTED)),
        hovertemplate="%{y}<br>%{x|%Y-%m-%d}: %{z:.0%}<extra></extra>",
    ))
    _style(fig, _title(logger, "Data coverage"), max(320, 24 * len(cov.columns) + 130),
           legend=False)
    fig.update_layout(margin=dict(l=250, r=34, t=76, b=64), hovermode="closest")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig
