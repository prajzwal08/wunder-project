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

import re

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

# Depth ramp, shallow -> deep: dark red at the surface to deep blue at 80 cm, so the
# ordering of the profile is legible without reading the legend. The path runs through
# purple rather than straight from red to blue, because a direct interpolation passes
# through near-white in the middle two steps and those lines vanish on this white ground.
DEPTH_HUES = ["#7f0000", "#b5192b", "#d4456f", "#9b4fa8", "#3a5fbf", "#0d2a6b"]

# Comparing entities (sites, fields, years). Capped at three overlaid.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MAX_SERIES = len(SERIES)

RAIN = "#7fa9d8"
AIR = "#d03b3b"
# The four soil water limits, drawn behind a moisture curve and as the marks on the WSF
# explainer, so the two always agree. Wet at the top of the profile, dry at the bottom:
# saturation, field capacity, the halfway point where WSF = 0.5, and residual.
THETA_SAT_C = "#0d2a6b"   # deep blue
THETA_FC_C = "#1a7a3c"    # green
THETA_HALF_C = "#d9a400"  # yellow, always dotted -- it is a derived midpoint, not a
                          # property read off the retention curve
THETA_R_C = "#7f0000"     # dark red
# The current year in a climatology panel, and its axis when a second axis shares the
# panel with it.
YEAR_INK = "#0d366b"
# Earlier years in a climatology panel, oldest first: one line each, pale enough to sit
# behind the current year but separable from one another. Sampled by position so a
# logger with three years of record and one with six put the same calendar year at
# roughly the same shade.
PAST_YEARS = ["#d4e2f4", "#b6cdea", "#98b6df", "#7a9fd3", "#5c88c5", "#3f70ad"]
# The year a reader has picked out. It needs its own strong colour rather than just full
# opacity: the pale end of the ramp above is background even at full strength and double
# width, so "2023 is selected" would be invisible on exactly the years hardest to see.
# Burnt orange reads against both the blue ramp and the black current year, and survives
# red-green colour blindness.
PICKED_YEAR = "#d1571f"
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

#: Left/right margin in a panel. Generous enough for a rotated axis title at desktop
#: width; `compact()` trims it for a phone, where 94 px a side is a quarter of the screen.
MARGIN_X = 56

# Precipitation is a total over an interval, so its axis is "mm per interval". The
# pandas alias is not a unit a reader should have to parse -- `mm 1h-1` was reaching
# the screen.
_FREQ_LABEL = {"30min": "mm/30 min", "1h": "mm/h", "6h": "mm/6 h",
               "1D": "mm/day", "1W": "mm/week"}


def _precip_units(freq: str) -> str:
    """Axis unit for precipitation bars binned at `freq`."""
    return _FREQ_LABEL.get(freq, f"mm/{freq}")


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
        # A floor, not a fixed width: `automargin` below grows these to fit whatever the
        # tick labels and axis titles actually need. Fixed 94 px gutters spent half of a
        # 375 px phone screen on empty margin.
        margin=dict(l=MARGIN_X, r=MARGIN_X, t=76 if title else 40, b=48),
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
        # The one thing that makes a figure work at 375 px and at 1500 px without the
        # caller having to know which it is: Plotly measures the labels it actually drew
        # and takes exactly the margin they need.
        automargin=True,
    )
    fig.update_xaxes(showgrid=False, **axis_common)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, **axis_common)
    return fig


def _panel(title: str | None = None, height: int = 400, *, secondary: bool = False,
           legend: bool = True) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]]) if secondary else go.Figure()
    return _style(fig, title, height, legend=legend)


def _wrap(text: str, width: int = 34) -> str:
    """Break a title onto `<br>` lines at word boundaries."""
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if len(trial) > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return "<br>".join(lines)


def compact(fig: go.Figure, *, title_width: int = 34) -> go.Figure:
    """Re-lay a finished figure for a narrow screen.

    Margins are not this function's job -- every panel sets `automargin`, so each already
    takes only the width its labels need at whatever size it is drawn, which is what
    makes one figure work at 375 px and at 1500 px. What `automargin` cannot decide is
    that 16 px tick type is too big for a 340 px plot, that a one-line title will run off
    the edge, or that an eight-entry legend stacked at the top will eat a third of the
    panel. Those are judgements about the viewport, which only the caller can make.

    Three changes, in order of how much they matter on a phone:

    1. **The legend moves below the plot.** At the top it pushes the data down and
       collides with the title; below, it costs only the bottom margin, which is sized
       here from how many lines it will take.
    2. **The title wraps** at `title_width` characters instead of overflowing.
    3. **Type shrinks a step**, and a single-panel figure loses a little height. A
       multi-row figure keeps its height -- its rows are already short, and shrinking
       them further is what turns four readable panels into four stripes.
    """
    rows = len({k for k in fig.layout if k.startswith("yaxis")}) or 1
    names = [t.name for t in fig.data if t.showlegend is not False and t.name]
    # How many lines the legend will take is a function of the rendered width, which is
    # the browser's business, not ours. Estimate it from the name lengths -- "2.5 cm"
    # packs several to a line, "Cumulative ET<sub>o</sub>" does not -- and round up.
    # Spare white space below a figure costs nothing; a clipped legend costs the key.
    longest = max((len(re.sub(r"<[^>]+>", "", n)) for n in names), default=6)
    per_line = 1 if longest > 11 else 2 if longest > 7 else 3
    bottom = 40 + 20 * max(1, -(-len(names) // per_line))

    title = fig.layout.title.text if fig.layout.title else None
    tall = fig.layout.height or 400

    fig.update_layout(
        margin=dict(l=8, r=8, t=44 if title else 16, b=bottom),
        height=(tall if rows > 1 else max(260, int(tall * 0.86))) + bottom - 48,
        font=dict(size=FS_TICK - 3),
        # Anchored to the figure's own bottom-left corner, not the plot area's: with
        # `automargin` the plot area starts wherever the y labels happen to end, which
        # left the legend indented by a different amount on every panel.
        legend=dict(orientation="h", xref="container", x=0.02, xanchor="left",
                    yref="container", y=0.012, yanchor="bottom",
                    font=dict(size=FS_LEGEND - 4)),
    )
    if title:
        fig.update_layout(title=dict(text=_wrap(title, title_width),
                                     font=dict(size=FS_TITLE - 5), y=0.985))
    fig.update_xaxes(title_font=dict(size=FS_AXIS_TITLE - 5),
                     tickfont=dict(size=FS_TICK - 3))
    fig.update_yaxes(title_font=dict(size=FS_AXIS_TITLE - 5),
                     tickfont=dict(size=FS_TICK - 3))
    return fig


def _title(logger: Logger | str | None, text: str) -> str:
    """The title a figure carries. Currently just `text`.

    Every figure used to end "— F1_1_ATMOS_SMST1", which the app's own page heading
    already says two lines above it; on a tab full of charts that is the logger's name
    repeated a dozen times and nothing else.

    `logger` is still threaded through all thirty-odd call sites deliberately. A caller
    outside the app -- a notebook, a figure for a paper -- has no page heading to lean
    on, and giving the suffix back is then a one-line change here rather than a hunt
    through every builder.
    """
    return text


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
        fig.update_yaxes(title_text=f"Precipitation ({_precip_units(freq)})",
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
    logger: Logger | str | None = None,
    ylim: tuple[float, float] | None = (0.0, 0.6),
    max_points: int | None = MAX_POINTS,
) -> go.Figure:
    """Thickness-weighted root-zone soil moisture, with rainfall."""
    rz = root_zone(df, "moisture")
    d = _decimate(df.assign(_rz=rz), max_points)
    depths = depths_of(depth_columns(df, "moisture"))
    label = f"RZSM ({depths[0]:g}–{depths[-1]:g} cm)" if depths else "RZSM"
    bar, freq = _rain_bars(df) if precip else (None, "")
    fig = _panel(_title(logger, "Root-zone soil moisture"), 460, secondary=bar is not None)

    if bar is not None:
        fig.add_trace(bar, secondary_y=True)
        fig.update_yaxes(title_text=f"Precipitation ({_precip_units(freq)})",
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
    fig.update_yaxes(title_text=f"Precipitation ({_precip_units(freq)})", **kw)
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


def evapotranspiration(
    df: pd.DataFrame,
    *,
    met: pd.DataFrame | None = None,
    ref: str | None = None,
    crop_coefficient: float = 1.0,
    out: pd.DataFrame | None = None,
    row_notes: bool = True,
    logger: Logger | str | None = None,
) -> go.Figure:
    """The whole ET story in four stacked rows on one x axis.

    Row 1, mm per day: rain beside evaporation. The evaporation bar carries ETo pale
    behind ETa solid, because what the soil supplied is a *part* of what the atmosphere
    asked, not a rival to it -- so the exposed pale head is the water stress, read
    straight off the bar. Rain and ET are both mm over the same day, which is what
    earns them one axis.

    Row 2: WSF, the water stress factor itself, 0 to 1.

    Row 3, mm: cumulative ETo against cumulative ETa. Both only rise, and the widening
    gap between them is the evaporation the soil could not supply -- the running total
    of the pale heads in row 1.

    Row 4, mm: the two running balances, cumulative `P - ETo` against cumulative
    `P - ETa`. These cross zero, which is why they are not on row 3 with the totals that
    cannot. Every running total restarts at the left edge of the window rather than on
    1 January; `climatology` has the year-to-date view.

    Four rows and no second y axis anywhere: each row carries one quantity at one
    scale, so nothing in the figure can be read as an alignment that is not there.

    Pass `out` to reuse a frame `stress.actual_et` has already produced. A caller
    redrawing on a slider wants that: evaluating the sigmoid over a quarter-million rows
    again, to change a coefficient the last row is merely multiplied by, is what makes an
    interactive control feel broken. `crop_coefficient` is then not used to *compute*
    anything -- the caller has already applied it -- but it is still read, because a
    figure drawn at Kc = 1.3 has to say so. It used to be silently dead on this path, so
    a reader dragging a Kc slider watched every ETa number move with nothing anywhere
    recording why.

    `row_notes` puts a short caption above each row saying what it shows. Four y-axis
    units tell a reader what is measured but not what they are looking at.
    """
    from .stress import actual_et

    ref = ref if ref is not None else logger
    name = _ref_name(ref) if ref is not None else None
    if out is None:
        try:
            out = actual_et(df, met, ref=name, crop_coefficient=crop_coefficient)
        except (FileNotFoundError, ValueError) as exc:
            return _panel(_title(logger, f"Evapotranspiration — {exc}".split(".")[0]), 320)
    if out.empty:
        return _panel(_title(logger, "Evapotranspiration — needs a weather station "
                                     "and soil parameters"), 320)

    # Rain comes from whichever frame carries the gauge -- the station's, when the soil
    # logger has none of its own.
    wx = met if met is not None and _has_data(met, PRECIP) else df
    rain = (wx[PRECIP].clip(lower=0.0).resample("1D").sum(min_count=1).reindex(out.index)
            if _has_data(wx, PRECIP) else None)

    # Kc never changes a label at 1, so the data tabs are untouched; off 1 it is stamped
    # on every ETa entry, and a downloaded PNG then carries its own assumption instead of
    # passing for the default figure.
    kc_note = "" if abs(crop_coefficient - 1.0) < 1e-9 else f", K<sub>c</sub> {crop_coefficient:g}"

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        row_heights=[0.26, 0.14, 0.13, 0.235, 0.235],
                        vertical_spacing=0.046 if row_notes else 0.030)
    _style(fig, _title(logger, "Evapotranspiration and the water balance"),
           1040 if row_notes else 980)
    # Seven series wrap the horizontal legend onto a second line, which then climbs into
    # the title. Put the title above the legend and give both room.
    # Eight series wrap the horizontal legend onto three lines, which then climbs into
    # the title above it and, with `row_notes`, collides with row 1's caption below it.
    # The top margin has to hold title + legend + that caption.
    # Eight series wrap the horizontal legend onto three lines. Anchor it to the
    # *container* top, not the plot area's: anchored to the plot area it moves down with
    # every increase in the top margin, so it kept landing on row 1's caption no matter
    # how much room was made for it.
    fig.update_layout(margin=dict(t=176 if row_notes else 136),
                      title=dict(y=0.985, yanchor="top"),
                      # Plotly flips the legend to reversed order as soon as a trace
                      # carries a fill, which put row 4's entries above row 1's. Pin it
                      # so the legend reads in the order the rows do.
                      legend_traceorder="normal",
                      legend=dict(xref="container", x=0.055, xanchor="left",
                                  yref="container", y=0.955, yanchor="top"))

    # -- row 1: the daily fluxes
    #
    # Rain stays bars -- it is a total over an interval, and a line between samples
    # would imply it was raining continuously. ETo and ETa are daily *rates* and are
    # drawn as lines, because the question asked of them is how they move and how far
    # apart they are, which two overlaid bars never let you trace. The band between the
    # lines is filled: that area *is* the demand the soil did not meet, so the reading
    # the stacked bars gave is kept, and made continuous.
    eta_line = _line(out.index, out["et"], f"ET<sub>a</sub> (supplied{kc_note})",
                     ET0_CUM, width=1.8, fmt=".2f")
    fig.add_trace(eta_line, row=1, col=1)
    eto_line = _line(out.index, out["et0"], "ET<sub>o</sub> (demand)", ET0_C,
                     width=1.6, dash="dot", fmt=".2f")
    # `tonexty` fills to the trace added immediately before it, so ETa goes in first.
    eto_line.update(fill="tonexty", fillcolor="rgba(237,161,0,0.22)")
    fig.add_trace(eto_line, row=1, col=1)
    fig.update_yaxes(title_text="ET (mm/day)", rangemode="tozero", row=1, col=1)

    # -- row 2: rain, on its own scale
    #
    # It used to share row 1 with the two ET series, which is what the shared-axis
    # argument asked for -- both are mm over the same day, so one scale let you read
    # supply against demand directly. That worked while everything was bars. It stops
    # working the moment ET is a line: one 23 mm storm sets the axis and the ET curves,
    # which never pass 5, are squashed into the bottom fifth of the row and effectively
    # disappear. Adjacent rows in the same units keep the comparison readable across,
    # and cost only that one rain day is no longer *beside* the ET it soaked.
    if rain is not None:
        fig.add_trace(go.Bar(
            x=rain.index, y=rain.values, name="Rain", marker_color=RAIN,
            marker_line_width=0,
            hovertemplate="%{y:.1f} mm<extra>Rain</extra>"), row=2, col=1)
        fig.update_layout(bargap=0.2)
        fig.update_yaxes(title_text="Rain (mm/day)", rangemode="tozero", row=2, col=1)
    else:
        fig.update_yaxes(title_text="Rain — no gauge", row=2, col=1)

    # -- row 2: the stress factor
    fig.add_trace(_line(out.index, out["wsf"], "WSF", RZ, width=1.6, fmt=".2f"),
                  row=3, col=1)
    fig.update_yaxes(title_text="WSF (—)", range=[0, 1.02], row=3, col=1)

    # -- row 3: the two running ET totals, in the same two shades as the bars above
    for col, label, colour in (("et0", "Cumulative ET<sub>o</sub>", ET0_C),
                               ("et", f"Cumulative ET<sub>a</sub>{kc_note}", ET0_CUM)):
        run = out[col].cumsum()
        fig.add_trace(_line(run.index, run.values, label, colour, width=1.8, fmt=".0f"),
                      row=4, col=1)
    fig.update_yaxes(title_text="Cumulative ET (mm)", rangemode="tozero", row=4, col=1)

    # -- row 4: the two running balances. Dashed for the demand, solid for what the soil
    # actually supplied -- the same pale/solid reading as the bars, in a mark that cannot
    # be drawn pale without disappearing.
    if rain is not None:
        filled = rain.fillna(0.0)
        for col, label, colour, dash in (
                ("et0", "P − ET<sub>o</sub>", RAIN, "dash"),
                ("et", "P − ET<sub>a</sub>", RAIN_CUM, None)):
            run = (filled - out[col]).cumsum()
            fig.add_trace(_line(run.index, run.values, label, colour,
                                width=1.8, dash=dash, fmt=".0f"), row=5, col=1)
        fig.add_hline(y=0, line=dict(color=AXIS, width=1, dash="dot"), row=5, col=1)
        fig.update_yaxes(title_text="Cumulative<br>P − ET (mm)", row=5, col=1)
    else:
        fig.update_yaxes(title_text="P − ET — no gauge", row=5, col=1)

    for row in (1, 2, 3, 4):
        fig.update_xaxes(showticklabels=False, row=row, col=1)

    if row_notes:
        # Anchored to each row's own domain, just above its top edge, so they sit in the
        # gap `vertical_spacing` opened for them and never over data.
        notes = ("what the atmosphere asked, what the soil gave, and the gap between",
                 "what fell",
                 "the stress factor behind that gap",
                 "the same two ET series, added up",
                 "rain minus each, added up")
        for row, note in enumerate(notes, start=1):
            axis = "y domain" if row == 1 else f"y{row} domain"
            fig.add_annotation(
                x=0, xref="x domain", y=1.0, yref=axis, yshift=13,
                text=note, showarrow=False, xanchor="left", yanchor="bottom",
                font=dict(family=FONT, size=FS_NOTE, color=MUTED))
    return fig


def limit_key(theta_sat: float, theta_fc: float, theta_r: float) -> list[dict]:
    """The four soil marks, wettest first: symbol, plain meaning, value, colour.

    The wording and the colours live here so the figure, the key printed under it and
    the dotted lines on the timeseries panels cannot drift apart. The app renders this;
    `wsf_explorer` draws the lines for the same four values in the same four colours.
    """
    return [
        {"symbol": "θ<sub>s</sub>", "label": "saturation",
         "meaning": "every pore full", "value": theta_sat, "colour": THETA_SAT_C},
        {"symbol": "θ<sub>fc</sub>", "label": "field capacity",
         "meaning": "what it holds against gravity", "value": theta_fc,
         "colour": THETA_FC_C},
        {"symbol": "θ<sub>c</sub>", "label": "stress begins",
         "meaning": "halfway from field capacity to residual",
         "value": 0.5 * (theta_fc + theta_r), "colour": THETA_HALF_C},
        {"symbol": "θ<sub>r</sub>", "label": "residual",
         "meaning": "water it never gives up", "value": theta_r,
         "colour": THETA_R_C},
    ]


def wsf_explorer(
    theta_sat: float,
    theta_fc: float,
    theta_r: float,
    *,
    theta: float | None = None,
    steepness: float = 100.0,
    title: str | None = None,
    annotate: bool = False,
) -> go.Figure:
    """The water stress function itself, with the four soil limits marked.

    A teaching figure: WSF against soil water content over `0 .. theta_sat`, with the
    four limits drawn where they fall.

    It used to carry a grey histogram of the water contents this probe has actually
    reached. That is gone: its bars were relative frequency scaled to fit *the WSF
    axis*, so a bar reaching 0.8 read as "WSF 0.8" and meant "this bin is 80% as common
    as the commonest". One quantity per axis, and in the one figure whose job is to
    teach that axis, most of all.

    The limits are **coloured vertical lines and nothing else**; what each one means
    belongs in a key under the figure, which `limit_key` supplies and the app renders.
    Four multi-line captions inside a plot this narrow fought each other and the curve.
    `annotate=True` puts them back for a standalone figure that has no key beneath it.

    The one caption that stays inside either way is the arrow at the midpoint reading
    "stress begins". It is not a property of the soil like the other three; it is a
    reading *of this curve*, and an arrow pointing at the crossing says "here" in a way
    no legend entry can.
    """
    from .stress import stress_factor

    x = np.linspace(0.0, max(theta_sat, theta_fc) * 1.02, 400)
    y = stress_factor(x, theta_fc=theta_fc, theta_r=theta_r, theta_sat=theta_sat,
                      steepness=steepness)

    fig = _panel(title or "Water stress function", 460)

    fig.add_trace(_line(x, y, "WSF", INK, width=2.2, fmt=".3f"))

    marks = limit_key(theta_sat, theta_fc, theta_r)
    half = marks[2]["value"]
    span = float(x[-1] - x[0]) or 1.0
    for i, m in enumerate(marks):
        stress = m["label"] == "water stress begins"
        fig.add_vline(x=m["value"], line=dict(color=m["colour"], width=1.8,
                                              dash="dot" if stress else "solid"))
        if not annotate or stress:
            continue
        near_right = (m["value"] - x[0]) / span > 0.72
        fig.add_annotation(
            x=m["value"], xref="x", y=1.0 if i % 2 == 0 else 0.84, yref="paper",
            text=f"{m['symbol']} — {m['label']}<br><i>{m['meaning']}</i>"
                 f"<br>{m['value']:.3f}",
            showarrow=False, xanchor="right" if near_right else "left",
            xshift=-5 if near_right else 5, yanchor="top", align="left",
            font=dict(family=FONT, size=FS_NOTE, color=m["colour"]))

    # The arrow stays whatever `annotate` says: see the docstring.
    fig.add_annotation(
        x=half, y=0.5, xref="x", yref="y",
        text=f"<b>stress begins</b><br>{half:.3f}",
        showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.8,
        arrowcolor=THETA_HALF_C, ax=0, ay=-58, xanchor="center", align="center",
        font=dict(family=FONT, size=FS_NOTE, color=THETA_HALF_C))

    if theta is not None:
        wsf = float(stress_factor(np.array([theta]), theta_fc=theta_fc,
                                  theta_r=theta_r, theta_sat=theta_sat,
                                  steepness=steepness)[0])
        fig.add_trace(go.Scatter(
            x=[theta], y=[wsf], mode="markers+text", name="this soil now",
            marker=dict(size=13, color=RZ, line=dict(color=SURFACE, width=2)),
            text=[f"  WSF {wsf:.2f}"], textposition="middle right",
            textfont=dict(family=FONT, size=FS_LEGEND, color=RZ),
            hovertemplate=f"θ {theta:.3f} → WSF {wsf:.3f}<extra></extra>"))

    fig.update_xaxes(title_text="Soil water content, θ (m<sup>3</sup> m<sup>-3</sup>)",
                     range=[float(x[0]), float(x[-1])])
    fig.update_yaxes(title_text="WSF (—)", range=[0, 1.06])
    return fig


def weekly_balance(
    df: pd.DataFrame,
    met: pd.DataFrame | None = None,
    *,
    ref: str | None = None,
    year: int | None = None,
    crop_coefficient: float = 1.0,
    logger: Logger | str | None = None,
) -> go.Figure:
    """Week by week from 1 January: rain, the demand, what evaporated, what was left.

    Top row, three upright bars a week: rain, ETo (what the atmosphere asked)
    and ETa (what the soil could supply). Reading them side by side is
    the whole point -- the gap between the two ET bars is the water stress, and
    whether rain clears them is whether the week paid for itself.

    Bottom row, one bar a week: P - ET, the net the profile gained or lost. Above
    zero the store filled, below zero it paid the difference. Its own row because a
    signed residual and the fluxes it comes from do not belong on one scale.

    The window is a calendar year, `year` defaulting to the most recent in the record,
    so the weeks line up with the cumulative panels beside it and a reader counting
    deficits is counting them from the same 1 January.
    """
    from .stress import actual_et

    ref = ref if ref is not None else logger
    try:
        out = actual_et(df, met, ref=_ref_name(ref) if ref is not None else None,
                        crop_coefficient=crop_coefficient)
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

    year = int(year if year is not None else daily.index.year.max())
    daily = daily[daily.index.year == year]
    if daily.empty:
        return _panel(_title(logger, f"Weekly water balance — nothing in {year}"), 320)

    totals = {c: _weekly_total(daily[c]) for c in daily.columns}
    weekly = pd.DataFrame(totals).dropna()
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
    _style(fig, _title(logger, f"Weekly water balance — {year}"), 540)

    # Two bars a week, the pair centred on the week they belong to: rain on the left,
    # evaporation on the right. The two ET terms share that one bar in two shades --
    # ETa is *part* of the demand, not a rival to it, so the exposed pale head
    # is the water stress, read directly off the bar. Explicit offsets rather than
    # Plotly's grouping, because the widths vary: a part-finished week is drawn narrow,
    # and grouped bars would then sit off its centre.
    half = width / 2.0
    for col, name, colour, opacity, offset in (
            ("precip", "rain", RAIN, 1.0, -half),
            ("et0", "ET<sub>o</sub> (demand)", ET0_C, 0.40, 0.0),
            ("et", "ET<sub>a</sub> (supplied)", ET0_CUM, 1.0, 0.0)):
        fig.add_trace(go.Bar(
            x=weekly.index, y=weekly[col], width=half, offset=offset, name=name,
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


#: Basemaps the map tab can draw. Both are free and need no API key: OpenStreetMap is
#: MapLibre's own built-in style, and the satellite layer is an Esri raster tile source
#: drawn under the markers on a blank ground. A Google basemap would need a billing
#: account and a secret in every deployment, which this app deliberately has none of.
#: Satellite first, and it is the default: on a food-forest plot the tree rows and the
#: field edges are what tell you where a logger is, and a street map of farmland is
#: mostly blank.
BASEMAPS = {"satellite": "Satellite (Esri)", "street": "OpenStreetMap"}

# Both basemaps are declared as explicit raster layers over a blank ground rather than
# by Plotly's built-in style name. The built-in "open-street-map" style supplies no
# attribution string, and MapLibre's attribution control then prints a literal
# `undefined` in the corner of the map. Declaring the source ourselves is the only way to
# put our own credit there -- and it costs nothing, since the built-in style fetches these
# very tiles.
_OSM_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
_OSM_CREDIT = "© OpenStreetMap contributors"
_ESRI_IMAGERY = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                 "World_Imagery/MapServer/tile/{z}/{y}/{x}")
_ESRI_CREDIT = "Esri · Maxar · Earthstar Geographics"

MAP_LIVE = "#1c5cab"
MAP_OFFLINE = "#9a9a9a"


def logger_map(
    loggers,
    *,
    selected=None,
    basemap: str = "satellite",
    height: int = 420,
) -> go.Figure:
    """Every logger on a basemap, for picking one without knowing its serial.

    Nobody remembers `z6-21176`, so the map is a real selector rather than decoration:
    the hover names the device, its field and what it actually reports, and the trace
    carries each serial in `customdata` so a click can be resolved back to a logger.

    Offline loggers are drawn grey rather than hidden -- `K2_ATMOS_SMST` being silent
    is information, and leaving it off the map would just raise the question of where
    it went.

    Two details exist to keep the word `undefined` off the map, which is what a reader
    was seeing. Both basemaps declare their tiles and their credit explicitly (see
    `_OSM_TILES` above), because Plotly's built-in style name carries no attribution and
    MapLibre's corner control prints `undefined` when it has none. And the hover string
    is assembled here into `text` rather than left to the browser to build out of
    `%{customdata[i]}`.

    None of this is visible from `write_image`: that uses a different renderer from the
    one Streamlit ships, so a PNG rendered here proves nothing about what the app draws.
    """
    loggers = list(loggers)
    if not loggers:
        return _panel("No loggers to map", height)

    chosen = getattr(selected, "serial", selected)
    live = [lg for lg in loggers if not lg.is_offline]
    off = [lg for lg in loggers if lg.is_offline]

    fig = go.Figure()
    for group, name, colour in ((live, "reporting", MAP_LIVE),
                                (off, "offline", MAP_OFFLINE)):
        if not group:
            continue
        fig.add_trace(go.Scattermap(
            lat=[lg.latitude for lg in group],
            lon=[lg.longitude for lg in group],
            mode="markers",
            name=name,
            # The hover string is built here rather than assembled in the browser from
            # `%{customdata[i]}`: one less thing that can come back `undefined` if index
            # handling differs between renderers. `customdata` still carries the serial,
            # because that is what a click has to resolve back to a logger.
            customdata=[[lg.serial] for lg in group],
            text=[f"<b>{lg.name}</b><br>{lg.serial}<br>{lg.field_name}"
                  f"<br>reports: {', '.join(lg.reporting_measures()) or 'nothing'}"
                  for lg in group],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                size=[20 if lg.serial == chosen else 12 for lg in group],
                color=colour,
                opacity=[1.0 if lg.serial == chosen else 0.85 for lg in group],
            ),
        ))

    # Frame the whole network, then lean towards the selection so a click zooms to its
    # field instead of leaving the reader to find which dot changed.
    lats = [lg.latitude for lg in loggers]
    lons = [lg.longitude for lg in loggers]
    pick = next((lg for lg in loggers if lg.serial == chosen), None)
    if pick is not None:
        centre, zoom = dict(lat=pick.latitude, lon=pick.longitude), 15.5
    else:
        centre = dict(lat=(min(lats) + max(lats)) / 2,
                      lon=(min(lons) + max(lons)) / 2)
        # The three sites are ~40 km apart; one zoom level shows them all.
        zoom = 8

    tiles, credit = ((_OSM_TILES, _OSM_CREDIT) if basemap == "street"
                     else (_ESRI_IMAGERY, _ESRI_CREDIT))
    fig.update_layout(map=dict(
        center=centre, zoom=zoom, style="white-bg",
        layers=[dict(sourcetype="raster", source=[tiles], below="traces",
                     sourceattribution=credit)],
    ))
    _style(fig, None, height)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), clickmode="event+select",
        # `_style` sets "x unified", which is meaningless on a map: there is no shared x
        # to gather points along.
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.75)", font=dict(size=FS_NOTE)),
    )
    return fig


def _wedge(x_centre: float, width: float, y_top: float, y_bottom: float,
           *, up: bool = True) -> str:
    """SVG path for an arrow of a given width: a shaft with a head at one end.

    `up` by default, because every flux this draws is evaporation and evaporation
    goes up. The picture is still laid out atmosphere-above-soil, so an upward head
    means water leaving the ground for the air, which is the thing being counted.
    """
    half = max(width, 0.6) / 2.0
    head = min(4.5, (y_top - y_bottom) * 0.34)
    if up:
        neck = y_top - head
        return (f"M {x_centre - half},{y_bottom} L {x_centre + half},{y_bottom} "
                f"L {x_centre + half},{neck} L {x_centre + half * 1.75},{neck} "
                f"L {x_centre},{y_top} L {x_centre - half * 1.75},{neck} "
                f"L {x_centre - half},{neck} Z")
    neck = y_bottom + head
    return (f"M {x_centre - half},{y_top} L {x_centre + half},{y_top} "
            f"L {x_centre + half},{neck} L {x_centre + half * 1.75},{neck} "
            f"L {x_centre},{y_bottom} L {x_centre - half * 1.75},{neck} "
            f"L {x_centre - half},{neck} Z")


def _wsf_shade(wsf: float) -> str:
    """Dark red at no water, deep blue at full -- the depth ramp's own two ends."""
    lo = (0x7f, 0x00, 0x00)
    hi = (0x0d, 0x2a, 0x6b)
    mid = (0xd4, 0x45, 0x6f)
    t = min(max(float(wsf), 0.0), 1.0)
    a, b, u = (lo, mid, t / 0.5) if t < 0.5 else (mid, hi, (t - 0.5) / 0.5)
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * u) for i in range(3))


def water_story(
    info: dict,
    *,
    theta: float,
    eto: float,
    kc: float = 1.0,
    narrow: bool = False,
    height: int | None = None,
    scale_mm: float = 8.0,
    title: str | None = "Where today's water goes",
) -> go.Figure:
    """The whole water story as one picture.

    The atmosphere asks for a certain amount of water; the soil can meet only part of
    it; what gets through is what the plants actually use; the rest is a shortfall you
    would have to irrigate. Three sections of prose said the same thing and made it look
    like three unrelated calculations. It is one sequence, and the only honest way to
    show that is to draw it.

    Two columns. The left one is the chain, running downwards because that is the
    direction the water goes: demand, then the gate the soil opens, then what came
    through beside what did not. The right one is the profile that gate is made of, with
    the water it would take to refill it standing beside it.

    `narrow=True` stacks those two instead, chain above soil, and makes the figure
    taller. Side by side is unreadable at phone width -- the columns run into each other
    -- and this is a drawing, so it cannot be rescued by shrinking the type the way a
    chart can.

    Every millimetre quantity -- ETo, ETa, the shortfall -- shares `scale_mm`, so their
    widths are comparable by eye. WSF and the soil fills are fractions, drawn as fills
    rather than arrows, so nothing can be read as a flux that is not one. Colours are the
    app's own throughout: a reader arrives already knowing what they mean.

    `info` is a `stress.root_zone_limits` dict; the arithmetic is the irrigation
    calculator's and is not duplicated here.
    """
    from .stress import stress_factor

    limits, depths = info["limits"], info["depths"]
    thick = info["thicknesses"]
    wsf = float(stress_factor(theta, theta_fc=info["theta_fc"],
                              theta_r=info["theta_r"], theta_sat=info["theta_sat"]))
    demand = max(kc * eto, 0.0)
    eta = wsf * demand
    short = max(demand - eta, 0.0)
    refill = sum(max(0.0, limits[d]["theta_fc"] - theta) * L * 10.0
                 for d, L in zip(depths, thick))
    full_store = sum((limits[d]["theta_fc"] - limits[d]["theta_r"]) * L * 10.0
                     for d, L in zip(depths, thick)) or 1.0

    # Every position below is expressed against these, so `narrow` only moves the two
    # blocks rather than needing a second copy of the drawing.
    height = height or ((820 if title is None else 850) if narrow else
                        (520 if title is None else 550))
    if narrow:
        # The chain needs more of the height when stacked: its captions wrap to two
        # lines there, and at 44 units the arrow labels landed on the closing sentence.
        cx0, cx1, ctop, cbot = 2.0, 92.0, 100.0, 50.0
        sx0, sx1, stop, sbot = 2.0, 92.0, 44.0, 4.0
    else:
        cx0, cx1, ctop, cbot = 0.0, 52.0, 100.0, 12.0
        sx0, sx1, stop, sbot = 56.0, 100.0, 100.0, 4.0
    cw = cx1 - cx0

    fig = go.Figure()
    _style(fig, title, height, legend=False)
    # The two block headings sit at the top of their own columns, so the figure needs a
    # title over both of them or the drawing opens with no statement of what it is.
    fig.update_layout(margin=dict(l=6, r=6, t=48 if title else 6, b=6),
                      plot_bgcolor=SURFACE)
    fig.update_xaxes(visible=False, range=[0, 100], fixedrange=True)
    fig.update_yaxes(visible=False, range=[0, 100], fixedrange=True)

    shapes, notes = [], []

    def label(x, y, text, *, size=FS_NOTE, color=INK_2, **kw):
        notes.append(dict(x=x, y=y, text=text, showarrow=False,
                          font=dict(family=FONT, size=size, color=color), **kw))

    # mm -> arrow width. Capped so a wet day cannot burst the column.
    def mm(v):
        return min(max(v, 0.0), scale_mm) / scale_mm * 17.0

    # ================= the chain ===========================================
    ch = lambda f: cbot + (ctop - cbot) * f       # noqa: E731  fraction of the block
    label(cx0 + 3, ch(1.00), "<b>THE ATMOSPHERE ASKS</b>", xanchor="left",
          yanchor="top", color=MUTED)
    label(cx0 + 3, ch(0.94), f"ET<sub>o</sub> <b>{eto:.1f}</b> mm/day"
          + (f"  ×  K<sub>c</sub> {kc:g}  =  <b>{demand:.1f}</b>"
             if abs(kc - 1.0) > 1e-9 else ""),
          xanchor="left", yanchor="top", size=FS_LEGEND, color=INK)
    shapes.append(dict(type="path",
                       path=_wedge(cx0 + cw * 0.5, mm(demand), ch(0.84), ch(0.68)),
                       fillcolor=ET0_C, opacity=0.45, line_width=0))

    # the gate
    gate0, gate1 = cx0 + 4, cx1 - 6
    shapes.append(dict(type="rect", x0=gate0, x1=gate1, y0=ch(0.57), y1=ch(0.65),
                       line=dict(color=AXIS, width=1, dash="dot"), fillcolor=GRID))
    if wsf > 0.005:
        shapes.append(dict(type="rect", x0=gate0, x1=gate0 + (gate1 - gate0) * wsf,
                           y0=ch(0.57), y1=ch(0.65), line_width=0, fillcolor=RZ))
    gap = "<br>" if narrow else "  "
    label(cx0 + 4, ch(0.53), f"<b>THE SOIL CAN MEET</b>{gap}WSF <b>{wsf:.2f}</b> — "
                             f"{100 * wsf:.0f}% of it",
          xanchor="left", yanchor="top", color=RZ)

    # what came through, and what did not
    # The gate caption wraps to two lines when narrow, so the arrows below it start
    # lower there or the second line lands on the shaft.
    a_top, a_bot = ch(0.40 if narrow else 0.45), ch(0.26 if narrow else 0.27)
    shapes.append(dict(type="path",
                       path=_wedge(cx0 + cw * 0.27, mm(eta), a_top, a_bot),
                       fillcolor=ET0_CUM, line_width=0))
    label(cx0 + cw * 0.27, a_bot - 2.5,
          f"ET<sub>a</sub> <b>{eta:.1f}</b> mm/day<br>the plants get this",
          xanchor="center", yanchor="top", color=ET0_CUM)
    if short > 0.01:
        shapes.append(dict(type="path",
                           path=_wedge(cx0 + cw * 0.72, mm(short), a_top, a_bot),
                           fillcolor="rgba(0,0,0,0)",
                           line=dict(color=ET0_CUM, width=1.4, dash="dot")))
        label(cx0 + cw * 0.72, a_bot - 2.5,
              f"<b>{short:.1f}</b> mm/day<br>asked for, not given",
              xanchor="center", yanchor="top", color=MUTED)

    label(cx0 + 4, ch(0.03 if narrow else 0.08),
          "The gap between those two is what<br>irrigation would have to cover."
          if narrow else
          "The gap between those two is what irrigation would have to cover.",
          xanchor="left", yanchor="top", color=MUTED)

    # ================= right column: the profile ===========================
    label(sx0, stop, "<b>THE SOIL IT COMES FROM</b>", xanchor="left", yanchor="top",
          color=MUTED)
    # The refill column is the one block with no heading over it -- its only label sat at
    # the foot, so the top right read as an unlabelled empty box. On the section
    # heading's own line, right-anchored, where nothing else reaches.
    label(sx1, stop, "<b>TO REFILL</b>", xanchor="right", yanchor="top", color=MUTED)

    # Proportional to thickness, but with a floor: 2.5 cm against 60 cm would otherwise
    # be a band too thin to put a word in, and an unreadable row is worse than a
    # slightly untrue one. The floor is stated in the caption under the figure.
    top, floor_y = stop - 10.0, sbot + 7.0
    span = top - floor_y
    raw = [L / sum(thick) for L in thick]
    heights = [max(r, 0.09) for r in raw]
    heights = [h / sum(heights) * span for h in heights]

    x0 = sx0 + (sx1 - sx0) * 0.20
    x1 = sx0 + (sx1 - sx0) * 0.70
    y = top
    for d, L, band in zip(depths, thick, heights):
        lim = limits[d]
        wet = min(max(theta / lim["theta_sat"], 0.0), 1.0)
        layer_wsf = float(stress_factor(theta, theta_fc=lim["theta_fc"],
                                        theta_r=lim["theta_r"],
                                        theta_sat=lim["theta_sat"]))
        shade = _wsf_shade(layer_wsf)
        shapes.append(dict(type="rect", x0=x0, x1=x1, y0=y - band + 0.8, y1=y,
                           line=dict(color=GRID, width=1), fillcolor=SURFACE))
        shapes.append(dict(type="rect", x0=x0, x1=x0 + (x1 - x0) * wet,
                           y0=y - band + 0.8, y1=y, line_width=0, fillcolor=shade))
        # A green tick where this layer's own field capacity falls, so the bar reads as
        # "how far from full" and not merely as a length. Same green as every other
        # field-capacity mark in the app.
        at_fc = x0 + (x1 - x0) * min(lim["theta_fc"] / lim["theta_sat"], 1.0)
        shapes.append(dict(type="line", x0=at_fc, x1=at_fc, y0=y - band + 0.8, y1=y,
                           line=dict(color=THETA_FC_C, width=1.4)))
        mid = y - band / 2
        label(x0 - 2, mid, f"{d} cm", xanchor="right", yanchor="middle")
        label(x1 + 2, mid, f"{layer_wsf:.2f}", xanchor="left", yanchor="middle",
              color=shade)
        y -= band
    # The key for these bands is printed under the figure by the caller, not inside it:
    # at this width an in-figure caption ran straight through the WSF column header.
    label(x1 + 2, top + 3, "WSF", xanchor="left", yanchor="bottom", color=MUTED)

    # what it would take to refill
    wx0, wx1 = sx1 - 7.0, sx1 - 1.0
    shapes.append(dict(type="rect", x0=wx0, x1=wx1, y0=floor_y, y1=top,
                       line=dict(color=GRID, width=1), fillcolor=SURFACE))
    fill = min(refill / full_store, 1.0) * span
    if fill > 0.3:
        shapes.append(dict(type="rect", x0=wx0, x1=wx1, y0=floor_y, y1=floor_y + fill,
                           line_width=0, fillcolor=RAIN))
    # Just the number at the foot: the column is headed now, so repeating "to refill"
    # under it said the same thing twice.
    label((wx0 + wx1) / 2, floor_y - 1.5, f"<b>{refill:.0f} mm</b>",
          xanchor="right" if narrow else "center", yanchor="top", color=RAIN_CUM)

    fig.update_layout(shapes=shapes, annotations=notes)
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
    rz = root_zone(df, measure)
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
                fig.update_yaxes(title_text=_precip_units(freq), row=i, col=1)
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
        s = root_zone(df, measure).dropna()
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
    # `_style` sets "x unified", which is wrong for a panel whose whole interaction is
    # "which line am I pointing at". Unified hover gathers *every* trace's point at that
    # x into one box, so a click reported all of them and the caller took the first --
    # always the oldest year, never the line under the cursor. "closest" makes hover name
    # the line you are on and a click return that one point.
    fig.update_layout(hovermode="closest", clickmode="event+select")

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
    "et0_cumulative": "Cumulative ETo",
    "rzsm": "Root-zone soil moisture",
    "wsf": "Water stress factor (WSF)",
    "et_cumulative": "Cumulative ETa",
    # One panel carrying both balances: P - ETo dashed, P - ETa solid, per year. They
    # were two panels and should not have been -- the gap between the pair *is* the
    # reading, and putting them on separate axes made the one comparison that matters
    # the one you could not make.
    "balance_cumulative": "Cumulative P − ET",
    "vpd_cumulative": "Cumulative vapour pressure deficit",
}

#: Kinds that need this logger's own weather station. Only four of the fourteen loggers
#: report met variables, so a caller listing kinds for a soil logger must drop these.
MET_KINDS = {"precip_cumulative", "et0_cumulative", "balance_cumulative",
             "et_cumulative", "wsf", "vpd_cumulative"}

#: Kinds that also need the site's soil parameters, so they need a `ref` to look the
#: site up and are skipped when it is missing or the site has not been extracted.
#: `balance_cumulative` is here as well as in MET_KINDS: its P - ETa half needs the
#: soil, and the panel degrades to the ETo line alone when the soil is missing.
SOIL_KINDS = {"et_cumulative", "wsf", "balance_cumulative"}

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
        s = root_zone(sm, "moisture").dropna()
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
            ylab = "Cumulative ETo (mm)"
        else:
            # P - ET0: the climatic water balance. Unlike the other cumulative kinds it
            # goes both ways, so the year-to-date value reads directly as a surplus or a
            # deficit rather than only as a distance from the median.
            if not _has_data(wx, PRECIP):
                return pd.Series(dtype="float64"), "", True
            daily = _water_balance(wx, freq=freq)["balance"]
            ylab = "Cumulative P − ET (mm)"
        if daily.empty:
            return pd.Series(dtype="float64"), "", True
        daily = daily.reindex(
            pd.date_range(daily.index.min(), daily.index.max(), freq=freq)
        ).fillna(0.0)
        return daily.groupby(daily.index.year).cumsum(), ylab, True

    if kind in ("et_cumulative", "wsf", "balance_a"):
        # All three come from the same pair -- the station's ETo and the soil's WSF --
        # so they are computed together and the kind only picks which one to return.
        # `balance_a` is not a public kind: it is the ETa half of `balance_cumulative`,
        # which `climatology` asks for separately and draws beside the ETo half.
        # Needs the site's soil parameters, hence `ref`.
        cumulative = kind != "wsf"
        if ref is None or not (_has_data(wx, RADIATION) and _has_data(wx, AIR_T)):
            return pd.Series(dtype="float64"), "", cumulative
        from .stress import profile_stress

        et0 = _reference_et(wx, freq=freq)
        if et0.empty:
            return pd.Series(dtype="float64"), "", cumulative
        try:
            wsf, _, _ = profile_stress(sm, ref=_ref_name(ref))
        except (FileNotFoundError, ValueError):
            return pd.Series(dtype="float64"), "", cumulative
        if wsf.empty:
            return pd.Series(dtype="float64"), "", cumulative
        daily_wsf = wsf.resample(freq).mean()

        if kind == "wsf":
            # A state, not an accumulation: 1 means the profile can meet whatever the
            # atmosphere asks, 0 means it can meet none of it.
            return daily_wsf.dropna(), "Water stress factor, WSF (—)", False

        actual = (et0 * daily_wsf).dropna()
        if actual.empty:
            return pd.Series(dtype="float64"), "", True
        if kind == "balance_a":
            if not _has_data(wx, PRECIP):
                return pd.Series(dtype="float64"), "", True
            rain = wx[PRECIP].clip(lower=0.0).resample(freq).sum(min_count=1)
            actual = (rain.reindex(actual.index).fillna(0.0) - actual).dropna()
            ylab = "Cumulative P − ET (mm)"
        else:
            ylab = "Cumulative ETa (mm)"
        actual = actual.reindex(
            pd.date_range(actual.index.min(), actual.index.max(), freq=freq)
        ).fillna(0.0)
        return actual.groupby(actual.index.year).cumsum(), ylab, True

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
                if median and not kind.startswith("balance") else None),
        "years": sorted(int(y) for y in past.index.year.unique()),
        "excluded": sorted(int(y) for y in partial),
    }


def _add_soil_limits(fig: go.Figure, df: pd.DataFrame,
                     ref: Logger | str | None) -> dict:
    """The four soil water limits, dotted, behind a moisture chart.

    Saturation, field capacity, the WSF = 0.5 midpoint and residual water content --
    thickness-weighted over the same depths as the moisture series itself
    (`stress.root_zone_limits`), so the curve and the lines are commensurable. A curve
    crossing the yellow line really is this profile at half stress.

    The same four colours mark the same four values in `wsf_explorer`, so a reader who
    has played with the function there recognises them here without a second legend.

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
    # Same four values, same four colours, same words as `limit_key` -- a reader who has
    # met them in the Understand tab meets them again here unchanged.
    lines = tuple(
        (key, f"{m['symbol']} {m['label']}".strip(), m["colour"], "dot")
        for key, m in zip(("theta_sat", "theta_fc", "theta_half", "theta_r"),
                          limit_key(info["theta_sat"], info["theta_fc"],
                                    info["theta_r"]))
    )
    for key, label, colour, dash in lines:
        # All four at the right, in one plain style, labelled inside the plot: the
        # curve's own year runs out at "today", so there is room there, and outside the
        # axis the text would be clipped by the margin.
        fig.add_hline(y=info[key], line=dict(color=colour, width=1, dash=dash),
                      annotation_text=f"{label} {info[key]:.3f}",
                      annotation_position="top right",
                      annotation_font=dict(family=FONT, size=FS_NOTE, color=colour))
        if key != "theta_half":
            continue
        # theta_c additionally gets a short arrow pointing down at its own line: it is
        # the one mark that says something about the plants rather than the soil.
        #
        # Text-less and placed left of the label, deliberately. The bold two-line
        # caption this replaced was pushed 46 px up from the line, which on these soils
        # -- where theta_fc sits only ~33 px above -- drove it straight through the
        # field-capacity label. An 18 px arrow beside the text reaches nothing.
        fig.add_annotation(
            x=0.80, xref="paper", y=info[key], yref="y", text="",
            showarrow=True, arrowhead=2, arrowsize=1.0, arrowwidth=1.6,
            arrowcolor=colour, ax=0, ay=-18)
    return info


def _year_traces(fig, series, *, colour, name, dash, highlight, fmt, click_targets,
                 current=False, showlegend=True):
    """One year's line, plus an invisible row of points that a click can land on.

    `current` is the year the reader came for: always black and always bold, whatever
    is selected. `highlight` is the year they have picked out, drawn bold in
    `PICKED_YEAR`; every other past year stays faded. Selecting the current year is
    therefore a no-op on the drawing, which is right -- it was already the bold one.


    Plotly only reports a selection when the cursor hits a *point*, and these are
    `mode="lines"` traces with no points in them -- which is why clicking a year did
    nothing. The companion trace is markers at roughly weekly spacing carrying the same
    year in `customdata`. Opacity 0.01 rather than 0: a fully transparent marker is not
    reliably hit-tested.
    """
    x = pd.to_datetime(series.index.dayofyear - 1, unit="D",
                       origin=pd.Timestamp("2000-01-01"))
    picked = highlight is not None and int(name) == int(highlight)
    bold = picked or current
    tr = _line(x, series.values,
               str(name),
               PICKED_YEAR if picked and not current else colour,
               width=2.8 if bold else 1.3, dash=dash, fmt=fmt)
    tr.update(opacity=1.0 if bold else 0.34, showlegend=showlegend,
              customdata=[int(name)] * len(x), legendgroup=str(name))
    fig.add_trace(tr)

    if click_targets:
        # Roughly one every three days, so there is a target wherever you point. They
        # sit *on* the line, so `hoverinfo="skip"` was backwards: it excluded them from
        # Plotly's hover machinery, which is the same machinery a click runs through.
        # Naming the year here is the point -- pointing at a line should say which year
        # it is, click or no.
        step = max(1, len(x) // 120)
        fig.add_trace(go.Scatter(
            x=x[::step], y=series.values[::step], mode="markers",
            marker=dict(size=16, color=colour, opacity=0.01),
            customdata=[int(name)] * len(x[::step]),
            showlegend=False, legendgroup=str(name),
            hovertemplate=f"<b>{name}</b><br>click to hold this year<extra></extra>"))


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
    highlight: int | None = None,
) -> go.Figure:
    """One line per year on a common calendar, so years can be held against each other.

    There is no min-max band and no median. A band said only "somewhere in here", and a
    median is not a year anyone can point at -- neither survives the question "is this
    year tracking 2024 or 2025?". Every year is drawn instead, oldest palest.

    `highlight` names the year to bring forward: it is drawn bold and fully opaque and
    every other year fades, the current year included. With `highlight=None` all years
    are drawn at full strength. Each line is shadowed by an invisible row of click
    targets, so the app can offer both a year button and a click on the line itself.

    **A single year is a valid panel.** When the record holds no complete earlier year --
    which is the case for all three TEROS21+TEROS12 loggers -- this draws that one year
    up to today rather than refusing. There is nothing to compare against yet, and that
    is a reason to say so in the title, not to withhold the data.

    `kind` is one of CLIMATOLOGY_KINDS. Cumulative kinds restart on 1 January, which is
    what makes "we are 90 mm behind by this date" a meaningful statement.
    `balance_cumulative` is the one two-series kind: P - ETo dashed against P - ETa
    solid, because the gap between them is the reading.
    """
    if measure is not None and kind == "rzsm" and measure != "moisture":
        # Only moisture is depth-averageable; see process.EXTENSIVE.
        raise ValueError("climatology(kind='rzsm') applies to soil moisture only")
    title = CLIMATOLOGY_KINDS.get(kind, kind)
    if df.empty:
        return _panel(_title(logger, f"{title} — no data"), 400)

    who = ref if ref is not None else logger
    s, ylab, cumulative = _climatology_series(df, kind, freq, who, met, soil)
    if s.empty:
        return _panel(_title(logger, f"{title} — not available on this logger"), 320)

    # The second series of the one two-series kind. Missing soil parameters simply leave
    # the panel with its ETo line, which is still worth drawing.
    second = None
    if kind == "balance_cumulative":
        try:
            cand, _, _ = _climatology_series(df, "balance_a", freq, who, met, soil)
        except (FileNotFoundError, ValueError):
            cand = pd.Series(dtype="float64")
        second = cand if not cand.empty else None

    # Only complete past years go into the comparison. A year whose record starts in May
    # accumulates from zero in May and reads as a freakishly dry year; one that stops in
    # August leaves a phantom gap. The current year is exempt -- being partial is the
    # whole point of it.
    newest = s.index.year.max()

    def drop_partial(series):
        if series is None or series.empty:
            return series, set()
        bad = {
            y for y, part in series.groupby(series.index.year)
            if y != newest
            and (part.index.min().dayofyear > 15 or part.index.max().dayofyear < 350)
        }
        return series[~series.index.year.isin(bad)], bad

    s, partial = drop_partial(s)
    second, _ = drop_partial(second)
    dropped = sorted(partial)
    if s.empty:
        return _panel(_title(logger, f"{title} — no usable record"), 320)

    years = sorted(int(y) for y in s.index.year.unique())
    only_one = len(years) == 1
    heading = f"{title} — {newest} so far" if only_one \
        else f"{title} — {newest} vs previous years"
    fig = _panel(_title(logger, heading), 470)
    if dropped:
        fig.add_annotation(
            text=f"{', '.join(str(y) for y in dropped)} excluded — incomplete year",
            showarrow=False, xref="paper", yref="paper", x=1, y=-0.16, xanchor="right",
            font=dict(family=FONT, size=FS_NOTE, color=MUTED))
        fig.update_layout(margin=dict(b=82))

    fmt = ".1f" if cumulative else ".3f"
    # Oldest palest, this year always black: the ramp encodes recency, so "the line just
    # under ours is last year" is readable without the legend. Sampled by position, so a
    # logger with three years and one with six put the same calendar year at a similar
    # shade.
    past = [y for y in years if y != newest]
    idx = (np.linspace(0, len(PAST_YEARS) - 1, len(past)).round().astype(int)
           if len(past) > 1 else [len(PAST_YEARS) - 1] * len(past))
    colours = {y: PAST_YEARS[i] for y, i in zip(past, idx)}
    colours[newest] = INK

    for year in years:
        part = s[s.index.year == year]
        if part.empty:
            continue
        pair = second is not None and not second[second.index.year == year].empty
        # One legend swatch per year, never two. It goes on the solid line where there
        # is a pair -- P - ETa is the one that says what actually happened.
        _year_traces(fig, part, colour=colours[year], name=year,
                     dash="dash" if pair else None, highlight=highlight, fmt=fmt,
                     click_targets=True, current=year == newest, showlegend=not pair)
        if pair:
            _year_traces(fig, second[second.index.year == year], colour=colours[year],
                         name=year, dash=None, highlight=highlight, fmt=fmt,
                         click_targets=False, current=year == newest, showlegend=True)

    now = s[s.index.year == newest]
    if len(now):
        # Where the year currently stands, called out rather than left to the axis.
        x = pd.to_datetime(now.index.dayofyear - 1, unit="D",
                           origin=pd.Timestamp("2000-01-01"))
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[now.values[-1]], mode="markers", showlegend=False,
            marker=dict(color=INK, size=7, line=dict(color=SURFACE, width=1.5)),
            hovertemplate="%{y:.1f}<extra>latest</extra>" if cumulative
            else "%{y:.3f}<extra>latest</extra>"))

    if second is not None and len(now):
        # Named at the line ends rather than in a footnote under the axis. The reader is
        # already looking there -- the end of the current year's line is where "how far
        # has this year got" is read -- so the name meets the eye at the question. A
        # footnote sat two inches from the thing it described and got skipped.
        tail = second[second.index.year == newest]
        pairs = [(now, "P − ET<sub>o</sub>", 14), (tail, "P − ET<sub>a</sub>", -14)]
        for part, label, shift in pairs:
            if part.empty:
                continue
            at = pd.to_datetime(part.index.dayofyear[-1] - 1, unit="D",
                                origin=pd.Timestamp("2000-01-01"))
            # To the *right* of the line end, not on it: the current year stops at
            # "today", so everything past that is open ground, and a label sitting on
            # its own curve is harder to read than one beside it.
            fig.add_annotation(
                x=at, y=float(part.values[-1]), xref="x", yref="y",
                text=f"<b>{label}</b>", showarrow=False,
                xanchor="left", xshift=9, yshift=shift,
                bgcolor="rgba(255,255,255,0.72)", borderpad=2,
                font=dict(family=FONT, size=FS_NOTE, color=INK))

    # `_style` sets "x unified", which is wrong for a panel whose whole interaction is
    # "which line am I pointing at". Unified hover gathers *every* trace's point at that
    # x into one box, so a click reported all of them and the caller took the first --
    # always the oldest year, never the line under the cursor. "closest" makes hover name
    # the line you are on and a click return that one point.
    fig.update_layout(hovermode="closest", clickmode="event+select")

    today = pd.Timestamp.now()
    fig.add_vline(x=pd.Timestamp(2000, today.month, today.day).timestamp() * 1000,
                  line=dict(color=AXIS, width=1, dash="dot"),
                  annotation_text="today", annotation_position="top",
                  annotation_font=dict(family=FONT, size=FS_NOTE, color=MUTED))
    fig.update_xaxes(tickformat="%b", dtick="M1")
    fig.update_yaxes(title_text=ylab,
                     range=[0, 0.6] if kind == "rzsm"
                     else [0, 1.02] if kind == "wsf" else None)
    if kind == "rzsm":
        # The lines that turn a moisture curve into a statement about the plants.
        _add_soil_limits(fig, soil if soil is not None else df, who)
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
