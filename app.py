"""WUNDER data viewer.

    streamlit run app.py     (or ./run.sh)

A thin layer over the `wunder` package — all the real work lives there, so the same figures
can be built in a notebook or exported for a paper.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

import wunder as w

st.set_page_config(page_title="WUNDER", page_icon="🌧", layout="wide")

# White, minimal, serif — matching the figures. The theme is pinned in
# .streamlit/config.toml so a viewer's OS dark mode can't invert the chrome around
# white charts.
st.markdown(
    """
    <style>
      #MainMenu, footer, header {visibility: hidden;}
      .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1550px;}
      html, body, [class*="css"] {font-family: "Times New Roman", Times, Georgia, serif;}
      h1 {font-size: 1.6rem; font-weight: 600; margin-bottom: 0.1rem;}
      .stTabs [data-baseweb="tab"] {font-size: 1rem; padding: 0.4rem 0.9rem;}
      .stTabs [data-baseweb="tab-list"] {gap: 0.2rem; border-bottom: 1px solid #e8e8e4;}
      [data-testid="stMetricValue"] {font-size: 1.3rem;}
      [data-testid="stMetricLabel"] {font-size: 0.82rem; color: #666;}
      section[data-testid="stSidebar"] {border-right: 1px solid #e8e8e4;
                                        background: #ffffff;}
      div[data-testid="stExpander"] details {border: 1px solid #e8e8e4; border-radius: 3px;}
    </style>
    """,
    unsafe_allow_html=True,
)

PRESETS = {"7 days": 7, "30 days": 30, "90 days": 90, "6 months": 182,
           "1 year": 365, "All": None}
MEASURES = {"moisture": "Soil moisture", "temperature": "Soil temperature",
            "matric_potential": "Matric potential", "conductivity": "Electrical conductivity"}

HAS_CACHE = (w.fetch_module.CACHE_DIR.exists()
             and any(w.fetch_module.CACHE_DIR.glob("*.parquet")))


@st.cache_data(show_spinner=False, ttl=3600)
def load(serial: str, token: int, days: int | None = None) -> pd.DataFrame:
    """A logger's full record.

    With a local Parquet cache, that is the cached 5-minute series. Without one -- a cloud
    container -- the whole history is pulled once and resampled to 30 minutes: 60 s and
    11 MB instead of 64 MB, with identical daily and cumulative statistics, and finer than
    the ~4000 points the figures decimate to anyway. Fetching only the selected window would
    be quicker still, but then the Summary tab has no earlier years to compare against.
    """
    if HAS_CACHE:
        return w.fetch(serial, refresh=bool(token))
    df = w.fetch(serial, start="2020-01-01", cache=False)
    return w.resample(df, "30min") if not df.empty else df


def show(fig, key: str) -> None:
    st.plotly_chart(fig, width='stretch', key=key,
                    config={"displaylogo": False, "displayModeBar": "hover",
                            "toImageButtonOptions": {"format": "png", "scale": 3}})


def name_of(lg) -> str:
    site = lg.site_name.replace("Voedselbos ", "").replace(" Herenboeren", "")
    return f"{lg.name} · {site}"


def time_range(key: str, *, default: str = "30 days"):
    """Preset periods plus an explicit start/end. Returns (days, start, end)."""
    choice = st.sidebar.selectbox("Period", [*PRESETS, "Custom…"], key=f"p{key}",
                                 index=list(PRESETS).index(default))
    if choice != "Custom…":
        return PRESETS[choice], None, None
    today = dt.date.today()
    pair = st.sidebar.date_input(
        "From / to", value=(today - dt.timedelta(days=30), today),
        min_value=dt.date(2022, 12, 1), max_value=today, key=f"d{key}",
    )
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return None, pd.Timestamp(pair[0]), pd.Timestamp(pair[1]) + pd.Timedelta(days=1)
    return None, None, None


def window(df: pd.DataFrame, days, start, end) -> pd.DataFrame:
    """Slice to the chosen period, anchored on the logger's own last record.

    Anchoring on the record rather than on today means a logger that stopped still shows
    its final weeks instead of an empty chart.
    """
    if df.empty:
        return df
    if start is not None or end is not None:
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df
    if days is None:
        return df
    return df[df.index >= df.index.max() - pd.Timedelta(days=days)]


st.sidebar.markdown("### WUNDER")
mode = st.sidebar.radio("Mode", ["Explore", "Compare"], horizontal=True,
                        label_visibility="collapsed")
st.sidebar.divider()

if "token" not in st.session_state:
    st.session_state.token = 0


# ==========================================================================
# COMPARE
# ==========================================================================
if mode == "Compare":
    kind = st.sidebar.radio("Compare", ["Fields at a site", "This year vs previous",
                                        "Variables"])
    st.title("Compare")

    # ---- field vs field within one site ----------------------------------
    if kind == "Fields at a site":
        candidates = [s_ for s_ in w.sites() if len(s_.by_field()) > 1]
        if not candidates:
            st.info("No site has more than one field.")
            st.stop()
        s_name = st.sidebar.selectbox("Site", [s_.name for s_ in candidates])
        site_ = next(s_ for s_ in candidates if s_.name == s_name)
        groups_ = site_.by_field()

        QUANTITIES = {
            "precip_cumulative": "Cumulative precipitation (this year)",
            "vpd_cumulative": "Cumulative VPD (this year)",
            "moisture": "Root-zone soil moisture",
            "temperature": "Root-zone soil temperature",
        }
        qty = st.sidebar.selectbox("Quantity", list(QUANTITIES),
                                   format_func=QUANTITIES.get)
        cumulative = qty.endswith("_cumulative")
        days, start, end = time_range("fld", default="1 year" if cumulative else "90 days")

        with st.spinner("Loading…"):
            per_field = {}
            for fk, lgs in groups_.items():
                frames = {}
                for l in lgs:
                    d = load(l.serial, st.session_state.token)
                    if not d.empty:
                        frames[l.name] = d
                if frames:
                    per_field[site_.fields.get(fk, fk)] = frames

        if len(per_field) < 2:
            st.warning("Fewer than two fields have data.")
            st.stop()

        series, notes, gaps = {}, [], []
        for fname, frames in per_field.items():
            if cumulative:
                # Rain and VPD come from the field's own ATMOS-41 station.
                col = (w.plot.PRECIP if qty == "precip_cumulative" else w.plot.VPD)
                src = next((n for n, d in frames.items()
                            if col in d.columns and d[col].notna().mean() > 0.5), None)
                if src is None and qty == "vpd_cumulative":
                    col = w.plot.VP
                    src = next((n for n, d in frames.items()
                                if col in d.columns and d[col].notna().mean() > 0.5), None)
                if src is None:
                    notes.append(f"{fname}: no weather station")
                    continue
                d = frames[src]
                s_ = w.cumulative_year(d[col].dropna(),
                                       how="sum" if qty == "precip_cumulative" else "mean")
                # A running total silently understates itself wherever the sensor was
                # down -- K2's rain gauge missed Jan, Feb and Jul 2026, so its curve sits
                # 285 mm below K1's and would read as a real site difference.
                this_year = d[col][d[col].index.year == d.index.max().year]
                expected = (d.index.max() - pd.Timestamp(d.index.max().year, 1, 1)) \
                    / pd.Timedelta(minutes=5)
                cov = this_year.notna().sum() / max(expected, 1)
                if cov < 0.9:
                    gaps.append(f"{fname} ({src}) has only {100 * cov:.0f}% of this "
                                f"year's readings — its total is an undercount")
                notes.append(f"{fname}: from {src}")
            else:
                s_ = w.field_series(frames, qty)
                n = sum(1 for d in frames.values() if w.depth_columns(d, qty))
                notes.append(f"{fname}: mean of {n} logger(s)")
            series[fname] = window(s_.to_frame("v"), days, start, end)["v"] if not s_.empty else s_

        series = {k: v for k, v in series.items() if v is not None and not v.empty}
        if len(series) < 2:
            st.warning("Not enough fields carry this quantity to compare.")
            st.stop()

        unit = {"precip_cumulative": "Cumulative precipitation (mm)",
                "vpd_cumulative": "Cumulative VPD (kPa d)",
                "moisture": "Root-zone soil moisture (m³ m⁻³)",
                "temperature": "Root-zone soil temperature (°C)"}[qty]
        show(w.plot.compare_series(
            series, ylabel=unit, title=f"{site_.name} — {QUANTITIES[qty]}",
            ylim=(0, 0.6) if qty == "moisture" else None,
            fmt=".1f" if cumulative else ".3f"), "fld")
        for g in gaps:
            st.warning(g)
        st.caption(" · ".join(notes) +
                   ("  ·  Cumulative totals restart on 1 January."
                    if cumulative else
                    "  ·  A field is several loggers tens of metres apart; the line is "
                    "their mean."))

    # ---- current year against the spread of earlier years -----------------
    elif kind == "This year vs previous":
        lg = st.sidebar.selectbox("Logger", w.loggers(), format_func=name_of)
        what = st.sidebar.selectbox(
            "Quantity", list(w.plot.CLIMATOLOGY_KINDS),
            format_func=w.plot.CLIMATOLOGY_KINDS.get,
        )
        full = load(lg.serial, st.session_state.token)
        if full.empty:
            st.error(f"{lg.name} returned no data.")
            st.stop()
        if full.index.year.nunique() < 2:
            st.warning(f"{lg.name} has only {full.index.year.nunique()} year of record — "
                       "there is nothing earlier to compare against.")
            st.stop()
        st.caption("The band spans every earlier year, the pale line is their median, and "
                   "the bold line is the current year. Cumulative totals restart on "
                   "1 January, so the gap at today's date is this year's surplus or "
                   "deficit.")
        show(w.plot.climatology(full, kind=what, logger=lg), "clim")

    # ---- one logger, several variables ------------------------------------
    else:
        lg = st.sidebar.selectbox("Logger", w.loggers(), format_func=name_of, key="lgv")
        days, start, end = time_range("var")
        full = load(lg.serial, st.session_state.token)
        df = window(full, days, start, end)
        if df.empty:
            st.error(f"{lg.name} has no data in this period.")
            st.stop()
        hk = set(w.housekeeping_columns())
        cols = [c for c in df.columns if df[c].notna().any() and c not in hk]
        chosen = st.multiselect("Variables", cols, default=cols[:3])
        norm = st.checkbox(
            "Overlay, scaled 0–1", value=False,
            help="Compares shape only. Magnitudes are not comparable once scaled.",
        )
        if chosen:
            show(w.plot.compare_variables(df, chosen, units=w.units(lg.serial),
                                          logger=lg, normalise=norm), "vars")
    st.stop()


# ==========================================================================
# EXPLORE
# ==========================================================================
sites = w.sites()
# Call the widget once and hold the result. Putting it inside a generator condition
# re-invokes it per iteration, which raises DuplicateWidgetID for any site after the first.
site_name = st.sidebar.selectbox("Site", [s.name for s in sites])
site = next(s for s in sites if s.name == site_name)
groups = site.by_field()
if len(groups) > 1:
    key = st.sidebar.selectbox("Field", list(groups),
                               format_func=lambda k: site.fields.get(k, k))
    choices = groups[key]
else:
    choices = next(iter(groups.values()))

lg = st.sidebar.selectbox("Logger", choices,
                          format_func=lambda x: x.name + ("  · offline" if x.is_offline else ""))
days, start, end = time_range("exp")

if st.sidebar.button("Refresh from server", width='stretch'):
    st.session_state.token += 1
    load.clear()
st.sidebar.caption(("Cache stale — refresh to update." if w.is_stale(lg)
                    else "Cache up to date.") + " Pulled at most once a day.")

with st.spinner(f"Loading {lg.name}…" + ("" if HAS_CACHE else "  (first view of a logger takes ~1 min)")):
    try:
        full = load(lg.serial, st.session_state.token)
    except w.FetchError as e:
        st.error(f"Could not reach the API: {e}")
        st.stop()

st.title(lg.name)
st.caption(f"{lg.serial}  ·  {lg.site_name} — {lg.field_name}  ·  "
           f"{lg.latitude:.4f}, {lg.longitude:.4f}  ·  {lg.elevation_m} m")

if full.empty:
    st.error(f"{lg.name} returned no data at all.")
    st.stop()

latest = full.index.max()
gap = (pd.Timestamp.now() - latest).days
if gap > 2:
    st.error(f"**Stopped reporting {latest:%d %B %Y}** — {gap} days ago. "
             f"Showing the {len(full):,} records before that.")

df = window(full, days, start, end)
if df.empty:
    st.warning(f"Nothing in that period; last record {latest:%Y-%m-%d}.")
    st.stop()

status = w.sensor_status(full)
live = w.active_measures(full)
live_cols = set(status.loc[status.live, "column"]) if not status.empty else set()
met = [c for c in lg.met_columns(df) if c in live_cols]

c1, c2, c3 = st.columns(3)
c1.metric("Records", f"{len(df):,}")
c2.metric("Shown", f"{df.index.min():%d %b %Y} — {df.index.max():%d %b %Y}")
c3.metric("Depths", ", ".join(live.get("moisture")
                              or live.get("matric_potential") or []) or "—")

if not status.empty:
    silent = status[~status.live & (status.coverage > 0.001)]
    if len(silent):
        with st.expander(f"{len(silent)} sensor(s) stopped reporting"):
            for _, r in silent.iterrows():
                st.markdown(f"`{r.column}` — {r.first:%b %Y} to **{r.last:%d %b %Y}**, "
                            f"{100 * r.coverage:.0f}% coverage, silent {r.days_silent} days")

names = ["Summary"]
if "moisture" in live:
    names += ["Soil moisture", "Root zone"]
if "temperature" in live:
    names += ["Soil temperature"]
if "matric_potential" in live:
    names += ["Water potential"]
if met:
    names += ["Weather", "Wind"]
names += ["Variables", "Coverage", "Loggers"]
tabs = dict(zip(names, st.tabs(names)))

if "Soil moisture" in tabs:
    with tabs["Soil moisture"]:
        show(w.plot.soil_moisture(df, precip=bool(met), logger=lg), "sm")
        if not met:
            st.caption("No rain gauge on this logger — see the field's ATMOS-41 station.")

if "Root zone" in tabs:
    with tabs["Root zone"]:
        method = st.radio("Weighting", ["trapezoid", "weighted"], horizontal=True,
                          format_func={"trapezoid": "Trapezoid",
                                       "weighted": "Layer-weighted"}.get,
                          help="Trapezoid assumes the profile varies linearly between "
                               "sensors (the equation in Bob's notes); layer-weighted "
                               "treats each sensor as representing its whole layer (what "
                               "the Ketelbroek notebook plots). Up to ~0.05 m³/m³ apart.")
        show(w.plot.root_zone_moisture(df, method=method, precip=bool(met), logger=lg), "rz")
        d = w.depths_of(w.depth_columns(df, "moisture"))
        if d:
            th = w.thicknesses(d)
            st.caption(f"Depths {', '.join(f'{x:g}' for x in d)} cm · thicknesses "
                       f"{', '.join(f'{x:g}' for x in th)} cm · profile {sum(th):g} cm")

if "Soil temperature" in tabs:
    with tabs["Soil temperature"]:
        show(w.plot.soil_temperature(df, air=bool(met), logger=lg), "st")

if "Water potential" in tabs:
    with tabs["Water potential"]:
        # These loggers carry no weather station, so VPD comes from the field's ATMOS-41.
        src = w.met_source(lg)
        met_df = None
        if src is not None and src.serial != lg.serial:
            met_df = window(load(src.serial, st.session_state.token), days, start, end)
        elif src is not None:
            met_df = df
        if met_df is not None and not met_df.empty:
            show(w.plot.matric_potential_vpd(df, met_df, logger=lg, met_logger=src), "mpv")
            if src.serial != lg.serial:
                st.caption(f"VPD is from **{src.name}**, the ATMOS-41 station for "
                           f"{src.field_name} — this logger has no weather sensor. "
                           "Soil supply below, atmospheric demand above.")
        else:
            show(w.plot.matric_potential(df, logger=lg), "mp")
            st.caption("No weather station available for VPD.")
        if "conductivity" in live:
            show(w.plot.conductivity(df, logger=lg), "ec")

if "Weather" in tabs:
    with tabs["Weather"]:
        show(w.plot.precipitation(df, cumulative=True, logger=lg), "pr")
        show(w.plot.temperature_radiation(df, logger=lg), "tr")
        show(w.plot.vpd_temperature(df, logger=lg), "vpd")

if "Wind" in tabs:
    with tabs["Wind"]:
        a, b = st.columns([1, 3])
        sectors = a.select_slider("Sectors", [8, 16], value=16)
        calm = a.slider("Calm threshold (m/s)", 0.0, 2.0, 0.5, 0.1,
                        help="Below this the vane direction is noise; excluded and "
                             "reported separately.")
        a.caption("Bars point in the direction the wind blows **from**.")
        with b:
            show(w.plot.wind_rose(df, sectors=sectors, calm_threshold=calm, logger=lg), "wr")

with tabs["Summary"]:
    if full.index.year.nunique() < 2:
        st.caption(f"Only {full.index.year.nunique()} year of record — nothing earlier to "
                   "compare against yet.")
    else:
        kinds = list(w.plot.CLIMATOLOGY_KINDS)
        if "moisture" not in live:
            kinds.remove("rzsm")
        if not met:
            kinds = [k for k in kinds if not k.startswith(("precip", "vpd"))]

        # Headline first: where this year stands today against the same date in earlier
        # complete years. The charts below show how it got there.
        standings = [(k, w.plot.climatology_standing(full, k)) for k in kinds]
        standings = [(k, v) for k, v in standings if v]
        if standings:
            cols = st.columns(len(standings))
            for col, (k, sd) in zip(cols, standings):
                dp = 0 if sd["cumulative"] else 3
                col.metric(
                    sd["label"],
                    f"{sd['current']:,.{dp}f}",
                    delta=f"{sd['delta']:+,.{dp}f} vs normal"
                          + (f" ({sd['pct']:+.0f}%)" if sd["pct"] is not None
                             and abs(sd["pct"]) < 1000 else ""),
                    delta_color="normal" if k != "vpd_cumulative" else "inverse",
                )
            ref = standings[0][1]
            st.caption(
                f"As of **{ref['as_of']:%d %B %Y}**, against the median of "
                f"{', '.join(str(y) for y in ref['years'])} on this date."
                + (f" {', '.join(str(y) for y in ref['excluded'])} excluded — incomplete."
                   if ref["excluded"] else "")
            )
        if not kinds:
            st.info("This logger has neither soil moisture nor a weather station.")
        for k in kinds:
            show(w.plot.climatology(full, kind=k, logger=lg), f"cl{k}")

with tabs["Variables"]:
    units = w.units(lg.serial)
    hk = set(w.housekeeping_columns())
    show_hk = st.checkbox("Include battery / logger diagnostics", value=False)
    cols = [c for c in df.columns if df[c].notna().any() and (show_hk or c not in hk)]
    for i, c in enumerate(st.multiselect("Variables", cols, default=cols[:1])):
        show(w.plot.timeseries(df, c, unit=units.get(c, ""), logger=lg), f"ts{i}")

with tabs["Coverage"]:
    show(w.plot.coverage_heatmap(full, logger=lg), "cov")
    st.caption("Fraction of the expected 288 daily samples present.")

with tabs["Loggers"]:
    st.caption("`ATMOS` = weather station · `WP` = water potential (TEROS21) · "
               "`SMST` = soil moisture and temperature")
    st.dataframe(w.overview(), width='stretch', hide_index=True)

st.sidebar.divider()
st.sidebar.caption(f"Showing {len(df):,} rows · 5-minute resolution")
