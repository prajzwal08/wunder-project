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
      /* ...but the arrow that re-opens a collapsed sidebar lives inside that header, so
         put it back: hiding it strands the user with no way to get the controls back. */
      header [data-testid="stExpandSidebarButton"],
      header [data-testid="stExpandSidebarButton"] * {visibility: visible;}
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

      /* Eleven tabs do not fit a phone. Let the strip scroll sideways rather than
         wrap into a block that pushes the charts off the screen. */
      .stTabs [data-baseweb="tab-list"] {overflow-x: auto; flex-wrap: nowrap;
                                         scrollbar-width: thin;}
      .stTabs [data-baseweb="tab"] {white-space: nowrap; flex: 0 0 auto;}

      @media (max-width: 640px) {
        /* Reclaim the desktop gutters; a chart 340 px wide needs every pixel. */
        .block-container {padding-top: 1rem; padding-bottom: 1rem;
                          padding-left: 0.7rem; padding-right: 0.7rem;}
        h1 {font-size: 1.3rem;}
        .stTabs [data-baseweb="tab"] {font-size: 0.9rem; padding: 0.35rem 0.6rem;}
        [data-testid="stMetricValue"] {font-size: 1.05rem;}
        [data-testid="stMetricLabel"] {font-size: 0.72rem;}
        /* Streamlit stacks columns at this width; kill the leftover side gaps so the
           stacked blocks sit flush rather than in a narrowing staircase. */
        [data-testid="stHorizontalBlock"] {gap: 0.6rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

#: Short labels in a row, the way a stock app shows a price history -- the label *is*
#: the control, so there is nothing to open and read.
PRESETS = {"1W": 7, "1M": 30, "3M": 90, "6M": 182, "1Y": 365, "All": None}
PRESET_HELP = {"1W": "7 days", "1M": "30 days", "3M": "90 days", "6M": "6 months",
               "1Y": "1 year", "All": "the whole record"}
MEASURES = {"moisture": "Soil moisture", "temperature": "Soil temperature",
            "matric_potential": "Matric potential", "conductivity": "Electrical conductivity"}

HAS_CACHE = (w.fetch_module.CACHE_DIR.exists()
             and any(w.fetch_module.CACHE_DIR.glob("*.parquet")))
# hasattr, not a plain call: Streamlit re-runs this script on a code change but keeps
# already-imported modules in sys.modules, so a new submodule can be missing from a package
# object left over from before it existed. Degrade to a live fetch rather than crash; a
# reboot of the app clears it properly.
HAS_PUBLISHED = hasattr(w, "publish") and w.publish.available()
# Same guard, same reason: `wunder.et` and `wunder.stress` are new submodules, and a
# Streamlit Cloud redeploy re-runs this script while keeping the already-imported
# package object in sys.modules -- so the new script can meet a `wunder` that predates
# them. Degrade to the pre-stress app rather than an AttributeError loop; a reboot of
# the Cloud app clears it properly.
HAS_STRESS = (hasattr(w, "stress") and hasattr(w, "soil_source")
              and hasattr(w.plot, "SOIL_KINDS"))


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
        # None lets fetch() decide: top up if the cache is more than a day old. Passing
        # False would mean "never hit the network", so the record would freeze at whatever
        # was cached the first time.
        return w.fetch(serial, refresh=True if token else None)
    if HAS_PUBLISHED:
        # Deployment path: the bulk comes off disk instantly and only the tail since the
        # last published timestamp crosses the network.
        return w.publish.read_current(serial)
    df = w.fetch(serial, start="2020-01-01", cache=False)
    return w.resample(df, "30min") if not df.empty else df


def on_phone() -> bool:
    """True when the browser looks like a phone.

    Plotly has no media query: a figure's margins, type sizes and height are baked into
    the layout when it is built, so something has to decide. Streamlit does not report
    the viewport width, and the User-Agent is the only signal available on the first
    render -- a JS round-trip would make every chart appear at the wrong size first and
    then jump. The sidebar toggle below exists because this guess can be wrong.
    """
    if "phone" in st.session_state:
        return bool(st.session_state.phone)
    try:
        ua = (st.context.headers.get("User-Agent") or "").lower()
    except Exception:
        return False
    return any(t in ua for t in ("iphone", "android", "ipod", "windows phone")) or (
        "mobile" in ua and "ipad" not in ua)


CHART_CONFIG = {"displaylogo": False, "displayModeBar": "hover",
                "toImageButtonOptions": {"format": "png", "scale": 3}}


def show(fig, key: str) -> None:
    if on_phone():
        fig = w.plot.compact(fig)
    st.plotly_chart(fig, width='stretch', key=key, config=CHART_CONFIG)


def name_of(lg) -> str:
    site = lg.site_name.replace("Voedselbos ", "").replace(" Herenboeren", "")
    return f"{lg.name} · {site}"


def time_range(key: str, *, default: str = "1M", where=None):
    """Preset periods plus an explicit start/end. Returns (days, start, end).

    A row of pills rather than a dropdown: the set is small and fixed, so a reader
    should see all of it at once and switch with one tap.

    The window is applied **server-side** -- the frame is re-sliced and re-decimated --
    rather than by Plotly's in-chart range buttons, which would look the same and behave
    worse. Every figure decimates to about 4000 points for the span it is handed, so a
    client-side zoom from a year down to a week would leave you reading 6-hourly means
    of 5-minute data. Re-slicing here gets the real resolution back.
    """
    where = where if where is not None else st.sidebar
    choice = where.segmented_control(
        "Period", [*PRESETS, "Custom…"], key=f"p{key}", default=default,
        label_visibility="collapsed",
        help="  ·  ".join(f"{k} = {v}" for k, v in PRESET_HELP.items()),
    ) or default
    if choice != "Custom…":
        return PRESETS[choice], None, None
    today = dt.date.today()
    pair = where.date_input(
        "From / to", value=(today - dt.timedelta(days=30), today),
        min_value=dt.date(2022, 12, 1), max_value=today, key=f"d{key}",
    )
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return None, pd.Timestamp(pair[0]), pd.Timestamp(pair[1]) + pd.Timedelta(days=1)
    return None, None, None


def with_period(key: str, render, *, default: str = "1M") -> None:
    """A chart with its own period pills above it, in its own fragment.

    Per chart rather than one picker for the page: the question you are asking of soil
    moisture is rarely the question you are asking of the wind rose, and a single
    control forced every panel to answer at the same span.

    The fragment is what makes that affordable. Without it, changing one chart's period
    re-runs the whole script and rebuilds every figure in every tab; with it, only this
    chart redraws, so a per-chart control is *faster* than the page-wide one it replaces.

    `render(days, start, end)` does its own windowing -- some panels need two frames,
    a soil logger's and its station's, sliced to the same span.
    """
    @st.fragment
    def panel() -> None:
        render(*time_range(key, default=default, where=st))

    panel()


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
        # Derived quantities need both halves of a field: its station for the demand
        # and its soil probe for the supply. Skipped entirely without the stress
        # module, exactly as the Summary tab is.
        if HAS_STRESS:
            QUANTITIES |= {
                "wsf": "Water stress factor (WSF)",
                "et_actual": "ETa (daily)",
                "et_cumulative": "Cumulative ETa (this year)",
                "et0_cumulative": "Cumulative ETo (this year)",
            }
        DERIVED = {"wsf", "et_actual", "et_cumulative", "et0_cumulative"}
        qty = st.sidebar.selectbox("Quantity", list(QUANTITIES),
                                   format_func=QUANTITIES.get)
        cumulative = qty.endswith("_cumulative")
        days, start, end = time_range("fld", default="1Y" if cumulative else "3M")

        with st.spinner("Loading…"):
            per_field, field_loggers = {}, {}
            for fk, lgs in groups_.items():
                frames = {}
                for l in lgs:
                    d = load(l.serial, st.session_state.token)
                    if not d.empty:
                        frames[l.name] = d
                if frames:
                    fname_ = site_.fields.get(fk, fk)
                    per_field[fname_] = frames
                    field_loggers[fname_] = lgs

        if len(per_field) < 2:
            st.warning("Fewer than two fields have data.")
            st.stop()

        series, notes, gaps = {}, [], []
        for fname, frames in per_field.items():
            if qty in DERIVED:
                # Both halves of the field: its station for the demand, its soil probe
                # for the supply. `met_source`/`soil_source` prefer the same field, so
                # this is the same pairing the Explore tabs use.
                base = field_loggers[fname][0]
                station, probe = w.met_source(base), w.soil_source(base)
                if station is None or probe is None:
                    notes.append(f"{fname}: no "
                                 + ("weather station" if station is None else "soil probe"))
                    continue
                met_d = (frames[station.name] if station.name in frames
                         else load(station.serial, st.session_state.token))
                soil_d = (frames[probe.name] if probe.name in frames
                          else load(probe.serial, st.session_state.token))
                try:
                    out = w.stress.actual_et(soil_d, met_d, ref=probe.serial)
                except (FileNotFoundError, ValueError) as exc:
                    notes.append(f"{fname}: {str(exc).split('.')[0]}")
                    continue
                if out.empty:
                    notes.append(f"{fname}: no overlapping soil and weather record")
                    continue
                s_ = {"wsf": out["wsf"],
                      "et_actual": out["et"],
                      "et_cumulative": w.cumulative_year(out["et"]),
                      "et0_cumulative": w.cumulative_year(out["et0"])}[qty]
                notes.append(f"{fname}: soil from {probe.name}, weather from {station.name}")
            elif cumulative:
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
                "temperature": "Root-zone soil temperature (°C)",
                "wsf": "Water stress factor, WSF (—)",
                "et_actual": "ETa (mm d⁻¹)",
                "et_cumulative": "Cumulative ETa (mm)",
                "et0_cumulative": "Cumulative ETo (mm)"}[qty]
        show(w.plot.compare_series(
            series, ylabel=unit, title=f"{site_.name} — {QUANTITIES[qty]}",
            ylim=(0, 0.6) if qty == "moisture" else (0, 1.02) if qty == "wsf" else None,
            fmt=".1f" if cumulative else ".2f" if qty in DERIVED else ".3f"), "fld")
        for g in gaps:
            st.warning(g)
        st.caption(" · ".join(notes) +
                   ("  ·  Each field's own instruments: one soil probe and one station, "
                    "not a field mean — WSF is defined against the soil under that "
                    "particular probe." if qty in DERIVED else
                    "  ·  Cumulative totals restart on 1 January." if cumulative else
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

# A click on the Map tab lands here, one rerun later. Widget state has to be seeded
# *before* the widget exists -- setting it afterwards raises -- so the map only records
# which logger was clicked and asks for a rerun, and the three selectboxes are primed
# here from that serial.
pending = st.session_state.pop("pending_logger", None)
if pending is not None:
    picked = w.logger(pending)
    st.session_state["exp_site"] = picked.site_name
    st.session_state["exp_field"] = picked.field_key
    st.session_state["exp_logger"] = picked

# Call the widget once and hold the result. Putting it inside a generator condition
# re-invokes it per iteration, which raises DuplicateWidgetID for any site after the first.
site_name = st.sidebar.selectbox("Site", [s.name for s in sites], key="exp_site")
site = next(s for s in sites if s.name == site_name)
groups = site.by_field()
# Keyed widgets keep their value across a site change, and a field or logger belonging to
# the site you just left is not in the new options -- which Streamlit refuses rather than
# ignores. Drop a stale selection so the widget falls back to its first option.
if st.session_state.get("exp_field") not in groups:
    st.session_state.pop("exp_field", None)
if len(groups) > 1:
    key = st.sidebar.selectbox("Field", list(groups), key="exp_field",
                               format_func=lambda k: site.fields.get(k, k))
    choices = groups[key]
else:
    choices = next(iter(groups.values()))

if st.session_state.get("exp_logger") not in choices:
    st.session_state.pop("exp_logger", None)
lg = st.sidebar.selectbox("Logger", choices, key="exp_logger",
                          format_func=lambda x: x.name + ("  · offline" if x.is_offline else ""))

st.sidebar.toggle(
    "Compact charts", value=on_phone(), key="phone",
    help="Trims chart margins and type for a narrow screen. Guessed from the browser; "
         "set it yourself if the guess is wrong.")

if st.sidebar.button("Refresh from server", width='stretch'):
    st.session_state.token += 1
    load.clear()
with st.spinner(f"Loading {lg.name}…"
                 + ("" if HAS_CACHE or HAS_PUBLISHED
                    else "  (first view of a logger takes ~1 min)")):
    try:
        full = load(lg.serial, st.session_state.token)
    except w.FetchError as e:
        st.error(f"Could not reach the API: {e}")
        st.stop()

st.sidebar.caption(
    (f"Data to {full.index.max():%d %b %H:%M}. " if not full.empty else "No data. ")
    + ("Topped up automatically once a day; the button forces it." if HAS_CACHE
       else "Published record, topped up live." if HAS_PUBLISHED
       else "Fetched live from the API.")
)

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

status = w.sensor_status(full)
live = w.active_measures(full)
live_cols = set(status.loc[status.live, "column"]) if not status.empty else set()
# From the full record, not a window: a met column that reported all last year but not
# in the last 30 days is still a met column, and deciding from a window meant the
# Weather tab could appear and disappear as the period changed.
met = [c for c in lg.met_columns(full) if c in live_cols]

# Weather and soil water come from whichever logger actually measures them: this one
# where possible, otherwise the field's ATMOS-41 and the nearest working soil probe.
# That is what lets every logger show ET and the stress factor, not only the four with
# their own weather sensors.
met_lg = w.met_source(lg)
soil_lg = w.soil_source(lg) if HAS_STRESS else None


@st.cache_data(show_spinner=False, ttl=3600)
def _source_frame(serial: str, token: int) -> pd.DataFrame:
    return load(serial, token)


def source_of(other, base: pd.DataFrame) -> pd.DataFrame | None:
    """`base` when `other` is this logger, else that logger's own record."""
    if other is None:
        return None
    if other.serial == lg.serial:
        return base
    got = _source_frame(other.serial, st.session_state.token)
    return got if not got.empty else None


met_full = source_of(met_lg, full)
soil_full = source_of(soil_lg, full)
borrowed = [f"{what} from **{who.name}**"
            for what, who in (("weather", met_lg), ("soil water", soil_lg))
            if who is not None and who.serial != lg.serial]

c1, c2, c3 = st.columns(3)
c1.metric("Records", f"{len(full):,}")
c2.metric("Record", f"{full.index.min():%d %b %Y} — {full.index.max():%d %b %Y}")
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
    names += ["Weather"]
if met_full is not None:
    names += ["Evapotranspiration"]
if HAS_STRESS and soil_full is not None:
    names += ["Understand"]
if met:
    names += ["Wind"]
names += ["Variables", "Coverage", "Loggers"]
tabs = dict(zip(names, st.tabs(names)))

def _empty(frame) -> bool:
    """Warn and report True when a period holds nothing for this logger."""
    if frame is None or frame.empty:
        st.warning(f"Nothing in that period; last record {latest:%Y-%m-%d}.")
        return True
    return False


if "Soil moisture" in tabs:
    with tabs["Soil moisture"]:
        def _sm(days, start, end) -> None:
            d = window(full, days, start, end)
            if _empty(d):
                return
            show(w.plot.soil_moisture(d, precip=bool(met), logger=lg), "sm")
            if not met:
                st.caption("No rain gauge on this logger — see the field's "
                           "ATMOS-41 station.")
        with_period("sm", _sm)

if "Root zone" in tabs:
    with tabs["Root zone"]:
        def _rz(days, start, end) -> None:
            d = window(full, days, start, end)
            if _empty(d):
                return
            show(w.plot.root_zone_moisture(d, precip=bool(met), logger=lg), "rz")
            depths = w.depths_of(w.depth_columns(d, "moisture"))
            if depths:
                th = w.thicknesses(depths)
                st.caption(
                    f"Depths {', '.join(f'{x:g}' for x in depths)} cm · thicknesses "
                    f"{', '.join(f'{x:g}' for x in th)} cm · profile {sum(th):g} cm  ·  "
                    "Each sensor stands for the slab around it, with boundaries at the "
                    "midpoints between sensors. The same weighting collapses the "
                    "per-layer WSF into the profile figure.")
        with_period("rz", _rz)

if "Soil temperature" in tabs:
    with tabs["Soil temperature"]:
        def _st(days, start, end) -> None:
            d = window(full, days, start, end)
            if _empty(d):
                return
            show(w.plot.soil_temperature(d, air=bool(met), logger=lg), "st")
        with_period("st", _st)

if "Water potential" in tabs:
    with tabs["Water potential"]:
        def _mp(days, start, end) -> None:
            d = window(full, days, start, end)
            if _empty(d):
                return
            # These loggers carry no weather station, so VPD comes from the field's
            # ATMOS-41, sliced to the same span.
            src = met_lg
            met_df = None
            if src is not None:
                met_df = window(source_of(src, full), days, start, end)
            if met_df is not None and not met_df.empty:
                show(w.plot.matric_potential_vpd(d, met_df, logger=lg, met_logger=src),
                     "mpv")
                if src.serial != lg.serial:
                    st.caption(f"VPD is from **{src.name}**, the ATMOS-41 station for "
                               f"{src.field_name} — this logger has no weather sensor. "
                               "Soil supply below, atmospheric demand above.")
            else:
                show(w.plot.matric_potential(d, logger=lg), "mp")
                st.caption("No weather station available for VPD.")
        with_period("mp", _mp)

        if "conductivity" in live:
            def _ec(days, start, end) -> None:
                d = window(full, days, start, end)
                if _empty(d):
                    return
                show(w.plot.conductivity(d, logger=lg), "ec")
            with_period("ec", _ec)

if "Weather" in tabs:
    with tabs["Weather"]:
        for _key, _builder, _default in (
                ("pr", lambda d: w.plot.precipitation(d, cumulative=True, logger=lg), "3M"),
                ("tr", lambda d: w.plot.temperature_radiation(d, logger=lg), "1M"),
                ("vpd", lambda d: w.plot.vpd_temperature(d, logger=lg), "1M")):
            # Bound as defaults, not captured: a closure over the loop variables would
            # leave all three panels drawing whatever the last iteration set.
            def _wx(days, start, end, key=_key, build=_builder) -> None:
                d = window(full, days, start, end)
                if _empty(d):
                    return
                show(build(d), key)
            with_period(_key, _wx, default=_default)

if "Evapotranspiration" in tabs:
    with tabs["Evapotranspiration"]:
        if borrowed:
            st.caption("This logger has no weather sensor of its own — "
                       + ", ".join(borrowed) + ".")

        def _et(days, start, end) -> None:
            met_win = (window(met_full, days, start, end)
                       if met_full is not None else full)
            soil_win = (window(soil_full, days, start, end)
                        if soil_full is not None else None)
            wb = w.water_balance(met_win)
            if wb.empty:
                st.warning("Not enough complete days in that period to compute ETo.")

            # ETa needs the site's soil parameters; a site whose model input has not
            # been prepared simply doesn't get the stress rows.
            try:
                aet = (w.stress.actual_et(soil_win, met_win, ref=soil_lg.serial)
                       if HAS_STRESS and soil_win is not None else None)
            except (FileNotFoundError, ValueError):
                aet = None

            if aet is None or aet.empty:
                if not wb.empty:
                    e1, e2, e3 = st.columns(3)
                    e1.metric("Mean ETo", f"{wb.et0.mean():.2f} mm/day")
                    e2.metric("Total ETo", f"{wb.et0.sum():,.0f} mm")
                    e3.metric("Total rainfall", f"{wb.precip.sum():,.0f} mm")
                st.info("ETa and WSF need this site's soil parameters, extracted from "
                        "a prepared model run (`forcing/extract_soil.py`).")
                show(w.plot.precipitation(met_win, cumulative=True, logger=met_lg), "et")
                return

            # Measured record, nothing to drag. Every control in the app lives in the
            # Understand tab, so a number read off this tab is always the same number.
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Total ETo", f"{aet.et0.sum():,.0f} mm",
                      help="Reference: the demand a well-watered grass sward would meet.")
            e2.metric("Total ETa", f"{aet.et.sum():,.0f} mm",
                      help="Actual: WSF \u00d7 Kc \u00d7 ETo, what this soil could supply.")
            e3.metric("Mean WSF", f"{aet.wsf.mean():.2f}",
                      help="Water stress factor: 1 = the soil can meet any demand, "
                           "0 = it can meet none of it. See the Understand tab.")
            e4.metric("Demand met", f"{100 * aet.et.sum() / aet.et0.sum():.0f}%",
                      help="ETa as a share of ETo over the period shown.")
            show(w.plot.evapotranspiration(soil_win, met=met_win, ref=soil_lg.serial,
                                           out=aet, logger=lg), "et")

            a = aet.attrs
            st.caption(
                f"**WSF** is evaluated at each depth against that depth's own soil and "
                f"then thickness-weighted \u2014 depths {', '.join(a['depths'])} cm at "
                f"**{a['site']}**. Profile values \u03b8_sat = {a['theta_sat']:.3f}, "
                f"\u03b8_fc = {a['theta_fc']:.3f}, half stress at "
                f"{a['theta_half']:.3f}, \u03b8_r = {a['theta_r']:.3f} "
                f"m\u00b3 m\u207b\u00b3.  \u00b7  {len(wb):,} complete days shown. "
                f"ETa is WSF \u00d7 ETo with K\u1d9c = 1: no claim is made about how a "
                f"food forest differs from the reference grass. To change K\u1d9c, and "
                f"for the method and an irrigation calculator, go to **Understand**."
            )

        with_period("et", _et, default="6M")

        st.caption(
            "**ETo is Makkink reference evapotranspiration** — the demand a well-watered "
            "short grass sward would meet, from global radiation and air temperature with "
            "the Dutch coefficient 0.65. This is the same quantity KNMI publishes as "
            "`EV24`, so it is comparable with the national record. It is *not* what this "
            "canopy actually transpires, which is what ETa estimates. Makkink also assumes "
            "the pyranometer sees open sky: a mast that the canopy has grown over "
            "under-reads radiation, and its ETo with it — which is the case at "
            "K1 Voedselbos."
        )

def _limit_key_html(marks) -> str:
    """The four soil marks as a compact coloured row under a figure.

    A key, not a legend: each line says what the mark *means*, which is the thing a
    reader who has not met theta_fc before actually needs. Wraps to one per line on a
    phone, because the alternative is four captions fighting inside the plot.
    """
    items = "".join(
        f'<div style="display:flex;align-items:baseline;gap:.45rem;'
        f'min-width:15rem;flex:1 1 15rem">'
        f'<span style="color:{m["colour"]};font-size:1.1rem;line-height:1">&#9646;</span>'
        f'<span><b style="color:{m["colour"]}">{m["symbol"]} {m["label"]}</b> '
        f'<span style="opacity:.62">{m["meaning"]}</span> '
        f'<b style="font-variant-numeric:tabular-nums">{m["value"]:.3f}</b></span></div>'
        for m in marks
    )
    return ('<div style="display:flex;flex-wrap:wrap;gap:.35rem 1.4rem;'
            f'font-size:.86rem;margin:-.4rem 0 .6rem">{items}</div>')


if "Understand" in tabs:
    with tabs["Understand"]:
        try:
            site_code = w.stress.site_of(soil_lg.serial)
            wsf_depths = [d for d in w.active_measures(soil_full).get("moisture", [])
                          if float(d) <= w.stress.MAX_DEPTH_CM]
            soil_limits = (w.stress.water_limits(site_code, wsf_depths)
                           if wsf_depths else {})
            profile = w.stress.root_zone_limits(soil_full, ref=soil_lg.serial)
        except (FileNotFoundError, ValueError) as exc:
            site_code, soil_limits, profile = None, {}, {}
            st.warning(f"No soil parameters for this logger — {str(exc).split('.')[0]}.")

        # ================= the picture ======================================
        if profile:
            st.caption(
                "The atmosphere asks for water, the soil meets part of it, and the rest "
                f"is what you would have to irrigate. Live for **{soil_lg.name}** — "
                "move a slider and the whole picture moves."
            )

            @st.fragment
            def story_panel(info: dict) -> None:
                depths, thick = info["depths"], info["thicknesses"]
                profile_cm = float(sum(thick))
                cur = w.root_zone(soil_full, "moisture",
                                  columns=info["columns"]).dropna()
                theta_now = float(cur.iloc[-1]) if len(cur) else info["theta_fc"]
                eto_now = 3.0
                if met_full is not None:
                    recent = w.reference_et(met_full).tail(7)
                    if len(recent):
                        eto_now = float(recent.mean())

                c_ctl, c_fig = st.columns([1, 3])
                with c_ctl:
                    theta = st.slider(
                        "\u03b8 — how wet the soil is", 0.0, 0.50,
                        round(theta_now, 3), 0.001, key="story_theta", format="%.3f",
                        help="Opens at this logger's latest root-zone reading.")
                    eto = st.slider("ETo — today's demand (mm/day)", 0.0, 8.0,
                                    round(eto_now, 1), 0.1, key="story_eto",
                                    help="Opens at this station's last 7 days.")
                    kc = st.slider("K\u1d9c — crop coefficient", 0.3, 1.5, 1.0, 0.05,
                                   key="story_kc",
                                   help="1.0 = this canopy uses what the reference "
                                        "grass would. It is what every other figure in "
                                        "the app assumes.")
                    if st.button("Back to today", width='stretch'):
                        for k in ("story_theta", "story_eto", "story_kc"):
                            st.session_state.pop(k, None)
                        st.rerun(scope="fragment")

                wsf = float(w.stress.stress_factor(
                    theta, theta_fc=info["theta_fc"], theta_r=info["theta_r"],
                    theta_sat=info["theta_sat"]))
                eta = wsf * kc * eto
                unmet = max(0.0, kc * eto - eta)
                refill = sum(
                    max(0.0, info["limits"][d]["theta_fc"] - theta) * L * 10.0
                    for d, L in zip(depths, thick))
                to_stress = ((theta - info["theta_half"]) * profile_cm * 10.0 / eta
                             if eta > 0 and theta > info["theta_half"] else 0.0)

                with c_fig:
                    show(w.plot.water_story(info, theta=theta, eto=eto, kc=kc,
                                            narrow=on_phone()), "story")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("WSF now", f"{wsf:.2f}",
                          help="The share of demand this soil can meet.")
                m2.metric("ETa", f"{eta:.2f} mm/day",
                          help="WSF × Kc × ETo — what the soil can actually supply.")
                m3.metric("Not getting", f"{unmet:.2f} mm/day", delta_color="inverse",
                          delta=(f"{100 * unmet / (kc * eto):.0f}% of demand"
                                 if kc * eto > 0 else None))
                m4.metric("To refill the root zone", f"{refill:.0f} mm",
                          help="Irrigation that would bring every layer back to its own "
                               "field capacity.")

                if theta > info["theta_half"] and eta > 0:
                    st.markdown(
                        f"At this demand and with no rain, stress begins in about "
                        f"**{to_stress:.0f} days** (\u03b8 falls to "
                        f"{info['theta_half']:.3f}). A floor, not a forecast: ETa falls "
                        "as the soil dries, so the real number is larger.")
                elif theta <= info["theta_half"]:
                    st.markdown(f"**This soil is already past the point where stress "
                                f"begins** (\u03b8 {theta:.3f} against "
                                f"{info['theta_half']:.3f}).")

                st.caption(
                    "Each soil band is filled to today's water content against that "
                    "layer's saturation, and the green tick is its own field capacity. "
                    "**Thin layers are drawn thicker than scale** so their labels fit — "
                    "the centimetre figure beside each band is the true one. Soil from "
                    f"**{info['site']}**, profile {profile_cm:g} cm. Refill assumes "
                    "water reaches every layer and none drains past the profile, so it "
                    "is an upper bound on what the soil will hold, not a schedule."
                )

            story_panel(profile)

        st.divider()
        st.caption("The detail, if you want it:")

        # ================= the detail, folded away ==========================
        with st.expander("What ETo and ETa actually are"):
            c_o, c_a = st.columns(2)
            with c_o:
                st.markdown(
                    "**ET<sub>o</sub> — the *reference***<br>"
                    "What a well-watered short grass sward would evaporate under this "
                    "weather. It is **demand**: the atmosphere asking. It depends only "
                    "on radiation and air temperature and *never* on how wet your soil "
                    "is, which is what makes it a fixed yardstick you can compare sites "
                    "and years against. Here it is Makkink with the Dutch coefficient "
                    "0.65 — the same quantity KNMI publishes as `EV24`.",
                    unsafe_allow_html=True)
            with c_a:
                st.markdown(
                    "**ET<sub>a</sub> — the *actual***<br>"
                    "What this soil could actually supply: "
                    "$ET_a = \\mathrm{WSF} \\cdot K_c \\cdot ET_o$. When the "
                    "profile is wet, WSF is near 1 and the two almost coincide; as it "
                    "dries they part, and **the gap between them is water the plants "
                    "asked for and did not get**. $K_c$ is 1 in every figure in the app "
                    "— no claim is made about how a food forest's canopy differs from "
                    "the reference grass.",
                    unsafe_allow_html=True)

        if soil_limits:
            with st.expander("The stress function itself, with sliders"):

                @st.fragment
                def wsf_panel(limits: dict, site: str) -> None:
                    c_ctl, c_fig = st.columns([1, 3])
                    with c_ctl:
                        depth = st.selectbox("Depth", list(limits), key="wsf_depth",
                                             format_func=lambda d: f"{d} cm")
                        L = limits[depth]
                        # Slider keys carry the depth, so changing depth builds fresh
                        # widgets at that layer's soil rather than keeping the last
                        # layer's numbers.
                        sfx = f"_{depth}"
                        if st.button("Reset to this soil", width='stretch'):
                            for k in [k for k in st.session_state
                                      if k.startswith("wsf_") and k.endswith(sfx)]:
                                del st.session_state[k]
                            st.rerun(scope="fragment")
                        theta = st.slider("\u03b8 — water content", 0.0, 0.60,
                                          float(L["theta_half"]), 0.001,
                                          key=f"wsf_theta{sfx}", format="%.3f")
                        t_sat = st.slider("\u03b8_s — saturation", 0.20, 0.65,
                                          float(L["theta_sat"]), 0.001,
                                          key=f"wsf_sat{sfx}", format="%.3f")
                        t_fc = st.slider("\u03b8_fc — field capacity", 0.05, 0.45,
                                         float(L["theta_fc"]), 0.001,
                                         key=f"wsf_fc{sfx}", format="%.3f")
                        t_r = st.slider("\u03b8_r — residual", 0.0, 0.20,
                                        float(L["theta_r"]), 0.001,
                                        key=f"wsf_r{sfx}", format="%.3f")
                        k_steep = st.slider("k — steepness", 10.0, 400.0,
                                            float(w.stress.STEEPNESS), 5.0,
                                            key=f"wsf_k{sfx}", format="%g")
                    with c_fig:
                        if t_fc <= t_r:
                            st.warning("Field capacity has to sit above residual.")
                            return
                        show(w.plot.wsf_explorer(
                            t_sat, t_fc, t_r, theta=theta, steepness=k_steep,
                            title=f"Water stress function — {depth} cm, {site}"),
                            "wsfx")
                        st.markdown(_limit_key_html(w.plot.limit_key(t_sat, t_fc, t_r)),
                                    unsafe_allow_html=True)

                wsf_panel(soil_limits, site_code)
                st.latex(r"\mathrm{WSF}(\theta) = \frac{1}{1 + \exp\!\left[-k\,"
                         r"\theta_{s}\left(\theta - \frac{\theta_{fc} + "
                         r"\theta_{r}}{2}\right)\right]}")

        if profile:
            with st.expander("The maths, layer by layer"):
                rows = "\n".join(
                    f"| {d} | {profile['limits'][d]['theta_sat']:.3f} "
                    f"| {profile['limits'][d]['theta_fc']:.3f} "
                    f"| {profile['limits'][d]['theta_half']:.3f} "
                    f"| {profile['limits'][d]['theta_r']:.3f} | {L_:.1f} |"
                    for d, L_ in zip(profile["depths"], profile["thicknesses"]))
                st.markdown(
                    "**The soil parameters are the model's own.** $\\theta_s$ and "
                    "$\\theta_r$ are read straight off the van Genuchten curve "
                    "*STEMMUS_SCOPE itself runs on* — SoilGrids texture through the "
                    "Schaap/Rosetta pedotransfer — and $\\theta_{fc}$ is that curve "
                    "evaluated at −33 kPa. Using the model's own soil is deliberate: "
                    "$\\mathrm{WSF} \\cdot ET_o$ and the model's transpiration then "
                    "rest on the same ground, so a disagreement between them means "
                    "something.\n\n"
                    "**WSF is computed per layer and only then weighted.** Each depth "
                    "gets its own soil and its own WSF; each sensor stands for the slab "
                    "around it, with boundaries at the midpoints between sensors:\n\n"
                    "| depth (cm) | $\\theta_s$ | $\\theta_{fc}$ | stress begins | "
                    "$\\theta_r$ | thickness (cm) |\n|---|---|---|---|---|---|\n"
                    + rows + "\n\n"
                    "**Why not average the moisture first?** Because the function is "
                    "curved. A wet 80 cm and a bone-dry 5 cm average to a "
                    "comfortable-looking profile, and one sigmoid on that average "
                    "reports no stress at all — exactly the situation the factor exists "
                    "to detect. At `F1_2_SMST2` the summer mean is **0.88** per layer "
                    "against **0.995** the other way round.\n\n"
                    "**That weighting is a modelling choice, not an identity.** WSF is a "
                    "ratio, so unlike stored water it does not add up over a profile. "
                    "Read it as *the share of root-zone demand the profile can meet, "
                    "each sensor standing for the slab around it*. The physically right "
                    "weight is root density, which this network cannot measure.\n\n"
                    "**It replaced an FAO-56 $K_s$** — a straight line from a threshold "
                    "at $\\theta_{fc} - p\\cdot\\mathrm{TAW}$ to zero at the "
                    "wilting point. FAO-56 needs a depletion fraction $p$ that nothing "
                    "here measures, and it has a kink at the threshold that no soil "
                    "exhibits."
                )

        if met_full is not None:
            with st.expander("See it on the real record, with K\u1d9c"):
                st.markdown(
                    "The same figure the **Evapotranspiration** tab shows, except that "
                    "here $K_c$ moves. Watch the **third row**: the gap between "
                    "cumulative ET<sub>o</sub> and ET<sub>a</sub> is the season's unmet "
                    "demand, and $K_c$ moves it directly. Nothing here changes any "
                    "other tab.", unsafe_allow_html=True)

                def _play(days, start, end) -> None:
                    et_win = window(met_full, days, start, end)
                    soil_et_win = window(soil_full, days, start, end)
                    try:
                        base = w.stress.actual_et(soil_et_win, et_win,
                                                  ref=soil_lg.serial)
                    except (FileNotFoundError, ValueError):
                        base = None
                    if base is None or base.empty:
                        st.info("Not enough overlapping soil and weather record in "
                                "that period.")
                        return
                    kc = st.slider("K\u1d9c — crop coefficient", 0.3, 1.5, 1.0, 0.05,
                                   key="und_kc")
                    out = base.copy()
                    out.attrs = dict(base.attrs)
                    out["et"] = out["wsf"] * kc * out["et0"]
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total ETo", f"{out.et0.sum():,.0f} mm",
                              help="Unchanged by Kc — the atmosphere does not know what "
                                   "is growing.")
                    k2.metric("Total ETa", f"{out.et.sum():,.0f} mm",
                              delta=(f"{out.et.sum() - base.et.sum():+,.0f} mm vs "
                                     "K\u1d9c = 1") if abs(kc - 1.0) > 1e-9 else None)
                    k3.metric("Mean WSF", f"{out.wsf.mean():.2f}",
                              help="Set by the soil, so Kc does not move it.")
                    k4.metric("Demand met",
                              f"{100 * out.et.sum() / out.et0.sum():.0f}%")
                    show(w.plot.evapotranspiration(soil_et_win, met=et_win,
                                                   ref=soil_lg.serial, out=out,
                                                   crop_coefficient=kc, logger=lg),
                         "et_play")

                with_period("und_et", _play, default="6M")

if "Wind" in tabs:
    with tabs["Wind"]:
        def _wind(days, start, end) -> None:
            d = window(full, days, start, end)
            if _empty(d):
                return
            a, b = st.columns([1, 3])
            # Inside the same fragment as the period pills, so sectors, calm threshold
            # and period all redraw this one chart rather than the whole app.
            sectors = a.select_slider("Sectors", [8, 16], value=16, key="wr_sectors")
            calm = a.slider("Calm threshold (m/s)", 0.0, 2.0, 0.5, 0.1, key="wr_calm",
                            help="Below this the vane direction is noise; excluded and "
                                 "reported separately.")
            a.caption("Bars point in the direction the wind blows **from**.")
            with b:
                show(w.plot.wind_rose(d, sectors=sectors, calm_threshold=calm,
                                      logger=lg), "wr")
        with_period("wr", _wind, default="3M")

with tabs["Summary"]:
    kinds = list(w.plot.CLIMATOLOGY_KINDS)
    if not HAS_STRESS:
        kinds = [k for k in kinds if k in {"precip_cumulative", "et0_cumulative",
                                           "balance_cumulative", "vpd_cumulative",
                                           "rzsm"}]
    if soil_full is None:
        kinds = [k for k in kinds
                 if k not in {"rzsm"} | (w.plot.SOIL_KINDS if HAS_STRESS else set())
                 or k == "balance_cumulative"]
    if met_full is None:
        kinds = [k for k in kinds if k not in w.plot.MET_KINDS]

    ref_serial = soil_lg.serial if soil_lg is not None else lg.serial
    newest_year = int(full.index.year.max())
    years_avail = sorted(int(y) for y in full.index.year.unique())

    # Half the width each: where this year stands on the left, where the logger stands on
    # the right. The map is an orientation device, not the subject of the tab.
    @st.fragment
    def header() -> None:
        left, right = st.columns([1, 1])
        with left:
            # The standing row needs a complete earlier year to be "normal" against. The
            # charts below do not -- a single year drawn to today is still the thing you
            # came to look at -- so a short record loses the metrics, not the panels.
            if len(years_avail) > 1:
                standings = [(k, w.plot.climatology_standing(
                    full, k, ref=ref_serial, met=met_full, soil=soil_full))
                    for k in kinds]
                standings = [(k, v) for k, v in standings if v]
                if standings:
                    # Two to a row: seven metrics across half a screen would each get
                    # about seventy pixels, which is narrower than "Cumulative P − ET".
                    for i in range(0, len(standings), 2):
                        for col, (k, sd) in zip(st.columns(2), standings[i:i + 2]):
                            dp = 0 if sd["cumulative"] else 3
                            col.metric(
                                sd["label"],
                                f"{sd['current']:,.{dp}f}",
                                delta=f"{sd['delta']:+,.{dp}f} vs normal"
                                      + (f" ({sd['pct']:+.0f}%)"
                                         if sd["pct"] is not None
                                         and abs(sd["pct"]) < 1000 else ""),
                                delta_color=("normal" if k != "vpd_cumulative"
                                             else "inverse"),
                            )
                    ref_sd = standings[0][1]
                    st.caption(
                        f"As of **{ref_sd['as_of']:%d %B %Y}**, against the median of "
                        f"{', '.join(str(y) for y in ref_sd['years'])} on this date."
                        + (f" {', '.join(str(y) for y in ref_sd['excluded'])} excluded "
                           "— incomplete." if ref_sd["excluded"] else "")
                    )
                else:
                    st.caption("Nothing to summarise for this logger yet.")
            else:
                st.caption(f"Only {newest_year} on record here — the charts below show "
                           "this year to date. There is no earlier complete year to "
                           "call normal yet.")
        with right:
            basemap = st.segmented_control(
                "Basemap", list(w.plot.BASEMAPS), key="basemap", default="satellite",
                format_func=w.plot.BASEMAPS.get, label_visibility="collapsed")
            fig_map = w.plot.logger_map(w.loggers(), selected=lg,
                                        basemap=basemap or "satellite",
                                        height=300 if on_phone() else 420)
            picked = st.plotly_chart(
                fig_map, width='stretch', key="map", on_select="rerun",
                selection_mode="points",
                config={"displaylogo": False, "displayModeBar": False},
            )
            st.caption("Tap a marker to switch logger · grey markers are offline")
            points = ((picked.selection or {}).get("points")
                      if picked is not None else None)
            if points:
                # The selection survives the rerun, so this fires once and then matches
                # the logger already showing -- no loop. The rerun is whole-app, not
                # fragment: changing logger changes every tab.
                serial = points[0]["customdata"][0]
                if serial != lg.serial:
                    st.session_state["pending_logger"] = serial
                    st.rerun()

    header()
    st.divider()

    if not kinds:
        st.info("This logger has neither soil moisture nor a weather station.")
    else:
        # One held year, shared by every panel: hold 2024 and 2024 comes forward
        # everywhere, so a reader comparing rainfall against soil moisture is comparing
        # the same year in both. Clicking is the only control -- a row of year buttons
        # said the same thing twice.
        picked_year = st.session_state.get("clim_year")
        if len(years_avail) > 1:
            st.caption(
                (f"Holding **{picked_year}**. Click its line again to let go, or click "
                 "another year." if picked_year is not None else
                 "This year is drawn bold and black; earlier years are faded behind it. "
                 "**Click any year's line** to hold it forward in every panel — hover "
                 "one to see which year it is.")
            )

        for k in kinds:
            fig_c = w.plot.climatology(full, kind=k, logger=lg, ref=ref_serial,
                                       met=met_full, soil=soil_full,
                                       highlight=picked_year)
            if on_phone():
                fig_c = w.plot.compact(fig_c)
            event = st.plotly_chart(
                fig_c, width='stretch', key=f"cl{k}", on_select="rerun",
                selection_mode="points", config=CHART_CONFIG,
            )
            # The invisible marker rows `climatology` lays along each line are what make
            # a click land at all -- `_line` is mode="lines" with no points in it.
            pts = (event.selection or {}).get("points") if event is not None else None
            if pts:
                # Take the first point that actually carries a year: the "latest value"
                # dot and anything else on the panel have no customdata, and with more
                # than one trace under the cursor the click can report several.
                year = next((pt.get("customdata") for pt in pts
                             if pt.get("customdata") is not None), None)
                year = year[0] if isinstance(year, (list, tuple)) else year
                if year is not None:
                    year = int(year)
                    # Clicking the held year lets go of it, so there is no separate
                    # "clear" control to find.
                    if year == picked_year:
                        st.session_state.pop("clim_year", None)
                        st.rerun()
                    elif year != newest_year or picked_year is not None:
                        st.session_state["clim_year"] = year
                        st.rerun()

        # Last, and on its own terms: the panels above are this year against other
        # years, one quantity each. This one is every term of the balance together,
        # against nothing but zero.
        if HAS_STRESS and met_full is not None and soil_full is not None:
            years_ = sorted(full.index.year.unique(), reverse=True)
            year_ = (st.radio("Year", years_, horizontal=True, key="wb_year")
                     if len(years_) > 1 else years_[0])
            show(w.plot.weekly_balance(soil_full, met_full, ref=soil_lg.serial,
                                       year=year_, logger=lg), "wb")
            st.caption("From 1 January, two bars a week: rain, and evaporation with "
                       "the demand (ETo) in pale behind what actually "
                       "evaporated — the exposed pale head *is* the water stress. "
                       "Below, the same weeks as P − ET: blue where the week put "
                       "water into the profile, brown where the store paid for it.")

    # Last on the page, deliberately: someone who wants the explanation will look for
    # it, and someone who does not should not scroll past it to reach the charts.
    st.divider()
    with st.container(border=True):
        st.markdown("**New to WSF, ETo and ETa?** Open the **Understand** tab — the "
                    "function, what the terms mean, and a calculator for how much you "
                    "would need to irrigate.")
        with st.popover("The short version"):
            st.markdown(
                "- **ETo** — *reference* evaporation: what a well-watered grass sward "
                "would use under this weather. Demand, and nothing to do with how wet "
                "your soil is.\n"
                "- **WSF** — the share of that demand this soil can actually meet. "
                "1 when it can meet all of it, 0 when it can meet none.\n"
                "- **ETa** — *actual* evaporation, ETo × WSF × Kc. The gap between ETa "
                "and ETo is water the plants asked for and did not get.")

with tabs["Variables"]:
    units = w.units(lg.serial)
    hk = set(w.housekeeping_columns())
    show_hk = st.checkbox("Include battery / logger diagnostics", value=False)
    # Offered from the whole record, not a window: a column that reported for a year and
    # then died is still worth plotting, and it would simply vanish from this list if the
    # choice were made from the last 30 days.
    cols = [c for c in full.columns if full[c].notna().any() and (show_hk or c not in hk)]
    for i, c in enumerate(st.multiselect("Variables", cols, default=cols[:1])):
        def _var(days, start, end, column=c, key=f"ts{i}") -> None:
            d = window(full, days, start, end)
            if _empty(d):
                return
            show(w.plot.timeseries(d, column, unit=units.get(column, ""), logger=lg), key)
        st.markdown(f"**{c}**")
        with_period(f"ts{i}", _var)

with tabs["Coverage"]:
    show(w.plot.coverage_heatmap(full, logger=lg), "cov")
    st.caption("Fraction of the expected 288 daily samples present.")

with tabs["Loggers"]:
    st.caption("`ATMOS` = weather station · `WP` = water potential (TEROS21) · "
               "`SMST` = soil moisture and temperature")
    st.dataframe(w.overview(), width='stretch', hide_index=True)

st.sidebar.divider()
st.sidebar.caption(f"{len(full):,} rows on record · 5-minute resolution")
