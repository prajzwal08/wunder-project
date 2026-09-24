# WUNDER data viewer

Browse the WUNDER soil-moisture logger network — three sites, fourteen loggers, five-minute
data back to 2023 — as an interactive app or as a Python library.

The network is run by the Water Resources department at ITC / University of Twente across
three Dutch agroforestry sites. This repo reads its public API, caches locally, and draws
publication-style figures.

---

## Install

```bash
mamba create -n wunder python=3.12 pandas plotly streamlit pyarrow pyyaml requests
mamba activate wunder
```

or `pip install -r requirements.txt`.

## Run the app

```bash
streamlit run app.py      # or ./run.sh
```

Opens at <http://localhost:8501>.

The first time you open a logger it downloads its full history (a minute or so, ~10 MB of
Parquet) into `data/raw/`. After that it is served from disk and refreshed at most once a
day. To pre-download everything — about ten minutes, 106 MB:

```python
import wunder; wunder.update_all()
```

### What's in it

**Explore** — one logger at a time:

| Tab | Shows |
|---|---|
| **Summary** | Half and half at the top: where this year stands on the left, and on the right a **map** of all 14 loggers on Esri satellite or OpenStreetMap, offline ones greyed rather than hidden — tap a marker to switch to that logger and every tab follows, and neither basemap needs an API key. Below, where this year stands today against the same date in previous complete years — cumulative rainfall, ETo, root-zone soil moisture, water stress factor, ETa, P − ET and VPD — each with the deviation from normal. The P − ET panel carries **both** balances, P − ETo dashed against P − ETa solid and each named at its own line end, because the gap between them is the reading. **Every earlier year is its own faded line**, oldest lightest, with this year always black and bold — no band and no median, because neither is a year you can point at. Click any year's line and it comes forward in orange in every panel at once; hover one to see which year it is, and click it again to let go. A logger with only one year of record still gets its charts, drawn to today. One quantity per panel, one axis per panel. The soil-moisture panel carries saturation, field capacity, the half-stress midpoint and residual water content as dotted lines, weighted over the same depths as the curve itself. Last comes the **weekly water balance**, week by week from 1 January: rain beside evaporation, with ETo pale behind ETa so the exposed head *is* the water stress, and P − ET as a signed bar beneath — blue where the week put water into the profile, brown where the store paid for it |
| Soil moisture | By depth, with rainfall bars on a reversed right axis |
| Root zone | Thickness-weighted profile average, with the four soil limits behind it |
| Soil temperature | By depth, with air temperature and a 0 °C line |
| Water potential | Matric potential against VPD — soil supply vs atmospheric demand. VPD comes from the field's ATMOS-41 station, since the TEROS21 loggers carry no weather sensor |
| Weather | Rainfall (with cumulative), temperature + radiation, VPD + air temperature |
| Evapotranspiration | Plain-language ETo (reference — what a well-watered grass sward would evaporate) against ETa (actual — what this soil could supply), a Kᶜ slider to play with, and four stacked rows on one x axis with no second y axis anywhere: rain beside evaporation in mm/day with ETo pale behind ETa so the exposed head *is* the water stress; WSF; cumulative ETo against cumulative ETa; and cumulative P − ETo against P − ETa. Measured record at Kᶜ = 1 with nothing to adjust; the method, a live Kᶜ and an irrigation calculator are in the **Understand** tab |
| **Understand** | Opens with **one picture of the whole chain** — the atmosphere asks, the soil opens a gate, what got through and what did not, the profile it came from and the millimetres that would refill it — with three sliders driving all of it. Under that, folded away: what ETo and ETa are, the stress function with its own sliders, the maths layer by layer, and the record with a live Kᶜ. It is **the only tab with anything to drag** — the data tabs are fixed, so a number read off one of them is always the same number. What ETo and ETa are in plain words; the evapotranspiration figure again with a live Kᶜ, so you can see what a different canopy assumption would do; the WSF curve with sliders on θ, θ_s, θ_fc, θ_r and the steepness, opening on the real soil under this logger; and **an irrigation calculator** — put a soil wetness and a day's demand together and it gives WSF, ETa, the demand the canopy is not getting, the millimetres that would refill the root zone, and how long until stress begins |
| Wind | Wind rose, 16 sectors, ordinal speed bins, calm excluded and reported |
| Variables | Any column, on demand |
| Coverage | Heatmap of when each sensor was reporting |
| Loggers | The reference table — who is who, what works, what broke |

**Compare** — across things:

- **Fields at a site** — Glanerbeek F1 vs F2, or Ketelbroek Voedselbos vs Grasveld. Soil
  quantities are the mean of that field's loggers; rain and VPD come from its own station.
  Also the derived pair, field against field: **water stress factor**, ETa (daily or
  cumulative) and cumulative ETo, each built from that field's own soil probe and
  its own weather station — named in the caption, since WSF is defined against the soil
  under one particular probe rather than a field mean.
- **This year vs previous** — the climatology view for any logger and quantity.
- **Variables** — several variables from one logger as stacked panels, or overlaid scaled
  0–1 to compare shape.

Panels only appear when the logger actually reports the data behind them.

## Use it as a library

The app is a thin layer; everything works without Streamlit installed.

```python
import wunder as w

w.loggers("glanerbeek")                      # every logger at a site
df = w.fetch("F1_1_ATMOS_SMST1", days=30)    # by device name or serial

df["rzsm"] = w.rzsm(df)                      # depth-weighted root-zone soil moisture
half_hourly = w.resample(df, "30min")        # sums rain, averages everything else

fig = w.plot.soil_moisture(df, precip=True)
fig.show()
```

| Call | Does |
|---|---|
| `w.sites()`, `w.loggers(site)`, `w.logger(ref)` | the registry; `ref` is a serial or a device name |
| `w.overview()` | reference table — who is who, what works, what broke |
| `w.fetch(ref, days=…, start=…, end=…)` | data, cached; `cache=False` fetches a window directly |
| `w.update(ref)`, `w.update_all()` | refresh the cache from the API |
| `w.sensor_status(df)` | per-sensor coverage, first/last reading, still-live flag |
| `w.active_measures(df)` | what is still reporting near the end of the record |
| `w.rzsm`, `w.rzst`, `w.root_zone` | depth-weighted profile averages |
| `w.reference_et(df)` | daily Makkink reference ET, ETo [mm d⁻¹] |
| `w.water_balance(df)` | daily rainfall, ETo and `P − ETo` |
| `w.stress.profile_stress(df, ref=…)` | WSF per layer and weighted into one series, with its provenance |
| `w.stress.root_zone_limits(df, ref=…)` | just θ_sat, θ_fc, the midpoint and θ_r, without computing the series |
| `w.stress.actual_et(df, ref=…)` | daily ETo, WSF and the ETa they imply |
| `w.stress.layer_limits(site)` | θ_sat, θ_fc, θ_r and θ_wp per SoilGrids layer |
| `w.met_source(ref)`, `w.soil_source(ref)` | which logger supplies weather / soil water for this one |
| `w.cumulative_year(s)` | running total that restarts each 1 January |
| `w.field_series(frames, measure)` | one series per field, averaged over its loggers |
| `w.wind_rose_table(df)` | direction × speed bins, with the calm fraction |
| `w.plot.*` | Plotly figures |

Figures are plain `plotly.graph_objects.Figure` — save with
`fig.write_image("fig.png", scale=3)` or `fig.write_html(...)`.

---

## The network

Device names are used everywhere, because nobody remembers serial numbers. `ATMOS` means a
weather station, `WP` means water potential (TEROS21), `SMST` means soil moisture and
temperature; `F1`/`F2`/`K1`/`K2`/`W1` is the field.

**Voedselbos Glanerbeek — Field 1** (52.223 N, 6.979 E, 36–42 m)

| Device | Serial | Sensors |
|---|---|---|
| `F1_1_ATMOS_SMST1` | z6-21176 | ATMOS-41 @200 cm + 5TM at 5, 10, 20, 40, 80 cm |
| `F1_2_SMST2` | z6-21178 | 5TM at 2.5, 5, 10, 20, 40, 80 cm |
| `F1_3_SMST3` | z6-25928 | TEROS11 at 2.5, 5, 10, 20, 40, 80 cm |
| `F1_4_WPST` | z6-25927 | TEROS21 water potential at 2.5, 5, 10, 20, 40, 80 cm |
| `F1_5_WP_SMST` | z6-28931 | TEROS21 + TEROS12 at 5, 20, 40 cm |
| `F1_6_WP_SMST` | z6-24636 | TEROS21 + TEROS12 at 5, 20, 40 cm |

**Voedselbos Glanerbeek — Field 2** (52.224 N, 6.980 E, 37–41 m)

| Device | Serial | Sensors |
|---|---|---|
| `F2_1_ATMOS_SMST1` | z6-21177 | ATMOS-41 @200 cm + 5TM at 5, 10, 20, 40, 80 cm |
| `F2_2_SMST2` | z6-21179 | 5TM at 2.5, 5, 10, 20, 40, 80 cm |
| `F2_3_WP_SMST` | z6-30173 | TEROS21 + TEROS12 at 5, 20, 40 cm |

**Voedselbos Ketelbroek** (51.769 N, 5.964–5.966 E, 21–26 m)

| Device | Serial | Sensors |
|---|---|---|
| `K1_ATMOS_SMST` (Voedselbos) | z6-21180 | ATMOS-41 @200 cm + 5TM at 5, 10, 20, 40, 80 cm |
| `K2_ATMOS_SMST` (Grasveld) | z6-08819 | ATMOS-41 @200 cm + 5TM at 5, 10, 20, 40, 80 cm |

**Wenumseveld Herenboeren — Field 1** (52.250–52.251 N, 5.970 E, 11–20 m)

| Device | Serial | Sensors |
|---|---|---|
| `W1_ATMOS_SMST1` | z6-08820 | ATMOS-41 @200 cm + 5TM at 2.5, 5, 10, 40, 80 cm |
| `W1_SMST2` | z6-08823 | 5TM at 5, 10, 20, 40 cm |
| `W1_SMST3` | z6-08822 | 5TM at 5, 10, 20, 40 cm |

Five devices carry an ATMOS-41 — exactly those with `ATMOS` in the name. Loggers without one
borrow met data from their field's station via `w.met_source()`.

Machine-readable form: `sites.yaml`, which separates what is *installed* (from `info.txt`)
from what is *observed* to report.

## Known data problems

Read this before trusting a number.

- **A column existing does not mean it has data.** The API returns a header that is not
  strictly per-device. `F2_2_SMST2` returns weather columns holding 92 records from its
  first day in 2023 and nothing in the three years since. Capability is derived from
  non-null coverage, never from `df.columns`.
- **Sensors fail mid-record.** `F1_2_SMST2`'s 20 cm probe reported for nine months and died
  in Feb 2024; `W1_ATMOS_SMST1`'s 5 cm probe ran at 98.8% until Aug 2026. `sensor_status()`
  gives each sensor's first reading, last reading and coverage.
- **`K2_ATMOS_SMST` stopped on 2026-08-01**, with three years of good record behind it. Its
  2026 rain gauge also has only 44% coverage — January, February and July missing — so its
  cumulative totals are undercounts. The app warns when a cumulative source is this gappy.
- **`K1_ATMOS_SMST`'s rain gauge is suspect for 2024–25.** It recorded 193 mm (2024) and
  215 mm (2025) at ~100% coverage, against ~800 mm normal for the Netherlands and 968 mm /
  617 mm at Glanerbeek in the same years. 2026 looks normal at 380 mm by September. Its
  year-on-year comparison therefore reads "+118% vs normal", which is the earlier years
  being wrong rather than this year being wet. **Not corrected** — flagged for whoever
  maintains the station.
- **The TEROS21 + TEROS12 loggers have short, intermittent records.** `z6-28931` at 66.7%
  coverage with every soil sensor starting 2025-06-19; `z6-24636` at 45–55%; `z6-30173`
  starting 2025-06-19. None has two complete calendar years of soil moisture.
- **Impossible timestamps.** One record dated 1987 (`z6-21178`), one 1989 (`z6-25927`), and
  161 dated 2038 on `z6-08819` — a 32-bit `time_t` overflow. Filtered on ingest; without it
  a daily resample would generate ~13,000 empty days.
- **Depth labels differ from `info.txt`.** A port documented as −2 cm appears as `2.5cm` in
  the API.
- **Two temperature series** on the TEROS21 + TEROS12 loggers, differing only by
  capitalisation: `Soil temperature <d>` and `Soil Temperature second measurement <d>`.
  Which physical sensor maps to which is unconfirmed.

## Method notes

- **Root-zone averages** weight each depth by its layer thickness, boundaries at the
  midpoints between sensors, so each sensor stands for the slab around it. A trapezoidal
  variant — the profile varying linearly between sensors — used to be selectable and was
  dropped: the two agree to well within the spread between loggers in the same field, and
  offering the choice implied the difference carried meaning that it does not. Sparse
  depths are dropped so the profile definition stays constant through time.
- **A depth-weighted mean of matric potential is refused**, not offered. Soil moisture is
  extensive — water volume adds up, so thickness weighting gives real stored water. Matric
  potential is an intensive state variable: it spans −10 to −1500 kPa, so a linear mean is
  dominated by the wettest layer and understates stress, and a plant extracts from the
  least-negative layer rather than experiencing the profile mean.
- **Reference ET, ETo, is Makkink**, `ETo = 0.65 · s/(s+γ) · Rs/λ`, from global radiation and air
  temperature. It is the Dutch standard — the same quantity KNMI publishes as `EV24` for
  every station in the country, so these numbers are comparable with the national record
  (Glanerbeek gives 541 and 551 mm for 2024 and 2025). Penman–Monteith would also need wind
  at a defined height and would be sensitive to VPD error; in this climate the radiation
  term dominates and the two agree closely for grass. **The coefficient 0.65 is fitted to
  daily totals**, so ETo is computed on whole days only: a day missing more than 10% of its
  readings is dropped rather than averaged from whatever hours happen to be present, since
  radiation over the daylight hours alone is about twice the 24-hour mean. Days are local
  days, as KNMI's are. ETo is the demand a well-watered *grass* sward would meet — not what
  a food forest actually transpires — and it assumes the pyranometer sees open sky, which
  K1 Voedselbos's canopy-covered mast does not.
- **The water stress factor, WSF, is a sigmoid in soil water content:**

      WSF(θ) = 1 / (1 + exp[ −k · θ_sat · (θ − (θ_fc + θ_r)/2) ])

  Half stress sits midway between field capacity and residual water content, and θ_sat
  sets how sharply the curve turns there — a coarse soil, holding more water at
  saturation, switches over a narrower band than a fine one. `k = 100` is the only free
  constant and it is dimensionless. On these soils WSF is 0.01–0.11 at residual, 0.5 at
  the midpoint and 0.90–0.99 at field capacity: it never reaches exactly 1 or 0, so a
  profile sitting at field capacity reads 0.98, not 1.00. The **Water stress** tab draws
  the curve with sliders on all four parameters.

  This replaced an FAO-56 `Kₛ`, a straight line from a threshold at `θ_fc − p·TAW` down
  to zero at the wilting point. Two reasons: FAO-56 needs a depletion fraction `p` that
  nothing at these sites measures, and it has a hard kink at the threshold that no soil
  exhibits. Every parameter of the sigmoid comes from the site's own retention curve.
- **WSF is computed per layer and then weighted, never the other way round.** Each sensor
  depth gets its own θ_fc, θ_r and θ_sat, its own WSF, and only the resulting factors are
  thickness-weighted into one profile number. This is not a detail: at `F1_2_SMST2` the
  summer mean is **0.88** computed per layer against **0.995** computed the other way,
  because 2.5–10 cm sit at 0.49–0.62 while 80 cm sits at 1.00. Averaging the moisture
  first lets a wet subsoil hide a bone-dry topsoil before the non-linearity ever sees it.

  The weighting is a modelling choice, not an identity: WSF is an intensive ratio, so
  unlike stored water it does not add up over a profile. Read it as *the share of
  root-zone demand the profile can meet, each sensor standing for the slab around it*.
  The physically right weight is root density, which this network cannot supply.
- **The soil parameters come from the soil STEMMUS_SCOPE itself runs on** — SoilGrids
  texture through Schaap/Rosetta pedotransfer, assembled by PyStemmusScope. θ_sat and θ_r
  are read straight off the van Genuchten curve and θ_fc is that curve evaluated at
  −33 kPa, over the top 100 cm, interpolated from the SoilGrids layer midpoints onto the
  sensor depths. Using the model's own soil is deliberate: `WSF · ETo` and the model's
  transpiration then rest on the same soil, so a disagreement between them means
  something. The −33 kPa convention is the model's own, verified by reproducing its
  `fieldMC` to 0.001 at every layer. θ_wp is still computed at −1500 kPa and quoted where
  it is useful, but it no longer drives the stress factor.
- **A logger uses the nearest instrument that actually measures each thing.** Weather
  comes from the field's ATMOS-41 (`met_source`) and soil water from the nearest working
  probe (`soil_source`), so every logger can show ET and stress — `F1_4_WPST` measures no
  soil moisture at all and borrows F1_3's. Substitutions are named in the app, never
  silent, and a logger with its own working sensor never borrows.
- **Cumulative series restart on 1 January**, which is what makes "we are 84 mm behind by
  this date" meaningful. Only complete past years enter a comparison — a year whose record
  starts in May would accumulate from zero in May and read as a freak drought.
- **Resampling aggregates each variable by its kind**, and there are three:
  *accumulated* (`Precipitation observed`) is **summed** — averaging would divide an hourly
  total by twelve; *circular* (`Wind direction observation`) is resolved as a **vector
  mean** weighted by wind speed — the arithmetic mean of 350° and 10° is 180°, due south
  for a northerly, and 17% of hours in this record differ from the correct value by more
  than 20°, with errors up to 180°; everything else is a *state* or a flux density and is
  **averaged**. Radiation belongs in the last group: W m⁻² is an instantaneous rate, so its
  mean is the mean irradiance.
- **Rainfall is drawn as bars**, being a total over an interval rather than a continuous
  value; the bar width adapts to the window.
- **Wind roses exclude calm records** (below 0.5 m/s by default) and report the fraction
  separately — about 30% of this network's record is that slow, where vane direction is
  noise. Directions are meteorological, i.e. the direction wind blows *from*, confirmed
  against the prevailing SSW/SW at Glanerbeek.
- **Depth colours are fixed per depth value**, so 20 cm is the same colour on every logger
  and in every panel — moisture, temperature, matric potential and EC alike. The ramp runs
  shallow to deep, dark red at the surface to deep blue at 80 cm, so the ordering of the
  profile is legible without reading the legend. It goes through purple rather than
  straight from red to blue: a direct interpolation passes through near-white in the
  middle two steps, and those lines vanish on this white ground.
- **Every chart carries its own period pills** — 1W · 1M · 3M · 6M · 1Y · All — directly
  above it, so soil moisture can be on a year while the wind rose is on a week. Each sits
  in its own fragment, so changing one redraws that chart alone rather than the page. The
  window is applied **server-side** rather than by Plotly's in-chart range buttons, which
  would look the same and read worse: every figure decimates to about 4000 points for the
  span it is given, so zooming a year down to a week in the browser would leave you
  reading 6-hourly means of 5-minute data. The Summary has no picker, a climatology being
  the whole record by definition.
- **One figure serves a phone and a laptop.** Every axis sets `automargin`, so a panel
  takes exactly the width its labels need rather than a fixed gutter — which used to
  spend half a 375 px screen on empty margin. `wunder.plot.compact(fig)` does what that
  cannot judge from inside the figure: it wraps the title, moves the legend below the
  plot, shrinks the type a step, and leaves a multi-row figure's height alone. The app
  applies it from a User-Agent guess, with a sidebar toggle to override — Streamlit
  cannot report the viewport width, so portrait and landscape are not distinguishable
  without asking.
- Series are decimated before plotting (~4000 points) so a three-year view stays responsive;
  precipitation is summed and everything else averaged.

## Data source

`http://majisysdemo.itc.utwente.nl/wunder/` — no key, HTTP only (there is no HTTPS).

| Endpoint | Returns |
|---|---|
| `get7days.py?location=<serial>&mindate=<t>&maxdate=<t>` | CSV for any window, despite the name |
| `getall.py?location=<serial>` | CSV, entire history |
| `get7days_json.py?location=<serial>&parameter=<name>` | JSON, one parameter |

Query timestamps are `%Y-%m-%dT%H_%M_%S` (underscores). The CSV has **two header lines** —
labels, then units — so `pd.read_csv(url, skiprows=[1])`, reading row 2 separately for units.
Native resolution is 5 minutes.

The server refreshes every two hours during daylight, so `fetch()` tops up at most once a
day. **Please keep it that way** — it is someone else's server, and no rate limit is
documented.

## The published dataset

`data/published/` holds the same record resampled to 30 minutes — 25 MB for all fourteen
loggers, committed to the repo. A deployment with no local cache reads it off disk and asks
the API only for records since the last published timestamp: **1.7 s instead of 60 s**, and
one small request instead of a full-history pull.

Cumulative and daily statistics are identical to the 5-minute record at this step; only
sub-half-hour detail is lost, and the figures decimate to ~4000 points regardless. Logger
diagnostics (battery, logger temperature, reference pressure) are dropped and values stored
as float32.

Refresh it after pulling new data:

```bash
python -m wunder.update_all      # or: python -c "import wunder; wunder.update_all()"
python -m wunder.publish
git add data/published && git commit -m "Refresh published dataset" && git push
```

The 5-minute cache in `data/raw/` stays local and is gitignored.

## Layout

```
app.py               Streamlit UI — thin; no logic of its own
wunder/metadata.py   registry: sites, loggers, sensors, column naming
wunder/fetch.py      API client + incremental Parquet cache
wunder/process.py    root-zone averages, resampling, wind binning, sensor lifetimes
wunder/et.py         Makkink reference ET (ETo) and the P − ETo water balance
wunder/stress.py     sigmoid water stress factor, ETa, retention curves
forcing/extract_soil.py   caches the model's soil hydraulics as JSON (needs the geo env)
model_input/soil/    SoilGrids van Genuchten parameters per site, committed
wunder/plots.py      Plotly figure builders
sites.yaml           registry data: installed vs observed, per logger
info.txt             original site notes from the network operator
trial/               original proof of concept, kept for provenance
wunder/publish.py    build/read the committed 30-minute dataset
data/published/      30-minute record, committed — what a deployment reads
data/raw/            5-minute cache, gitignored, rebuilt from the API
```
