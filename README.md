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
| **Summary** | Where this year stands today against the same date in previous complete years: cumulative rainfall, cumulative VPD, root-zone soil moisture, each with the deviation from normal and a band spanning earlier years |
| Soil moisture | By depth, with rainfall bars on a reversed right axis |
| Root zone | Depth-weighted profile average, trapezoid or layer-weighted |
| Soil temperature | By depth, with air temperature and a 0 °C line |
| Water potential | Matric potential against VPD — soil supply vs atmospheric demand. VPD comes from the field's ATMOS-41 station, since the TEROS21 loggers carry no weather sensor |
| Weather | Rainfall (with cumulative), temperature + radiation, VPD + air temperature |
| Wind | Wind rose, 16 sectors, ordinal speed bins, calm excluded and reported |
| Variables | Any column, on demand |
| Coverage | Heatmap of when each sensor was reporting |
| Loggers | The reference table — who is who, what works, what broke |

**Compare** — across things:

- **Fields at a site** — Glanerbeek F1 vs F2, or Ketelbroek Voedselbos vs Grasveld. Soil
  quantities are the mean of that field's loggers; rain and VPD come from its own station.
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
  midpoints between sensors. Two definitions are available: `trapezoid` (the profile varies
  linearly between sensors) and `weighted` (each sensor represents its whole layer). They
  differ by up to ~0.05 m³ m⁻³ here. Sparse depths are dropped so the profile definition
  stays constant through time.
- **A depth-weighted mean of matric potential is refused**, not offered. Soil moisture is
  extensive — water volume adds up, so thickness weighting gives real stored water. Matric
  potential is an intensive state variable: it spans −10 to −1500 kPa, so a linear mean is
  dominated by the wettest layer and understates stress, and a plant extracts from the
  least-negative layer rather than experiencing the profile mean.
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
- **Depth colours are fixed per depth value**, so 20 cm is the same colour on every logger.
  The palette is validated for colour-vision deficiency rather than chosen by eye.
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

## Layout

```
app.py               Streamlit UI — thin; no logic of its own
wunder/metadata.py   registry: sites, loggers, sensors, column naming
wunder/fetch.py      API client + incremental Parquet cache
wunder/process.py    root-zone averages, resampling, wind binning, sensor lifetimes
wunder/plots.py      Plotly figure builders
sites.yaml           registry data: installed vs observed, per logger
info.txt             original site notes from the network operator
trial/               original proof of concept, kept for provenance
data/raw/            Parquet cache (gitignored, rebuilt from the API)
```
