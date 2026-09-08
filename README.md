# WUNDER data viewer

Browse the WUNDER soil-moisture logger network — three sites, fourteen loggers, five-minute
data back to 2023 — as an interactive app or as a Python library.

![sites](https://img.shields.io/badge/sites-3-blue) ![loggers](https://img.shields.io/badge/loggers-14-blue)

## Install

```bash
mamba create -n wunder python=3.12 pandas plotly streamlit pyarrow pyyaml requests
mamba activate wunder
```

or `pip install -r requirements.txt`.

## Run the app

```bash
streamlit run app.py
```

Opens at <http://localhost:8501>. Pick a site, then a logger, then a period.

The first time you open a logger it downloads its full history (a minute or so, ~10 MB of
Parquet) into `data/raw/`. After that it is served from disk and refreshed at most once a day.
To pre-download everything — about ten minutes, 106 MB — run:

```python
import wunder; wunder.update_all()
```

## Use it as a library

The app is a thin layer; everything works without Streamlit.

```python
import wunder as w

w.loggers("glanerbeek")                  # every logger at a site
df  = w.fetch("F1_1_ATMOS_SMST1", days=30)   # by device name or serial

df["rzsm"] = w.rzsm(df)                  # depth-weighted root-zone soil moisture
half_hourly = w.resample(df, "30min")    # sums rain, averages everything else

fig = w.plot.soil_moisture(df, precip=True)
fig = w.plot.wind_rose(df)
fig.show()
```

Useful entry points:

| Call | Does |
|---|---|
| `w.sites()`, `w.loggers(site)`, `w.logger(ref)` | the registry; `ref` is a serial or a device name |
| `w.overview()` | the reference table — who is who, what works, what broke |
| `w.fetch(ref, days=…, start=…, end=…)` | data, cached |
| `w.sensor_status(df)` | per-sensor coverage, first/last reading, still-live flag |
| `w.active_measures(df)` | what is still reporting near the end of the record |
| `w.rzsm`, `w.rzst`, `w.root_zone` | depth-weighted profile averages |
| `w.plot.*` | Plotly figures |

## Which logger is which?

Serial numbers are unmemorable, so device names are used everywhere. `ATMOS` in a name means
a weather station, `WP` means water potential, `SMST` means soil moisture and temperature.

Full table in `sites.yaml`, and in the app's **Loggers** tab.

Things worth knowing before trusting a plot:

- **Sensors fail mid-record.** `F1_2_SMST2`'s 20 cm probe reported for nine months and died in
  Feb 2024; `W1_ATMOS_SMST1`'s 5 cm probe ran at 98.8% until Aug 2026. Use `sensor_status()`.
- **`K2_ATMOS_SMST` stopped on 2026-08-01** but has three years of good record behind it.
- **A column existing does not mean it has data.** `F2_2_SMST2` returns weather columns that
  hold 92 records from its first day in 2023 and nothing since.
- **Two loggers emitted impossible timestamps** (1987, 1989, and 161 records dated 2038 — a
  32-bit `time_t` overflow). These are filtered on ingest.

## Hosting

The API is publicly reachable from outside the UT network (verified), so cloud deployment
works.

**Streamlit Community Cloud** — free, public URL, redeploys on push:

1. Push this repo to GitHub (public repo; free tier).
2. Go to <https://share.streamlit.io>, connect the repo, set the entry point to `app.py`.
3. It installs from `requirements.txt` and gives you a URL.

`data/` is gitignored, so the cloud instance has no Parquet cache. The app detects this and
fetches only the selected window straight from the API — about 3 seconds for 30 days, no
storage needed. Nothing to configure.

Two caveats: free apps sleep after about a week idle and wake on the next visit, and the
"Everything" period is slower without a cache since it pulls the full record each time.

Alternatives: **Hugging Face Spaces** (also free, also runs Streamlit); or ask ITC to run it
next to the existing `majisysdemo` service if it should look official.

## Data source

`http://majisysdemo.itc.utwente.nl/wunder/` — `get7days.py` (any date range, despite the
name), `getall.py` (full history), `get7days_json.py` (per parameter). No key, HTTP only.
CSV with two header lines: labels, then units.

Please keep refreshes to about once a day; it is someone else's server.

## Layout

```
wunder/metadata.py   registry: sites, loggers, sensors, column naming
wunder/fetch.py      API client + incremental Parquet cache
wunder/process.py    RZSM/RZST, resampling, wind binning, sensor lifetimes
wunder/plots.py      Plotly figure builders
app.py               Streamlit UI
sites.yaml           the registry data
trial/               original proof of concept, kept for reference
```

The logger registry, including which sensors are installed and which are still
reporting, lives in `sites.yaml`.
API and network details.
