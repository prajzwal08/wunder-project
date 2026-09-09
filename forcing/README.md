# Building STEMMUS_SCOPE input from the WUNDER network

This directory turns a **place** and a **date range** into the two NetCDF files
STEMMUS_SCOPE needs to run. Running the model itself is `model/`.

Both engines read the same two files — the MATLAB model through PyStemmusScope,
and the Python port `stemmus-scope-py`. The MATLAB path needs a strict superset
of what the port needs, so building for it satisfies both.

---

## 1. Set up, once

```bash
conda activate geo
earthengine authenticate          # opens a browser; only needed the first time
```

You need `earthengine-api`, `xarray`, `netCDF4`, `pandas`, `pyarrow`, `scipy`,
`pyyaml`, `matplotlib`, and a Google Earth Engine account (free for research).

Nothing else needs downloading first. Everything is fetched on demand.

---

## 2. Build the input

```bash
python forcing/make_input.py --site NL-Gl1 --plot
```

That is the whole thing. It writes:

```
model_input/forcing/FLX_NL-Gl1_FLUXNET2015_FULLSET_2023-2026.nc     4.8 MB
model_input/ic/NL-Gl1_2023-05-19_InitialCondition.nc                 12 kB
plots/forcing_NL-Gl1.png                                            (--plot)
```

and prints what it did — where every variable came from, the bias corrections it
fitted, and the checks it ran afterwards.

The first run for a site takes a couple of minutes: it fetches ERA5-Land, MODIS
LAI, Mauna Loa CO₂ and the static fields from Earth Engine. Those are cached in
`~/data/wunder/`, so a rebuild takes seconds. The cache is shared across every
site and every rebuild, which is why it lives outside the repo. Both directories
are gitignored — the outputs regenerate from sources the scripts can re-fetch.

### All five stations

```bash
python forcing/make_input.py --all --plot
```

| Code | Station | Weather station |
|---|---|---|
| `NL-Gl1` | Voedselbos Glanerbeek — Field 1 | `z6-21176` |
| `NL-Gl2` | Voedselbos Glanerbeek — Field 2 | `z6-21177` |
| `NL-Ke1` | Voedselbos Ketelbroek — food forest | `z6-21180` |
| `NL-Ke2` | Voedselbos Ketelbroek — grass field | `z6-08819` |
| `NL-We1` | Wenumseveld Herenboeren — Field 1 | `z6-08820` |

### A particular period

```bash
python forcing/make_input.py --site NL-Gl1 --start 2024-01-01 --end 2024-12-31
```

Leave both out and you get the station's whole record. `--start` then defaults to
its first day with both met and soil data, and `--end` to as far as ERA5-Land
reaches — currently about five days behind today.

### A site that has no logger

```bash
python forcing/make_input.py --code NL-Ams --lat 52.37 --lon 4.90 \
    --start 2024-01-01 --end 2024-12-31 --reference-height 10
```

Any latitude and longitude. No configuration, no entry in `sites.yaml`, no
instruments. The file is built entirely from ERA5-Land and comes out with the
same variables in the same shape, so a model run cannot tell the difference.

This is not a fallback bolted on afterwards — it is how the pipeline is built.
See *How it works*.

### Every option

| Flag | What it does |
|---|---|
| `--site CODE` | a station from `sites.yaml`; repeat for several |
| `--all` | every station in `sites.yaml` |
| `--code / --lat / --lon` | a location not in `sites.yaml` |
| `--start`, `--end` | `YYYY-MM-DD`; both optional |
| `--logger REF` | use a different weather station — a serial (`z6-21176`) or a device name (`F1_1_ATMOS_SMST1`). `--logger ""` builds from ERA5-Land alone |
| `--reference-height M` | measurement height in metres; also sets the canopy cap |
| `--plot` | write a one-page figure of every variable to `plots/` |
| `--out DIR` | where the `.nc` files go (default `model_input/`) |
| `--cache DIR` | where downloads are cached (default `~/data/wunder`) |
| `--refresh` | re-download instead of reusing the cache |
| `--no-verify` | skip the checks that otherwise run afterwards |

### Which station supplies the data

With `--site`, the station is the one `sites.yaml` names for that code, and the
code refers to that station's own field — `NL-Gl1` is Glanerbeek Field 1 and uses
`z6-21176`, and nothing from Field 2 ever enters it. `--logger` overrides that
when you want a different mast; it refuses a soil-only logger rather than quietly
producing a reanalysis-only file. With `--code/--lat/--lon` there is no station at
all and everything comes from ERA5-Land.

---

## 3. Check what you built

```bash
python forcing/verify.py                    # all of them
python forcing/verify.py --site NL-Gl1
```

Three kinds of check, each catching a different class of mistake:

- **Structural** — every variable both engines index is present, no NaN, uniform
  30-minute steps, `qc` flags in range. Both loaders index by name with no guard
  and neither checks for NaN, so these would otherwise surface as a bare
  `KeyError` deep inside a run.
- **Physical** — ranges and annual totals a Dutch half-hour has to satisfy, plus
  the `units` attribute of every variable. A unit error produces a file that is
  perfectly well formed and wrong by a factor of a thousand.
- **Alignment** — weights time-of-day by `SWdown` to find solar noon and checks
  it sits where the sun does, **in both winter and summer**. This is the one test
  a daylight-saving error cannot survive.

A station with a known, documented problem reports it as `KNOWN` rather than
failing — currently only `NL-Ke1`, whose mast is under a canopy (see below).

Look at the figure too. A forcing file is 57,000 rows of float32 that a model
will read and nobody will ever open, and most of the ways it can be wrong are
obvious in a picture and invisible in a summary statistic.

**Reading the figure.** Every panel is the raw 30-minute series — exactly the
rows the model reads, with no aggregation, so there is no ambiguity about what a
line means. A variable with a diurnal cycle draws as a filled envelope whose
thickness is the day–night range. **Red** means not measured at this station, and
the corner label names the source. Precipitation is the one panel rescaled for
display, to millimetres per half-hour, because a rate of 2.2 × 10⁻⁴ is unreadable
on an axis; the axis names both units.

---

## 4. Run the model

That is a separate job, in its own directory:

```bash
python model/run_model.py  --site NL-Gl1 --days 7     # the Python port
python model/run_matlab.py --site NL-Gl1 --days 7     # the MATLAB model
```

See **[`model/README.md`](../model/README.md)**.

---

## How it works

Every file is built in two passes over one 30-minute UTC grid:

```
1. ERA5-Land     fill every variable, everywhere        qc = 2
2. this station  overwrite where it has data            qc = 0
```

Three consequences follow from that order, and they are the reason for it.

**The baseline comes first, rather than patching gaps afterwards.** Filling gaps
*from* reanalysis leaves a hole wherever no source had data, and neither engine
has a NaN guard. Filling everything first means a hole cannot exist — and it
means a place with no instruments still produces a complete, runnable file, which
is why `--lat/--lon` works at all.

**One file per weather station, never per site.** Sensors from different masts
are never combined, even at the same site: Glanerbeek F1 and F2 are separate
files, and so are Ketelbroek K1 and K2. Borrowing one station's data to cover
another's gap reads well until the two turn out to be different microclimates —
K1 and K2 are 175 m apart and differ by a factor of four in wind speed. Anything
a station does not supply comes from ERA5-Land, which is at least uniform and
documented.

**Excluding a bad sensor is configuration, not code.** Both Ketelbroek rain
gauges read a fraction of the true total, so those stations list
`Precipitation observed` in `exclude_columns` and the baseline simply shows
through.

### Bias correction

Because the baseline underlies the whole record rather than a few gaps, it is
fitted against the station over their entire overlap *before* being overlaid, so
the stretches it supplies sit continuously with the measured ones instead of
stepping at the joins. Three kinds of fit:

| Fit | Variables | Why |
|---|---|---|
| least squares with intercept | `Tair`, `Psurf`, vapour pressure | the offset is physical — ERA5-Land's 9 km cell sits at a different mean elevation, so `Psurf` differs by a near-constant |
| least squares through the origin | `SWdown`, `Wind` | proportional error that must stay zero-anchored. An intercept would put radiation at midnight |
| volume match, `Σobs / Σera5` | `Precip` | reanalysis gets the amount roughly right and the timing poorly (r ≈ 0.4). Least squares fits a positive intercept and adds ~0.007 mm to *every* dry half-hour — 56 mm/yr of rain that never fell, drizzling in 100% of timesteps instead of the observed 7% |

`LWdown` gets no correction. Nothing in the network measures downwelling
longwave, so there is nothing to fit against.

---

## Where each variable comes from

| Variable | Unit | Source |
|---|---|---|
| `Tair` | K | station air temperature, `+273.15` |
| `SWdown` | W m⁻² | station pyranometer — confirmed to be incoming shortwave: exactly 0 at every night half-hour, annual mean 105–111 W m⁻² matching the Dutch norm |
| `LWdown` | W m⁻² | **ERA5-Land only** — measured nowhere in the network |
| `Psurf` | Pa | station air pressure, `×1000` |
| `Precip` | kg m⁻² s⁻¹ | station rain gauge summed over the half-hour, `÷1800` |
| `Wind` | m s⁻¹ | station anemometer |
| `VPD`, `RH`, `Qair` | hPa, %, kg kg⁻¹ | derived from the merged `Tair`, vapour pressure and `Psurf` |
| `CO2air` | ppm | NOAA GML Mauna Loa daily means |
| `LAI` | m² m⁻² | MODIS MCD15A3H, 3×3 block of 500 m pixels |

Statics: `elevation` from Copernicus GLO-30, `canopy_height` from ETH Global
Canopy Height 2020, `IGBP_veg_short/long` from MODIS MCD12Q1, and
`latitude`/`longitude`/`reference_height` from `sites.yaml`.

`VPD`, `RH` and `Qair` are derived once at the end rather than merged
separately. All three are functions of the same three primitives, and any
half-hour may take `Tair` from the station and vapour pressure from ERA5-Land, so
deriving them together keeps the trio mutually consistent. It matters because the
MATLAB path recomputes vapour pressure back out of our `RH` — an inconsistent
trio would put two different values into one run.

### Units

Neither engine checks a `units` attribute. It reads the numbers and converts as
if they were in the expected unit, so a wrong one is not an error but a run that
finishes and is quietly wrong. `verify.py` checks every attribute against this
table, traced to the code that consumes each variable:

| Variable | Unit | What the consumer does with it |
|---|---|---|
| `Tair` | K | `Tair - 273.15` |
| `SWdown`, `LWdown` | W m⁻² | passed through unconverted, into `Rin_.dat` / `Rli_.dat` |
| `Psurf` | Pa | `Psurf / 100` to hPa |
| `Precip` | kg m⁻² s⁻¹ | `Precip / 10`, "mm/s to cm/s" — numerically identical to mm s⁻¹ |
| `RH` | **% on 0–100**, not a fraction | fed to `calculate_ea`, whose docstring specifies 0–100; the port divides by 100 at load |
| `VPD` | hPa | |
| `Qair` | kg kg⁻¹ | |
| `Wind` | m s⁻¹ | clamped to ≥ 0.05. The reference file's `"mm/s"` attribute is wrong (`OPEN_ISSUES.md` #7) |
| `CO2air` | ppm | `CO2air * 1e-6` to a molar fraction |
| `LAI` | m² m⁻² | clamped to ≥ 0.01 |

---

## Reading the provenance

Every value carries a flag saying where it came from:

| `qc` | meaning |
|---|---|
| 0 | measured by this file's own weather station |
| 2 | ERA5-Land baseline, bias-corrected against that station |
| 3 | parameterised from measured `Tair` / vapour pressure / `SWdown` |

The build prints the fraction at each level per variable, and the same summary
goes into the file's global attributes alongside the fitted bias coefficients.
**Read them before trusting a run.**

| Station | Measured fraction | Note |
|---|---|---|
| `NL-Gl1`, `NL-Gl2` | 99.9% | clean |
| `NL-We1` | 99.9% | clean |
| `NL-Ke1` | 83.2% | precipitation is entirely ERA5-Land |
| `NL-Ke2` | 62.8% | patchy record; met stops 2026-06-21 |

---

## Things to know before you trust a run

**Timestamps are converted, not offset.** The loggers stamp Europe/Amsterdam
wall-clock time *including* daylight saving — solar noon in the raw record jumps
by an hour on exactly the EU transition dates. Reading it as a fixed UTC+02
offset would shift every winter half-hour. Output is UTC.

**`NL-Ke1`'s mast is under the canopy.** It reads 0.46 m s⁻¹ mean wind and
79 W m⁻² mean radiation, against 1.6–2.0 and 115–128 at every other station —
including `NL-Ke2`, 175 m away in the open at the same site. ETH canopy height
gives 14 m there against 4–5 m elsewhere, which confirms it independently. The
values are real and kept as measured, but they are **sub-canopy** forcing and
STEMMUS_SCOPE will apply its own canopy attenuation on top of them.

**Both Ketelbroek rain gauges under-read**, at roughly a quarter of the true
total, so precipitation at `NL-Ke1` and `NL-Ke2` is ERA5-Land throughout.

**Canopy height is capped** at 0.4 × the measurement height — 0.8 m for the 2 m
WUNDER masts. A canopy near or above the mast would put the model's reference
level inside the canopy, and the surface-layer relations it derives
(`d ≈ 0.67·hc`, `z0 ≈ 0.1·hc`) would stop meaning anything.

**LAI is a 500 m satellite product.** That is the right scale for a
one-dimensional ecosystem column and the same source PLUMBER2 uses, but it
describes the surrounding landscape rather than an individual plot. Quality
control keeps only pixels MODIS flags good, with working detectors, where the
main radiative-transfer algorithm succeeded — dropping about 56% of observations
and most of every Dutch winter. Those gaps are filled from the **climatology**
for that time of year rather than interpolated through: a cubic through a
two-month hole, which is what `laiprocessing` does, gave a maximum LAI of 15.2
and erased the seasonal cycle entirely.

**A fresh clone has no logger cache.** `data/raw/` is gitignored. The build falls
back automatically — the 5-minute cache if present, then the 30-minute snapshot
in `data/published/` which every clone has, then the MajiSys API. If none
answers it says so and builds from ERA5-Land alone rather than failing. The
report line `station data from:` says which happened.

---

## The scripts

`make_input.py` is the only one most people need; it calls the others.

| Script | Does |
|---|---|
| `make_input.py` | the whole build: fetch what is missing, assemble, verify |
| `run_model.py` | run the Python port on a built file |
| `verify.py` | check produced files; safe to run any time |
| `plot_forcing.py` | redraw the figures from existing files |
| `download_era5land.py` | ERA5-Land at a point, from Earth Engine |
| `download_lai.py` | MODIS LAI, 3×3 pixels, quality-filtered and smoothed |
| `download_co2.py` | NOAA Mauna Loa CO₂ onto the 30-minute grid |
| `download_statics.py` | elevation, canopy height, IGBP class; prints, does not write |
| `build_forcing.py` | assemble and write, assuming everything is cached |
