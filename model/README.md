# Running STEMMUS_SCOPE

Two engines, reading the **same two files** that `forcing/` builds:

```bash
python model/run_model.py  --site NL-Gl1 --days 7     # the Python port
python model/run_matlab.py --site NL-Gl1 --days 7     # the MATLAB model
```

Build the input first if you have not:

```bash
python forcing/make_input.py --site NL-Gl1
```

See [`forcing/README.md`](../forcing/README.md).

---

## Why both

They read identical input, so the comparison is clean. Where the two agree, the
forcing is doing what it should. Where they disagree, the difference belongs to
an implementation rather than to the data — which is the question
`~/stemmus-scope-py/STEMMUSSCOPE_Python_Migration.md` exists to answer.

That is not hypothetical here. See *What the first runs showed*, below.

| | `run_model.py` | `run_matlab.py` |
|---|---|---|
| engine | `~/stemmus-scope-py`, the Python port | compiled MATLAB, `~/STEMMUSSCOPEexe/STEMMUS_SCOPE` |
| needs | nothing beyond the `geo` env | MATLAB Runtime R2023a (installed; no licence needed) |
| site named by | `sites.yaml` code | `Location=NL-Gl1` in a generated `config_file.txt` |
| vegetation parameters | set in `run_model.py` | `input_data.xlsx`, keyed on `IGBP_veg_long` |
| output | `runs/<CODE>/<CODE>_run.nc` | `runs/<CODE>/matlab/` |

---

## How to run

### One station, one process

The default, and what to use first.

```bash
python model/run_matlab.py --site NL-Gl1 --days 7      # ~20 minutes
python model/run_matlab.py --site NL-Gl1               # the whole record
```

**Start with `--days 7`.** A week proves the configuration — runtime, the MATLAB
Runtime, the input translation, the parameters — for about twenty minutes of
compute, before you commit to days of it.

### Several stations at once

```bash
python model/run_matlab.py --all --jobs 5
python model/run_matlab.py --site NL-Gl1 --site NL-We1 --jobs 2
```

One process per station, following the pattern in
`~/stemmus_scope_sites/run_model_4sites.py`. Three things to be clear about:

- **`--jobs` does not make a single station faster.** Each station is one
  sequential simulation of coupled half-hourly steps and cannot be split. Five
  stations at `--jobs 5` still take as long as the slowest one.
- **Contention is not the constraint.** This machine has 128 cores and 503 GB, so
  five concurrent runs cost nothing. The constraint is wall-clock time per
  station.
- **Each run gets its own `WorkDir`** (`runs/<CODE>/matlab/`), so different
  stations cannot collide. PyStemmusScope timestamps its input and output
  directories to the minute, so two runs of the *same* station started within the
  same minute would.

Without `--jobs`, several `--site` flags run one after another.

### Long runs

A full 3.3-year record is **57,744 half-hourly steps at ~3.6 s/step ≈ 58 hours**.
Run it detached:

```bash
nohup python model/run_matlab.py --site NL-Gl1 > runs/NL-Gl1/run.log 2>&1 &
tail -f runs/NL-Gl1/run.log
```

Run it as **one continuous simulation, not year-by-year chunks**. STEMMUS_SCOPE
carries soil state forward, and restarting each year would reset the profile to
the ERA5-Land initial condition — discarding exactly the multi-year drying and
wetting that the 40 and 80 cm probes exist to test. The cost is that output is
written at the end, so a late failure loses the run; that is the right trade once
a short run has proved the setup.

### All options

| Flag | Effect |
|---|---|
| `--site CODE` | station to run; repeat for several |
| `--all` | every station in `sites.yaml` |
| `--jobs N` | run N stations concurrently (default 1) |
| `--start`, `--end` | `YYYY-MM-DD`. Omit both for the whole record |
| `--days N` | run N days from `--start` |
| `--out DIR` | where results go (default `runs/`) |

Both runners take the same flags.

### Roughly how long

| Window | Steps | MATLAB |
|---|---|---|
| 2 days | 96 | 6 min |
| 1 week | 336 | 20 min |
| 1 year | 17,520 | 18 h |
| full record | 57,744 | 58 h |

---

## What `run_model.py` records

`runs/<CODE>/<CODE>_run.nc`:

- fluxes — `Rntot`, `lEtot`, `Htot`, `Gtot`, and the split of latent heat into
  canopy transpiration `lEctot` and soil evaporation `lEstot`, plus `Actot` (GPP)
- `SoilMoisture` and `SoilTemperature` at **5, 10, 20, 40 and 80 cm** — chosen to
  be the WUNDER probe depths, so a run can be compared against the soil loggers
  in its own field with no regridding

and `runs/<CODE>/<CODE>_parameters.json`, the exact parameters the run used.

> ### The vegetation parameters are a placeholder
>
> The port has no IGBP-to-parameter lookup, and its shipped `parameters.toml` is
> Scots pine at NL-Loo: 30 m tall, measured from a 30 m tower. Handing that to a
> 0.8 m canopy under a 2 m mast would not fail — it would produce numbers.
>
> `run_model.py` therefore sets the **geometry** correctly, since that is
> site-specific and unambiguous: canopy height from `sites.yaml`, roughness
> `z0 = 0.1 hc`, displacement `d = 0.67 hc`, measurement height, coordinates.
> The **biochemistry** is generic C3 herbaceous — a starting point, not a
> calibration. Anyone drawing conclusions about these sites needs to revisit it.

---

## What the first runs showed

A two-day run at NL-Gl1 (1–2 July 2024) through the Python port:

```
Rntot   115.46      lEtot    29.90      Htot     63.90      Gtot     21.54
                    lEctot    0.01      lEstot   29.89      Actot     5.12
energy balance: Rn 115.5 = LE 29.9 + H 63.9 + G 21.5  ->  residual 0.1 W/m2
```

The energy balance closes to 0.1 W m⁻², so the model is internally consistent.
But **canopy transpiration is essentially zero**: `lEctot` is 0.01 W m⁻² against
29.89 for soil evaporation. At midday it is 0.02 W m⁻², 0.03% of latent heat,
and its maximum over the whole run is 0.057 W m⁻².

That cannot be right. The same steps give GPP of 11.21 at midday — the canopy is
assimilating carbon, so its stomata are open, so it must be transpiring. The
resulting Bowen ratio of 2.14 matches the symptom in `OPEN_ISSUES.md` #9
(NL-Loo: 2.5, against an observed 0.5–1.5).

**This rules out the leading explanation.** Issue #10 attributes that Bowen ratio
to LAI collapsing to zero in winter for an evergreen site. In this run LAI is
**2.05**, soil moisture at 5 cm is 0.256 m³ m⁻³ (unstressed), and photosynthesis
is active — and transpiration is still ~0. Whatever suppresses it is not the LAI
forcing.

Running the MATLAB engine on the same two files is the next step: it separates a
defect in the port from anything in the input.

---

## MATLAB specifics

`run_matlab.py` generates the `config_file.txt` PyStemmusScope expects and drives
the compiled executable. Three details that are easy to get wrong:

- **`Location` is the site code**, not a lat/lon pair. PyStemmusScope matches it
  against `[A-Z]{2}-([A-z]|\d){3}` to choose site mode over global mode.
- **The forcing is found by substring.** It scans `ForcingPath` for a filename
  *containing* the code and raises `ValueError` on two matches, so exactly one
  generation of a station's file may sit there. `make_input.py` overwrites in
  place, so this only bites if old files are moved there by hand.
- **The initial condition is found by glob**, `InitialConditionPath/<CODE>*.nc`,
  and merged with `open_mfdataset` — so two files matching a station would be
  silently combined rather than refused.

The MATLAB Runtime is installed at both `/usr/local/MATLAB/MATLAB_Runtime/R2023a`
and `/opt/matlab/MATLAB_Runtime/R2023a` on this machine. `run_matlab.py` checks
which one actually has the libraries and sets `LD_LIBRARY_PATH` itself, because
getting it wrong produces a linker error that says nothing about MATLAB.

---

## Evaluating a run

The reason the stations are kept separate all the way through: **`NL-Gl1`'s run
is compared against the soil loggers in Field 1, never Field 2.** The forcing
came from one mast and the evaluation comes from the ground under it.

| Station | Soil loggers in its own field |
|---|---|
| `NL-Gl1` | `z6-21176`, `z6-21178`, `z6-25928`, `z6-25927`, `z6-28931`, `z6-24636` |
| `NL-Gl2` | `z6-21177`, `z6-21179`, `z6-30173` |
| `NL-Ke1` | `z6-21180` |
| `NL-Ke2` | `z6-08819` |
| `NL-We1` | `z6-08820`, `z6-08823`, `z6-08822` |

Read them with `wunder.fetch`, and note two things from `sites.yaml` before
comparing: `z6-25928`'s moisture only starts 2024-05-08, and `z6-21178` loses its
20 cm probe in February 2024.

The soil probes were deliberately kept out of the initial condition — that is
ERA5-Land at every site — so this comparison stays independent rather than partly
self-referential.
