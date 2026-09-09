# Running STEMMUS_SCOPE

```bash
python model/run_matlab.py --site NL-Gl1 --days 7
```

Runs the compiled MATLAB model on the files `forcing/` builds, through
PyStemmusScope.

Build the input first if you have not:

```bash
python forcing/make_input.py --site NL-Gl1
```

See [`forcing/README.md`](../forcing/README.md).

---

## Which engine

The MATLAB model, via the compiled executable at
`~/STEMMUSSCOPEexe/STEMMUS_SCOPE` and MATLAB Runtime R2022b. No licence needed.

There is also a Python port at `~/stemmus-scope-py`, driven by
`model/run_model.py`, but it is **not the path in use**: on identical forcing it
produces near-zero canopy transpiration where MATLAB puts half the latent heat
there. That is a defect in the port, not in the input — see *What the first runs
showed* below — and fixing it is deferred.

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

### Roughly how long

| Window | Steps | MATLAB |
|---|---|---|
| 2 days | 96 | 6 min |
| 1 week | 336 | 20 min |
| 1 year | 17,520 | 18 h |
| full record | 57,744 | 58 h |

---

## What the first run showed

A two-day trial at NL-Gl1 (1–2 July 2024):

```
Rntot   90.18      lEtot   50.00      Htot   40.22      Gtot    0.23
                   lEctot  25.00      lEstot 24.99      Actot   9.47
Bowen H/LE = 0.80        transpiration = 50.0% of latent heat
```

That is a physically correct Dutch grassland in July: Bowen 0.80 against a
literature 0.3–0.8, half the latent heat as transpiration, and a two-day mean
ground heat flux of 0.23 W m⁻² — which has to integrate to ≈ 0 over full diurnal
cycles, and does. **The forcing is validated end to end**, from the logger record
through to a closed energy balance.

### Why the Python port is not used

The same two files, run through `~/stemmus-scope-py` for the same two days:

| | MATLAB | Python port |
|---|---|---|
| `Rntot` | 90.18 | 115.46 |
| `lEtot` | 50.00 | 29.90 |
| `Htot` | 40.22 | 63.90 |
| `Gtot` (2-day mean) | 0.23 | 21.54 |
| `lEctot` transpiration | 25.00 | **0.01** |
| Bowen ratio | 0.80 | 2.14 |
| transpiration share of LE | 50.0% | **0.04%** |

The port's canopy transpiration is 0.04% of latent heat while its own GPP runs at
5.12 µmol m⁻² s⁻¹ — a canopy fixing carbon has open stomata and must transpire.
Its two-day mean `Gtot` of 21.5 W m⁻² is non-physical for the same reason.

This also **rules out the standing explanation** for the port's high Bowen ratio.
`OPEN_ISSUES.md` #10 attributes it to LAI collapsing to zero in winter at an
evergreen site. Here LAI is **2.05** — the same value, from the same file, that
MATLAB transpires normally with — soil moisture at 5 cm is 0.256 m³ m⁻³ and
unstressed, and photosynthesis is active. Transpiration is still ~0, so the LAI
forcing is not what suppresses it.

Fixing that is deferred. `model/run_model.py` still exists if you want to
reproduce the comparison.

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

### The MATLAB Runtime is R2022b, not R2023a

The compiled model needs `libmwmclmcrrt.so.9.13` — **Runtime 9.13 = R2022b**.
This machine has three runtimes and the matching one is not in a
`MATLAB_Runtime/` directory at all: it is inside a full install at
**`/opt/matlab/R2022b`**. The two `MATLAB_Runtime/R2023a` directories are 9.14,
and the shell profile points at one of them.

`run_matlab.py` therefore searches by the library the executable actually asks
for rather than by directory name, and prepends to `LD_LIBRARY_PATH`. Pointed at
the wrong runtime, the model exits 255 with "Could not find version 9.13 of the
MATLAB Runtime" — which PyStemmusScope swallows into a bare
`CalledProcessError` that says nothing about MATLAB.

`hdf5storage` must also be **≥ 0.2.2** in the `geo` env: 0.1.19 uses
`np.unicode_`, removed in NumPy 2.0, and fails while writing
`forcing_globals.mat`.

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
