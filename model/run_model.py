"""Run STEMMUS_SCOPE on a station's forcing, through the Python port.

    python model/run_model.py --site NL-Gl1 --days 7
    python model/run_model.py --site NL-Gl1 --start 2024-06-01 --end 2024-08-31

Output goes to `runs/<CODE>/`: a NetCDF of the fluxes and soil state, and the
exact parameters the run used.

**The vegetation parameters here are a placeholder, and that matters.** The port
has no IGBP-to-parameter lookup -- its `parameters.toml` is Scots pine at NL-Loo,
30 m tall, measured from a 30 m tower. Handing that to a 0.8 m canopy under a 2 m
mast would not fail; it would produce numbers. What is set below is the geometry,
which is site-specific and unambiguous, plus generic C3 herbaceous biochemistry.
The biochemistry is a starting point, not a calibration: `OPEN_ISSUES.md` #9/#10
is a standing reminder of what happens when vegetation parameters and the site
disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

PORT = Path.home() / "stemmus-scope-py"
FORCING_DIR = REPO / "model_input" / "forcing"
IC_DIR = REPO / "model_input" / "ic"
RUNS_DIR = REPO / "runs"

SOIL_PROPERTY = (Path.home() / "STEMMUS_SCOPE_model" / "STEMMUS_SCOPE_old"
                 / "STEMMUS_SCOPE" / "input" / "SoilProperty")
SCOPE_INPUT = Path.home() / "SCOPE" / "input"
OPTIPAR = SCOPE_INPUT / "fluspect_parameters" / "Optipar2017_ProspectD.mat"
ATMO_DIR = SCOPE_INPUT / "radiationdata"
SOIL_SPECTRUM = SCOPE_INPUT / "soil_spectra" / "soilnew.txt"

#: Temperature dependence of photosynthesis. Generic C3, unchanged from the
#: port's own defaults -- these are not site properties.
TDP = {
    "delHaV": 72000.0, "delSV": 710.0, "delHdV": 220000.0,
    "delHaR": 53000.0, "delSR": 490.0, "delHdR": 150000.0,
    "delHaKc": 59430.0, "delHaKo": 36000.0, "delHaT": 37000.0,
    "Q10": 2.0,
    "s1": 0.3, "s2": 0.5, "s3": 280.0, "s4": 0.3, "s5": 313.0, "s6": 0.8,
}

#: Leaf optics. Moved off the port's Scots-pine values towards generic
#: herbaceous: less chlorophyll and much less dry matter per unit area than a
#: conifer needle.
LEAFBIO = {
    "Cab": 40.0, "Cca": 10.0, "Cdm": 0.005, "Cw": 0.010,
    "Cs": 0.0, "Cant": 0.0, "Cbc": 0.0, "Cp": 0.0,
    "N": 1.5, "V2Z": 0, "fqe": 0.01,
    "rho_thermal": 0.01, "tau_thermal": 0.01,
}

LEAFBIO_EBAL = {
    "Type": "C3", "Vcmax25": 60.0,
    "BallBerrySlope": 8.0, "BallBerry0": 0.01,
    "RdPerVcmax25": 0.015, "Kn0": 2.48,
    "Knalpha": 2.83, "Knbeta": 0.114,
    "stressfactor": 1.0, "TDP": TDP,
    "emis": 0.98, "rho_thermal": 0.01, "tau_thermal": 0.005,
}

SOILPAR_BASE = {"BSMBrightness": 0.5, "SMC": 0.25}

#: Depths to record, chosen to line up with the WUNDER probes so a run can be
#: compared against the soil loggers in its own field without regridding.
DEPTH_TARGETS_CM = [5, 10, 20, 40, 80]


def canopy_config(canopy_height_m: float, lai: float) -> dict:
    """Canopy geometry for a short herbaceous cover.

    Roughness and displacement height follow the usual `z0 = 0.1 hc`,
    `d = 0.67 hc`. With the 0.8 m canopy cap that puts `d + z0` at 0.62 m, safely
    under the 2 m mast -- which is the whole reason the cap exists.
    """
    return {
        "LAI": lai,
        "hc": canopy_height_m,
        "leafwidth": 0.02,        # a grass blade, not a 2 mm pine needle
        "LIDFa": -0.35,
        "LIDFb": -0.15,
        "kV": 0.6394,
        "zo": 0.1 * canopy_height_m,
        "d": 0.67 * canopy_height_m,
        "Cd": 0.3,
        "rwc": 1.0,
    }


def find_input(code: str) -> tuple[Path, Path]:
    forcing = sorted(FORCING_DIR.glob(f"FLX_{code}_*.nc"))
    ic = sorted(IC_DIR.glob(f"{code}_*_InitialCondition.nc"))
    if not forcing or not ic:
        raise SystemExit(
            f"no input for {code}. Build it first:\n"
            f"    python forcing/make_input.py --site {code}"
        )
    if len(forcing) > 1:
        raise SystemExit(f"more than one forcing file for {code}: {forcing}")
    return forcing[0], ic[0]


def window(forcing_path: Path, start, end, days) -> tuple[int, int]:
    """Resolve a date range to the index window the port slices on."""
    import xarray as xr

    with xr.open_dataset(forcing_path) as ds:
        time = pd.DatetimeIndex(ds["time"].values)
    if start:
        i0 = int(time.searchsorted(pd.Timestamp(start)))
    else:
        i0 = 0
    if end:
        i1 = int(time.searchsorted(pd.Timestamp(end)))
    elif days:
        i1 = min(i0 + days * 48, len(time))
    else:
        i1 = len(time)
    if i1 <= i0:
        raise SystemExit(f"empty window: {time[i0]} to {end or days}")
    return i0, i1


def layer_midpoints_cm(ms) -> np.ndarray:
    """Depth below the surface of each layer midpoint, top to bottom."""
    dz = ms.DeltZ[::-1]                      # STEMMUS stores bottom-to-top
    tops = np.concatenate([[0.0], np.cumsum(dz[:-1])])
    return tops + dz / 2.0


def run(code: str, i_start: int, i_end: int, out_dir: Path) -> Path:
    from wunder.metadata import forcing_site

    cfg = forcing_site(code)
    forcing_path, ic_path = find_input(code)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(PORT))
    from scope.engine import SCOPEEngine
    from stemmus.engine import STEMMUSEngine
    from stemmus_scope_coupler.coupler import ModelCoupler
    from stemmus_scope_coupler.io.forcing_loader import load_forcing, slice_forcing

    print(f"{code}  {cfg.get('label','')}")
    print(f"  forcing {forcing_path.name}")
    print(f"  ic      {ic_path.name}")

    forcing = slice_forcing(load_forcing(str(forcing_path)),
                            i_start=i_start, i_end=i_end)
    n = len(forcing)
    mean_lai = float(np.mean([f.LAI for f in forcing]))
    print(f"  window  {n} steps ({n / 48:.1f} days), mean LAI {mean_lai:.2f}")

    soil = STEMMUSEngine()
    soil.initialize(str(SOIL_PROPERTY), str(ic_path),
                    cfg["latitude"], cfg["longitude"])
    mids = layer_midpoints_cm(soil.ms)
    idx = [int(np.argmin(np.abs(mids - d))) for d in DEPTH_TARGETS_CM]
    print(f"  soil    {soil.ms.NL} layers to {soil.ms.Tot_Depth:.0f} cm; "
          f"recording {[f'{mids[i]:.1f}' for i in idx]} cm")

    canopy_cfg = canopy_config(cfg["canopy_height_m"], mean_lai)
    canopy = SCOPEEngine()
    canopy.initialize(
        optipar_path=str(OPTIPAR),
        atmo_dir=str(ATMO_DIR) + "/",
        soil_spec_path=str(SOIL_SPECTRUM),
        lat=cfg["latitude"], lon=cfg["longitude"],
        leafbio=LEAFBIO,
        canopy_cfg=canopy_cfg,
        leafbio_ebal=LEAFBIO_EBAL,
        soilpar={**SOILPAR_BASE,
                 "BSMlat": cfg["latitude"], "BSMlon": cfg["longitude"]},
        options={"mode": "multilayer"},
        z_meas=cfg["reference_height_m"],
    )
    print(f"  canopy  hc {canopy_cfg['hc']:.2f} m, z0 {canopy_cfg['zo']:.3f}, "
          f"d {canopy_cfg['d']:.3f}, mast {cfg['reference_height_m']:.1f} m")

    coupler = ModelCoupler(soil, canopy, n_soil_layers=soil.ms.NL)

    # SCOPE's own flux dictionary, which the coupler does not pass through.
    # `SCOPEEngine.update` stashes it on `_last_fluxes`, so it can be read after
    # each step -- `run_test_week.py` monkey-patches `update` to get at the same
    # thing, which is not necessary.
    keys = ["Rntot", "lEtot", "Htot", "Gtot", "lEctot", "lEstot", "Actot"]
    rec = {k: np.full(n, np.nan) for k in keys}
    rec["time"] = np.zeros(n)
    theta = np.full((n, len(idx)), np.nan)
    tsoil = np.full((n, len(idx)), np.nan)

    print(f"  running {n} steps ...", flush=True)
    failures = 0
    for i, step in enumerate(forcing):
        rec["time"][i] = step.t
        try:
            coupler.step(step, i)
        except Exception as exc:  # noqa: BLE001 - record and continue
            failures += 1
            if failures <= 3:
                print(f"    step {i} failed: {type(exc).__name__}: {exc}")
            continue

        fluxes = getattr(canopy, "_last_fluxes", None) or {}
        for k in keys:
            value = fluxes.get(k, np.nan)
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            # SCOPE occasionally returns a non-converged energy balance. The
            # reference driver screens the same way rather than letting a
            # 10^6 W/m2 latent heat into a daily mean.
            rec[k][i] = value if np.isfinite(value) and abs(value) < 2000.0 else np.nan
        # STEMMUS stores the profile bottom-to-top; reverse for depth order.
        soil_exch = coupler.last_soil_exchange
        if soil_exch is not None:
            profile = np.asarray(soil_exch.theta_L).ravel()[::-1]
            theta[i] = [profile[j] if j < len(profile) else np.nan for j in idx]
        if getattr(soil.ms, "TT", None) is not None:
            profile = np.asarray(soil.ms.TT).ravel()[::-1]
            tsoil[i] = [profile[j] if j < len(profile) else np.nan for j in idx]
        if (i + 1) % 480 == 0:
            print(f"    {i + 1}/{n}", flush=True)

    if failures:
        print(f"  {failures} of {n} steps failed")

    import xarray as xr

    time = pd.to_datetime(rec["time"], unit="s", utc=True)
    ds = xr.Dataset(
        {k: ("time", rec[k]) for k in keys}
        | {"SoilMoisture": (("time", "depth"), theta),
           "SoilTemperature": (("time", "depth"), tsoil)},
        coords={"time": time.tz_localize(None),
                "depth": [float(mids[i]) for i in idx]},
        attrs={
            "station": cfg.get("label", code),
            "forcing": forcing_path.name,
            "initial_condition": ic_path.name,
            "engine": "stemmus-scope-py (Python port)",
            "canopy_height_m": canopy_cfg["hc"],
            "reference_height_m": cfg["reference_height_m"],
            "vegetation_parameters": (
                "PLACEHOLDER — generic C3 herbaceous, not calibrated to this site"
            ),
            "failed_steps": failures,
        },
    )
    path = out_dir / f"{code}_run.nc"
    ds.to_netcdf(path)

    (out_dir / f"{code}_parameters.json").write_text(json.dumps(
        {"site": {k: cfg[k] for k in
                  ("code", "latitude", "longitude", "canopy_height_m",
                   "reference_height_m", "igbp_short")},
         "canopy": canopy_cfg, "leafbio": LEAFBIO,
         "leafbio_ebal": {k: v for k, v in LEAFBIO_EBAL.items() if k != "TDP"},
         "window": {"i_start": i_start, "i_end": i_end,
                    "from": str(time[0]), "to": str(time[-1])}},
        indent=2))

    good = np.isfinite(rec["lEtot"]) & np.isfinite(rec["Htot"])
    if good.sum():
        le, h = rec["lEtot"][good].mean(), rec["Htot"][good].mean()
        print(f"  mean LE {le:7.2f} W/m2, H {h:7.2f} W/m2, "
              f"Bowen {h / le if le else float('nan'):.2f}")
    print(f"  -> {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", action="append", dest="sites")
    parser.add_argument("--all", action="store_true",
                        help="every station in sites.yaml")
    parser.add_argument("--jobs", type=int, default=1,
                        help="stations to run concurrently (default 1)")
    parser.add_argument("--start", help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD")
    parser.add_argument("--days", type=int, help="run this many days from --start")
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    if args.all:
        from wunder.metadata import forcing_sites

        codes = list(forcing_sites())
    elif args.sites:
        codes = args.sites
    else:
        raise SystemExit("specify --site CODE (repeatable) or --all")

    jobs = []
    for code in codes:
        forcing_path, _ = find_input(code)
        i0, i1 = window(forcing_path, args.start, args.end, args.days)
        jobs.append((code, i0, i1, args.out / code))

    if args.jobs <= 1 or len(jobs) == 1:
        for job in jobs:
            run(*job)
        return 0

    from multiprocessing import Pool

    print(f"running {len(jobs)} stations, {args.jobs} at a time")
    with Pool(processes=min(args.jobs, len(jobs))) as pool:
        pool.starmap(run, jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
