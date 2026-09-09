"""Check produced forcing files before a run trusts them.

Three kinds of check, in increasing order of how much they can tell you:

  structural   the things that make an engine fail. Both loaders index their
               variables by name with no guard, so a missing one is a bare
               KeyError deep inside a run; and neither has a NaN guard.
  physical     ranges and totals a Dutch half-hour must satisfy. Catches unit
               errors, which otherwise produce a file that looks perfectly well
               formed and is wrong by a factor of 1000.
  alignment    that the diurnal cycle sits where the sun does. A timezone or
               period-labelling error survives every other check on this list.

Exit status is non-zero if any check fails, so this can gate a rebuild.

Usage:
    python forcing/verify.py [--site NL-Gl1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

FORCING_DIR = Path(__file__).resolve().parent.parent / "model_input" / "forcing"
IC_DIR = Path(__file__).resolve().parent.parent / "model_input" / "ic"

#: Every name the two engines index. PyStemmusScope reads the statics too
#: (`forcing_io.py` 96-103); the Python port reads only the time series.
ENGINE_VARIABLES = [
    "Tair", "SWdown", "LWdown", "VPD", "Psurf", "Precip", "Wind", "RH",
    "CO2air", "Qair", "LAI",
]
ENGINE_STATICS = [
    "latitude", "longitude", "elevation", "reference_height", "canopy_height",
    "IGBP_veg_long",
]

IC_VARIABLES = ["skt", "stl1", "stl2", "stl3", "stl4",
                "swvl1", "swvl2", "swvl3", "swvl4"]

#: Physically possible for a Dutch half-hour. Wide enough to admit any real
#: value, narrow enough that a unit error cannot slip through.
RANGES = {
    "Tair": (250.0, 320.0),
    "SWdown": (0.0, 1400.0),
    "LWdown": (100.0, 500.0),
    "VPD": (0.0, 80.0),
    "Psurf": (90000.0, 110000.0),
    "Precip": (0.0, 0.02),
    "Wind": (0.0, 40.0),
    "RH": (0.0, 100.0),
    "Qair": (0.0, 0.05),
    "CO2air": (350.0, 500.0),
    "LAI": (0.0, 10.0),
}

#: Annual means a Dutch site must land inside, given the record here.
ANNUAL = {
    "SWdown": (95.0, 145.0),      # W/m2; 2024-25 at Glanerbeek measured 105-111
    "LWdown": (250.0, 380.0),
    "Tair": (279.0, 288.0),
}

PRECIP_MM_YEAR = (500.0, 1200.0)

#: The unit each variable must be written in, and what makes it so. Both engines
#: read these by name and neither checks the attribute, so a wrong unit is not an
#: error anywhere -- it is a run that completes and is quietly wrong. Traced to
#: the code that consumes each one:
#:
#:   Tair    PyStemmusScope forcing_io.py:67  `Tair - 273.15`      -> K
#:   Psurf   forcing_io.py:68                 `Psurf / 100` to hPa -> Pa
#:   Precip  forcing_io.py:75                 `Precip / 10`, "mm/s to cm/s"
#:                                            -> kg/m2/s, numerically mm/s
#:   CO2air  forcing_io.py:69-71              `CO2air * 1e-6`      -> ppm
#:   RH      forcing_io.py:84 feeds calculate_ea, whose docstring says "as a
#:           percentage (e.g. ranging from 0 - 100)"; the Python port divides by
#:           100 at load                      -> % on 0-100, NOT a fraction
#:   SWdown  forcing_io.py:77 and port load_inputs.py:71 both pass it through
#:           unconverted, straight into Rin_.dat  -> W/m2
#:   LWdown  same, into Rli_.dat              -> W/m2
#:   VPD, Qair, LAI                            -> hPa, kg/kg, m2/m2
EXPECTED_UNITS = {
    "Tair": "K",
    "SWdown": "W/m2",
    "LWdown": "W/m2",
    "VPD": "hPa",
    "Psurf": "Pa",
    "Precip": "kg/m2/s",
    "Wind": "m/s",
    "RH": "%",
    "Qair": "kg/kg",
    "CO2air": "ppm",
    "LAI": "m2/m2",
}

#: Checks a station is known to fail, and why. These are downgraded from failure
#: to a stated warning -- the check still runs and still reports, so the
#: condition cannot quietly become normal, but it does not gate a rebuild.
#: Nothing belongs here without a matching explanation in sites.yaml.
KNOWN = {
    "NL-Ke1": {
        "SWdown annual mean": (
            "the K1 mast stands under the food-forest canopy (ETH 2020 gives 14 m "
            "there against 4-5 m at every other station), so it measures ~30% less "
            "radiation and 75% less wind than the open field 175 m away. Real, "
            "measured, and documented in sites.yaml -- but sub-canopy forcing, "
            "which STEMMUS_SCOPE will attenuate a second time."
        ),
    },
}


class Checker:
    def __init__(self, label: str, known: dict[str, str] | None = None):
        self.label = label
        self.known = known or {}
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def check(self, ok: bool, message: str) -> bool:
        if ok:
            return True
        for prefix, reason in self.known.items():
            if message.startswith(prefix):
                self.warnings.append(f"{message}\n          expected: {reason}")
                return False
        self.failures.append(message)
        return False

    def note(self, message: str) -> None:
        self.notes.append(message)

    def report(self) -> bool:
        print(f"\n{self.label}")
        for line in self.notes:
            print(f"    {line}")
        for line in self.warnings:
            print(f"    KNOWN {line}")
        for line in self.failures:
            print(f"    FAIL  {line}")
        if not self.failures:
            print("    all checks passed"
                  + (f" ({len(self.warnings)} known exception)" if self.warnings else ""))
        return not self.failures


def verify_forcing(path: Path) -> bool:
    code = next((k for k in KNOWN if k in path.name), None)
    c = Checker(path.name, KNOWN.get(code))
    ds = xr.open_dataset(path)

    # --- structural ---
    missing = [v for v in ENGINE_VARIABLES if v not in ds]
    c.check(not missing, f"missing variables the engines index: {missing}")
    missing_static = [v for v in ENGINE_STATICS if v not in ds]
    c.check(not missing_static, f"missing statics PyStemmusScope reads: {missing_static}")

    time = pd.DatetimeIndex(ds["time"].values)
    c.check(time.is_monotonic_increasing, "time is not strictly increasing")
    c.check(not time.duplicated().any(), "time has duplicates")
    steps = time.to_series().diff().dropna().unique()
    c.check(len(steps) == 1 and steps[0] == pd.Timedelta("30min"),
            f"time step is not a uniform 30 min: {steps[:3]}")

    for name in ENGINE_VARIABLES:
        if name not in ds:
            continue
        values = ds[name].values.ravel()
        nan = int(np.isnan(values).sum())
        c.check(nan == 0, f"{name} has {nan} NaN (neither engine has a NaN guard)")
        low, high = RANGES[name]
        out = int(((values < low) | (values > high)).sum())
        c.check(out == 0, f"{name} has {out} values outside [{low}, {high}]")

    for name, expected in EXPECTED_UNITS.items():
        if name not in ds:
            continue
        written = ds[name].attrs.get("units")
        c.check(written == expected,
                f"{name} units attribute is {written!r}, expected {expected!r}")

    # Qair dims: the reference PLUMBER2 file has this wrong, as (dim_0, dim_1,
    # dim_2) of a different length. Reproducing that bug would be easy.
    if "Qair" in ds:
        c.check(ds["Qair"].dims == ("x", "y", "time"),
                f"Qair dims are {ds['Qair'].dims}, expected ('x','y','time')")

    for name in [v for v in ds.data_vars if v.endswith("_qc")]:
        levels = set(np.unique(ds[name].values).astype(int).tolist())
        c.check(levels <= {0, 1, 2, 3}, f"{name} has unexpected levels {levels}")

    # --- physical ---
    years = len(time) / (2 * 24 * 365.25)
    for name, (low, high) in ANNUAL.items():
        if name not in ds:
            continue
        mean = float(np.nanmean(ds[name].values))
        c.check(low <= mean <= high,
                f"{name} annual mean {mean:.1f} outside [{low}, {high}]")
        c.note(f"{name:7s} mean {mean:8.2f}")

    if "Precip" in ds:
        total = float(np.nansum(ds["Precip"].values)) * 1800 / years
        c.check(PRECIP_MM_YEAR[0] <= total <= PRECIP_MM_YEAR[1],
                f"precipitation {total:.0f} mm/yr outside {PRECIP_MM_YEAR}")
        c.note(f"Precip  {total:8.0f} mm/yr")

    # --- alignment ---
    # The check a timezone error cannot survive. Weighting time-of-day by
    # SWdown gives solar noon; in UTC it must sit at 12:00 minus the longitude,
    # in BOTH seasons. A DST error moves winter and summer apart by an hour.
    if "SWdown" in ds and "longitude" in ds:
        lon = float(ds["longitude"].values.ravel()[0])
        expected = 12.0 - lon / 15.0
        sw = pd.Series(ds["SWdown"].values.ravel(), index=time)
        hours = {}
        for label, months in (("DJF", (12, 1, 2)), ("JJA", (6, 7, 8))):
            sel = sw[sw.index.month.isin(months)]
            tod = sel.index.hour + sel.index.minute / 60
            hours[label] = float(np.average(tod, weights=sel)) if sel.sum() else np.nan
        c.note(f"solar noon UTC: DJF {hours['DJF']:.2f} h, JJA {hours['JJA']:.2f} h "
               f"(expected ~{expected:.2f})")
        for label, value in hours.items():
            c.check(abs(value - expected) < 0.6,
                    f"{label} solar noon {value:.2f} h is far from {expected:.2f} — "
                    "timezone or period-labelling error")
        c.check(abs(hours["DJF"] - hours["JJA"]) < 0.5,
                f"solar noon differs by {abs(hours['DJF'] - hours['JJA']):.2f} h "
                "between winter and summer — daylight saving was not removed")

    measured = {}
    for name in ENGINE_VARIABLES:
        flag = f"{name}_qc"
        if flag in ds:
            measured[name] = float((ds[flag].values == 0).mean())
    if measured:
        c.note("measured fraction: " + " ".join(
            f"{k}={v:.3f}" for k, v in sorted(measured.items())))

    ds.close()
    return c.report()


def verify_ic(path: Path) -> bool:
    c = Checker(f"{path.name}")
    ds = xr.open_dataset(path)
    missing = [v for v in IC_VARIABLES if v not in ds]
    c.check(not missing, f"missing initial-condition variables: {missing}")
    for name in IC_VARIABLES:
        if name not in ds:
            continue
        value = float(ds[name].values)
        c.check(np.isfinite(value), f"{name} is not finite")
        if name.startswith(("skt", "stl")):
            c.check(230.0 < value < 330.0, f"{name} = {value:.1f} K is not a temperature")
        else:
            c.check(0.0 <= value <= 1.0, f"{name} = {value:.3f} is not a water fraction")
    c.note("  ".join(f"{v}={float(ds[v].values):.3f}" for v in IC_VARIABLES if v in ds))
    ds.close()
    return c.report()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", action="append", dest="sites")
    parser.add_argument("--forcing-dir", type=Path, default=FORCING_DIR)
    parser.add_argument("--ic-dir", type=Path, default=IC_DIR)
    args = parser.parse_args()

    forcing = sorted(args.forcing_dir.glob("FLX_*.nc"))
    ics = sorted(args.ic_dir.glob("*_InitialCondition.nc"))
    if args.sites:
        forcing = [p for p in forcing if any(s in p.name for s in args.sites)]
        ics = [p for p in ics if any(s in p.name for s in args.sites)]

    if not forcing:
        print(f"no forcing files in {args.forcing_dir}")
        return 1

    ok = True
    for path in forcing:
        ok &= verify_forcing(path)
    for path in ics:
        ok &= verify_ic(path)

    # Distinct stations must not have produced identical files -- equal sizes are
    # expected (same dimensions, no compression) but equal contents would mean a
    # code path silently ignored the station.
    if len(forcing) > 1:
        signatures = {}
        for path in forcing:
            with xr.open_dataset(path) as ds:
                signatures[path.name] = float(np.nansum(ds["Tair"].values))
        duplicates = [k for k, v in signatures.items()
                      if list(signatures.values()).count(v) > 1]
        if duplicates:
            print(f"\nFAIL  these files have identical Tair: {duplicates}")
            ok = False
        else:
            print(f"\nall {len(forcing)} stations produced distinct records")

    print("\n" + ("VERIFIED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
