"""Assemble one station's STEMMUS_SCOPE input files.

Produces the two files both engines read:

    <out>/forcing/FLX_<CODE>_FLUXNET2015_FULLSET_<y0>-<y1>.nc
    <out>/ic/<CODE>_<run start>_InitialCondition.nc

Everything upstream must already be on disk -- run these first, in any order:

    python forcing/download_era5land.py     the baseline, and the soil state
    python forcing/download_lai.py          MODIS LAI
    python forcing/download_co2.py          Mauna Loa CO2
    python forcing/download_statics.py      elevation / canopy height / IGBP,
                                            which are then entered in sites.yaml

Files are overwritten in place rather than versioned. PyStemmusScope finds a
station's forcing by scanning `ForcingPath` for a filename *containing* the code
and raises `ValueError("Multiple forcing files exist...")` on two matches, so a
second generation sitting beside the first would break the run.

Usage:
    python forcing/build_forcing.py [--site NL-Gl1] [--out ~/data/wunder]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path.home() / "data" / "wunder"
CO2_PATH = OUT_DIR / "co2_maunaloa_30min.csv"
LAI_DIR = OUT_DIR / "lai"


def load_co2(index: pd.DatetimeIndex, path: Path = CO2_PATH) -> pd.Series:
    """Mauna Loa CO2 on the forcing grid."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python forcing/download_co2.py"
        )
    series = pd.read_csv(path, comment="#", index_col="time", parse_dates=True)["co2"]
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    covered = series.reindex(series.index.union(index)).interpolate(method="time")
    out = covered.reindex(index)
    if out.isna().any():
        raise ValueError(
            f"CO2 does not cover {index.min()} .. {index.max()}; "
            "re-run forcing/download_co2.py"
        )
    return out.rename("CO2air")


def load_lai(code: str, index: pd.DatetimeIndex, directory: Path = LAI_DIR) -> pd.Series:
    """MODIS LAI on the forcing grid, held across each day."""
    path = directory / f"lai_{code}_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python forcing/download_lai.py --site {code}"
        )
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from download_lai import to_grid

    daily = pd.read_parquet(path)["LAI"]
    if daily.index.tz is None:
        daily.index = daily.index.tz_localize("UTC")
    return to_grid(daily, index)


def build(code: str, out_dir: Path, *, cache_dir=None) -> dict:
    """Build and write one station's forcing and initial-condition files."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from wunder import era5land, export, gapfill
    from wunder.metadata import forcing_site

    cfg = forcing_site(code)
    forcing, qc, report = gapfill.build(code, cache_dir=cache_dir)
    index = forcing.index

    forcing["CO2air"] = load_co2(index)
    forcing["LAI"] = load_lai(code, index)
    qc["CO2air"] = export.QC_ERA5  # a background record, never a site measurement

    statics = {
        "latitude": cfg["latitude"],
        "longitude": cfg["longitude"],
        "reference_height": cfg["reference_height_m"],
        "canopy_height": cfg["canopy_height_m"],
        "elevation": cfg.get("elevation_dem_m", cfg["elevation_m"]),
        "IGBP_veg_short": cfg["igbp_short"],
        "IGBP_veg_long": cfg["igbp_long"],
    }

    provenance = {
        name: " ".join(
            f"qc{level}={fraction:.4f}"
            for level, fraction in sorted(levels.items())
        )
        for name, levels in report.qc_fractions.items()
    }
    attrs = {
        "title": f"STEMMUS_SCOPE forcing for {code}",
        "station": cfg.get("label", code),
        "met_logger": cfg["met_logger"],
        "history": (
            f"built {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by "
            "wunder/forcing/build_forcing.py"
        ),
        "source": (
            "WUNDER ZENTRA logger network (MajiSys API); ERA5-Land via Google "
            "Earth Engine; MODIS MCD15A3H LAI; NOAA GML Mauna Loa CO2"
        ),
        "qc_convention": "; ".join(
            f"{k} = {v}" for k, v in sorted(export.QC_MEANING.items())
        ),
        "qc_fractions": "; ".join(f"{k}: {v}" for k, v in sorted(provenance.items())),
        "bias_fits": "; ".join(f.describe() for f in report.fits.values()),
        "timezone": "UTC (loggers report Europe/Amsterdam wall-clock, converted)",
        "canopy_height_note": (
            "capped at 0.8 m: the ATMOS-41 is at 2.0 m, and a canopy above the "
            "measurement height would put the reference level inside the canopy"
        ),
    }

    ds = export.build_dataset(forcing, qc, statics, attrs)

    years = f"{index[0].year}-{index[-1].year}"
    forcing_path = out_dir / "forcing" / f"FLX_{code}_FLUXNET2015_FULLSET_{years}.nc"
    export.write_forcing(ds, forcing_path, start=index[0])

    start = index[0]
    values = era5land.initial_condition(code, start)
    ic_path = out_dir / "ic" / f"{code}_{start:%Y-%m-%d}_InitialCondition.nc"
    export.write_initial_condition(
        values, cfg["latitude"], cfg["longitude"], start, ic_path,
        attrs={"source": "ERA5-Land via Google Earth Engine",
               "station": cfg.get("label", code)},
    )

    return {"code": code, "forcing": forcing_path, "ic": ic_path,
            "report": report, "steps": len(index)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", action="append", dest="sites")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from wunder.metadata import forcing_sites

    failures = 0
    for code in args.sites or list(forcing_sites()):
        try:
            result = build(code, args.out)
        except Exception as exc:  # noqa: BLE001 - report and continue to the next
            failures += 1
            print(f"\n{code}: FAILED — {type(exc).__name__}: {exc}")
            continue
        print(f"\n{'=' * 70}")
        print("\n".join(result["report"].lines()))
        print(f"  wrote {result['forcing']}")
        print(f"  wrote {result['ic']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
