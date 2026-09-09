"""Build ready-to-run STEMMUS_SCOPE input for a site and a date range.

One command, two files out:

    <out>/forcing/FLX_<CODE>_FLUXNET2015_FULLSET_<y0>-<y1>.nc
    <out>/ic/<CODE>_<start>_InitialCondition.nc

Everything else -- ERA5-Land, MODIS LAI, Mauna Loa CO2, elevation, canopy height
and land cover -- is fetched as needed and cached, so a second run over the same
window is fast.

Two ways to name a site:

    --site NL-Gl1                     a station in sites.yaml. Its weather
                                      station is used wherever it has data, and
                                      ERA5-Land fills the rest.

    --code NL-Xyz --lat 52.1 --lon 5.2
                                      anywhere at all. No logger is involved and
                                      the file is built from ERA5-Land alone,
                                      which is complete by construction.

Both produce the same variables in the same shape, so a run does not need to know
which kind of site it is looking at.

Examples:
    python forcing/make_input.py --site NL-Gl1
    python forcing/make_input.py --site NL-Gl1 --start 2024-01-01 --end 2024-12-31
    python forcing/make_input.py --all
    python forcing/make_input.py --code NL-Ams --lat 52.37 --lon 4.90 \\
        --start 2024-01-01 --end 2024-12-31

Requires Google Earth Engine credentials (`earthengine authenticate`) the first
time; after that everything is cached under --out.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

#: Where the two model-input files land: in the working folder, beside the code
#: that made them, because they are the deliverable and people look for them here.
OUT_DIR = HERE.parent / "model_input"

#: Where the downloads live. Deliberately NOT in the repo: ~11 MB of ERA5-Land,
#: MODIS and CO2 that is reused across every build and every site, has nothing to
#: do with any one run, and would be re-downloaded pointlessly if it were tied to
#: a clone. Shared with `data/raw/` in spirit -- an input cache, not an output.
CACHE_DIR = Path.home() / "data" / "wunder"

#: A canopy may be at most this fraction of the measurement height. STEMMUS_SCOPE
#: derives displacement height and roughness from the canopy (d ~ 0.67*hc,
#: z0 ~ 0.1*hc), so a canopy approaching the mast puts the reference level inside
#: it and the log-wind profile stops meaning anything. At the WUNDER stations the
#: ATMOS-41 is at 2.0 m, so this caps the canopy at 0.8 m; a taller mast at a
#: future site scales with it rather than needing the rule rewritten.
CANOPY_FRACTION_OF_MAST = 0.4

#: Used when a site is given as a bare lat/lon, with no mast to measure from.
#: With the fraction above this puts the canopy cap at 1.6 m. Override with
#: --reference-height when the real measurement height is known.
DEFAULT_REFERENCE_HEIGHT_M = 4.0


@dataclass
class Site:
    """Everything needed to build one file, however the site was specified."""

    code: str
    latitude: float
    longitude: float
    start: pd.Timestamp
    end: pd.Timestamp
    reference_height_m: float
    label: str = ""
    met_logger: str | None = None
    logger_override: str | None = None
    exclude_columns: tuple[str, ...] = ()
    canopy_height_m: float | None = None
    elevation_m: float | None = None
    igbp_short: str | None = None
    igbp_long: str | None = None


def resolve(args) -> list[Site]:
    """Turn command-line arguments into the sites to build."""
    from wunder.metadata import forcing_sites

    known = forcing_sites()
    out: list[Site] = []

    if args.code:
        if args.lat is None or args.lon is None:
            raise SystemExit("--code needs --lat and --lon")
        if args.start is None:
            raise SystemExit("--code needs --start (there is no logger to infer it from)")
        out.append(
            Site(
                code=args.code,
                latitude=args.lat,
                longitude=args.lon,
                start=pd.Timestamp(args.start, tz="UTC"),
                end=pd.Timestamp(args.end, tz="UTC") if args.end else None,
                reference_height_m=args.reference_height
                or DEFAULT_REFERENCE_HEIGHT_M,
            )
        )
        return out

    wanted = args.sites or (list(known) if args.all else None)
    if not wanted:
        raise SystemExit(
            "specify --site CODE, --all, or --code/--lat/--lon for a new location.\n"
            f"sites.yaml knows: {', '.join(known)}"
        )

    for code in wanted:
        if code not in known:
            raise SystemExit(f"unknown site {code!r}; sites.yaml has {', '.join(known)}")
        cfg = known[code]
        out.append(
            Site(
                code=code,
                latitude=cfg["latitude"],
                longitude=cfg["longitude"],
                start=pd.Timestamp(args.start or cfg["run_start"], tz="UTC"),
                end=pd.Timestamp(args.end, tz="UTC") if args.end else None,
                reference_height_m=args.reference_height or cfg["reference_height_m"],
                label=cfg.get("label", ""),
                met_logger=args.logger if args.logger is not None else cfg["met_logger"],
                logger_override=args.logger,
                exclude_columns=tuple(cfg["exclude_columns"]),
                canopy_height_m=cfg.get("canopy_height_m"),
                elevation_m=cfg.get("elevation_dem_m"),
                igbp_short=cfg.get("igbp_short"),
                igbp_long=cfg.get("igbp_long"),
            )
        )
    return out


def ensure_era5land(ee, site: Site, cache: Path, refresh: bool) -> pd.Timestamp:
    """Fetch the ERA5-Land baseline and soil state if not already cached."""
    from download_era5land import era5land_end, fetch_site

    store = cache / "era5land"
    store.mkdir(parents=True, exist_ok=True)
    available = era5land_end(ee)
    end = min(site.end, available) if site.end is not None else available

    hourly = store / f"era5land_{site.code}_hourly.parquet"
    if hourly.exists() and not refresh:
        have = pd.read_parquet(hourly)
        if have.index.tz is None:
            have.index = have.index.tz_localize("UTC")
        if have.index.min() <= site.start and have.index.max() >= end - pd.Timedelta("1h"):
            print(f"  ERA5-Land   cached ({len(have)} hours)")
            return end

    print(f"  ERA5-Land   fetching {site.start.date()} .. {end.date()}")
    frames = fetch_site(
        ee, site.latitude, site.longitude,
        str(site.start.date()), str((end + pd.Timedelta(days=1)).date()),
    )
    for kind, frame in frames.items():
        frame.to_parquet(store / f"era5land_{site.code}_{kind}.parquet")
    return end


def ensure_lai(ee, site: Site, cache: Path, refresh: bool) -> None:
    from download_lai import composite, fetch_raw, smooth

    store = cache / "lai"
    store.mkdir(parents=True, exist_ok=True)
    path = store / f"lai_{site.code}_daily.parquet"
    if path.exists() and not refresh:
        print(f"  LAI         cached")
        return

    end = site.end or pd.Timestamp.now(tz="UTC")
    raw = fetch_raw(ee, site.latitude, site.longitude,
                    str(site.start.date()), str((end + pd.Timedelta(days=8)).date()))
    series, stats = composite(raw)
    daily = smooth(series)
    daily.rename("LAI").to_frame().to_parquet(path)
    print(f"  LAI         {stats['dates']} composites, "
          f"{stats['pixels_per_date']} pixels each, "
          f"mean {daily.mean():.2f}")


def ensure_co2(cache: Path, refresh: bool) -> None:
    from download_co2 import DEFAULT_START, download, parse_daily, to_30min

    path = cache / "co2_maunaloa_30min.csv"
    if path.exists() and not refresh:
        print(f"  CO2         cached")
        return
    daily = parse_daily(download())
    series = to_30min(daily, DEFAULT_START)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# Mauna Loa daily mean CO2, NOAA GML, interpolated to 30 min.\n")
        series.to_csv(handle, index_label="time")
    print(f"  CO2         {series.iloc[-1]:.1f} ppm at {series.index[-1].date()}")


def ensure_statics(ee, site: Site) -> dict:
    """Fill any static the site did not already carry, by sampling GEE."""
    needed = [
        site.canopy_height_m is None,
        site.elevation_m is None,
        site.igbp_short is None,
    ]
    if not any(needed):
        print("  statics     from sites.yaml")
        return {
            "elevation": site.elevation_m,
            "canopy_height": site.canopy_height_m,
            "igbp_short": site.igbp_short,
            "igbp_long": site.igbp_long,
        }

    from download_statics import statics_for

    sampled = statics_for(ee, site.latitude, site.longitude)
    cap = CANOPY_FRACTION_OF_MAST * site.reference_height_m
    canopy = min(float(sampled["canopy_height_50m"]), cap)
    print(f"  statics     elevation {sampled['elevation']:.1f} m, "
          f"canopy {sampled['canopy_height_50m']:.1f} m capped to {canopy:.2f} m "
          f"({CANOPY_FRACTION_OF_MAST:g} x {site.reference_height_m:g} m mast), "
          f"IGBP {sampled['igbp_short']}")
    return {
        "elevation": site.elevation_m
        if site.elevation_m is not None
        else float(sampled["elevation"]),
        "canopy_height": site.canopy_height_m
        if site.canopy_height_m is not None
        else canopy,
        "igbp_short": site.igbp_short or sampled["igbp_short"],
        "igbp_long": site.igbp_long or sampled["igbp_long"],
    }


def build_site(ee, site: Site, out: Path, cache: Path, refresh: bool) -> dict:
    """Fetch what is missing, then write the two files."""
    from build_forcing import load_co2, load_lai
    from wunder import era5land, export, gapfill

    print(f"\n{site.code}  {site.label}".rstrip())
    print(f"  ({site.latitude:.5f}, {site.longitude:.5f})  mast {site.reference_height_m:g} m")

    end = ensure_era5land(ee, site, cache, refresh)
    ensure_lai(ee, site, cache, refresh)
    ensure_co2(cache, refresh)
    statics = ensure_statics(ee, site)

    last = end.floor("D") - pd.Timedelta(export.FREQ)
    forcing, qc, report = gapfill.build(
        site.code, site.start, last, store=cache / "era5land",
        met_logger=site.logger_override,
    )
    index = forcing.index
    forcing["CO2air"] = load_co2(index, cache / "co2_maunaloa_30min.csv")
    forcing["LAI"] = load_lai(site.code, index, cache / "lai")
    qc["CO2air"] = export.QC_ERA5

    ds = export.build_dataset(
        forcing, qc,
        {
            "latitude": site.latitude,
            "longitude": site.longitude,
            "reference_height": site.reference_height_m,
            "canopy_height": statics["canopy_height"],
            "elevation": statics["elevation"],
            "IGBP_veg_short": statics["igbp_short"],
            "IGBP_veg_long": statics["igbp_long"],
        },
        _attributes(site, report, statics),
    )

    years = f"{index[0].year}-{index[-1].year}"
    forcing_path = out / "forcing" / f"FLX_{site.code}_FLUXNET2015_FULLSET_{years}.nc"
    export.write_forcing(ds, forcing_path, start=index[0])

    values = era5land.initial_condition(site.code, index[0], store=cache / "era5land")
    ic_path = out / "ic" / f"{site.code}_{index[0]:%Y-%m-%d}_InitialCondition.nc"
    export.write_initial_condition(
        values, site.latitude, site.longitude, index[0], ic_path,
        attrs={"source": "ERA5-Land via Google Earth Engine", "station": site.label},
    )

    print(f"  built       {len(index)} steps, {index[0]:%Y-%m-%d} .. {index[-1]:%Y-%m-%d}")
    for name, levels in sorted(report.qc_fractions.items()):
        measured = levels.get(export.QC_MEASURED, 0.0)
        print(f"                {name:8s} {measured * 100:6.2f}% measured")
    print(f"  -> {forcing_path}")
    print(f"  -> {ic_path}")
    return {"forcing": forcing_path, "ic": ic_path, "report": report,
            "data": forcing, "qc": qc, "code": site.code, "label": site.label}


def _attributes(site: Site, report, statics: dict) -> dict:
    from wunder import export

    return {
        "title": f"STEMMUS_SCOPE forcing for {site.code}",
        "station": site.label or site.code,
        "met_logger": site.met_logger or "none — ERA5-Land only",
        "history": (
            f"built {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by "
            "forcing/make_input.py"
        ),
        "source": (
            "WUNDER ZENTRA logger network (MajiSys API); ERA5-Land via Google "
            "Earth Engine; MODIS MCD15A3H LAI; NOAA GML Mauna Loa CO2; "
            "Copernicus GLO-30 elevation; ETH canopy height; MODIS MCD12Q1 IGBP"
        ),
        "qc_convention": "; ".join(
            f"{k} = {v}" for k, v in sorted(export.QC_MEANING.items())
        ),
        "qc_fractions": "; ".join(
            f"{name}: " + " ".join(f"qc{lvl}={frac:.4f}" for lvl, frac in sorted(levels.items()))
            for name, levels in sorted(report.qc_fractions.items())
        ),
        "bias_fits": "; ".join(f.describe() for f in report.fits.values()) or "none",
        "timezone": "UTC (loggers report Europe/Amsterdam wall-clock, converted)",
        "canopy_height_note": (
            f"{statics['canopy_height']:.2f} m, capped at "
            f"{CANOPY_FRACTION_OF_MAST:g} x the {site.reference_height_m:g} m "
            "measurement height so the reference level stays above the canopy"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    where = parser.add_argument_group("which site")
    where.add_argument("--site", action="append", dest="sites",
                       help="station code from sites.yaml; repeatable")
    where.add_argument("--all", action="store_true", help="every station in sites.yaml")
    where.add_argument("--code", help="code for a site not in sites.yaml")
    where.add_argument("--lat", type=float, help="latitude, with --code")
    where.add_argument("--lon", type=float, help="longitude, with --code")
    where.add_argument("--logger", metavar="REF",
                       help="use this weather station instead of the one sites.yaml "
                            "names for the site — a serial (z6-21176) or a device "
                            "name (F1_1_ATMOS_SMST1). Pass an empty string to build "
                            "from ERA5-Land alone.")

    when = parser.add_argument_group("which dates")
    when.add_argument("--start", help="YYYY-MM-DD (default: the site's run_start)")
    when.add_argument("--end", help="YYYY-MM-DD (default: as far as ERA5-Land reaches)")

    how = parser.add_argument_group("options")
    how.add_argument("--out", type=Path, default=OUT_DIR,
                     help="where the two .nc files are written "
                          "(default: model_input/ in the repo)")
    how.add_argument("--cache", type=Path, default=CACHE_DIR,
                     help=f"where downloads are cached (default {CACHE_DIR})")
    how.add_argument("--reference-height", type=float,
                     help="measurement height in m; sets the canopy cap")
    how.add_argument("--refresh", action="store_true",
                     help="re-download instead of using the cache")
    how.add_argument("--no-verify", action="store_true",
                     help="skip the checks that normally run afterwards")
    how.add_argument("--plot", action="store_true",
                     help="also write a one-page figure of every forcing variable")
    how.add_argument("--plot-dir", type=Path, default=None,
                     help="where those figures go (default: plots/ in the repo)")
    args = parser.parse_args()

    from download_era5land import _initialise

    sites = resolve(args)
    ee = _initialise()

    built, failed = [], []
    for site in sites:
        try:
            result = build_site(ee, site, args.out, args.cache, args.refresh)
            if args.plot:
                from plot_forcing import plot_forcing

                from plot_forcing import PLOT_DIR

                figure = plot_forcing(
                    result["data"], result["qc"], result["code"],
                    result["label"], args.plot_dir or PLOT_DIR,
                )
                print(f"  -> {figure}")
            built.append(result)
        except Exception as exc:  # noqa: BLE001 - report and carry on to the next
            failed.append(site.code)
            print(f"  FAILED — {type(exc).__name__}: {exc}")

    if built and not args.no_verify:
        print(f"\n{'=' * 70}\nverifying")
        from verify import verify_forcing, verify_ic

        ok = all(verify_forcing(r["forcing"]) for r in built)
        ok &= all(verify_ic(r["ic"]) for r in built)
        print("\n" + ("VERIFIED" if ok else "CHECKS FAILED"))
        if not ok:
            return 1

    if failed:
        print(f"\nfailed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
