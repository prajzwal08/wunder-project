"""Sample the per-station static fields, from Google Earth Engine.

PyStemmusScope reads `latitude`, `longitude`, `elevation`, `reference_height`,
`canopy_height` and `IGBP_veg_long` out of the forcing file with no guard
(`forcing_io.py` lines 96-103), so a missing one is a `KeyError` while the MATLAB
path builds `forcing_globals.mat`. The first two come from sites.yaml and
`reference_height` is the ATMOS-41 mast height; this script supplies the rest.

Three sources, all point samples:

  elevation      COPERNICUS/DEM/GLO30, 30 m. Not the loggers' own GPS altitude,
                 which scatters 36-42 m across one Glanerbeek field and 11-20 m
                 at Wenumseveld -- a spread larger than the real relief.
  canopy_height  ETH Global Canopy Height 2020, 10 m. Reported at the point and
                 as a 50 m mean, because one 10 m pixel on a mast in a clearing
                 is not the stand.
  IGBP           MODIS MCD12Q1 LC_Type1, 500 m -- which *is* the IGBP scheme, so
                 no translation from another classification is needed.

None of these is written to sites.yaml automatically. They are judgement calls at
these sites: a 500 m land-cover pixel over a one-hectare food forest is mostly
the surrounding farmland, and a voedselbos planted in the 2010s reads low in a
2020 canopy product. The script prints what the data says and leaves a person to
decide.

Usage:
    python forcing/download_statics.py [--site NL-Gl1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEM = "COPERNICUS/DEM/GLO30"
CANOPY = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"
CANOPY_SD = "users/nlang/ETH_GlobalCanopyHeightSD_2020_10m_v1"
LANDCOVER = "MODIS/061/MCD12Q1"

#: MCD12Q1 LC_Type1 is the IGBP scheme. Short names follow the FLUXNET usage that
#: `IGBP_veg_short` carries in a PLUMBER2 file.
IGBP = {
    1: ("ENF", "Evergreen Needleleaf Forests"),
    2: ("EBF", "Evergreen Broadleaf Forests"),
    3: ("DNF", "Deciduous Needleleaf Forests"),
    4: ("DBF", "Deciduous Broadleaf Forests"),
    5: ("MF", "Mixed Forests"),
    6: ("CSH", "Closed Shrublands"),
    7: ("OSH", "Open Shrublands"),
    8: ("WSA", "Woody Savannas"),
    9: ("SAV", "Savannas"),
    10: ("GRA", "Grasslands"),
    11: ("WET", "Permanent Wetlands"),
    12: ("CRO", "Croplands"),
    13: ("URB", "Urban and Built-up Lands"),
    14: ("CVM", "Cropland/Natural Vegetation Mosaics"),
    15: ("SNO", "Permanent Snow and Ice"),
    16: ("BSV", "Barren"),
    17: ("WAT", "Water Bodies"),
}


def _initialise():
    import ee

    try:
        ee.Initialize()
    except Exception:  # noqa: BLE001 - fall back to the project in the credentials
        import json

        project = json.loads(
            (Path.home() / ".config" / "earthengine" / "credentials").read_text()
        ).get("project")
        if not project:
            raise
        ee.Initialize(project=project)
    return ee


def _reduce(ee, image, point, scale, reducer=None, buffer_m: float | None = None):
    region = point.buffer(buffer_m) if buffer_m else point
    reducer = reducer or ee.Reducer.first()
    result = image.reduceRegion(reducer=reducer, geometry=region, scale=scale)
    return result.getInfo()


def statics_for(ee, lat: float, lon: float) -> dict:
    """Elevation, canopy height and IGBP class at one point."""
    point = ee.Geometry.Point(lon, lat)
    out: dict = {}

    dem = ee.ImageCollection(DEM).select("DEM").mosaic()
    out["elevation"] = _reduce(ee, dem, point, 30).get("DEM")

    canopy = ee.Image(CANOPY).select("b1").rename("canopy_height")
    out["canopy_height_point"] = _reduce(ee, canopy, point, 10).get("canopy_height")
    out["canopy_height_50m"] = _reduce(
        ee, canopy, point, 10, reducer=ee.Reducer.mean(), buffer_m=50
    ).get("canopy_height")
    sd = ee.Image(CANOPY_SD).select("b1").rename("sd")
    out["canopy_height_sd"] = _reduce(ee, sd, point, 10).get("sd")

    # Most recent land-cover year available.
    cover = ee.ImageCollection(LANDCOVER).select("LC_Type1")
    latest = ee.Image(cover.sort("system:time_start", False).first())
    out["igbp_year"] = ee.Date(latest.get("system:time_start")).get("year").getInfo()
    code = _reduce(ee, latest, point, 500).get("LC_Type1")
    out["igbp_code"] = code
    short, long = IGBP.get(int(code), ("?", "unknown")) if code is not None else ("?", "unknown")
    out["igbp_short"], out["igbp_long"] = short, long

    # What the neighbourhood looks like, since a 500 m pixel over a small plot is
    # mostly its surroundings.
    hist = _reduce(
        ee, latest, point, 500, reducer=ee.Reducer.frequencyHistogram(), buffer_m=1000
    ).get("LC_Type1") or {}
    total = sum(hist.values()) or 1
    out["igbp_neighbourhood"] = {
        IGBP.get(int(k), ("?", "?"))[0]: round(v / total, 3)
        for k, v in sorted(hist.items(), key=lambda kv: -kv[1])
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", action="append", dest="sites")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from wunder.metadata import forcing_sites

    ee = _initialise()
    for code in args.sites or list(forcing_sites()):
        cfg = forcing_sites()[code]
        values = statics_for(ee, cfg["latitude"], cfg["longitude"])
        print(f"\n{code}  {cfg.get('label', '')}")
        print(f"  ({cfg['latitude']:.5f}, {cfg['longitude']:.5f})")
        print(f"  elevation        {values['elevation']:.1f} m  "
              f"(logger GPS says {cfg['elevation_m']} m)")
        print(f"  canopy_height    {values['canopy_height_point']:.1f} m at the point, "
              f"{values['canopy_height_50m']:.1f} m mean over 50 m "
              f"(sd {values['canopy_height_sd']:.1f})")
        print(f"  IGBP {values['igbp_year']}       {values['igbp_short']} — "
              f"{values['igbp_long']}")
        print(f"  within 1 km      {values['igbp_neighbourhood']}")
    print("\nNothing written. These are judgement calls at these sites — put the "
          "values you accept into sites.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
