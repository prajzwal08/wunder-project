"""Soil water stress: how much of the atmospheric demand the soil can actually meet.

`wunder.et` gives ET0, the demand a well-watered sward would meet. This module gives
the factor that closes the gap to what a *real*, sometimes-dry soil supplies:

    ET_actual = Ks * Kc * ET0

`Ks` is the FAO-56 water stress coefficient. It is 1 while the profile still holds
readily available water and falls linearly to 0 at the wilting point:

    TAW  = theta_fc - theta_wp          total available water
    RAW  = p * TAW                      readily available -- taken without stress
    Ks   = 1                                             while theta >= theta_fc - RAW
    Ks   = (theta - theta_wp) / ((1 - p) * TAW)          below that, clipped to [0, 1]

**theta_fc and theta_wp come from the model's own soil.** They are read off the van
Genuchten curve that STEMMUS_SCOPE runs on -- SoilGrids texture through the
Schaap/Rosetta pedotransfer, assembled by PyStemmusScope and cached as JSON by
`forcing/extract_soil.py`:

    theta(h) = theta_r + (theta_s - theta_r) / (1 + (alpha*h)^n)^(1 - 1/n)
    theta_fc = theta(336 cm)     = theta at -33 kPa
    theta_wp = theta(15300 cm)   = theta at -1500 kPa

Using the model's soil rather than an independent one is deliberate: `Ks * ET0` and
the model's own transpiration then rest on the same soil, so a disagreement between
them means something. Verified against the file rather than assumed -- STEMMUS_SCOPE
carries its own `fieldMC`, and theta evaluated at -33 kPa reproduces it to 0.001 at
every layer, which pins its field-capacity convention to -33 kPa.

**The profile stops at 100 cm.** SoilGrids supplies a 100-200 cm layer, but no sensor
in this network goes below 80 cm and nothing that deep is root zone here.

The alternative source -- fitting retention curves to the three loggers that pair a
TEROS21 tensiometer with a TEROS12 moisture probe -- is kept in `retention_curve` and
`observed_limits` as an *independent check*, not as the primary path. Those fits give
usable theta_fc and theta_wp but unusable van Genuchten parameters, because alpha is
set by the air-entry region below 9 kPa which the TEROS21 cannot measure. See
`soil_retention_notes.txt`.

    import wunder as w
    ks, info = w.stress.root_zone_stress(w.fetch("F1_3_SMST3"), site="NL-Gl1")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metadata import forcing_sites, measures
from .process import depth_columns, root_zone

#: Where `forcing/extract_soil.py` caches the parameters.
SOIL_DIR = Path(__file__).resolve().parent.parent / "model_input" / "soil"

#: Suction defining field capacity [kPa]. -33 is what STEMMUS_SCOPE itself uses --
#: confirmed by reproducing its `fieldMC` to 0.001 at every layer. FAO-56 allows
#: -10 kPa for coarse sandy soils, which would roughly double the available water;
#: it is left as a parameter, but changing it puts Ks on a different soil definition
#: from the model, so do it knowingly.
FIELD_CAPACITY_KPA = -33.0
#: Suction defining the permanent wilting point [kPa]. FAO-56's 15 bar.
WILTING_KPA = -1500.0
#: 1 kPa of suction in cm of water, since van Genuchten alpha is in cm-1.
CM_PER_KPA = 10.197

#: Nothing in this network measures below 80 cm, and SoilGrids' 100-200 cm layer is
#: not root zone here.
MAX_DEPTH_CM = 100.0

#: FAO-56 Table 22 depletion fraction: the share of available water a plant takes
#: freely, so `1 - p` is the share over which stress develops. 0.5 is the value for
#: deciduous trees and orchards, the closest listed analogue to a food forest.
DEFAULT_P = 0.5


# --- the model's soil ------------------------------------------------------


def van_genuchten(head_cm, theta_r, theta_s, alpha, n):
    """Water content at a suction head [cm], van Genuchten with m = 1 - 1/n.

    The Mualem restriction, which is what gives the closed form for unsaturated
    conductivity and is the form STEMMUS_SCOPE uses.
    """
    return theta_r + (theta_s - theta_r) / (1.0 + (alpha * head_cm) ** n) ** (1.0 - 1.0 / n)


def site_of(ref: str) -> str:
    """Forcing site code for a logger name or serial, e.g. 'F1_3_SMST3' -> 'NL-Gl1'."""
    from .metadata import logger as _logger

    serial = _logger(ref).serial
    for code, entry in forcing_sites().items():
        if serial in entry.get("soil_loggers", []):
            return code
    raise ValueError(f"{ref} belongs to no forcing site in sites.yaml")


def soil_layers(site: str, *, max_depth_cm: float = MAX_DEPTH_CM) -> pd.DataFrame:
    """The site's SoilGrids layers, down to `max_depth_cm`.

    Raises with the command to run if the site has not been extracted yet.
    """
    path = SOIL_DIR / f"{site}_soil.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no soil parameters for {site}. Build them under the `geo` environment:\n"
            f"    python forcing/extract_soil.py --site {site}\n"
            "which needs the model input to have been prepared for that site first."
        )
    payload = json.loads(path.read_text())
    frame = pd.DataFrame(payload["layers"])
    frame = frame[frame["top_cm"] < max_depth_cm].reset_index(drop=True)
    frame.attrs["provenance"] = payload.get("provenance", "")
    frame.attrs["source"] = payload.get("source", "")
    return frame


def layer_limits(site: str, *, field_capacity_kpa: float = FIELD_CAPACITY_KPA,
                 wilting_kpa: float = WILTING_KPA, **kw) -> pd.DataFrame:
    """theta_fc, theta_wp and TAW per SoilGrids layer, from its van Genuchten curve."""
    layers = soil_layers(site, **kw)
    head_fc = abs(field_capacity_kpa) * CM_PER_KPA
    head_wp = abs(wilting_kpa) * CM_PER_KPA
    args = (layers["ResidualMC"], layers["SaturatedMC"],
            layers["Coefficient_Alpha"], layers["Coefficient_n"])
    out = layers[["top_cm", "bottom_cm", "midpoint_cm"]].copy()
    out["theta_s"] = layers["SaturatedMC"]
    out["theta_r"] = layers["ResidualMC"]
    out["theta_fc"] = van_genuchten(head_fc, *args)
    out["theta_wp"] = van_genuchten(head_wp, *args)
    out["taw"] = out["theta_fc"] - out["theta_wp"]
    out.attrs.update(layers.attrs)
    return out


def water_limits(site: str, depths: list[str], **kw) -> dict[str, dict]:
    """`{depth: limits}` for sensor depths, interpolated from the layer midpoints.

    The sensors sit at 2.5, 5, 10, 20, 40 and 80 cm; the SoilGrids layers are
    0-5, 5-15, 15-30, 30-60 and 60-100, whose midpoints are 2.5, 10, 22.5, 45 and
    80. Interpolating in depth between midpoints is smoother than snapping each
    sensor into whichever layer contains it, and it happens to be exact at 2.5 and
    80 cm. Outside the midpoint range the nearest value is held flat rather than
    extrapolated.
    """
    layers = layer_limits(site, **kw)
    mid = layers["midpoint_cm"].to_numpy(dtype="float64")
    out = {}
    for depth in depths:
        z = float(depth)
        theta_fc = float(np.interp(z, mid, layers["theta_fc"].to_numpy()))
        theta_wp = float(np.interp(z, mid, layers["theta_wp"].to_numpy()))
        out[depth] = {
            "depth": depth,
            "theta_fc": theta_fc,
            "theta_wp": theta_wp,
            "taw": theta_fc - theta_wp,
            "source": "soilgrids",
            "site": site,
        }
    return out


# --- the stress factor -----------------------------------------------------


def stress_factor(theta, theta_fc: float, theta_wp: float, p: float = DEFAULT_P):
    """FAO-56 water stress coefficient Ks, in [0, 1].

    1 while the soil still holds readily available water, then a straight line down
    to 0 at the wilting point.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p is a fraction of available water, got {p}")
    taw = theta_fc - theta_wp
    if taw <= 0:
        raise ValueError(
            f"field capacity {theta_fc:.3f} is not above wilting point {theta_wp:.3f}"
        )
    ks = (theta - theta_wp) / ((1.0 - p) * taw)
    return ks.clip(0.0, 1.0) if hasattr(ks, "clip") else float(np.clip(ks, 0.0, 1.0))


def adjust_p(p: float, etc_mm_day) -> float:
    """FAO-56's demand correction to the depletion fraction (eq. 83).

    A plant stresses sooner on a 6 mm/day than on a 2 mm/day, because the roots
    cannot keep up even from moist soil. Clipped to the range the table covers.
    """
    return float(np.clip(p + 0.04 * (5.0 - np.nanmean(np.asarray(etc_mm_day,
                                                                dtype="float64"))),
                         0.1, 0.8))


def _depth_of(column: str) -> str:
    return column.split()[-1].removesuffix("cm")


def root_zone_limits(
    df: pd.DataFrame,
    *,
    site: str | None = None,
    ref: str | None = None,
    limits: dict[str, dict] | None = None,
    p: float = DEFAULT_P,
    method: str = "trapezoid",
    min_coverage: float = 0.9,
    max_depth_cm: float = MAX_DEPTH_CM,
) -> dict:
    """This profile's own theta_fc, theta_wp and stress threshold [m3 m-3].

    The three numbers a root-zone moisture series should be read against, on the same
    weighting as the series itself. `{}` when the frame carries no usable moisture.

    Separated from `root_zone_stress` because a figure often wants only the limits --
    two horizontal lines behind a moisture curve -- and computing Ks over a
    quarter-million rows to get them would be waste.
    """
    if site is None:
        if ref is None:
            raise ValueError("give site= or ref= so the soil can be looked up")
        site = site_of(ref)

    cols = depth_columns(df, "moisture")
    if cols and min_coverage:
        keep = [c for c in cols if df[c].notna().mean() >= min_coverage]
        cols = keep or cols
    # Nothing below the profile the soil file describes.
    cols = [c for c in cols if float(_depth_of(c)) <= max_depth_cm]
    if not cols:
        return {}

    if limits is None:
        limits = water_limits(site, [_depth_of(c) for c in cols],
                              max_depth_cm=max_depth_cm)

    bounds = pd.DataFrame(
        [[limits[_depth_of(c)]["theta_fc"] for c in cols],
         [limits[_depth_of(c)]["theta_wp"] for c in cols]],
        columns=cols,
    )
    weighted = root_zone(bounds, "moisture", method=method, columns=cols)
    theta_fc, theta_wp = float(weighted.iloc[0]), float(weighted.iloc[1])
    return {
        "site": site,
        "columns": cols,
        "depths": [_depth_of(c) for c in cols],
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "taw": theta_fc - theta_wp,
        "threshold": theta_fc - p * (theta_fc - theta_wp),
        "p": p,
        "method": method,
        "source": "soilgrids",
        "limits": {_depth_of(c): limits[_depth_of(c)] for c in cols},
    }


def root_zone_stress(
    df: pd.DataFrame,
    *,
    p: float = DEFAULT_P,
    method: str = "trapezoid",
    **kw,
) -> tuple[pd.Series, dict]:
    """Root-zone Ks through time, with the provenance of the numbers behind it.

    Give either `site` (a forcing code like 'NL-Gl1') or `ref` (a logger name or
    serial, from which the site is looked up).

    The profile's theta_fc and theta_wp are depth-weighted with *the same* weights,
    columns and method as the soil moisture itself -- by running the limits back
    through `process.root_zone` rather than re-deriving a weighting here. That is
    what keeps "Ks = 1" meaning *this* profile is at field capacity, rather than
    some other average of some other set of depths.
    """
    info = root_zone_limits(df, p=p, method=method, **kw)
    if not info:
        return pd.Series(dtype="float64", name="Ks"), {"reason": "no soil moisture"}

    theta = root_zone(df, "moisture", method=method, columns=info["columns"])
    ks = stress_factor(theta, info["theta_fc"], info["theta_wp"], p).rename("Ks")
    return ks.dropna(), info


def actual_et(
    df: pd.DataFrame,
    met: pd.DataFrame | None = None,
    *,
    crop_coefficient: float = 1.0,
    p: float = DEFAULT_P,
    adjust_for_demand: bool = False,
    **kw,
) -> pd.DataFrame:
    """Daily ET0, Ks and the water-limited ET they imply [mm d-1].

    Supply and demand come from different instruments and that is the point: the
    soil moisture is this logger's own, while ET0 needs a weather station, and
    eleven of the fourteen loggers have none. Pass `ref` and the field's ATMOS-41 is
    found and fetched automatically -- the same `met_source` rule the water-potential
    figure already uses for VPD, so a soil logger and its station are paired the same
    way everywhere in the package. Pass `met` explicitly to override, or to avoid the
    fetch when the frame is already in hand.

    `crop_coefficient` (Kc) defaults to 1, so this reports `Ks * ET0` and makes no
    claim about how a food forest's canopy differs from the reference grass. Kc for
    this vegetation is genuinely unknown, and inventing one would bury a guess inside
    a number that otherwise rests on measurements.
    """
    from .et import reference_et

    station = None
    if met is None:
        met, station = _weather_for(df, kw.get("ref"))
    et0 = reference_et(met if met is not None else df)
    if et0.empty:
        return pd.DataFrame(columns=["et0", "ks", "et"])

    if adjust_for_demand:
        p = adjust_p(p, et0 * crop_coefficient)
    ks, info = root_zone_stress(df, p=p, **kw)
    if ks.empty:
        return pd.DataFrame(columns=["et0", "ks", "et"])

    out = pd.DataFrame({"et0": et0, "ks": ks.resample("1D").mean()}).dropna()
    out["et"] = out["ks"] * crop_coefficient * out["et0"]
    out.attrs.update(info)
    out.attrs["crop_coefficient"] = crop_coefficient
    out.attrs["station"] = station
    return out


def _weather_for(df: pd.DataFrame, ref: str | None):
    """(met frame, station name) for a soil logger -- its own, or its field's ATMOS-41.

    Returns the frame unchanged when the logger carries its own weather sensors, so
    the four ATMOS-41 loggers never trigger a second fetch.
    """
    from .et import RADIATION
    from .fetch import fetch as _fetch
    from .metadata import met_source

    if RADIATION in df.columns and df[RADIATION].notna().any():
        return df, "own"
    if ref is None:
        return None, None
    station = met_source(ref)
    if station is None:
        return None, None
    return _fetch(station.serial), station.name


# --- independent check: the in-situ retention curves -----------------------
#
# Three loggers pair a TEROS21 tensiometer with a TEROS12 moisture probe at 5, 20
# and 40 cm. Fitting those gives theta_fc and theta_wp measured on site, which is a
# genuine check on the SoilGrids values above. It is NOT the primary source: the two
# probes sit in separate holes, and the van Genuchten parameters that come out are
# unusable because alpha is fixed by the air-entry region below 9 kPa, which the
# TEROS21 cannot measure. See soil_retention_notes.txt.

#: TEROS21's specified range is -9 kPa and drier. Wetter than that it reports a
#: floor -- at F2_3 5 cm, -0.1 kPa is the single most common value in the record,
#: 12.5% of it. Those are end-stop readings, not measurements.
VALID_KPA = 9.0
#: Log-spaced suction bins [kPa, positive] for the non-parametric curve.
PSI_BINS = (0.0, 1.0, 3.0, 10.0, 33.0, 100.0, 330.0, 1000.0, 1500.0, 1e5)
MIN_BIN = 50


def _pairs(df: pd.DataFrame, depth: str, freq: str = "1h") -> pd.DataFrame:
    """Hourly (suction, water content) pairs at one depth."""
    psi_col = f"{measures()['matric_potential']} {depth}cm"
    theta_col = f"{measures()['moisture']} {depth}cm"
    if psi_col not in df.columns or theta_col not in df.columns:
        return pd.DataFrame(columns=["psi", "theta"])
    pair = df[[psi_col, theta_col]].resample(freq).mean().dropna()
    pair.columns = ["psi", "theta"]
    return pair[(pair["psi"] < 0) & (pair["theta"] > 0.01)]


def retention_curve(df: pd.DataFrame, depth: str, *, freq: str = "1h",
                    bins: tuple[float, ...] = PSI_BINS,
                    min_bin: int = MIN_BIN) -> pd.DataFrame:
    """Observed retention at one depth: median theta per suction bin.

    Columns `psi_kpa` (the bin's own median, not its midpoint), `theta`, `n`, and
    the 10th/90th percentiles, which are the honest width of the relation.
    Non-parametric: a van Genuchten fit over four decades of suction gave alpha
    against its bound in a third of cases, for the reason in the module docstring.
    """
    pair = _pairs(df, depth, freq)
    if pair.empty:
        return pd.DataFrame(columns=["psi_kpa", "theta", "theta_lo", "theta_hi", "n"])
    grouped = pair.groupby(pd.cut(-pair["psi"], bins), observed=True)
    out = grouped.agg(
        psi_kpa=("psi", lambda s: float(-s.median())),
        theta=("theta", "median"),
        theta_lo=("theta", lambda s: float(s.quantile(0.10))),
        theta_hi=("theta", lambda s: float(s.quantile(0.90))),
        n=("theta", "size"),
    )
    return out[out["n"] >= min_bin].sort_values("psi_kpa").reset_index(drop=True)


def observed_limits(df: pd.DataFrame, depth: str, *,
                    field_capacity_kpa: float = FIELD_CAPACITY_KPA,
                    wilting_kpa: float = WILTING_KPA, **kw) -> dict | None:
    """theta_fc and theta_wp measured on site, or None if the record cannot give them.

    None when fewer than four suction bins survive, or when the observed suctions do
    not bracket both targets -- F2_3 has one hour drier than -1500 kPa at 20 cm and
    none at 40 cm, so its wilting point is not observed and is not invented here.
    """
    curve = retention_curve(df, depth, **kw)
    if len(curve) < 4:
        return None
    x = np.log(curve["psi_kpa"].to_numpy())
    y = curve["theta"].to_numpy()
    targets = (abs(field_capacity_kpa), abs(wilting_kpa))
    if not all(x.min() <= np.log(t) <= x.max() for t in targets):
        return None
    theta_fc, theta_wp = (float(np.interp(np.log(t), x, y)) for t in targets)
    return {
        "depth": depth,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "taw": theta_fc - theta_wp,
        "source": "observed",
        "bins": int(len(curve)),
        "n": int(curve["n"].sum()),
    }


def compare_limits(df: pd.DataFrame, site: str, depths: list[str] | None = None,
                   **kw) -> pd.DataFrame:
    """SoilGrids limits beside the observed ones, for the depths that have both."""
    if depths is None:
        depths = [_depth_of(c) for c in depth_columns(df, "matric_potential")]
    model = water_limits(site, depths, **kw)
    rows = []
    for depth in depths:
        obs = observed_limits(df, depth)
        row = {"depth": depth,
               "model_fc": round(model[depth]["theta_fc"], 3),
               "model_wp": round(model[depth]["theta_wp"], 3),
               "model_taw": round(model[depth]["taw"], 3)}
        if obs:
            row |= {"obs_fc": round(obs["theta_fc"], 3),
                    "obs_wp": round(obs["theta_wp"], 3),
                    "obs_taw": round(obs["taw"], 3),
                    "d_fc": round(obs["theta_fc"] - model[depth]["theta_fc"], 3),
                    "d_wp": round(obs["theta_wp"] - model[depth]["theta_wp"], 3)}
        rows.append(row)
    return pd.DataFrame(rows)
