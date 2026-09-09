"""Pull the SoilGrids-derived soil hydraulics out of a run's `soil_parameters.mat`.

PyStemmusScope builds that file from SoilGrids texture through the Schaap/Rosetta
pedotransfer functions, on the standard SoilGrids depth boundaries
0, 5, 15, 30, 60, 100, 200 cm. It is the soil the model actually runs on, so it is
also the soil the water stress factor should use -- otherwise `Ks * ET0` and the
model's own transpiration rest on two different soils and any disagreement between
them is uninterpretable.

The `.mat` is MATLAB v7.3, i.e. HDF5, so reading it needs `h5py`, which the light
`wunder` environment does not have. This script runs under `geo` and caches the
numbers as JSON in `model_input/soil/`, which `wunder.stress` then reads with the
standard library alone.

    conda activate geo
    python forcing/extract_soil.py --site NL-Gl1

Re-run it whenever the forcing input is rebuilt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
OUT_DIR = REPO / "model_input" / "soil"

#: SoilGrids layer boundaries [cm] that PyStemmusScope selects, from its
#: `soil_io.py`: valid depths are 0, 5, 15, 30, 60, 100 and 200.
BOUNDARIES = [0, 5, 15, 30, 60, 100, 200]

#: Variables worth keeping. The first four are the van Genuchten parameters the
#: retention curve needs; the rest are context for judging them.
WANTED = [
    "Coefficient_Alpha",   # van Genuchten alpha [cm-1]
    "Coefficient_n",       # van Genuchten n [-]
    "SaturatedMC",         # theta_s [m3 m-3]
    "ResidualMC",          # theta_r [m3 m-3]
    "fieldMC",             # the model's own field capacity [m3 m-3]
    "porosity",
    "SaturatedK",
    "FOS",                 # fraction of sand
    "FOC",                 # fraction of clay
    "MSOC",                # soil organic carbon
]


def find_mat(code: str) -> Path:
    """The most recent `soil_parameters.mat` written for this site."""
    matches = sorted(RUNS.glob(f"{code}/matlab/input/*/soil_parameters.mat"))
    if not matches:
        raise SystemExit(
            f"no soil_parameters.mat for {code}. It is written when the model input is "
            f"prepared:\n    python model/run_matlab.py --site {code} --days 1"
        )
    return matches[-1]


def extract(code: str) -> dict:
    import h5py

    path = find_mat(code)
    with h5py.File(path, "r") as handle:
        data = {}
        for key in WANTED:
            if key not in handle:
                continue
            values = np.atleast_1d(np.array(handle[key]).squeeze())
            data[key] = [float(v) for v in values]

    n = len(data["Coefficient_Alpha"])
    layers = [
        {
            "top_cm": BOUNDARIES[i],
            "bottom_cm": BOUNDARIES[i + 1],
            "midpoint_cm": (BOUNDARIES[i] + BOUNDARIES[i + 1]) / 2.0,
            **{key: data[key][i] for key in data if len(data[key]) == n},
        }
        for i in range(n)
    ]
    return {
        "site": code,
        "source": str(path.relative_to(REPO)),
        "provenance": ("SoilGrids texture through Schaap/Rosetta pedotransfer, as "
                       "assembled by PyStemmusScope.soil_io"),
        "units": {"Coefficient_Alpha": "cm-1", "Coefficient_n": "-",
                  "SaturatedMC": "m3 m-3", "ResidualMC": "m3 m-3",
                  "fieldMC": "m3 m-3", "SaturatedK": "cm s-1"},
        "layers": layers,
        "scalars": {k: v[0] for k, v in data.items() if len(v) == 1},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", action="append", dest="sites", required=True)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for code in args.sites:
        payload = extract(code)
        out = OUT_DIR / f"{code}_soil.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"{code}: {len(payload['layers'])} layers -> {out.relative_to(REPO)}")
        for layer in payload["layers"]:
            print(f"   {layer['top_cm']:>3}-{layer['bottom_cm']:<3} cm  "
                  f"alpha {layer['Coefficient_Alpha']:.4f} cm-1  "
                  f"n {layer['Coefficient_n']:.3f}  "
                  f"theta_s {layer['SaturatedMC']:.3f}  "
                  f"theta_r {layer['ResidualMC']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
