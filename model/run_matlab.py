"""Run the MATLAB STEMMUS_SCOPE on a station's forcing, through PyStemmusScope.

    python forcing/run_matlab.py --site NL-Gl1 --days 7

Uses the compiled model at ~/STEMMUSSCOPEexe/STEMMUS_SCOPE with the MATLAB
Runtime R2023a, so no MATLAB licence is needed.

The point of having this alongside `run_model.py` is that the two engines read
the *same* two files. Where they agree, the forcing is doing what it should.
Where they disagree, the difference is attributable to an implementation rather
than to the input -- which is exactly the question
`STEMMUSSCOPE_Python_Migration.md` exists to answer, and the reason the Python
port's near-zero canopy transpiration is worth checking against MATLAB before
anyone concludes anything about these sites.

`Location` is the site code rather than a lat/lon pair. PyStemmusScope matches it
against `[A-Z]{2}-([A-z]|\\d){3}` to choose site mode, then finds the forcing by
scanning `ForcingPath` for a filename *containing* the code -- so exactly one
generation of a station's file may sit there, and the initial condition must be
named `<CODE>*.nc`. Both hold for what `make_input.py` writes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

EXE = Path.home() / "STEMMUSSCOPEexe" / "STEMMUS_SCOPE"
FORCING_DIR = REPO / "model_input" / "forcing"
IC_DIR = REPO / "model_input" / "ic"
RUNS_DIR = REPO / "runs"

MODEL_INPUT = (Path.home() / "STEMMUS_SCOPE_model" / "STEMMUS_SCOPE_new"
               / "STEMMUS_SCOPE" / "input")
SOIL_PROPERTY = (Path.home() / "STEMMUS_SCOPE_model" / "STEMMUS_SCOPE_old"
                 / "STEMMUS_SCOPE" / "input" / "SoilProperty")

#: The compiled model was built against MATLAB Runtime **9.13 = R2022b**, and
#: says so only obliquely: pointed at the wrong one it reports "Could not find
#: version 9.13 of the MATLAB Runtime" and exits 255, which PyStemmusScope
#: surfaces as a bare CalledProcessError.
#:
#: This machine has three, and the one that matches is not in a
#: `MATLAB_Runtime/` directory at all -- it is inside a full R2022b install.
#: The shell profile points at R2023a (9.14), which is why this searches by the
#: library the executable actually asks for rather than by directory name.
REQUIRED_MCR_SO = "libmwmclmcrrt.so.9.13"

MCR_CANDIDATES = [
    Path("/opt/matlab/R2022b"),
    Path("/opt/matlab/MATLAB_Runtime/R2023a"),
    Path("/usr/local/MATLAB/MATLAB_Runtime/R2023a"),
    Path("/opt/matlab/MATLAB_Runtime/v910"),
]


def mcr_environment() -> dict:
    """`LD_LIBRARY_PATH` for the runtime the compiled model actually needs.

    Selected by looking for `REQUIRED_MCR_SO`, not by directory name: the
    matching runtime here lives inside a full MATLAB install, and the
    similarly-named `MATLAB_Runtime/R2023a` directories are the wrong version.
    """
    for root in MCR_CANDIDATES:
        if not (root / "runtime" / "glnxa64" / REQUIRED_MCR_SO).exists():
            continue
        parts = [root / "runtime" / "glnxa64", root / "bin" / "glnxa64",
                 root / "sys" / "os" / "glnxa64",
                 root / "extern" / "bin" / "glnxa64"]
        env = dict(os.environ)
        # Prepended, not appended: the shell profile already points at R2023a,
        # and the first matching soname on the path is the one that loads.
        env["LD_LIBRARY_PATH"] = ":".join(
            [str(p) for p in parts if p.exists()]
            + [env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(":")
        print(f"  runtime {root}")
        return env
    raise SystemExit(
        f"{REQUIRED_MCR_SO} (MATLAB Runtime R2022b) not found in "
        + ", ".join(map(str, MCR_CANDIDATES))
        + "\nThe compiled model at "
        + str(EXE) + " was built against it."
    )


def write_config(code: str, start: str, end: str, work_dir: Path) -> Path:
    """The `config_file.txt` PyStemmusScope reads, for one station."""
    work_dir.mkdir(parents=True, exist_ok=True)
    config = work_dir / "config_file.txt"
    config.write_text(
        f"WorkDir={work_dir}/\n"
        f"SoilPropertyPath={SOIL_PROPERTY}/\n"
        f"ForcingPath={FORCING_DIR}/\n"
        f"InitialConditionPath={IC_DIR}/\n"
        f"Location={code}\n"
        f"directional={MODEL_INPUT}/directional/\n"
        f"fluspect_parameters={MODEL_INPUT}/fluspect_parameters/\n"
        f"leafangles={MODEL_INPUT}/leafangles/\n"
        f"radiationdata={MODEL_INPUT}/radiationdata/\n"
        f"soil_spectrum={MODEL_INPUT}/soil_spectrum/\n"
        f"input_data={MODEL_INPUT}/input_data.xlsx\n"
        f"StartTime={start}\n"
        f"EndTime={end}\n"
        "InputPath=\n"
        "OutputPath=\n"
    )
    return config


def resolve_window(code: str, start, end, days) -> tuple[str, str]:
    """Clamp a requested window to what the forcing file actually covers.

    `_slice_forcing_file` raises rather than clipping if either end falls
    outside, so it is better to find that out here with a readable message.
    """
    import xarray as xr

    matches = sorted(FORCING_DIR.glob(f"FLX_{code}_*.nc"))
    if not matches:
        raise SystemExit(
            f"no forcing for {code}. Build it:\n"
            f"    python forcing/make_input.py --site {code}"
        )
    with xr.open_dataset(matches[0]) as ds:
        time = pd.DatetimeIndex(ds["time"].values)

    first = max(pd.Timestamp(start), time[0]) if start else time[0]
    if end:
        last = min(pd.Timestamp(end), time[-1])
    elif days:
        last = min(first + pd.Timedelta(days=days) - pd.Timedelta("30min"), time[-1])
    else:
        last = time[-1]
    if last <= first:
        raise SystemExit(f"empty window for {code}: {first} to {last}")
    return first.strftime("%Y-%m-%dT%H:%M"), last.strftime("%Y-%m-%dT%H:%M")


def run(code: str, start: str, end: str, out_dir: Path) -> Path:
    from PyStemmusScope import StemmusScope

    work_dir = out_dir / code / "matlab"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    config = write_config(code, start, end, work_dir)

    print(f"{code}  MATLAB engine")
    print(f"  window  {start} .. {end}")
    print(f"  config  {config}")

    os.environ.update(mcr_environment())
    model = StemmusScope(config_file=str(config), model_src_path=str(EXE))

    print("  setup ...", flush=True)
    input_dir = model.setup(Location=code, StartTime=start, EndTime=end)
    print(f"  input   {input_dir}")
    written = sorted(p.name for p in Path(input_dir).parent.glob("*.dat"))
    print(f"          {len(written)} .dat files: {', '.join(written[:8])}"
          + (" ..." if len(written) > 8 else ""))

    print("  running ...", flush=True)
    result = model.run()
    print(f"  output  {model.config['OutputPath']}")
    if result:
        tail = str(result).strip().splitlines()[-3:]
        for line in tail:
            print(f"          {line}")
    return Path(model.config["OutputPath"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--site", action="append", dest="sites")
    parser.add_argument("--all", action="store_true",
                        help="every station in sites.yaml")
    parser.add_argument("--start", help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD")
    parser.add_argument("--days", type=int, help="run this many days from --start")
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    parser.add_argument("--jobs", type=int, default=1,
                        help="stations to run concurrently (default 1). Does not "
                             "make any single station faster -- each is one "
                             "sequential simulation.")
    args = parser.parse_args()

    if args.all:
        from wunder.metadata import forcing_sites

        codes = list(forcing_sites())
    elif args.sites:
        codes = args.sites
    else:
        raise SystemExit("specify --site CODE (repeatable) or --all")

    jobs = [(c, *resolve_window(c, args.start, args.end, args.days), args.out)
            for c in codes]

    if args.jobs <= 1 or len(jobs) == 1:
        for job in jobs:
            run(*job)
        return 0

    # One process per station. Each writes to its own runs/<CODE>/matlab/, so
    # they cannot collide; the model itself is sequential and cannot be split,
    # so this shortens a batch and never a single station.
    from multiprocessing import Pool

    print(f"running {len(jobs)} stations, {args.jobs} at a time")
    with Pool(processes=min(args.jobs, len(jobs))) as pool:
        pool.starmap(run, jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
