"""Daily evapotranspiration from the Glanerbeek eddy tower and from STEMMUS-SCOPE.

    python -m wunder.flux        # rebuild data/flux/NL-Gl1_daily.csv

The output is committed, cleared for publication on 2026-11-25, and holds only the daily
values the dashboard shows. The half-hourly eddy record it comes from is not public.
A deployment without the file simply does not show the comparison.

Both series are converted to mm per local (Europe/Amsterdam) day, the day the loggers and
ETo use. Two assumptions, neither confirmed by the provider: the tower's time stamps are
fixed UTC+2 at the end of the interval (inferred from the best lag against radiation), and
the fluxes are in W m-2.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "flux" / "NL-Gl1_daily.csv"
SITE = "NL-Gl1"
#: The period that is published, and shown.
YEAR, MONTHS = 2025, (6, 7, 8)

PAPER01 = Path.home() / "paper01" / "GlanerbeekEddy"
EDDY = PAPER01 / "Gla_2024_2025_Reddyproc_U50_qc01_results.csv"
#: One-year run; see the project notes before replacing it.
MODEL = PAPER01 / "year1" / "out" / "y1_ml_fix_sw" / "fluxes.csv"
MODEL_START = "2024-11-12"

LATENT_HEAT = 2.45e6      # J kg-1
TO_MM_DAY = 86400 / LATENT_HEAT
MIN_MEASURED = 24         # half-hours with a measured (fqc 0) flux
MIN_GOOD = 46             # ... measured or best-class gap-filled (fqc 0 or 1)


def _local_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local calendar day of each UTC interval start."""
    return (index.tz_localize("UTC").tz_convert("Europe/Amsterdam")
            .tz_localize(None).floor("D"))


def eddy_daily(path=EDDY) -> pd.DataFrame:
    """`eddy` [mm d-1], blank on days with too few measured half-hours."""
    e = pd.read_csv(path)
    stamp = pd.DatetimeIndex(pd.to_datetime(e["DateTime"], format="mixed"))
    day = _local_days(stamp - pd.Timedelta("150min"))
    fqc = e["LE_U50_fqc"]
    good = fqc.isin([0, 1]).groupby(day).sum()
    measured = (fqc == 0).groupby(day).sum()
    et = e["LE_U50_f"].groupby(day).mean() * TO_MM_DAY
    keep = (good >= MIN_GOOD) & (measured >= MIN_MEASURED)
    return pd.DataFrame({"eddy": et.where(keep), "eddy_measured": measured})


def model_daily(path=MODEL, start=MODEL_START) -> pd.Series:
    """STEMMUS-SCOPE ET [mm d-1] on complete local days."""
    le = pd.read_csv(path, skiprows=[1])["lEtot"]
    day = _local_days(pd.date_range(start, periods=len(le), freq="30min"))
    grouped = le.groupby(day)
    return (grouped.mean() * TO_MM_DAY)[grouped.size() >= MIN_GOOD].rename("model")


def build(out=OUT) -> pd.DataFrame:
    frame = eddy_daily().join(model_daily(), how="outer").sort_index()
    # Only what the dashboard shows. This file is committed to a public repository.
    frame = frame[(frame.index.year == YEAR) & frame.index.month.isin(MONTHS)]
    frame.index.name = "date"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.round(4).to_csv(out)
    return frame


def read(path=OUT) -> pd.DataFrame:
    """The daily frame, empty when this machine does not have it."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["eddy", "eddy_measured", "model"])
    return pd.read_csv(path, index_col="date", parse_dates=True)


if __name__ == "__main__":
    print(build().describe())
