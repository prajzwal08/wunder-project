"""Build and read the published dataset.

A cloud deployment has no Parquet cache, so without this it re-downloads a logger's whole
history (~134 MB of CSV, ~60 s) every time its container restarts. That is slow for the
viewer and rude to someone else's server.

The published dataset is the same record resampled to 30 minutes and committed to the repo:
about 25 MB for all fourteen loggers, small enough for git, and finer than the ~4000 points
the figures decimate to anyway. A deployment reads it instantly and then asks the API only
for records since the last published timestamp.

Rebuild after pulling fresh data:

    python -m wunder.publish

and commit the result.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from .fetch import CACHE_DIR, _read_cache, fetch
from .metadata import Logger, loggers
from .metadata import logger as _logger
from .process import resample

PUBLISHED_DIR = Path(__file__).resolve().parent.parent / "data" / "published"

#: Resolution of the published series. Cumulative and daily statistics are identical to the
#: 5-minute record at this step; only sub-half-hour detail is lost.
FREQ = "30min"

MANIFEST = "manifest.json"

#: Diagnostics for whoever services the loggers; not part of the scientific record.
_DROP_COLUMNS = {"Battery Level in %", "BatteryLevel in Volt", "Logger Temperature",
                 "Reference Pressure"}


def path_for(serial: str, directory: Path | None = None) -> Path:
    return (directory or PUBLISHED_DIR) / f"{serial}.parquet"


def available(directory: Path | None = None) -> bool:
    d = directory or PUBLISHED_DIR
    return d.exists() and any(d.glob("*.parquet"))


def read(ref: str | Logger, directory: Path | None = None) -> pd.DataFrame:
    """The published series for one logger, or an empty frame if not published."""
    lg = ref if isinstance(ref, Logger) else _logger(ref)
    p = path_for(lg.serial, directory)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def read_current(
    ref: str | Logger,
    *,
    directory: Path | None = None,
    top_up: bool = True,
) -> pd.DataFrame:
    """Published series, extended to now from the API.

    This is what a deployment without a local cache should call: the bulk arrives from disk
    instantly, and only the tail since the last published timestamp crosses the network.
    """
    lg = ref if isinstance(ref, Logger) else _logger(ref)
    base = read(lg, directory)
    if base.empty:
        return base
    if not top_up:
        return base

    try:
        tail = fetch(lg, start=base.index.max() - dt.timedelta(hours=1), cache=False)
    except Exception:  # noqa: BLE001 -- stale data beats no data
        return base
    if tail.empty:
        return base

    tail = resample(tail, FREQ)
    out = pd.concat([base, tail])
    return out[~out.index.duplicated(keep="last")].sort_index()


def build(
    *,
    directory: Path | None = None,
    freq: str = FREQ,
    verbose: bool = True,
) -> pd.DataFrame:
    """Rebuild the published dataset from the local cache.

    Requires the full cache (`wunder.update_all()`), since it downsamples what is on disk
    rather than re-fetching.
    """
    d = directory or PUBLISHED_DIR
    d.mkdir(parents=True, exist_ok=True)

    if not CACHE_DIR.exists() or not any(CACHE_DIR.glob("*.parquet")):
        raise RuntimeError(
            f"no local cache in {CACHE_DIR}. Run wunder.update_all() first — publish "
            "downsamples the cache rather than re-fetching."
        )

    rows, manifest = [], {}
    for lg in loggers():
        raw, meta = _read_cache(lg.serial, None)
        if raw.empty:
            if verbose:
                print(f"{lg.name:20s} no cached data, skipped")
            continue
        out = resample(raw, freq)
        # Trim what a viewer never plots. Battery voltage and logger temperature are
        # diagnostics for whoever services the loggers, not part of the record.
        out = out.drop(columns=[c for c in out.columns if c in _DROP_COLUMNS],
                       errors="ignore")
        # float32 is ~7 significant digits -- far beyond any of these sensors' precision,
        # and halves the file.
        out = out.astype({c: "float32" for c in out.columns
                          if pd.api.types.is_float_dtype(out[c])})
        p = path_for(lg.serial, d)
        out.to_parquet(p, compression="zstd", compression_level=9)
        size = p.stat().st_size / 1e6
        manifest[lg.serial] = {
            "name": lg.name,
            "rows": len(out),
            "start": str(out.index.min()),
            "end": str(out.index.max()),
            "mb": round(size, 2),
        }
        rows.append({"Logger": lg.name, "Serial": lg.serial, "Rows": len(out),
                     "From": out.index.min(), "To": out.index.max(), "MB": round(size, 2)})
        if verbose:
            print(f"{lg.name:20s} {len(raw):>7,} -> {len(out):>6,} rows   {size:5.1f} MB")

    (d / MANIFEST).write_text(json.dumps(
        {"freq": freq, "built": str(dt.datetime.now()), "loggers": manifest}, indent=2))

    df = pd.DataFrame(rows)
    if verbose and len(df):
        print(f"\n{len(df)} loggers, {df['MB'].sum():.1f} MB total -> {d}")
    return df


def manifest(directory: Path | None = None) -> dict:
    p = (directory or PUBLISHED_DIR) / MANIFEST
    return json.loads(p.read_text()) if p.exists() else {}


if __name__ == "__main__":
    build()
