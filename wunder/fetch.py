"""MajiSys API client with an incremental Parquet cache.

The full record per logger is large -- z6-21176 is 347k rows / 134 MB of CSV back to
2023-05-18 -- so a full pull happens once and later refreshes ask only for the new tail.

    df = fetch("F1_1_ATMOS_SMST1")          # everything, from cache
    df = fetch("z6-21176", days=30)         # last 30 days
    df = fetch("z6-21176", refresh=True)    # top up from the API first
"""

from __future__ import annotations

import datetime as dt
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

from .metadata import Logger, logger as _logger

BASE = "http://majisysdemo.itc.utwente.nl/wunder"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Constant per logger; kept out of the cache rather than repeated on every row.
_DROP = ["ID", "lat", "lon"]

_TIMEFMT = "%Y-%m-%dT%H_%M_%S"

# Loggers occasionally stamp a record before their clock has a valid date -- z6-21178 has one
# row dated 1987-05-04, z6-25927 one dated 1989-10-20. The network's earliest real record is
# 2022-12-09. A single stray timestamp is harmless in itself, but it makes resample("1D")
# generate ~13k empty days, so drop anything outside a plausible window.
MIN_VALID = dt.datetime(2020, 1, 1)
FUTURE_TOLERANCE = dt.timedelta(days=1)


class FetchError(RuntimeError):
    pass


def _url(serial: str, start: dt.datetime | None, end: dt.datetime | None) -> str:
    if start is None and end is None:
        return f"{BASE}/getall.py?location={serial}"
    # get7days.py honours any window despite the name.
    end = end or dt.datetime.now()
    return (
        f"{BASE}/get7days.py?location={serial}"
        f"&mindate={start.strftime(_TIMEFMT)}&maxdate={end.strftime(_TIMEFMT)}"
    )


def _download(url: str, timeout: int = 600) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001 -- surface the URL, the API gives poor errors
        raise FetchError(f"{url} failed: {e}") from e


def _parse(text: str) -> tuple[pd.DataFrame, dict[str, str], dict]:
    """CSV with two header lines (labels, units) -> frame indexed by Date, plus units and site info."""
    if not text.strip():
        return pd.DataFrame(), {}, {}
    head = text.split("\n", 2)
    if len(head) < 2:
        return pd.DataFrame(), {}, {}
    labels = head[0].rstrip("\r").split(",")
    units = [u.strip("[]") for u in head[1].rstrip("\r").split(",")]
    unit_map = dict(zip(labels, units))

    df = pd.read_csv(io.StringIO(text), skiprows=[1])
    if df.empty:
        return pd.DataFrame(), unit_map, {}

    info = {}
    for c in _DROP:
        if c in df.columns:
            v = df[c].dropna()
            if len(v):
                info[c] = v.iloc[0]
    df = df.drop(columns=[c for c in _DROP if c in df.columns])

    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = drop_bad_timestamps(df)
    return df, unit_map, info


def drop_bad_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records stamped before the network existed or implausibly far ahead."""
    if df.empty:
        return df
    hi = pd.Timestamp(dt.datetime.now() + FUTURE_TOLERANCE)
    keep = (df.index >= pd.Timestamp(MIN_VALID)) & (df.index <= hi)
    return df[keep]


def _paths(serial: str, cache_dir: Path | None) -> tuple[Path, Path]:
    d = Path(cache_dir) if cache_dir else CACHE_DIR
    return d / f"{serial}.parquet", d / f"{serial}.meta.json"


def _read_cache(serial: str, cache_dir: Path | None) -> tuple[pd.DataFrame, dict]:
    pq, meta = _paths(serial, cache_dir)
    if not pq.exists():
        return pd.DataFrame(), {}
    df = pd.read_parquet(pq)
    m = json.loads(meta.read_text()) if meta.exists() else {}
    return df, m


def _write_cache(serial: str, df: pd.DataFrame, meta: dict, cache_dir: Path | None) -> None:
    pq, mp = _paths(serial, cache_dir)
    pq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pq)
    mp.write_text(json.dumps(meta, indent=2, default=str))


def update(
    ref: str | Logger,
    *,
    cache_dir: Path | None = None,
    full: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Bring the cache up to date and return the whole record.

    First call downloads the full history (slow -- minutes, >100 MB per logger). Later calls
    request only from the last cached timestamp. `full=True` forces a complete re-pull.
    """
    lg = ref if isinstance(ref, Logger) else _logger(ref)
    cached, meta = _read_cache(lg.serial, cache_dir)

    if full or cached.empty:
        start = None
        if verbose:
            print(f"{lg.label}: full history (this takes a minute)...")
    else:
        # Re-request the last hour too; the server backfills late-arriving records.
        start = cached.index.max().to_pydatetime() - dt.timedelta(hours=1)
        if verbose:
            print(f"{lg.label}: since {start:%Y-%m-%d %H:%M}...")

    new, units, info = _parse(_download(_url(lg.serial, start, None)))

    if new.empty and cached.empty:
        meta = {"serial": lg.serial, "rows": 0, "offline": True, "updated": dt.datetime.now()}
        _write_cache(lg.serial, pd.DataFrame(), meta, cache_dir)
        if verbose:
            print(f"{lg.label}: no data returned (offline?)")
        return pd.DataFrame()

    if cached.empty:
        df = new
    elif new.empty:
        df = cached
    else:
        df = pd.concat([cached, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()

    meta = {
        "serial": lg.serial,
        "name": lg.name,
        "units": units or meta.get("units", {}),
        "info": info or meta.get("info", {}),
        "rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "updated": dt.datetime.now(),
    }
    _write_cache(lg.serial, df, meta, cache_dir)
    if verbose:
        added = len(df) - len(cached)
        print(f"{lg.label}: {len(df):,} rows ({added:+,}), {df.index.min():%Y-%m-%d} -> {df.index.max():%Y-%m-%d %H:%M}")
    return df


# The server refreshes every 2 hours, daytime only, but once a day is plenty for viewing
# and keeps our load on someone else's server negligible.
MAX_AGE_HOURS = 24


def is_stale(ref: str | Logger, *, max_age_hours: float = MAX_AGE_HOURS,
             cache_dir: Path | None = None) -> bool:
    """True if the cache is missing or older than `max_age_hours`."""
    lg = ref if isinstance(ref, Logger) else _logger(ref)
    _, meta = _read_cache(lg.serial, cache_dir)
    updated = meta.get("updated")
    if not updated:
        return True
    age = dt.datetime.now() - pd.Timestamp(updated).to_pydatetime()
    return age > dt.timedelta(hours=max_age_hours)


def fetch(
    ref: str | Logger,
    *,
    start: str | dt.datetime | None = None,
    end: str | dt.datetime | None = None,
    days: int | None = None,
    refresh: bool | None = None,
    max_age_hours: float = MAX_AGE_HOURS,
    cache: bool = True,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Return a logger's record, from cache, optionally sliced.

    Accepts a serial ('z6-21176') or a device name ('F1_1_ATMOS_SMST1').

    Refreshes at most once a day by default: `refresh=None` tops up only if the cache is
    older than `max_age_hours`. `refresh=True` forces it, `refresh=False` never hits the
    network (except to seed an empty cache).

    `cache=False` skips disk entirely and fetches just the requested window from the API --
    use it where there is no persistent storage, such as a cloud deployment.
    """
    lg = ref if isinstance(ref, Logger) else _logger(ref)

    if not cache:
        # Straight to the API for just this window -- no disk, no full history. This is what
        # makes cloud deployment practical: the full cache is ~106 MB and seeding it takes
        # ~10 minutes, which no free container will tolerate on cold start.
        if days is not None and start is None:
            start = dt.datetime.now() - dt.timedelta(days=days)
        lo = pd.Timestamp(start).to_pydatetime() if start is not None else dt.datetime(2020, 1, 1)
        hi = pd.Timestamp(end).to_pydatetime() if end is not None else None
        df, _, _ = _parse(_download(_url(lg.serial, lo, hi)))
        return df

    df, _ = _read_cache(lg.serial, cache_dir)
    want = (
        df.empty
        if refresh is False
        else refresh or is_stale(lg, max_age_hours=max_age_hours, cache_dir=cache_dir)
    )
    if want:
        df = update(lg, cache_dir=cache_dir, verbose=False)
    if df.empty:
        return df

    if days is not None:
        start = df.index.max() - pd.Timedelta(days=days)
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def units(ref: str | Logger, cache_dir: Path | None = None) -> dict[str, str]:
    """Column -> unit, from the CSV's second header line."""
    lg = ref if isinstance(ref, Logger) else _logger(ref)
    _, meta = _read_cache(lg.serial, cache_dir)
    return meta.get("units", {})


def cache_status(cache_dir: Path | None = None) -> pd.DataFrame:
    """What is cached for each logger -- rows, span, last update."""
    from .metadata import loggers

    rows = []
    for lg in loggers():
        _, meta = _read_cache(lg.serial, cache_dir)
        rows.append(
            {
                "Logger": lg.name,
                "Serial": lg.serial,
                "Site": lg.site_name,
                "Rows": meta.get("rows", 0),
                "From": meta.get("start"),
                "To": meta.get("end"),
                "Cached": meta.get("updated"),
            }
        )
    return pd.DataFrame(rows)


def update_all(*, cache_dir: Path | None = None, full: bool = False) -> None:
    """Seed or refresh every logger. First run downloads ~1.5 GB and takes ~15 minutes."""
    from .metadata import loggers

    for lg in loggers():
        try:
            update(lg, cache_dir=cache_dir, full=full)
        except FetchError as e:
            print(f"{lg.label}: FAILED -- {e}")
