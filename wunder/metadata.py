"""Logger registry loaded from sites.yaml.

Identity, coordinates and installed sensors come from info.txt via sites.yaml. What a logger
*actually* reports is a different question -- the API returns columns that are entirely empty
(see CLAUDE.md), so anything that matters should be checked against real data with
`Logger.reporting()`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import pandas as pd

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "sites.yaml"

# Depths, shallowest first. Used to order series so legends read top-down.
DEPTH_ORDER = ["2.5", "5", "10", "20", "40", "80"]

# Fixed depth -> colour, so a depth is the same colour in every figure.
# Carried over from trial/Ketelbroek_DataReport.ipynb cell 5.
DEPTH_COLORS = {
    "2.5": "blue",
    "5": "orange",
    "10": "green",
    "20": "red",
    "40": "purple",
    "80": "brown",
}


def _depth_key(depth: str) -> float:
    return float(depth)


@dataclass(frozen=True)
class Installed:
    """One sensor in the ground, as documented in info.txt."""

    depth: str  # label used in API column names, e.g. "2.5"
    documented_cm: int  # what info.txt says, which is sometimes different
    sensor: str  # 5TM | TEROS11 | TEROS12 | TEROS21

    @property
    def depth_cm(self) -> float:
        return float(self.depth)


@dataclass(frozen=True)
class Logger:
    name: str  # e.g. "F1_1_ATMOS_SMST1" -- what people actually recognise
    serial: str  # e.g. "z6-21176"
    site_key: str
    site_name: str
    field_key: str
    field_name: str
    latitude: float
    longitude: float
    elevation_m: int
    weather_station: dict | None
    installed: tuple[Installed, ...]
    observed: dict
    issues: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    # -- identity -----------------------------------------------------------

    @property
    def label(self) -> str:
        """Name plus serial, for dropdowns and titles. Never show the serial alone."""
        return f"{self.name} ({self.serial})"

    @property
    def location(self) -> str:
        return f"{self.site_name} - {self.field_name}"

    @property
    def has_weather_station(self) -> bool:
        return self.weather_station is not None

    @property
    def stopped(self):
        """Date the logger stopped reporting, or None if it is still live.

        Distinct from having no data: z6-08819 stopped on 2026-08-01 but has three years of
        good record behind it, which should still be plotted.
        """
        return self.observed.get("stopped")

    @property
    def is_offline(self) -> bool:
        return self.stopped is not None or bool(self.observed.get("offline"))

    # -- what it measures ---------------------------------------------------

    def installed_depths(self, measure: str) -> list[str]:
        """Depths where a sensor capable of `measure` was installed."""
        capable = {s for s, ms in _registry()["sensors"].items() if measure in ms}
        depths = {i.depth for i in self.installed if i.sensor in capable}
        return sorted(depths, key=_depth_key)

    def observed_depths(self, measure: str) -> list[str]:
        """Depths that returned non-null data in the snapshot. See `reporting()` for live."""
        return sorted(self.observed.get(measure, []), key=_depth_key)

    def dead_depths(self, measure: str) -> list[str]:
        """Installed but silent -- a fault, not an absent sensor."""
        if self.is_offline:
            return []
        live = set(self.observed_depths(measure))
        return [d for d in self.installed_depths(measure) if d not in live]

    def reporting_measures(self) -> list[str]:
        """Measures with at least one live depth in the snapshot."""
        return [m for m in _registry()["measures"] if self.observed.get(m)]

    def column(self, measure: str, depth: str) -> str:
        """API column name. Prefixes are case-sensitive and inconsistent -- see CLAUDE.md."""
        return f"{_registry()['measures'][measure]} {depth}cm"

    def columns(self, measure: str, *, observed: bool = True) -> list[str]:
        depths = self.observed_depths(measure) if observed else self.installed_depths(measure)
        return [self.column(measure, d) for d in depths]

    def reporting(self, df) -> dict[str, list[str]]:
        """Measure -> depths carrying non-null data in `df`.

        The authoritative check. The snapshot in sites.yaml is only a hint; sensors die.
        """
        out: dict[str, list[str]] = {}
        for measure in _registry()["measures"]:
            depths = [
                d
                for d in DEPTH_ORDER
                if (c := self.column(measure, d)) in df.columns and df[c].notna().any()
            ]
            if depths:
                out[measure] = depths
        return out

    def met_columns(self, df=None) -> list[str]:
        """Met columns that carry data. Pass `df` to check for real rather than trust the snapshot."""
        cols = _registry()["met_columns"]
        if df is not None:
            return [c for c in cols if c in df.columns and df[c].notna().any()]
        return list(cols) if self.observed.get("met") else []


@dataclass(frozen=True)
class Site:
    key: str
    name: str
    fields: dict[str, str]
    loggers: tuple[Logger, ...] = field(default_factory=tuple)

    def by_field(self) -> dict[str, list[Logger]]:
        """Loggers grouped by field, for a two-level dropdown (Glanerbeek has nine)."""
        out: dict[str, list[Logger]] = {k: [] for k in self.fields}
        for lg in self.loggers:
            out.setdefault(lg.field_key, []).append(lg)
        return {k: v for k, v in out.items() if v}


@functools.lru_cache(maxsize=1)
def _registry() -> dict:
    with open(REGISTRY_PATH) as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=1)
def _sites() -> tuple[Site, ...]:
    out = []
    for s in _registry()["sites"]:
        fields = s.get("fields", {})
        lgs = tuple(
            Logger(
                name=d["name"],
                serial=d["serial"],
                site_key=s["key"],
                site_name=s["name"],
                field_key=d["field"],
                field_name=fields.get(d["field"], d["field"]),
                latitude=d["latitude"],
                longitude=d["longitude"],
                elevation_m=d["elevation_m"],
                weather_station=d.get("weather_station"),
                installed=tuple(Installed(**i) for i in d.get("installed", [])),
                observed=d.get("observed", {}),
                issues=tuple(d.get("issues", [])),
                notes=tuple(d.get("notes", [])),
            )
            for d in s["loggers"]
        )
        out.append(Site(key=s["key"], name=s["name"], fields=fields, loggers=lgs))
    return tuple(out)


# -- public lookups ---------------------------------------------------------


def sites() -> list[Site]:
    return list(_sites())


def site(key: str) -> Site:
    for s in _sites():
        if s.key == key or s.name.lower() == key.lower():
            return s
    raise KeyError(f"unknown site {key!r}; have {[s.key for s in _sites()]}")


def loggers(site_key: str | None = None) -> list[Logger]:
    if site_key is None:
        return [lg for s in _sites() for lg in s.loggers]
    return list(site(site_key).loggers)


def logger(ref: str) -> Logger:
    """Look up by serial ('z6-21176') or device name ('F1_1_ATMOS_SMST1'), case-insensitive."""
    r = ref.strip().lower()
    for lg in loggers():
        if lg.serial.lower() == r or lg.name.lower() == r:
            return lg
    raise KeyError(f"unknown logger {ref!r}")


def met_source(ref: str | Logger) -> Logger | None:
    """The weather-station logger whose met data applies to `ref`.

    The TEROS21 water-potential loggers carry no weather station, so pairing matric
    potential with VPD needs the ATMOS-41 logger from the same field — falling back to the
    same site, then to any station still reporting. Returns None if nothing is available.
    """
    lg = ref if isinstance(ref, Logger) else logger(ref)
    if lg.has_weather_station and not lg.is_offline:
        return lg
    stations = [x for x in loggers() if x.has_weather_station and not x.is_offline]
    for pool in (
        [x for x in stations if x.site_key == lg.site_key and x.field_key == lg.field_key],
        [x for x in stations if x.site_key == lg.site_key],
        stations,
    ):
        if pool:
            return pool[0]
    return None


def soil_source(ref: str | Logger) -> Logger | None:
    """The logger whose soil moisture applies to `ref`.

    Mirrors `met_source`, for the other half of the pairing. Two loggers here measure no
    soil moisture at all -- `F1_4_WPST` is matric potential only -- and an offline one
    measures nothing, so anything derived from soil water (root-zone moisture, the FAO-56
    stress factor, water-limited ET) needs a stand-in.

    Preference is same field, then same site, then the whole network, and *within* each
    pool the physically nearest logger, since soil varies over metres. Returns the logger
    itself when it has its own moisture, so the common case costs nothing and never
    silently substitutes a neighbour for a working probe.
    """
    lg = ref if isinstance(ref, Logger) else logger(ref)
    if lg.observed_depths("moisture") and not lg.is_offline:
        return lg
    pool_all = [x for x in loggers()
                if x.observed_depths("moisture") and not x.is_offline
                and x.serial != lg.serial]
    for pool in (
        [x for x in pool_all
         if x.site_key == lg.site_key and x.field_key == lg.field_key],
        [x for x in pool_all if x.site_key == lg.site_key],
        pool_all,
    ):
        if pool:
            return min(pool, key=lambda x: _separation_m(lg, x))
    return None


def _separation_m(a: Logger, b: Logger) -> float:
    """Ground distance between two loggers [m], flat-earth -- they are metres apart."""
    import math

    dy = (a.latitude - b.latitude) * 111_320.0
    dx = ((a.longitude - b.longitude) * 111_320.0
          * math.cos(math.radians((a.latitude + b.latitude) / 2.0)))
    return math.hypot(dx, dy)


def measures() -> dict[str, str]:
    """measure key -> API column prefix."""
    return dict(_registry()["measures"])


@functools.lru_cache(maxsize=1)
def forcing_sites() -> dict[str, dict]:
    """Station code -> settings for building STEMMUS_SCOPE forcing.

    One entry per weather station, not per site: Glanerbeek F1 and F2 are
    separate, and so are Ketelbroek K1 and K2. Sensors from different stations
    are never combined, so each forcing file describes exactly one mast.

    Keys are the FLUXNET-style codes both engines use to find a station's files.
    They must match PyStemmusScope's site pattern `[A-Z]{2}-([A-z]|\\d){3}`, and
    no code may be a substring of another -- `get_forcing_file` scans a directory
    for filenames merely *containing* the code and refuses on two matches.
    """
    out: dict[str, dict] = {}
    for code, raw in (_registry().get("forcing") or {}).items():
        cfg = dict(raw)
        met = logger(cfg["met_logger"])
        cfg["code"] = code
        cfg["latitude"] = met.latitude
        cfg["longitude"] = met.longitude
        cfg["elevation_m"] = met.elevation_m
        cfg["met"] = met
        cfg["soil"] = [logger(s) for s in cfg.get("soil_loggers") or []]
        cfg["exclude_columns"] = list(cfg.get("exclude_columns") or [])
        out[code] = cfg

    clashes = [
        (a, b) for a in out for b in out if a != b and a in b
    ]
    if clashes:
        raise ValueError(
            f"station codes must not contain one another: {clashes}. "
            "PyStemmusScope's get_forcing_file would match both."
        )
    return out


def forcing_site(code: str) -> dict:
    """One station's forcing settings, by code."""
    try:
        return forcing_sites()[code]
    except KeyError:
        known = ", ".join(forcing_sites()) or "none"
        raise KeyError(f"unknown station code {code!r}; sites.yaml has: {known}") from None


def housekeeping_columns() -> list[str]:
    return list(_registry()["housekeeping_columns"])


def overview() -> "pd.DataFrame":
    """The reference table: who is who, what they measure, what is broken.

    Shown in the app's Loggers tab -- nobody recognises a bare serial number.
    """
    import pandas as pd

    rows = []
    for lg in loggers():
        dead = {m: lg.dead_depths(m) for m in measures()}
        dead = {m: d for m, d in dead.items() if d}
        rows.append(
            {
                "Site": lg.site_name,
                "Field": lg.field_name,
                "Logger": lg.name,
                "Serial": lg.serial,
                "Weather station": "yes" if lg.has_weather_station else "-",
                "Measures": ", ".join(sorted(lg.reporting_measures())) or "-",
                "Depths": ", ".join(lg.observed_depths("moisture") or lg.observed_depths("matric_potential")) or "-",
                "Status": "OFFLINE" if lg.is_offline else "ok",
                "Problems": "; ".join(
                    [f"{m} {','.join(d)}cm dead" for m, d in dead.items()] + list(lg.issues)
                ),
                "Latitude": lg.latitude,
                "Longitude": lg.longitude,
                "Elevation (m)": lg.elevation_m,
            }
        )
    return pd.DataFrame(rows)
