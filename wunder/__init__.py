"""WUNDER logger data — fetch, process and plot.

    from wunder import loggers, fetch, rzsm, plot

    df  = fetch("F1_1_ATMOS_SMST1", days=30)   # by device name or serial
    fig = plot.soil_moisture(df, precip=True)

First use of a logger downloads its full history (back to 2023-05-18, a few minutes) into
data/raw/. After that it is cached and refreshed at most once a day.

Nothing here imports Streamlit; the app is a separate, thin layer.
"""

from . import et
from . import stress
from . import fetch as fetch_module
from . import publish
from . import plots as plot
from .et import makkink, reference_et, water_balance
from .fetch import FetchError, cache_status, fetch, is_stale, units, update, update_all
from .metadata import (
    DEPTH_COLORS,
    DEPTH_ORDER,
    Logger,
    Site,
    housekeeping_columns,
    logger,
    loggers,
    measures,
    met_source,
    soil_source,
    overview,
    site,
    sites,
)
from .process import (
    active_measures,
    circular_mean,
    coverage,
    cumulative_year,
    field_series,
    depth_columns,
    depths_of,
    resample,
    root_zone,
    rzsm,
    rzst,
    sensor_status,
    summary,
    thicknesses,
    wind_rose_table,
)

__all__ = [
    "DEPTH_COLORS",
    "DEPTH_ORDER",
    "FetchError",
    "Logger",
    "Site",
    "active_measures",
    "cache_status",
    "circular_mean",
    "coverage",
    "cumulative_year",
    "depth_columns",
    "depths_of",
    "et",
    "fetch",
    "field_series",
    "housekeeping_columns",
    "is_stale",
    "logger",
    "loggers",
    "makkink",
    "measures",
    "met_source",
    "overview",
    "fetch_module",
    "plot",
    "publish",
    "reference_et",
    "resample",
    "root_zone",
    "rzsm",
    "rzst",
    "sensor_status",
    "site",
    "sites",
    "soil_source",
    "stress",
    "summary",
    "thicknesses",
    "units",
    "update",
    "update_all",
    "water_balance",
    "wind_rose_table",
]
