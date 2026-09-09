"""One page showing every variable that goes into the model.

The point is not decoration. A forcing file is 57,000 rows of float32 that will
be read by a model and never looked at again, and most of the ways it can be
wrong -- a unit slip, an inverted seasonal cycle, a variable that is quietly all
reanalysis -- are obvious in a picture and invisible in a summary statistic.

Every panel is the raw 30-minute series -- exactly the rows the model reads, with
no aggregation and so no ambiguity about what a line means. A variable with a
strong diurnal cycle draws as a filled envelope whose thickness is the day-night
range, and a single bad half-hour stays a visible spike instead of being averaged
into invisibility. Half-hours that did not come from the station's own sensors
are marked in a second colour; `LWdown` is entirely reanalysis at every WUNDER
site, and that should be plain to see rather than buried in an attribute.

Written as a static PNG rather than an interactive figure because it is a
build artefact: something to glance at once before a run, and to attach to a
note about why a run looked odd.
"""

from __future__ import annotations

from pathlib import Path

#: Alongside the code rather than beside the data. These are small, they are
#: meant to be looked at and shared, and keeping them in the working folder means
#: they turn up next to the scripts that made them.
PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: Panel order, and whether the stored value is rescaled for display. Only
#: precipitation is: everything else is plotted in the unit the file holds.
PANELS = [
    ("Tair", "Air temperature", "K", "mean"),
    ("SWdown", "Shortwave down", "W m$^{-2}$", "mean"),
    ("LWdown", "Longwave down", "W m$^{-2}$", "mean"),
    ("Precip", "Precipitation", "mm per 30 min", "rate"),
    ("VPD", "Vapour pressure deficit", "hPa", "mean"),
    ("RH", "Relative humidity", "%", "mean"),
    ("Qair", "Specific humidity", "kg kg$^{-1}$", "mean"),
    ("Psurf", "Surface pressure", "Pa", "mean"),
    ("Wind", "Wind speed", "m s$^{-1}$", "mean"),
    ("CO2air", "CO$_2$", "ppm", "mean"),
    ("LAI", "Leaf area index", "m$^2$ m$^{-2}$", "mean"),
]

MEASURED = "#1a1a1a"
FILLED = "#c2492e"

#: Variables no weather station in the network measures, and what they are
#: instead. Without this a panel with no `_qc` in the file reads as "unmeasured",
#: which is true of LWdown but wrong for CO2 (a real observation, just not from
#: this site) and badly wrong for RH and Qair (derived from measured quantities,
#: and simply not given their own flag -- the reference PLUMBER2 file omits it
#: too).
NEVER_MEASURED = {
    "LWdown": "ERA5-Land",
    "CO2air": "Mauna Loa",
    "LAI": "MODIS",
}
DERIVED = {"RH": "Tair, ea", "Qair": "Tair, ea, Psurf", "VPD": "Tair, ea"}

#: Where the plotted unit is not the unit in the file. Only precipitation: it is
#: stored as a rate, which is what both engines read, but a rate of 2.2e-04 is
#: unreadable on an axis, so the panel shows the daily total instead. Naming both
#: on the axis stops the figure being mistaken for the file.
STORED_UNIT = {"Precip": "file: kg m$^{-2}$ s$^{-1}$"}


def plot_forcing(forcing: pd.DataFrame, qc: pd.DataFrame, code: str,
                 label: str, out_dir: Path) -> Path:
    """Write the one-page summary and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [p for p in PANELS if p[0] in forcing.columns]
    rows = len(panels)
    fig, axes = plt.subplots(rows, 1, figsize=(11, 1.55 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    index = forcing.index
    span_years = (index[-1] - index[0]).days / 365.25

    for ax, (name, title, unit, how) in zip(axes, panels):
        # Every half-hour the model will actually read, not a daily summary. At
        # ~57,000 points over 3.3 years this is ~35 samples per pixel, so a
        # variable with a strong diurnal cycle draws as a filled envelope whose
        # thickness *is* the day-night range -- and anything anomalous shows up
        # as a spike rather than being averaged away.
        # Precipitation is the one variable shown in something other than its
        # stored unit: a rate of 2.2e-04 is unreadable on an axis, so it is
        # scaled to the millimetres that fell in each half-hour. The axis names
        # both units.
        series = forcing[name] * 1800 if how == "rate" else forcing[name]

        # A line width that suits 57,000 points of a diurnal cycle renders a
        # smooth daily series (LAI, CO2) almost invisible, so scale it by how
        # much the variable actually moves between steps.
        values = series.to_numpy()
        spread = np.nanmax(values) - np.nanmin(values)
        jitter = np.nanmean(np.abs(np.diff(values))) / spread if spread else 0.0
        width = 0.18 if jitter > 0.02 else 0.7

        ax.plot(series.index, values, color=MEASURED,
                linewidth=width, alpha=0.9, solid_joinstyle="round")

        # Half-hours that did not come from this station's own sensors.
        flag = qc[name] if name in qc.columns else None
        fraction = None
        if flag is not None:
            filled = series[flag != 0]
            if len(filled):
                ax.plot(filled.index, filled.to_numpy(), ".", color=FILLED,
                        markersize=0.8, linewidth=0, alpha=0.7)
            fraction = float((flag != 0).mean())

        if name in NEVER_MEASURED:
            note, colour = NEVER_MEASURED[name], FILLED
        elif fraction is None:
            note, colour = f"derived from {DERIVED.get(name, 'measured inputs')}", "#666666"
        elif fraction > 0.999:
            note, colour = "all filled", FILLED
        elif fraction < 0.001:
            note, colour = "all measured", "#666666"
        else:
            note = f"{fraction * 100:.1f}% filled"
            colour = FILLED if fraction > 0.5 else "#666666"
        stored = STORED_UNIT.get(name)
        axis_label = f"{title}\n[{unit}]" + (f"\n{stored}" if stored else "")
        ax.set_ylabel(axis_label, fontsize=7.5, linespacing=1.4)
        # Above the axes, not inside them. Relative humidity spends most of the
        # record pinned at 100%, so anything drawn within the panel is hidden by
        # the data it is describing.
        ax.set_title(note, loc="right", fontsize=6.5, color=colour, pad=2.0)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[-1].xaxis.set_major_locator(
        mdates.YearLocator() if span_years > 2 else mdates.MonthLocator(interval=2)
    )
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Average only over variables a station could have measured. Including
    # LWdown, CO2 and LAI would report a low number that says nothing about the
    # station and everything about which variables have no sensor.
    measurable = {
        n: float((qc[n] == 0).mean())
        for n, *_ in panels
        if n in qc.columns and n not in NEVER_MEASURED
    }
    overall = np.mean(list(measurable.values())) if measurable else float("nan")
    title = f"{code}"
    if label:
        title += f" — {label}"
    fig.suptitle(
        f"{title}\n{index[0]:%Y-%m-%d} to {index[-1]:%Y-%m-%d}  ·  {len(index):,} "
        f"half-hours  ·  {overall * 100:.1f}% of the {len(measurable)} measurable "
        "variables came from the station  ·  red = not measured here",
        fontsize=9, y=0.998, va="top",
    )
    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    path = out_dir / f"forcing_{code}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    """Plot from already-written NetCDF files, without rebuilding them."""
    import argparse
    import sys

    import xarray as xr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", action="append", dest="sites")
    parser.add_argument("--forcing-dir", type=Path,
                        default=PLOT_DIR.parent / "model_input" / "forcing")
    parser.add_argument("--out", type=Path, default=PLOT_DIR)
    args = parser.parse_args()

    paths = sorted(args.forcing_dir.glob("FLX_*.nc"))
    if args.sites:
        paths = [p for p in paths if any(s in p.name for s in args.sites)]
    if not paths:
        print(f"no forcing files in {args.forcing_dir}")
        return 1

    for path in paths:
        with xr.open_dataset(path) as ds:
            index = pd.DatetimeIndex(ds["time"].values)
            forcing = pd.DataFrame(
                {n: ds[n].values.ravel() for n, *_ in PANELS if n in ds}, index=index
            )
            qc = pd.DataFrame(
                {n: ds[f"{n}_qc"].values.ravel().astype(int)
                 for n, *_ in PANELS if f"{n}_qc" in ds},
                index=index,
            )
            code = path.name.split("_")[1]
            label = str(ds.attrs.get("station", ""))
        print(plot_forcing(forcing, qc, code, label, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
