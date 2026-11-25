"""What each chart is, the equation behind it, and why it matters.

One entry per chart, keyed by the name the app passes to `show(..., about=...)`. Markdown
with `$...$` maths; no Streamlit and no Plotly, so a notebook can print the same text and
there is one place to correct a sentence.

House rules: `what` one or two sentences, `how` the equation with its symbols and nothing
else, `why` one or two sentences with the main limitation. Every constant is the one in
the code (`et.py`, `stress.py`, `process.py`, `flux.py`); change them together. This file
is public: findings from the private flux data do not belong in it.
"""

from __future__ import annotations

#: What the collapsed header says under every chart.
LABEL = "What is it, and why does it matter?"

_THETA = ("$\\theta=\\sum\\theta_i L_i/\\sum L_i$, where $\\theta_i$ is the water content "
          "(m³/m³) at sensor $i$ and $L_i$ is the thickness of the soil layer it "
          "represents.")

_LIMITS = ("Field capacity is $\\theta_{fc}=\\theta(-33\\,\\mathrm{kPa})$, $\\theta_r$ is "
           "the residual water content, and stress begins at "
           "$\\theta_c=(\\theta_{fc}+\\theta_r)/2$. These values are calculated from "
           "SoilGrids soil properties.")

_MAKKINK = ("$ET_o=0.65\\,\\dfrac{s}{s+\\gamma}\\,\\dfrac{R_s}{\\lambda}$ (Makkink, "
            "mm/day), where $R_s$ is daily global radiation (MJ/m²), $s$ is the slope of "
            "the saturation vapour pressure curve at air temperature, $\\gamma$ is the "
            "psychrometric constant and $\\lambda$ is the latent heat of vaporisation "
            "(MJ/kg).")

_WSF = ("For each soil layer $i$, $\\mathrm{WSF}_i="
        "\\left[1+e^{-100\\,\\theta_{s,i}(\\theta_i-\\theta_{c,i})}\\right]^{-1}$, where "
        "$\\theta_i$ is the water content, $\\theta_s$ is saturation and "
        "$\\theta_c=(\\theta_{fc}+\\theta_r)/2$ is midway between field capacity and "
        "residual water content. WSF is the average over layers, weighted by thickness.")

_ETA = ("Makkink $ET_a=\\mathrm{WSF}\\times K_c\\times ET_o$, where WSF is the water stress factor "
        "(0–1) and the crop coefficient is $K_c=1$.")

_ZERO = " Missing days count as zero."

EXPLAIN: dict[str, dict[str, str]] = {
    # -- Summary
    "rzsm": {
        "what": "The average water content of the root zone. Each line is one year; the "
                "current year is black.",
        "how": _THETA + " " + _LIMITS,
        "why": "When the line is below *stress begins*, the soil is dry and can supply "
               "less than about half of the water demand.",
    },
    "wsf": {
        "what": "The water stress factor (WSF) is the fraction of the water demand "
                "(reference evapotranspiration, ETo) that the soil can supply. It ranges "
                "from 0 to 1.",
        "how": _WSF,
        "why": "When WSF is below 0.5 (dotted line), the soil is dry. Years can be "
               "compared at one logger. Loggers in the same field should not be compared, "
               "because they use the same soil properties.",
    },
    "precip_cumulative": {
        "what": "The total rainfall since 1 January. One millimetre is one litre per "
                "square metre.",
        "how": "Daily rainfall is added up." + _ZERO,
        "why": "It shows whether the current year is wetter or drier than previous years. "
               "A flat line is a dry period or a period without data.",
    },
    "et0_cumulative": {
        "what": "The total reference evapotranspiration (ETo) since 1 January. ETo is "
                "the water use of well-watered short grass.",
        "how": _MAKKINK + _ZERO,
        "why": "ETo is the water demand of the atmosphere. It does not depend on soil "
               "moisture. KNMI publishes the same quantity as `EV24`.",
    },
    "et_cumulative": {
        "what": "The total Makkink ETa since 1 January. Makkink ETa is the actual "
                "evapotranspiration calculated as ETo × WSF.",
        "how": _ETA + " $ET_o$ is the reference evapotranspiration." + _ZERO,
        "why": "When Makkink ETa rises more slowly than ETo, dry soil limits "
               "evapotranspiration.",
    },
    "balance_cumulative": {
        "what": "Total rainfall minus total evapotranspiration (P − ET) since 1 January. "
                "The dashed line uses the demand (ETo) and the solid line uses Makkink ETa "
                "(ETo × WSF).",
        "how": "Daily rainfall minus daily ET is added up. Drainage and runoff are not "
               "included." + _ZERO,
        "why": "A falling solid line means that evapotranspiration exceeds rainfall. The "
               "gap between the two lines is the demand that the soil did not supply.",
    },
    "vpd_cumulative": {
        "what": "The sum of daily vapour pressure deficit (VPD) since 1 January. VPD is "
                "the drying power of the air.",
        "how": "$\\mathrm{VPD}=e_s(T)-e_a$, where $e_s$ is the saturation vapour pressure "
               "at air temperature $T$ and $e_a$ is the actual vapour pressure. Daily "
               "means are added up (kPa × days)." + _ZERO,
        "why": "High VPD increases the water demand. Plants can respond by closing their "
               "stomata, which reduces growth.",
    },
    "weekly_balance": {
        "what": "The upper chart shows weekly rainfall, demand (ETo, pale) and "
                "Makkink ETa (ETo × WSF, solid). The lower chart shows rainfall minus "
                "Makkink ETa.",
        "how": _ETA + " Drainage and runoff are not included.",
        "why": "A brown bar means that Makkink ETa exceeded rainfall in that week and soil water "
               "decreased. The exposed pale part is the demand that the soil did not "
               "supply.",
    },
    # -- Soil water
    "soil_moisture": {
        "what": "The soil water content at each depth, from shallow (red) to deep (blue). "
                "Rainfall is shown where it is measured.",
        "how": "Dielectric sensors measure the volume of water per volume of soil "
               "(m³/m³). The typical accuracy is ±0.03. Frozen water is not measured.",
        "why": "It shows how deep rainfall or irrigation infiltrates. A decrease in the "
               "deep layers means that the soil water store is being depleted.",
    },
    "root_zone": {
        "what": "The average water content of the soil profile. Rainfall is shown where "
                "it is measured.",
        "how": _THETA + " The stored water is $10\\,\\theta\\sum L_i$ mm, with $L_i$ in cm.",
        "why": "A falling line means that the stored soil water is decreasing. The value "
               "includes water that roots cannot extract.",
    },
    "matric_potential": {
        "what": "The matric potential (kPa) describes how strongly the soil holds water. "
                "A more negative value means a drier soil. Vapour pressure deficit (VPD), "
                "the drying power of the air, is shown where available.",
        "how": "Field capacity is about −33 kPa and the permanent wilting point is about "
               "−1500 kPa. The sensors cannot measure values above about −9 kPa.",
        "why": "Roots must overcome this potential to take up water. The values can be "
               "compared between different soils.",
    },
    "conductivity": {
        "what": "The electrical conductivity of the soil at each depth (mS/cm).",
        "how": "It increases with water content, dissolved salts and temperature (about "
               "2% per °C).",
        "why": "A change that is not explained by water content or temperature indicates "
               "a change in dissolved salts or nutrients.",
    },
    # -- Temperature
    "soil_temperature": {
        "what": "The soil temperature at each depth. Air temperature is shown where it is "
                "measured.",
        "how": "The amplitude of temperature variations decreases with depth $z$ as "
               "$e^{-z/d}$. The damping depth $d$ is about 0.1 m for the daily cycle and "
               "2 m for the yearly cycle.",
        "why": "Soil temperature controls germination, root growth and microbial "
               "activity. Mulch and shade reduce the daily variation.",
    },
    # -- Weather
    "precipitation": {
        "what": "The bars show rainfall per interval. The line shows the total over the "
                "selected period.",
        "how": "Rain gauge totals are used without a wind correction. Missing data count "
               "as zero.",
        "why": "It shows the amount and timing of rainfall. Intense rainfall can produce "
               "runoff.",
    },
    "temperature_radiation": {
        "what": "Air temperature, root-zone soil temperature and incoming solar "
                "radiation.",
        "how": "Air temperature and radiation are measured at 2 m. Soil temperature is "
               "averaged over depth, weighted by layer thickness.",
        "why": "Radiation and air temperature determine the reference evapotranspiration "
               "(ETo).",
    },
    "vpd_temperature": {
        "what": "Vapour pressure deficit (VPD), the drying power of the air, and air "
                "temperature.",
        "how": "$\\mathrm{VPD}=e_s(T)-e_a$ (kPa), where $e_s$ is the saturation vapour "
               "pressure at air temperature $T$ and $e_a$ is the actual vapour pressure. "
               "VPD is 0 when the air is saturated.",
        "why": "High VPD increases the water demand. Plants can respond by closing their "
               "stomata, which reduces growth.",
    },
    "evapotranspiration": {
        "what": "Daily reference evapotranspiration (ETo), Makkink ETa, rainfall, the "
                "water stress factor (WSF), and the totals since the start of the "
                "selected period.",
        "how": _MAKKINK + "\n\n" + _ETA + "\n\n" + _WSF,
        "why": "The shaded area is the demand that the soil did not supply. Makkink ETa is "
               "calculated from measured soil moisture, measured weather and SoilGrids "
               "soil properties. It is not measured directly.",
    },
    "flux": {
        "what": "Daily evapotranspiration (ET) from June to August: measured by eddy "
                "covariance, simulated by STEMMUS-SCOPE, and calculated as Makkink ETo and "
                "Makkink ETa (ETo × WSF). Eddy covariance is the measured actual ET.",
        "how": "The tower ET is $LE/\\lambda$, where $LE$ is the latent heat flux and "
               "$\\lambda=2.45$ MJ/kg. A day is used if at least 24 of its 48 half-hours "
               "are measured and at most 2 are neither measured nor best-class "
               "gap-filled.\n\n"
               + _MAKKINK + "\n\n" + _ETA,
        "why": "The tower measurements are used to evaluate the model and the calculated "
               "ET. The tower represents a larger area than one soil sensor.",
    },
    "wind_rose": {
        "what": "The frequency of wind **from** each direction. Colours show wind speed.",
        "how": "The frequency is a percentage of all valid records at 2 m. Calm records "
               "are reported separately.",
        "why": "It shows the prevailing wind direction, which can be used to position "
               "hedges and windbreaks.",
    },
    # -- Diagnostics
    "timeseries": {
        "what": "One variable as recorded by the logger.",
        "how": "Long periods are shown as averages; rainfall is shown as totals. The data "
               "are not quality-checked.",
        "why": "Gaps, constant values and sudden jumps usually indicate a sensor problem.",
    },
    "coverage": {
        "what": "The fraction of expected readings received, for each sensor and day.",
        "how": "The number of readings received is divided by the number expected per "
               "day.",
        "why": "Missing data can affect totals and averages in the other charts.",
    },
}


def text(key: str) -> str:
    """An entry as Markdown, or "" when the chart has none."""
    entry = EXPLAIN.get(key)
    if not entry:
        return ""
    return "\n\n".join(f"**{head}** {entry[part]}" for head, part in
                       (("What it is.", "what"), ("How.", "how"),
                        ("Why it matters.", "why")))
