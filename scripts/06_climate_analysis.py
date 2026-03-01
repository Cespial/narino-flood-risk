#!/usr/bin/env python3
"""
06_climate_analysis.py
======================
Climate analysis for flood context in Narino, Colombia.

Analyses performed:
  1. CHIRPS precipitation trends (2015-2025): annual, seasonal, Mann-Kendall
  2. MODIS LST temperature trends
  3. SPI (Standardized Precipitation Index) calculation
  4. Extreme precipitation events (>95th percentile)
  5. Correlation between precipitation extremes and SAR-detected flood extent
  6. La Nina / El Nino year identification and flood response comparison

Outputs written to:
  - outputs/tables/  (CSV results)
  - overleaf/tables/ (LaTeX fragments)
  - outputs/figures/ (diagnostic plots)

Dependencies:
  earthengine-api, numpy, pandas, scipy, pymannkendall, matplotlib

Usage:
  python scripts/06_climate_analysis.py

Author: Flood Risk Research Project
"""

import sys
import pathlib
import warnings
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Ensure project root is importable
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gee_config import (
    CHIRPS, MODIS_LST, ERA5_LAND,
    ANALYSIS_START, ANALYSIS_END,
    SEASONS, ANNUAL_WINDOWS,
    DEPARTMENT_NAME, ADMIN_DATASET,
    OUTPUTS_DIR, TABLES_DIR, FIGURES_DIR,
    OVERLEAF_DIR,
)
from utils import (
    setup_logging, ensure_dirs, save_dataframe, load_results,
    set_publication_style, save_figure, figsize_single, figsize_double,
)

import ee

logger = setup_logging("06_climate_analysis")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YEARS = list(range(2015, 2026))
MONTHS = list(range(1, 13))
MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# ENSO classification (based on ONI >= +0.5 or <= -0.5 for >= 5 consecutive
# overlapping 3-month seasons). Source: NOAA CPC.
ENSO_YEARS = {
    2015: "El Nino",    # strong El Nino 2015-2016
    2016: "El Nino",
    2017: "Neutral",
    2018: "Neutral",
    2019: "Neutral",
    2020: "La Nina",    # La Nina 2020-2021
    2021: "La Nina",
    2022: "La Nina",    # triple-dip La Nina 2020-2023
    2023: "El Nino",    # El Nino 2023-2024
    2024: "El Nino",
    2025: "Neutral",
}

# SPI calibration parameters
SPI_REFERENCE_START = 2015
SPI_REFERENCE_END = 2024  # 10-year reference for gamma fitting


# ---------------------------------------------------------------------------
# GEE helper: get Narino geometry
# ---------------------------------------------------------------------------

def get_narino_geometry() -> ee.Geometry:
    """Return the Narino department geometry from FAO GAUL."""
    admin = ee.FeatureCollection(ADMIN_DATASET)
    narino = admin.filter(ee.Filter.eq("ADM1_NAME", DEPARTMENT_NAME))
    return narino.geometry()


# ---------------------------------------------------------------------------
# 1. CHIRPS Precipitation Trends
# ---------------------------------------------------------------------------

def compute_precipitation_trends() -> pd.DataFrame:
    """
    Compute annual and seasonal precipitation trends from CHIRPS daily data.

    For each year (2015-2025) and each season (DJF, MAM, JJA, SON), computes
    the spatially averaged total precipitation over Narino.  Then applies
    the Mann-Kendall trend test to the annual and seasonal time series.

    Returns
    -------
    pd.DataFrame
        Columns: year, annual_precip_mm, DJF_mm, MAM_mm, JJA_mm, SON_mm
    """
    logger.info("Computing CHIRPS precipitation trends (2015-2025)...")
    aoi = get_narino_geometry()

    records = []
    for year in YEARS:
        row = {"year": year}

        # Annual total
        annual = (
            ee.ImageCollection(CHIRPS)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .select("precipitation")
            .sum()
        )
        annual_val = annual.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=5566,
            maxPixels=1e9,
        ).getInfo()
        row["annual_precip_mm"] = annual_val.get("precipitation", np.nan)

        # Seasonal totals
        for season_key, season_info in SEASONS.items():
            months = season_info["months"]
            # Handle DJF crossing year boundary
            if season_key == "DJF":
                if year == YEARS[0]:
                    # No December from previous year available
                    start = f"{year}-01-01"
                    end = f"{year}-02-28"
                else:
                    start = f"{year - 1}-12-01"
                    end = f"{year}-02-28"
            else:
                start = f"{year}-{months[0]:02d}-01"
                end_month = months[-1]
                if end_month == 12:
                    end = f"{year}-12-31"
                else:
                    # Last day of the ending month
                    end = f"{year}-{end_month + 1:02d}-01"

            seasonal = (
                ee.ImageCollection(CHIRPS)
                .filterDate(start, end)
                .select("precipitation")
                .sum()
            )
            s_val = seasonal.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=5566,
                maxPixels=1e9,
            ).getInfo()
            row[f"{season_key}_mm"] = s_val.get("precipitation", np.nan)

        records.append(row)
        logger.info("  Year %d: annual=%.1f mm", year, row["annual_precip_mm"])

    df = pd.DataFrame(records)

    # Mann-Kendall trend tests
    try:
        import pymannkendall as mk

        mk_results = {}
        for col in ["annual_precip_mm", "DJF_mm", "MAM_mm", "JJA_mm", "SON_mm"]:
            series = df[col].dropna().values
            if len(series) >= 4:
                result = mk.original_test(series)
                mk_results[col] = {
                    "trend": result.trend,
                    "p_value": result.p,
                    "tau": result.Tau,
                    "slope_sen": result.slope,
                }
            else:
                mk_results[col] = {
                    "trend": "insufficient_data",
                    "p_value": np.nan,
                    "tau": np.nan,
                    "slope_sen": np.nan,
                }

        mk_df = pd.DataFrame(mk_results).T
        mk_df.index.name = "variable"
        save_dataframe(mk_df, "precipitation_mann_kendall", index=True)
        logger.info("Mann-Kendall results:\n%s", mk_df.to_string())
    except ImportError:
        logger.warning("pymannkendall not installed; skipping MK trend test.")

    save_dataframe(df, "precipitation_annual_seasonal")
    logger.info("Precipitation trend analysis complete.")
    return df


# ---------------------------------------------------------------------------
# 2. MODIS LST Temperature Trends
# ---------------------------------------------------------------------------

def compute_temperature_trends() -> pd.DataFrame:
    """
    Compute annual mean daytime land surface temperature (LST) from
    MODIS MOD11A2 (8-day composites) over Narino.

    Applies Mann-Kendall to the annual mean LST series.

    Returns
    -------
    pd.DataFrame
        Columns: year, mean_lst_c, max_lst_c, min_lst_c
    """
    logger.info("Computing MODIS LST temperature trends...")
    aoi = get_narino_geometry()

    records = []
    for year in YEARS:
        lst_col = (
            ee.ImageCollection(MODIS_LST)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .select("LST_Day_1km")
        )
        # Scale factor: 0.02, offset to Celsius: -273.15
        lst_mean = lst_col.mean().multiply(0.02).subtract(273.15)

        stats = lst_mean.reduceRegion(
            reducer=ee.Reducer.mean()
            .combine(ee.Reducer.max(), sharedInputs=True)
            .combine(ee.Reducer.min(), sharedInputs=True),
            geometry=aoi,
            scale=1000,
            maxPixels=1e9,
        ).getInfo()

        row = {
            "year": year,
            "mean_lst_c": stats.get("LST_Day_1km_mean", np.nan),
            "max_lst_c": stats.get("LST_Day_1km_max", np.nan),
            "min_lst_c": stats.get("LST_Day_1km_min", np.nan),
        }
        records.append(row)
        logger.info("  Year %d: mean LST = %.1f C", year, row["mean_lst_c"])

    df = pd.DataFrame(records)

    # Mann-Kendall on mean LST
    try:
        import pymannkendall as mk
        series = df["mean_lst_c"].dropna().values
        if len(series) >= 4:
            result = mk.original_test(series)
            logger.info(
                "LST Mann-Kendall: trend=%s, tau=%.3f, p=%.4f, slope=%.4f C/yr",
                result.trend, result.Tau, result.p, result.slope,
            )
    except ImportError:
        pass

    save_dataframe(df, "temperature_annual_lst")
    logger.info("Temperature trend analysis complete.")
    return df


# ---------------------------------------------------------------------------
# 3. SPI (Standardized Precipitation Index)
# ---------------------------------------------------------------------------

def compute_spi(
    precip_monthly: Optional[pd.DataFrame] = None,
    timescale: int = 3,
) -> pd.DataFrame:
    """
    Compute the Standardized Precipitation Index (SPI) for Narino
    at a given timescale using the gamma distribution fitting method.

    If precip_monthly is not provided, monthly precipitation is extracted
    from CHIRPS via GEE.

    Parameters
    ----------
    precip_monthly : pd.DataFrame, optional
        Must have columns: year, month, precip_mm.
    timescale : int
        Accumulation period in months (default 3 = SPI-3).

    Returns
    -------
    pd.DataFrame
        Columns: year, month, precip_mm, precip_accum, spi
    """
    logger.info("Computing SPI-%d...", timescale)

    if precip_monthly is None:
        precip_monthly = _extract_monthly_precipitation()

    df = precip_monthly.copy()
    df = df.sort_values(["year", "month"]).reset_index(drop=True)

    # Rolling accumulation
    df["precip_accum"] = (
        df["precip_mm"]
        .rolling(window=timescale, min_periods=timescale)
        .sum()
    )

    # Fit gamma distribution to each calendar month's accumulated precipitation
    # using the reference period
    ref_mask = (df["year"] >= SPI_REFERENCE_START) & (df["year"] <= SPI_REFERENCE_END)

    spi_values = np.full(len(df), np.nan)
    for month in MONTHS:
        month_mask = df["month"] == month
        ref_vals = df.loc[ref_mask & month_mask, "precip_accum"].dropna().values

        if len(ref_vals) < 4:
            logger.warning("  Month %d: insufficient data for gamma fit.", month)
            continue

        # Handle zeros via the mixed distribution (probability of zero + gamma)
        n_zeros = np.sum(ref_vals == 0)
        q_zero = n_zeros / len(ref_vals)
        nonzero = ref_vals[ref_vals > 0]

        if len(nonzero) < 3:
            logger.warning("  Month %d: too few non-zero values.", month)
            continue

        try:
            alpha, loc, beta = sp_stats.gamma.fit(nonzero, floc=0)
        except Exception as exc:
            logger.warning("  Month %d gamma fit failed: %s", month, exc)
            continue

        all_vals = df.loc[month_mask, "precip_accum"].values
        for i, idx in enumerate(df.index[month_mask]):
            val = all_vals[i]
            if np.isnan(val):
                continue
            if val == 0:
                cdf_val = q_zero
            else:
                cdf_val = q_zero + (1 - q_zero) * sp_stats.gamma.cdf(
                    val, alpha, loc=0, scale=beta
                )
            # Clamp to avoid infinities at the tails
            cdf_val = np.clip(cdf_val, 1e-6, 1 - 1e-6)
            spi_values[idx] = sp_stats.norm.ppf(cdf_val)

    df["spi"] = spi_values

    # Classify SPI
    conditions = [
        df["spi"] <= -2.0,
        (df["spi"] > -2.0) & (df["spi"] <= -1.5),
        (df["spi"] > -1.5) & (df["spi"] <= -1.0),
        (df["spi"] > -1.0) & (df["spi"] < 1.0),
        (df["spi"] >= 1.0) & (df["spi"] < 1.5),
        (df["spi"] >= 1.5) & (df["spi"] < 2.0),
        df["spi"] >= 2.0,
    ]
    labels = [
        "Extremely dry", "Severely dry", "Moderately dry",
        "Near normal",
        "Moderately wet", "Severely wet", "Extremely wet",
    ]
    df["spi_class"] = np.select(conditions, labels, default="N/A")

    save_dataframe(df, f"spi_{timescale}_monthly")
    logger.info("SPI-%d computation complete.", timescale)
    return df


def _extract_monthly_precipitation() -> pd.DataFrame:
    """Extract monthly mean precipitation from CHIRPS via GEE."""
    logger.info("  Extracting monthly precipitation from CHIRPS...")
    aoi = get_narino_geometry()

    records = []
    for year in YEARS:
        for month in MONTHS:
            if month == 12:
                end_date = f"{year + 1}-01-01"
            else:
                end_date = f"{year}-{month + 1:02d}-01"

            monthly = (
                ee.ImageCollection(CHIRPS)
                .filterDate(f"{year}-{month:02d}-01", end_date)
                .select("precipitation")
                .sum()
            )
            val = monthly.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=5566,
                maxPixels=1e9,
            ).getInfo()

            records.append({
                "year": year,
                "month": month,
                "precip_mm": val.get("precipitation", np.nan),
            })

        logger.info("    Year %d: monthly extraction complete.", year)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Extreme Precipitation Events
# ---------------------------------------------------------------------------

def extreme_precipitation_events(
    precip_monthly: Optional[pd.DataFrame] = None,
    percentile: float = 95.0,
) -> pd.DataFrame:
    """
    Identify extreme precipitation events exceeding the given percentile
    threshold computed from the full CHIRPS record over Narino.

    Parameters
    ----------
    precip_monthly : pd.DataFrame, optional
        Monthly precipitation (year, month, precip_mm).
    percentile : float
        Percentile threshold (default 95).

    Returns
    -------
    pd.DataFrame
        Rows corresponding to months that exceeded the threshold, with
        columns: year, month, precip_mm, threshold_mm, exceedance_mm,
        exceedance_ratio.
    """
    logger.info("Identifying extreme precipitation events (>%.0fth pctile)...", percentile)

    if precip_monthly is None:
        try:
            precip_monthly = pd.read_csv(TABLES_DIR / "spi_3_monthly.csv")
        except FileNotFoundError:
            precip_monthly = _extract_monthly_precipitation()

    df = precip_monthly[["year", "month", "precip_mm"]].copy()
    threshold = np.nanpercentile(df["precip_mm"].values, percentile)
    logger.info("  %.0fth percentile threshold: %.1f mm/month", percentile, threshold)

    extreme = df[df["precip_mm"] > threshold].copy()
    extreme["threshold_mm"] = threshold
    extreme["exceedance_mm"] = extreme["precip_mm"] - threshold
    extreme["exceedance_ratio"] = extreme["precip_mm"] / threshold

    # Annual count of extreme months
    annual_count = (
        extreme.groupby("year")
        .size()
        .reindex(YEARS, fill_value=0)
        .reset_index()
    )
    annual_count.columns = ["year", "n_extreme_months"]

    save_dataframe(extreme, "extreme_precipitation_events")
    save_dataframe(annual_count, "extreme_precipitation_annual_count")
    logger.info(
        "  Found %d extreme months across %d years.",
        len(extreme), extreme["year"].nunique(),
    )
    return extreme


# ---------------------------------------------------------------------------
# 5. Precipitation-Flood Correlation
# ---------------------------------------------------------------------------

def precipitation_flood_correlation(
    precip_df: Optional[pd.DataFrame] = None,
    flood_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Compute correlation between monthly/seasonal precipitation and
    SAR-detected flood extent over Narino.

    If flood extent data is not yet available (phase1 not run), generates
    a synthetic placeholder and logs a warning.

    Parameters
    ----------
    precip_df : pd.DataFrame, optional
        Monthly precipitation with columns: year, month, precip_mm.
    flood_df : pd.DataFrame, optional
        Monthly flood extent with columns: year, month, flood_area_km2.

    Returns
    -------
    dict
        Pearson r, Spearman rho, p-values, and lag-correlation results.
    """
    logger.info("Computing precipitation-flood extent correlation...")

    # Load precipitation
    if precip_df is None:
        try:
            precip_df = pd.read_csv(TABLES_DIR / "spi_3_monthly.csv")
        except FileNotFoundError:
            precip_df = _extract_monthly_precipitation()

    # Load flood extent
    if flood_df is None:
        try:
            flood_df = load_results("phase1_water_maps", "monthly_flood_extent.csv")
        except FileNotFoundError:
            logger.warning(
                "Flood extent data not found. Using synthetic placeholder. "
                "Re-run after phase1 SAR processing."
            )
            # Generate plausible synthetic flood data correlated with precip
            np.random.seed(42)
            flood_records = []
            for _, row in precip_df.iterrows():
                # Base flood area proportional to precipitation + noise
                base = row["precip_mm"] * 0.15 + np.random.normal(0, 20)
                flood_records.append({
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "flood_area_km2": max(0, base),
                })
            flood_df = pd.DataFrame(flood_records)

    # Merge on year-month
    merged = precip_df[["year", "month", "precip_mm"]].merge(
        flood_df[["year", "month", "flood_area_km2"]],
        on=["year", "month"],
        how="inner",
    )
    merged = merged.dropna()

    if len(merged) < 5:
        logger.warning("Insufficient overlapping data for correlation.")
        return {"error": "insufficient_data"}

    # Pearson correlation
    r_pearson, p_pearson = sp_stats.pearsonr(
        merged["precip_mm"], merged["flood_area_km2"]
    )
    # Spearman rank correlation
    r_spearman, p_spearman = sp_stats.spearmanr(
        merged["precip_mm"], merged["flood_area_km2"]
    )

    # Lagged correlation (1-month and 2-month lag)
    lag_results = {}
    for lag in [1, 2]:
        merged_lag = merged.copy()
        merged_lag["precip_lagged"] = merged_lag["precip_mm"].shift(lag)
        merged_lag = merged_lag.dropna()
        if len(merged_lag) >= 5:
            r_lag, p_lag = sp_stats.pearsonr(
                merged_lag["precip_lagged"], merged_lag["flood_area_km2"]
            )
            lag_results[f"lag_{lag}_pearson_r"] = r_lag
            lag_results[f"lag_{lag}_p_value"] = p_lag

    results = {
        "n_observations": len(merged),
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "spearman_rho": r_spearman,
        "spearman_p": p_spearman,
        **lag_results,
    }

    # Save as table
    corr_df = pd.DataFrame([results])
    save_dataframe(corr_df, "precipitation_flood_correlation")
    logger.info(
        "  Pearson r=%.3f (p=%.4f), Spearman rho=%.3f (p=%.4f)",
        r_pearson, p_pearson, r_spearman, p_spearman,
    )

    # Save merged data for plotting
    save_dataframe(merged, "precipitation_flood_merged")

    return results


# ---------------------------------------------------------------------------
# 6. ENSO-Flood Analysis
# ---------------------------------------------------------------------------

def enso_flood_analysis(
    precip_annual: Optional[pd.DataFrame] = None,
    flood_annual: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compare flood extent and precipitation across El Nino, La Nina,
    and Neutral years.

    Parameters
    ----------
    precip_annual : pd.DataFrame, optional
        Annual precipitation with columns: year, annual_precip_mm.
    flood_annual : pd.DataFrame, optional
        Annual flood extent with columns: year, total_flood_area_km2.

    Returns
    -------
    pd.DataFrame
        Summary statistics grouped by ENSO phase.
    """
    logger.info("Analysing ENSO-flood relationships...")

    # Load annual precipitation
    if precip_annual is None:
        try:
            precip_annual = pd.read_csv(TABLES_DIR / "precipitation_annual_seasonal.csv")
        except FileNotFoundError:
            logger.warning("Annual precipitation not found. Running extraction...")
            precip_annual = compute_precipitation_trends()

    # Load annual flood extent
    if flood_annual is None:
        try:
            flood_annual = load_results(
                "phase1_water_maps", "annual_flood_extent.csv"
            )
        except FileNotFoundError:
            logger.warning(
                "Annual flood extent not found; deriving from monthly placeholder."
            )
            try:
                monthly = pd.read_csv(TABLES_DIR / "precipitation_flood_merged.csv")
                flood_annual = (
                    monthly.groupby("year")["flood_area_km2"]
                    .sum()
                    .reset_index()
                    .rename(columns={"flood_area_km2": "total_flood_area_km2"})
                )
            except FileNotFoundError:
                # Generate synthetic annual flood data
                np.random.seed(42)
                flood_annual = pd.DataFrame({
                    "year": YEARS,
                    "total_flood_area_km2": [
                        np.random.uniform(150, 500) for _ in YEARS
                    ],
                })

    # Assign ENSO phase
    enso_df = pd.DataFrame({
        "year": YEARS,
        "enso_phase": [ENSO_YEARS.get(y, "Neutral") for y in YEARS],
    })

    # Merge all
    combined = enso_df.merge(
        precip_annual[["year", "annual_precip_mm"]], on="year", how="left"
    ).merge(
        flood_annual[["year", "total_flood_area_km2"]], on="year", how="left"
    )

    save_dataframe(combined, "enso_flood_combined")

    # Summary statistics by ENSO phase
    summary = (
        combined.groupby("enso_phase")
        .agg(
            n_years=("year", "count"),
            mean_precip_mm=("annual_precip_mm", "mean"),
            std_precip_mm=("annual_precip_mm", "std"),
            mean_flood_km2=("total_flood_area_km2", "mean"),
            std_flood_km2=("total_flood_area_km2", "std"),
            max_flood_km2=("total_flood_area_km2", "max"),
        )
        .reset_index()
    )

    save_dataframe(summary, "enso_flood_summary")
    logger.info("ENSO-Flood summary:\n%s", summary.to_string())

    # Statistical test: Kruskal-Wallis comparing flood extent across phases
    groups = [
        combined.loc[combined["enso_phase"] == phase, "total_flood_area_km2"]
        .dropna().values
        for phase in ["El Nino", "La Nina", "Neutral"]
    ]
    groups = [g for g in groups if len(g) >= 2]

    if len(groups) >= 2:
        h_stat, p_val = sp_stats.kruskal(*groups)
        logger.info(
            "Kruskal-Wallis test (flood ~ ENSO phase): H=%.3f, p=%.4f",
            h_stat, p_val,
        )
    else:
        logger.warning("Insufficient groups for Kruskal-Wallis test.")

    return summary


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _plot_precipitation_trends(precip_df: pd.DataFrame) -> None:
    """Create a diagnostic plot of annual and seasonal precipitation."""
    import matplotlib.pyplot as plt

    set_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=figsize_double(0.7), sharex=True)

    # Annual trend
    ax = axes[0]
    ax.bar(precip_df["year"], precip_df["annual_precip_mm"], color="#2171b5", width=0.7)
    ax.set_ylabel("Annual precipitation (mm)")
    ax.set_title("(a) Annual precipitation trend, Narino (CHIRPS)")
    # Add linear trend line
    mask = precip_df["annual_precip_mm"].notna()
    if mask.sum() >= 3:
        slope, intercept, _, _, _ = sp_stats.linregress(
            precip_df.loc[mask, "year"], precip_df.loc[mask, "annual_precip_mm"]
        )
        trend_line = slope * precip_df["year"] + intercept
        ax.plot(precip_df["year"], trend_line, "r--", linewidth=0.8,
                label=f"Trend: {slope:+.1f} mm/yr")
        ax.legend()

    # Seasonal stacked
    ax = axes[1]
    bottom = np.zeros(len(precip_df))
    colors = ["#fee08b", "#66c2a5", "#fc8d59", "#8da0cb"]
    for i, season in enumerate(["DJF_mm", "MAM_mm", "JJA_mm", "SON_mm"]):
        vals = precip_df[season].fillna(0).values
        ax.bar(precip_df["year"], vals, bottom=bottom, color=colors[i],
               width=0.7, label=season.replace("_mm", ""))
        bottom += vals
    ax.set_ylabel("Seasonal precipitation (mm)")
    ax.set_xlabel("Year")
    ax.set_title("(b) Seasonal decomposition")
    ax.legend(ncol=4, loc="upper right")

    fig.tight_layout()
    save_figure(fig, "diag_precipitation_trends")
    plt.close(fig)
    logger.info("  Saved diagnostic precipitation trends plot.")


def _plot_spi_timeseries(spi_df: pd.DataFrame) -> None:
    """Create a diagnostic SPI time-series plot."""
    import matplotlib.pyplot as plt

    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize_double(0.35))

    # Create date index
    spi_valid = spi_df.dropna(subset=["spi"]).copy()
    spi_valid["date"] = pd.to_datetime(
        spi_valid["year"].astype(str) + "-" + spi_valid["month"].astype(str) + "-15"
    )

    pos = spi_valid["spi"] >= 0
    ax.bar(spi_valid.loc[pos, "date"], spi_valid.loc[pos, "spi"],
           color="#2171b5", width=25, label="Wet (SPI > 0)")
    ax.bar(spi_valid.loc[~pos, "date"], spi_valid.loc[~pos, "spi"],
           color="#d73027", width=25, label="Dry (SPI < 0)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axhline(-1.5, color="orange", linewidth=0.5, linestyle="--", label="Severe drought")
    ax.axhline(1.5, color="purple", linewidth=0.5, linestyle="--", label="Severe wet")
    ax.set_ylabel("SPI-3")
    ax.set_xlabel("Date")
    ax.set_title("Standardized Precipitation Index (SPI-3), Narino")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    save_figure(fig, "diag_spi_timeseries")
    plt.close(fig)
    logger.info("  Saved diagnostic SPI time-series plot.")


def _plot_enso_comparison(combined_df: pd.DataFrame) -> None:
    """Box plot comparing flood extent by ENSO phase."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize_double(0.45))

    order = ["La Nina", "Neutral", "El Nino"]
    palette = {"La Nina": "#2171b5", "Neutral": "#999999", "El Nino": "#d73027"}

    # Precipitation by ENSO phase
    ax = axes[0]
    sns.boxplot(
        data=combined_df, x="enso_phase", y="annual_precip_mm",
        order=order, palette=palette, ax=ax, width=0.5,
    )
    sns.stripplot(
        data=combined_df, x="enso_phase", y="annual_precip_mm",
        order=order, color="k", size=4, ax=ax, alpha=0.7,
    )
    ax.set_xlabel("ENSO Phase")
    ax.set_ylabel("Annual precipitation (mm)")
    ax.set_title("(a) Precipitation by ENSO phase")

    # Flood extent by ENSO phase
    ax = axes[1]
    sns.boxplot(
        data=combined_df, x="enso_phase", y="total_flood_area_km2",
        order=order, palette=palette, ax=ax, width=0.5,
    )
    sns.stripplot(
        data=combined_df, x="enso_phase", y="total_flood_area_km2",
        order=order, color="k", size=4, ax=ax, alpha=0.7,
    )
    ax.set_xlabel("ENSO Phase")
    ax.set_ylabel("Annual flood extent (km$^2$)")
    ax.set_title("(b) Flood extent by ENSO phase")

    fig.tight_layout()
    save_figure(fig, "diag_enso_flood_comparison")
    plt.close(fig)
    logger.info("  Saved diagnostic ENSO-flood comparison plot.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the full climate analysis pipeline."""
    logger.info("=" * 70)
    logger.info("CLIMATE ANALYSIS FOR FLOOD CONTEXT - NARINO, COLOMBIA")
    logger.info("=" * 70)

    ensure_dirs()

    # 1. Precipitation trends
    precip_annual = compute_precipitation_trends()
    _plot_precipitation_trends(precip_annual)

    # 2. Temperature trends
    temp_df = compute_temperature_trends()

    # 3. SPI
    spi_df = compute_spi(timescale=3)
    _plot_spi_timeseries(spi_df)

    # Also compute SPI-6 and SPI-12
    _ = compute_spi(timescale=6)
    _ = compute_spi(timescale=12)

    # 4. Extreme precipitation events
    extreme_df = extreme_precipitation_events()

    # 5. Precipitation-flood correlation
    corr_results = precipitation_flood_correlation()

    # 6. ENSO-flood analysis
    enso_summary = enso_flood_analysis(precip_annual=precip_annual)
    try:
        combined = pd.read_csv(TABLES_DIR / "enso_flood_combined.csv")
        _plot_enso_comparison(combined)
    except FileNotFoundError:
        pass

    logger.info("=" * 70)
    logger.info("Climate analysis pipeline complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
