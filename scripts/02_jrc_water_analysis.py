#!/usr/bin/env python3
"""
02_jrc_water_analysis.py
========================
Analysis of the JRC Global Surface Water (GSW) v1.4 dataset for Narino.

This module:
  1. Loads all JRC GSW layers (occurrence, change, seasonality, recurrence,
     transitions).
  2. Computes flood frequency maps from monthly water history.
  3. Calculates seasonal water dynamics (wet-season vs. dry-season extent).
  4. Determines water occurrence trends (increasing / decreasing).
  5. Validates SAR-derived water extents against JRC reference.

References:
  - Pekel et al. (2016). Nature 540:418-422.
    "High-resolution mapping of global surface water and its long-term
    changes."

Usage:
    python 02_jrc_water_analysis.py
    python 02_jrc_water_analysis.py --no-export

Author : Narino Flood Risk Research Project
Date   : 2026-02-26
"""

import sys
import pathlib
import argparse
from typing import Dict, Tuple

import ee

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gee_config as cfg  # noqa: E402
from scripts.utils import (  # noqa: E402
    get_study_area,
    get_study_area_geometry,
    get_municipalities,
    setup_logging,
    safe_getinfo,
    export_to_drive,
    export_table_to_drive,
    monitor_tasks,
)

log = setup_logging("02_jrc_water")


# ===========================================================================
# JRC LAYER ACCESS
# ===========================================================================

def get_jrc_layers(region: ee.Geometry) -> Dict[str, ee.Image]:
    """
    Load all JRC Global Surface Water v1.4 layers clipped to *region*.

    Layers returned
    ---------------
    - **occurrence** : % of time water present (0-100).
    - **change_abs** : Absolute change in occurrence between first/last year.
    - **change_norm** : Normalised change.
    - **seasonality** : Number of months water present per year.
    - **recurrence** : Frequency of interannual water recurrence (0-100 %).
    - **transitions** : Categorical layer encoding water-class transitions.
    - **max_extent** : Maximum observed water extent (binary).

    Parameters
    ----------
    region : ee.Geometry
        Clipping geometry.

    Returns
    -------
    dict[str, ee.Image]
        Dictionary of named JRC layers.
    """
    gsw = ee.Image(cfg.JRC_GSW).clip(region)

    layers = {
        "occurrence": gsw.select("occurrence"),
        "change_abs": gsw.select("change_abs"),
        "change_norm": gsw.select("change_norm"),
        "seasonality": gsw.select("seasonality"),
        "recurrence": gsw.select("recurrence"),
        "transitions": gsw.select("transition"),
        "max_extent": gsw.select("max_extent"),
    }
    log.info("JRC GSW v1.4 layers loaded (%d bands)", len(layers))
    return layers


# ===========================================================================
# FLOOD FREQUENCY
# ===========================================================================

def compute_flood_frequency(region: ee.Geometry) -> ee.Image:
    """
    Compute a classified flood frequency map from JRC monthly water history.

    For each pixel, counts the number of months where water was detected
    across the full JRC record (1984-present), normalises to a percentage,
    then classifies using ``gee_config.FLOOD_FREQUENCY_CLASSES``.

    Parameters
    ----------
    region : ee.Geometry
        Analysis region.

    Returns
    -------
    ee.Image
        Multi-band image with:
        - ``'flood_freq_pct'`` : continuous frequency (0-100 %).
        - ``'flood_freq_class'`` : classified frequency (1-5).
    """
    # JRC Monthly History: band 'water' -> 0 = no data, 1 = not water, 2 = water
    monthly = (
        ee.ImageCollection(cfg.JRC_GSW_MONTHLY)
        .filterBounds(region)
    )

    # Count water observations and total valid observations per pixel
    def _is_water(img):
        return img.select("water").eq(2).rename("is_water")

    def _is_valid(img):
        return img.select("water").gte(1).rename("is_valid")

    water_count = monthly.map(_is_water).sum().rename("water_count")
    valid_count = monthly.map(_is_valid).sum().rename("valid_count")

    # Frequency as percentage of valid observations
    freq_pct = (
        water_count
        .divide(valid_count.max(1))  # avoid division by zero
        .multiply(100)
        .rename("flood_freq_pct")
        .clip(region)
    )

    # Classify into flood frequency categories
    classified = ee.Image(0).rename("flood_freq_class")
    for idx, (_, cls) in enumerate(cfg.FLOOD_FREQUENCY_CLASSES.items(), start=1):
        low, high = cls["range"]
        mask = freq_pct.gte(low).And(freq_pct.lt(high))
        classified = classified.where(mask, idx)
    classified = classified.selfMask().clip(region)

    result = freq_pct.addBands(classified)
    log.info("Flood frequency map computed (5 classes)")
    return result


# ===========================================================================
# SEASONAL DYNAMICS
# ===========================================================================

def seasonal_dynamics(region: ee.Geometry) -> Dict[str, ee.Image]:
    """
    Compute seasonal water extent differences for Narino.

    Uses Colombia's bimodal precipitation pattern defined in
    ``gee_config.SEASONS`` to compare wet-season and dry-season water
    extents from JRC monthly water history.

    Parameters
    ----------
    region : ee.Geometry
        Analysis region.

    Returns
    -------
    dict[str, ee.Image]
        Keys: season abbreviations (``'DJF'``, ``'MAM'``, ``'JJA'``,
        ``'SON'``) mapped to mean water frequency images, plus
        ``'wet_dry_diff'`` (wet - dry difference).
    """
    monthly = (
        ee.ImageCollection(cfg.JRC_GSW_MONTHLY)
        .filterBounds(region)
    )

    seasonal_images = {}
    for season_code, season_info in cfg.SEASONS.items():
        months = season_info["months"]
        season_filter = ee.Filter.inList(
            "month", months
        )

        # Filter by month using a computed property
        def _add_month(img):
            return img.set("month", ee.Date(img.get("system:time_start")).get("month"))

        monthly_with_month = monthly.map(_add_month)
        season_coll = monthly_with_month.filter(season_filter)

        # Mean water occurrence for this season
        water_freq = (
            season_coll
            .map(lambda img: img.select("water").eq(2).rename("water"))
            .mean()
            .multiply(100)
            .rename(f"water_freq_{season_code}")
            .clip(region)
        )
        seasonal_images[season_code] = water_freq
        log.info(
            "Seasonal water extent computed: %s (%s) - months %s",
            season_code, season_info["label"], months,
        )

    # Wet vs dry difference
    # Wet seasons: MAM (first rains) + SON (peak floods)
    # Dry seasons: DJF + JJA
    wet_mean = seasonal_images["MAM"].add(seasonal_images["SON"]).divide(2)
    dry_mean = seasonal_images["DJF"].add(seasonal_images["JJA"]).divide(2)
    diff = wet_mean.subtract(dry_mean).rename("wet_dry_diff").clip(region)
    seasonal_images["wet_dry_diff"] = diff

    log.info("Wet-dry difference computed (positive = more water in wet season)")
    return seasonal_images


# ===========================================================================
# WATER OCCURRENCE TRENDS
# ===========================================================================

def water_trend_analysis(
    region: ee.Geometry,
    start_year: int = 2000,
    end_year: int = 2023,
) -> ee.Image:
    """
    Analyse trends in annual water occurrence from JRC yearly history.

    Fits a linear regression to annual water occurrence per pixel to
    detect increasing or decreasing surface water trends.

    Parameters
    ----------
    region : ee.Geometry
        Analysis region.
    start_year, end_year : int
        Year range for trend analysis.

    Returns
    -------
    ee.Image
        Multi-band image with:
        - ``'water_trend_slope'`` : slope of linear fit (% / year).
        - ``'water_trend_direction'`` : -1 (decreasing), 0 (stable),
          +1 (increasing).
    """
    yearly = (
        ee.ImageCollection(cfg.JRC_GSW_YEARLY)
        .filterBounds(region)
        .filterDate(f"{start_year}-01-01", f"{end_year + 1}-01-01")
    )

    def _annual_water_pct(img):
        """Compute percentage of water for a yearly image."""
        # JRC yearly: 1 = not water, 2 = seasonal water, 3 = permanent water
        water = img.select("waterClass").gte(2).rename("water_pct")
        year = ee.Date(img.get("system:time_start")).get("year")
        return water.set("year", year).toFloat()

    annual_water = yearly.map(_annual_water_pct)

    # Add a constant band and a time band for linear regression
    def _add_time(img):
        year = ee.Number(img.get("year"))
        time_band = ee.Image(year).rename("time").toFloat()
        constant = ee.Image(1).rename("constant").toFloat()
        return img.addBands(time_band).addBands(constant)

    regression_input = annual_water.map(_add_time)

    # Linear regression: water_pct = a * time + b
    regression = regression_input.select(["constant", "time", "water_pct"]).reduce(
        ee.Reducer.linearRegression(numX=2, numY=1)
    )

    coefficients = regression.select("coefficients").arrayProject([0]).arrayFlatten(
        [["constant", "slope"]]
    )
    slope = coefficients.select("slope").rename("water_trend_slope").clip(region)

    # Classify direction
    direction = (
        ee.Image(0)
        .where(slope.gt(0.005), 1)     # increasing
        .where(slope.lt(-0.005), -1)    # decreasing
        .rename("water_trend_direction")
        .clip(region)
    )

    result = slope.addBands(direction)
    log.info("Water trend analysis: %d-%d (%d years)", start_year, end_year, end_year - start_year + 1)
    return result


# ===========================================================================
# VALIDATION: SAR vs JRC
# ===========================================================================

def validate_sar_with_jrc(
    sar_water: ee.Image,
    region: ee.Geometry,
    scale: int = 30,
    num_samples: int = 10000,
) -> Dict[str, ee.Number]:
    """
    Validate a SAR-derived water mask against JRC occurrence data.

    Computes agreement metrics between the SAR binary water mask and a
    JRC-derived reference water mask (occurrence >= 50 %).

    Parameters
    ----------
    sar_water : ee.Image
        Binary SAR water mask (1 = water).
    region : ee.Geometry
        Validation region.
    scale : int
        Sampling scale in meters.
    num_samples : int
        Number of random validation points.

    Returns
    -------
    dict[str, ee.Number]
        Dictionary with validation metrics:
        ``'overall_accuracy'``, ``'precision'``, ``'recall'``, ``'f1'``,
        ``'kappa'``, ``'n_samples'``.
    """
    # JRC reference: pixels with occurrence >= 50% considered water
    jrc = ee.Image(cfg.JRC_GSW).select("occurrence").clip(region)
    jrc_water = jrc.gte(50).rename("jrc_water")

    # Stack SAR and JRC
    stacked = sar_water.rename("sar_water").addBands(jrc_water)

    # Stratified random sample
    sample = stacked.stratifiedSample(
        numPoints=num_samples,
        classBand="jrc_water",
        region=region,
        scale=scale,
        seed=cfg.CV_PARAMS["random_state"],
        geometries=False,
    )

    # Confusion matrix
    error_matrix = sample.errorMatrix("jrc_water", "sar_water")

    overall_accuracy = error_matrix.accuracy()
    kappa = error_matrix.kappa()

    # Precision and recall from the matrix (class 1 = water)
    matrix_array = error_matrix.array()

    # True positives, false positives, false negatives
    tp = ee.Number(matrix_array.get([1, 1]))
    fp = ee.Number(matrix_array.get([0, 1]))
    fn = ee.Number(matrix_array.get([1, 0]))

    precision = tp.divide(tp.add(fp).max(1))
    recall = tp.divide(tp.add(fn).max(1))
    f1 = precision.multiply(recall).multiply(2).divide(precision.add(recall).max(0.001))

    metrics = {
        "overall_accuracy": overall_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
        "n_samples": sample.size(),
    }

    log.info("SAR vs JRC validation prepared (%d sample points)", num_samples)
    return metrics


# ===========================================================================
# EXPORT PIPELINE
# ===========================================================================

def run_jrc_analysis(export: bool = True) -> list:
    """
    Execute the full JRC water analysis pipeline and export results.

    Returns
    -------
    list[ee.batch.Task]
        Started GEE export tasks.
    """
    region = get_study_area_geometry()
    tasks = []

    # --- 1. JRC core layers ---
    log.info("Loading JRC GSW layers ...")
    jrc_layers = get_jrc_layers(region)

    if export:
        for name, image in jrc_layers.items():
            task = export_to_drive(
                image=image.toFloat(),
                description=f"narino_jrc_{name}",
                region=region,
                scale=cfg.EXPORT_SCALE,
            )
            tasks.append(task)

    # --- 2. Flood frequency ---
    log.info("Computing flood frequency map ...")
    flood_freq = compute_flood_frequency(region)

    if export:
        task = export_to_drive(
            image=flood_freq.toFloat(),
            description="narino_jrc_flood_frequency",
            region=region,
            scale=cfg.EXPORT_SCALE,
        )
        tasks.append(task)

    # --- 3. Seasonal dynamics ---
    log.info("Computing seasonal water dynamics ...")
    seasonal = seasonal_dynamics(region)

    if export:
        # Stack seasonal bands into one image for export
        seasonal_stack = (
            seasonal["DJF"]
            .addBands(seasonal["MAM"])
            .addBands(seasonal["JJA"])
            .addBands(seasonal["SON"])
            .addBands(seasonal["wet_dry_diff"])
        )
        task = export_to_drive(
            image=seasonal_stack.toFloat(),
            description="narino_jrc_seasonal_dynamics",
            region=region,
            scale=cfg.EXPORT_SCALE,
        )
        tasks.append(task)

    # --- 4. Water trends ---
    log.info("Computing water occurrence trends ...")
    trends = water_trend_analysis(region)

    if export:
        task = export_to_drive(
            image=trends.toFloat(),
            description="narino_jrc_water_trends",
            region=region,
            scale=cfg.EXPORT_SCALE,
        )
        tasks.append(task)

    # --- 5. Municipal-level JRC stats ---
    log.info("Computing municipal-level JRC statistics ...")
    municipalities = get_municipalities()

    # Zonal mean of JRC occurrence per municipality
    occurrence = jrc_layers["occurrence"]
    muni_stats = occurrence.reduceRegions(
        collection=municipalities,
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.percentile([10, 50, 90]),
            sharedInputs=True,
        ),
        scale=cfg.EXPORT_SCALE,
    )

    if export:
        task = export_table_to_drive(
            collection=muni_stats,
            description="narino_jrc_municipal_occurrence_stats",
            file_format="CSV",
        )
        tasks.append(task)

    log.info("JRC analysis complete. Export tasks: %d", len(tasks))
    return tasks


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def main() -> None:
    """Command-line interface for JRC water analysis."""
    parser = argparse.ArgumentParser(
        description="JRC Global Surface Water analysis for Narino",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip exporting to Google Drive.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Block and monitor export tasks until completion.",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("JRC Global Surface Water Analysis Pipeline")
    log.info("=" * 60)

    tasks = run_jrc_analysis(export=not args.no_export)

    if args.monitor and tasks:
        monitor_tasks(tasks)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
