#!/usr/bin/env python3
"""
01_sar_water_detection.py
=========================
Sentinel-1 SAR-based surface water detection for the Narino department.

Processing chain:
  1. Load Sentinel-1 GRD IW collection, filter by region, date, and orbit.
  2. Apply focal-median speckle filtering.
  3. Implement Otsu automatic thresholding on the VV band.
  4. Generate binary water masks per image.
  5. Create monthly composites (maximum water extent per month).
  6. Export annual maximum flood extent maps (2015-2025).

References:
  - Otsu N. (1979). IEEE Trans. Syst. Man Cybern. 9(1):62-66.
  - Liang & Liu (2020). Water 12(6):1584.  SAR-based flood mapping.

Usage:
    python 01_sar_water_detection.py               # full pipeline
    python 01_sar_water_detection.py --year 2022    # single year

Author : Narino Flood Risk Research Project
Date   : 2026-02-26
"""

import sys
import pathlib
import argparse
from typing import Optional, Tuple

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
    apply_speckle_filter,
    setup_logging,
    safe_getinfo,
    export_to_drive,
    monitor_tasks,
)

log = setup_logging("01_sar_water")


# ===========================================================================
# SENTINEL-1 COLLECTION
# ===========================================================================

def get_s1_collection(
    start_date: str,
    end_date: str,
    region: ee.Geometry,
    orbit_pass: str = cfg.S1_PARAMS["orbitProperties_pass"],
    polarization: str = "VV",
) -> ee.ImageCollection:
    """
    Load and pre-filter a Sentinel-1 GRD IW collection.

    Parameters
    ----------
    start_date, end_date : str
        ISO date strings (``'YYYY-MM-DD'``).
    region : ee.Geometry
        Spatial filter geometry.
    orbit_pass : str
        ``'ASCENDING'`` or ``'DESCENDING'`` (default from config).
    polarization : str
        Band to verify presence of (``'VV'`` or ``'VH'``).

    Returns
    -------
    ee.ImageCollection
        Filtered S1 GRD collection with speckle filtering applied.
    """
    collection = (
        ee.ImageCollection(cfg.S1_COLLECTION)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", cfg.S1_PARAMS["instrumentMode"]))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", polarization))
        .select([polarization])
    )

    # Apply speckle filter to each image
    radius = cfg.SAR_WATER_THRESHOLDS["speckle_filter_radius"]
    filtered = collection.map(lambda img: apply_speckle_filter(img, radius_m=radius))

    log.info(
        "S1 collection: %s to %s, orbit=%s, pol=%s",
        start_date, end_date, orbit_pass, polarization,
    )
    return filtered


# ===========================================================================
# OTSU AUTOMATIC THRESHOLDING
# ===========================================================================

def _compute_histogram(
    image: ee.Image,
    band: str,
    region: ee.Geometry,
    scale: int,
    num_buckets: int = 256,
) -> ee.Dictionary:
    """
    Compute a histogram of *band* values within *region*.

    Returns the GEE dictionary produced by ``ee.Reducer.histogram()``.
    """
    histogram = image.select(band).reduceRegion(
        reducer=ee.Reducer.histogram(maxBuckets=num_buckets),
        geometry=region,
        scale=scale,
        maxPixels=1e10,
        bestEffort=True,
    )
    return ee.Dictionary(histogram.get(band))


def otsu_threshold(
    image: ee.Image,
    band: str = "VV",
    region: Optional[ee.Geometry] = None,
    scale: int = cfg.S1_PARAMS["resolution"],
    num_buckets: int = 256,
) -> ee.Number:
    """
    Compute the Otsu threshold for bimodal separation of *band* values.

    The Otsu method finds the threshold that minimises the intra-class
    variance (equivalently, maximises the inter-class variance) of pixel
    intensities, assuming a bimodal distribution (water vs. non-water).

    The computation is performed entirely server-side in GEE.

    Parameters
    ----------
    image : ee.Image
        SAR image (typically VV band in dB).
    band : str
        Band name to threshold.
    region : ee.Geometry, optional
        Region for histogram computation. Defaults to Narino.
    scale : int
        Scale in meters for the histogram reduction.
    num_buckets : int
        Number of histogram bins.

    Returns
    -------
    ee.Number
        Optimal threshold value (dB).
    """
    if region is None:
        region = get_study_area_geometry()

    hist_dict = _compute_histogram(image, band, region, scale, num_buckets)

    # Extract histogram arrays
    counts = ee.Array(hist_dict.get("histogram"))
    means = ee.Array(hist_dict.get("bucketMeans"))

    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    sum_all = counts.multiply(means).reduce(ee.Reducer.sum(), [0]).get([0])

    # Convert to lists for cumulative sums (server-side)
    counts_list = counts.toList()
    means_list = means.toList()
    size = counts_list.size()

    def _otsu_iterate(index, state):
        """Iterate over histogram bins accumulating between-class variance."""
        state = ee.Dictionary(state)
        index = ee.Number(index)

        count_i = ee.Number(counts_list.get(index))
        mean_i = ee.Number(means_list.get(index))

        w0 = ee.Number(state.get("w0")).add(count_i)
        sum0 = ee.Number(state.get("sum0")).add(count_i.multiply(mean_i))

        w1 = ee.Number(total).subtract(w0)

        # Guard against division by zero
        mu0 = ee.Algorithms.If(w0.gt(0), sum0.divide(w0), ee.Number(0))
        mu1_num = ee.Number(sum_all).subtract(sum0)
        mu1 = ee.Algorithms.If(w1.gt(0), mu1_num.divide(w1), ee.Number(0))

        between_var = w0.multiply(w1).multiply(
            ee.Number(mu0).subtract(ee.Number(mu1)).pow(2)
        )

        best_var = ee.Number(state.get("best_var"))
        best_thresh = ee.Number(state.get("best_thresh"))

        new_best_var = ee.Algorithms.If(between_var.gt(best_var), between_var, best_var)
        new_best_thresh = ee.Algorithms.If(between_var.gt(best_var), mean_i, best_thresh)

        return ee.Dictionary({
            "w0": w0,
            "sum0": sum0,
            "best_var": new_best_var,
            "best_thresh": new_best_thresh,
        })

    initial_state = ee.Dictionary({
        "w0": ee.Number(0),
        "sum0": ee.Number(0),
        "best_var": ee.Number(0),
        "best_thresh": ee.Number(cfg.SAR_WATER_THRESHOLDS["vv_default"]),
    })

    result = ee.Dictionary(
        ee.List.sequence(0, size.subtract(1)).iterate(_otsu_iterate, initial_state)
    )

    threshold = ee.Number(result.get("best_thresh"))

    # Clamp to physically reasonable range
    vv_min, vv_max = cfg.SAR_WATER_THRESHOLDS["vv_range"]
    threshold = threshold.max(vv_min).min(vv_max)

    return threshold


# ===========================================================================
# WATER DETECTION
# ===========================================================================

def detect_water_sar(
    image: ee.Image,
    threshold: Optional[ee.Number] = None,
    band: str = "VV",
    region: Optional[ee.Geometry] = None,
    min_area_ha: float = cfg.SAR_WATER_THRESHOLDS["min_water_area_ha"],
) -> ee.Image:
    """
    Create a binary water mask from a single SAR image.

    If no threshold is provided, the Otsu method is applied automatically.

    Parameters
    ----------
    image : ee.Image
        Speckle-filtered Sentinel-1 image (VV band, dB).
    threshold : ee.Number, optional
        Backscatter threshold (dB). Values **below** this are classified
        as water.  Computed via Otsu if not supplied.
    band : str
        Band name to threshold.
    region : ee.Geometry, optional
        Region for Otsu computation. Defaults to Narino.
    min_area_ha : float
        Minimum water-body area in hectares. Smaller clusters are removed
        to reduce commission errors from wet soil or shadow.

    Returns
    -------
    ee.Image
        Binary water mask (1 = water, 0 = non-water) named ``'water'``.
    """
    if region is None:
        region = get_study_area_geometry()

    if threshold is None:
        threshold = otsu_threshold(image, band=band, region=region)

    water = image.select(band).lt(threshold).rename("water")

    # Remove small clusters (connected-component filtering)
    min_pixels = ee.Number(min_area_ha).multiply(10000).divide(
        ee.Number(cfg.S1_PARAMS["resolution"]).pow(2)
    ).int()

    connected = water.selfMask().connectedPixelCount(maxSize=256)
    water_clean = water.updateMask(connected.gte(min_pixels)).unmask(0).rename("water")

    # Copy timestamp
    water_clean = water_clean.copyProperties(image, image.propertyNames())
    water_clean = water_clean.set("water_threshold_dB", threshold)

    return water_clean


def _add_water_band(image: ee.Image) -> ee.Image:
    """Map function: detect water and add as a band (for collection mapping)."""
    region = get_study_area_geometry()
    return detect_water_sar(image, region=region)


# ===========================================================================
# MONTHLY AND ANNUAL COMPOSITES
# ===========================================================================

def monthly_water_composite(
    year: int,
    month: int,
    region: ee.Geometry,
) -> ee.Image:
    """
    Create a monthly maximum water extent composite from Sentinel-1.

    The composite takes the **maximum** water detection value across all
    images in the month, capturing the peak flood extent.

    Parameters
    ----------
    year : int
        Calendar year.
    month : int
        Calendar month (1-12).
    region : ee.Geometry
        Spatial filter geometry.

    Returns
    -------
    ee.Image
        Binary water extent (1 = water detected at least once) for the
        month, with properties ``'year'``, ``'month'``, ``'n_images'``.
    """
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")

    s1 = get_s1_collection(
        start_date=start.format("YYYY-MM-dd"),
        end_date=end.format("YYYY-MM-dd"),
        region=region,
    )

    water_collection = s1.map(_add_water_band)
    n_images = water_collection.size()

    composite = (
        water_collection
        .select("water")
        .max()
        .rename("water_monthly_max")
        .set("year", year)
        .set("month", month)
        .set("n_images", n_images)
    )

    return composite


def annual_max_extent(
    year: int,
    region: ee.Geometry,
) -> ee.Image:
    """
    Generate the annual maximum flood extent map for a given year.

    Aggregates all monthly max composites into a single binary layer
    representing the maximum water extent observed during the year.

    Parameters
    ----------
    year : int
        Calendar year (2015-2025).
    region : ee.Geometry
        Spatial filter geometry.

    Returns
    -------
    ee.Image
        Binary annual flood extent with properties ``'year'`` and
        ``'n_months_with_data'``.
    """
    monthly_images = []
    for month in range(1, 13):
        composite = monthly_water_composite(year, month, region)
        monthly_images.append(composite)

    monthly_collection = ee.ImageCollection.fromImages(monthly_images)

    annual = (
        monthly_collection
        .select("water_monthly_max")
        .max()
        .rename("water_annual_max")
        .set("year", year)
    )

    return annual


def compute_sar_water_frequency(
    start_year: int,
    end_year: int,
    region: ee.Geometry,
) -> ee.Image:
    """
    Compute SAR-based water occurrence frequency (0-100 %).

    For each pixel, calculates the fraction of monthly composites in which
    water was detected, expressed as a percentage.

    Parameters
    ----------
    start_year, end_year : int
        Inclusive year range.
    region : ee.Geometry
        Spatial filter.

    Returns
    -------
    ee.Image
        Single-band image ``'sar_water_frequency'`` (0-100).
    """
    all_monthly = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            composite = monthly_water_composite(year, month, region)
            all_monthly.append(composite)

    collection = ee.ImageCollection.fromImages(all_monthly)
    total_months = collection.size()

    water_sum = collection.select("water_monthly_max").sum()
    frequency = water_sum.divide(total_months).multiply(100).rename("sar_water_frequency")

    log.info(
        "SAR water frequency computed for %d-%d (%d months)",
        start_year, end_year, len(all_monthly),
    )
    return frequency


# ===========================================================================
# EXPORT PIPELINE
# ===========================================================================

def export_annual_maps(
    start_year: int = 2015,
    end_year: int = 2025,
    export: bool = True,
) -> list:
    """
    Generate and optionally export annual max flood extent maps.

    Parameters
    ----------
    start_year, end_year : int
        Year range.
    export : bool
        If True, start Drive export tasks.

    Returns
    -------
    list[ee.batch.Task]
        List of started export tasks (empty if *export* is False).
    """
    region = get_study_area_geometry()
    tasks = []

    for year in range(start_year, end_year + 1):
        log.info("Processing annual max flood extent: %d", year)
        annual = annual_max_extent(year, region)

        if export:
            task = export_to_drive(
                image=annual.toFloat(),
                description=f"narino_sar_flood_max_{year}",
                region=region,
                scale=cfg.S1_PARAMS["resolution"],
                folder="narino_flood_risk",
            )
            tasks.append(task)

    # Also export the multi-year water frequency
    log.info("Computing multi-year SAR water frequency (%d-%d)", start_year, end_year)
    frequency = compute_sar_water_frequency(start_year, end_year, region)

    if export:
        task = export_to_drive(
            image=frequency.toFloat(),
            description=f"narino_sar_water_frequency_{start_year}_{end_year}",
            region=region,
            scale=cfg.S1_PARAMS["resolution"],
            folder="narino_flood_risk",
        )
        tasks.append(task)

    log.info("Total export tasks started: %d", len(tasks))
    return tasks


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def main() -> None:
    """Command-line interface for SAR water detection pipeline."""
    parser = argparse.ArgumentParser(
        description="Sentinel-1 SAR water detection for Narino",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Process a single year (default: all 2015-2025).",
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
    log.info("Sentinel-1 SAR Water Detection Pipeline")
    log.info("=" * 60)

    if args.year:
        start_year = end_year = args.year
    else:
        start_year, end_year = 2015, 2025

    tasks = export_annual_maps(
        start_year=start_year,
        end_year=end_year,
        export=not args.no_export,
    )

    if args.monitor and tasks:
        monitor_tasks(tasks)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
