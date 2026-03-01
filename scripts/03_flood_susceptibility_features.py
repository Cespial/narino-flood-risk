#!/usr/bin/env python3
"""
03_flood_susceptibility_features.py
====================================
Generate all predictor variables for the ML flood susceptibility model.

Feature categories:
  - **Topographic** : elevation, slope, aspect, curvature, HAND, TWI, SPI
  - **Hydrologic**  : distance to rivers, drainage density
  - **Climate**     : annual rainfall, max monthly rainfall, soil moisture
  - **Land cover**  : ESA WorldCover 2021 classes
  - **Proximity**   : distance to roads
  - **Population**  : WorldPop 100 m density
  - **Water history**: JRC occurrence, SAR-derived water frequency

All features are stacked into a single multi-band image, and stratified
training samples are extracted for downstream ML modelling.

Usage:
    python 03_flood_susceptibility_features.py
    python 03_flood_susceptibility_features.py --no-export
    python 03_flood_susceptibility_features.py --samples-only

Author : Narino Flood Risk Research Project
Date   : 2026-02-26
"""

import sys
import pathlib
import argparse
from typing import Dict, List, Optional

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
    get_dem,
    compute_hand,
    compute_twi,
    compute_spi,
    setup_logging,
    safe_getinfo,
    export_to_drive,
    export_table_to_drive,
    monitor_tasks,
)

log = setup_logging("03_features")

# Number of training samples per class (flood / non-flood)
SAMPLES_PER_CLASS = 5000
SAMPLE_SCALE = 30  # meters


# ===========================================================================
# TOPOGRAPHIC FEATURES
# ===========================================================================

def compute_topographic_features(
    dem: ee.Image,
    region: ee.Geometry,
) -> ee.Image:
    """
    Derive topographic predictor variables from SRTM DEM.

    Bands produced:
      ``elevation``, ``slope``, ``aspect``, ``curvature``, ``hand``,
      ``twi``, ``spi``

    Parameters
    ----------
    dem : ee.Image
        Digital elevation model (SRTM 30 m).
    region : ee.Geometry
        Analysis region.

    Returns
    -------
    ee.Image
        Multi-band image with 7 topographic features.
    """
    log.info("Computing topographic features ...")

    elevation = dem.select("elevation").rename("elevation")

    terrain = ee.Terrain.products(dem)
    slope = terrain.select("slope").rename("slope")
    aspect = terrain.select("aspect").rename("aspect")

    # Plan curvature: Laplacian of the DEM approximated with a 3x3 kernel
    # Negative = concave (collects water), positive = convex
    laplacian_kernel = ee.Kernel.laplacian8(normalize=True)
    curvature = dem.select("elevation").convolve(laplacian_kernel).rename("curvature")

    # HAND
    hand = compute_hand(dem, region)

    # TWI
    twi = compute_twi(dem)

    # SPI
    spi = compute_spi(dem)

    stack = (
        elevation
        .addBands(slope)
        .addBands(aspect)
        .addBands(curvature)
        .addBands(hand)
        .addBands(twi)
        .addBands(spi)
    ).clip(region)

    log.info("Topographic features: %d bands", 7)
    return stack


# ===========================================================================
# HYDROLOGIC FEATURES
# ===========================================================================

def compute_hydrologic_features(region: ee.Geometry) -> ee.Image:
    """
    Compute hydrologic predictor variables.

    - **dist_rivers** : Euclidean distance to nearest river pixel (m), derived
      from JRC permanent water (occurrence >= 75 %).
    - **drainage_density** : Proportion of river pixels in a 1-km
      neighbourhood, approximating drainage density.

    Parameters
    ----------
    region : ee.Geometry
        Analysis region.

    Returns
    -------
    ee.Image
        Two-band image: ``dist_rivers``, ``drainage_density``.
    """
    log.info("Computing hydrologic features ...")

    # River network proxy from JRC occurrence (>=75% = permanent/frequent)
    jrc_occ = ee.Image(cfg.JRC_GSW).select("occurrence").clip(region)
    river_mask = jrc_occ.gte(75).rename("rivers")

    # Euclidean distance to nearest river pixel
    # fastDistanceTransform outputs distance^2 in pixel units
    max_dist_px = 334  # ~10 km at 30 m
    dist_sq = river_mask.Not().selfMask().fastDistanceTransform(
        neighborhood=max_dist_px,
        units="pixels",
        metric="squared_euclidean",
    )
    dist_rivers = dist_sq.sqrt().multiply(30).rename("dist_rivers").clip(region)

    # Drainage density: fraction of river pixels within a 1 km radius
    drainage_density = river_mask.focal_mean(
        radius=1000,
        kernelType="circle",
        units="meters",
    ).rename("drainage_density").clip(region)

    result = dist_rivers.addBands(drainage_density)
    log.info("Hydrologic features computed (dist_rivers, drainage_density)")
    return result


# ===========================================================================
# CLIMATE FEATURES
# ===========================================================================

def compute_climate_features(
    region: ee.Geometry,
    start_date: str = cfg.ANALYSIS_START,
    end_date: str = cfg.ANALYSIS_END,
) -> ee.Image:
    """
    Compute climate predictor variables.

    - **rainfall_annual** : Mean annual precipitation (mm/year) from CHIRPS.
    - **rainfall_max_monthly** : Maximum monthly rainfall (mm/month).
    - **soil_moisture** : Mean volumetric soil water content from ERA5-Land.

    Parameters
    ----------
    region : ee.Geometry
        Analysis region.
    start_date, end_date : str
        Temporal window for climatological averages.

    Returns
    -------
    ee.Image
        Three-band image: ``rainfall_annual``, ``rainfall_max_monthly``,
        ``soil_moisture``.
    """
    log.info("Computing climate features (%s to %s) ...", start_date, end_date)

    # ---- CHIRPS daily precipitation ----
    chirps = (
        ee.ImageCollection(cfg.CHIRPS)
        .filterBounds(region)
        .filterDate(start_date, end_date)
    )

    # Annual mean: sum daily precip per year, then average across years
    # Use Python-side loop to avoid nested ee.List.map serialisation issues
    annual_images = []
    for yr in range(2015, 2026):
        annual = (
            chirps
            .filter(ee.Filter.calendarRange(yr, yr, "year"))
            .sum()
            .rename("annual_precip")
        )
        annual_images.append(annual)

    annual_precip_coll = ee.ImageCollection(annual_images)
    rainfall_annual = annual_precip_coll.mean().rename("rainfall_annual").clip(region)

    # Max monthly rainfall
    monthly_images = []
    for yr in range(2015, 2026):
        for mo in range(1, 13):
            start = ee.Date.fromYMD(yr, mo, 1)
            end = start.advance(1, "month")
            monthly = chirps.filterDate(start, end).sum().rename("monthly_precip")
            monthly_images.append(monthly.set("year", yr).set("month", mo))

    monthly_coll = ee.ImageCollection(monthly_images)

    # Mean of the maximum monthly precipitation per year
    max_monthly_images = []
    for yr in range(2015, 2026):
        yr_max = (
            monthly_coll
            .filter(ee.Filter.eq("year", yr))
            .max()
            .rename("max_monthly")
        )
        max_monthly_images.append(yr_max)

    max_monthly_coll = ee.ImageCollection(max_monthly_images)
    rainfall_max_monthly = max_monthly_coll.mean().rename("rainfall_max_monthly").clip(region)

    # ---- ERA5-Land soil moisture ----
    era5 = (
        ee.ImageCollection(cfg.ERA5_LAND)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .select("volumetric_soil_water_layer_1")
    )
    soil_moisture = era5.mean().rename("soil_moisture").clip(region)

    result = rainfall_annual.addBands(rainfall_max_monthly).addBands(soil_moisture)
    log.info("Climate features computed (rainfall_annual, rainfall_max_monthly, soil_moisture)")
    return result


# ===========================================================================
# LAND COVER FEATURES
# ===========================================================================

def compute_land_cover_features(region: ee.Geometry) -> ee.Image:
    """
    Load ESA WorldCover 2021 (10 m) land cover classification.

    The original integer classes are preserved. An additional ``ndvi_mean``
    band is computed from Sentinel-2 annual composite as a continuous
    vegetation proxy.

    Parameters
    ----------
    region : ee.Geometry
        Analysis region.

    Returns
    -------
    ee.Image
        Two-band image: ``land_cover`` (categorical int) and ``ndvi_mean``.
    """
    log.info("Computing land cover features ...")

    # ESA WorldCover 2021
    worldcover = (
        ee.ImageCollection(cfg.WORLDCOVER)
        .first()
        .select("Map")
        .rename("land_cover")
        .clip(region)
    )

    # Mean annual NDVI from Sentinel-2 Surface Reflectance (2021)
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate("2021-01-01", "2021-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    def _compute_ndvi(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")
        return ndvi.copyProperties(img, img.propertyNames())

    ndvi_coll = s2.map(_compute_ndvi)
    ndvi_mean = ndvi_coll.mean().rename("ndvi_mean").clip(region)

    result = worldcover.addBands(ndvi_mean)
    log.info("Land cover features computed (land_cover, ndvi_mean)")
    return result


# ===========================================================================
# PROXIMITY FEATURES
# ===========================================================================

def compute_proximity_features(region: ee.Geometry) -> ee.Image:
    """
    Compute road-proximity feature using Oxford/MAP friction surface.

    The friction surface (travel time per unit distance) serves as an
    excellent proxy for road proximity: low friction = near roads,
    high friction = far from roads.  We invert and normalise so that
    higher values indicate greater distance from transport networks.

    Parameters
    ----------
    region : ee.Geometry
        Analysis region.

    Returns
    -------
    ee.Image
        Single-band image ``'dist_roads'`` (friction-based proxy, 0-1).
    """
    log.info("Computing proximity features (Oxford friction surface) ...")

    # Oxford/MAP friction surface 2019 v5.1 (~1 km resolution)
    # Lower friction = closer to roads/infrastructure
    friction = (
        ee.Image("projects/malariaatlasproject/assets/accessibility/friction_surface/2019_v5_1")
        .select("friction")
        .clip(region)
    )

    # Normalise friction to 0-1 range within the study area
    # Higher values = farther from roads (more friction)
    stats = friction.reduceRegion(
        reducer=ee.Reducer.percentile([2, 98]),
        geometry=region,
        scale=1000,
        maxPixels=1e9,
        bestEffort=True,
    )
    p2 = ee.Number(stats.get("friction_p2"))
    p98 = ee.Number(stats.get("friction_p98"))

    dist_roads = (
        friction
        .subtract(p2)
        .divide(p98.subtract(p2))
        .clamp(0, 1)
        .rename("dist_roads")
        .toFloat()
    )

    log.info("Proximity features computed (dist_roads via friction surface)")
    return dist_roads


# ===========================================================================
# POPULATION FEATURES
# ===========================================================================

def compute_population_features(
    region: ee.Geometry,
    year: int = 2020,
) -> ee.Image:
    """
    Load WorldPop population density (100 m resolution).

    Parameters
    ----------
    region : ee.Geometry
        Analysis region.
    year : int
        Target year for population data.

    Returns
    -------
    ee.Image
        Single-band image ``'pop_density'`` (people per 100 m pixel).
    """
    log.info("Loading population features (WorldPop %d) ...", year)

    pop = (
        ee.ImageCollection(cfg.WORLDPOP)
        .filterBounds(region)
        .filter(ee.Filter.eq("year", year))
        .filter(ee.Filter.eq("country", "COL"))
        .first()
        .rename("pop_density")
        .clip(region)
    )

    return pop


# ===========================================================================
# WATER HISTORY FEATURES
# ===========================================================================

def compute_water_history_features(region: ee.Geometry) -> ee.Image:
    """
    Compute water history predictor variables.

    - **flood_frequency_jrc** : JRC water occurrence (0-100 %).
    - **sar_water_frequency** : Sentinel-1 derived water frequency (0-100 %).

    For the SAR frequency, this function computes a simplified proxy using
    monthly composites over a representative period.

    Parameters
    ----------
    region : ee.Geometry
        Analysis region.

    Returns
    -------
    ee.Image
        Two-band image: ``flood_frequency_jrc``, ``sar_water_frequency``.
    """
    log.info("Computing water history features ...")

    # JRC occurrence
    jrc_occ = (
        ee.Image(cfg.JRC_GSW)
        .select("occurrence")
        .rename("flood_frequency_jrc")
        .clip(region)
    )

    # SAR water frequency - use importlib because module name starts with digit
    import importlib
    sar_module = importlib.import_module("scripts.01_sar_water_detection")
    compute_sar_water_frequency = sar_module.compute_sar_water_frequency

    sar_freq = compute_sar_water_frequency(
        start_year=2018,
        end_year=2023,
        region=region,
    )

    result = jrc_occ.addBands(sar_freq)
    log.info("Water history features computed (jrc_occurrence, sar_frequency)")
    return result


# ===========================================================================
# FEATURE STACK
# ===========================================================================

def stack_all_features(
    region: Optional[ee.Geometry] = None,
) -> ee.Image:
    """
    Assemble the full multi-band feature stack for flood susceptibility.

    All features are resampled / reprojected to a common 30 m grid and
    combined into a single ``ee.Image``.

    Parameters
    ----------
    region : ee.Geometry, optional
        Analysis region (defaults to Narino).

    Returns
    -------
    ee.Image
        Multi-band image with all predictor features listed in
        ``gee_config.SUSCEPTIBILITY_FEATURES``.
    """
    if region is None:
        region = get_study_area_geometry()

    dem = get_dem()

    log.info("Building full feature stack ...")

    topo = compute_topographic_features(dem, region)
    hydro = compute_hydrologic_features(region)
    climate = compute_climate_features(region)
    lulc = compute_land_cover_features(region)
    roads = compute_proximity_features(region)
    pop = compute_population_features(region)
    water = compute_water_history_features(region)

    # Stack all features
    feature_stack = (
        topo
        .addBands(hydro)
        .addBands(climate)
        .addBands(lulc)
        .addBands(roads)
        .addBands(pop)
        .addBands(water)
    ).toFloat()

    # Log expected band count (avoid server-side getInfo to prevent
    # "Too many concurrent aggregations" errors)
    expected = cfg.SUSCEPTIBILITY_FEATURES
    log.info(
        "Feature stack assembled (expected %d bands: %s)",
        len(expected),
        ", ".join(expected),
    )

    return feature_stack


# ===========================================================================
# TRAINING SAMPLE GENERATION
# ===========================================================================

def _create_flood_label(region: ee.Geometry) -> ee.Image:
    """
    Create a binary flood label image for training data generation.

    Flood pixels (label = 1): JRC occurrence >= 25 % **or** HAND < 5 m
    Non-flood pixels (label = 0): JRC occurrence < 5 % **and** HAND >= 30 m

    Pixels not meeting either criterion are masked out to create clear
    separation between classes.

    Parameters
    ----------
    region : ee.Geometry
        Analysis region.

    Returns
    -------
    ee.Image
        Binary label image (0 = non-flood, 1 = flood) with ambiguous
        pixels masked.
    """
    jrc_occ = ee.Image(cfg.JRC_GSW).select("occurrence").clip(region)
    dem = get_dem()
    hand = compute_hand(dem, region)

    # Flood class: historically wet areas OR very low HAND
    flood = jrc_occ.gte(25).Or(hand.lt(5)).rename("label")

    # Non-flood class: dry areas AND high HAND
    non_flood = jrc_occ.lt(5).And(hand.gte(30)).rename("label")

    # Combine: 1 = flood, 0 = non-flood; mask ambiguous pixels
    label = ee.Image(0).rename("label")
    label = label.where(flood, 1)
    label = label.updateMask(flood.Or(non_flood))

    return label.clip(region).toByte()


def generate_training_samples(
    feature_stack: ee.Image,
    region: Optional[ee.Geometry] = None,
    samples_per_class: int = SAMPLES_PER_CLASS,
    scale: int = SAMPLE_SCALE,
    seed: int = cfg.CV_PARAMS["random_state"],
) -> ee.FeatureCollection:
    """
    Generate stratified random training samples from the feature stack.

    Samples are drawn from clearly flood-prone and clearly non-flood areas
    to maximise class separability. Each sample point includes all feature
    band values and a binary ``label`` (1 = flood, 0 = non-flood).

    Parameters
    ----------
    feature_stack : ee.Image
        Multi-band feature image.
    region : ee.Geometry, optional
        Sampling region (defaults to Narino).
    samples_per_class : int
        Number of samples per class.
    scale : int
        Sampling scale in meters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ee.FeatureCollection
        Point samples with feature values and ``label`` property.
    """
    if region is None:
        region = get_study_area_geometry()

    log.info(
        "Generating training samples: %d per class at %d m",
        samples_per_class, scale,
    )

    label = _create_flood_label(region)
    stack_with_label = feature_stack.addBands(label)

    # Stratified sampling
    samples = stack_with_label.stratifiedSample(
        numPoints=samples_per_class,
        classBand="label",
        region=region,
        scale=scale,
        seed=seed,
        geometries=True,  # keep coordinates for spatial CV
    )

    log.info("Training samples generated")
    return samples


# ===========================================================================
# EXPORT PIPELINE
# ===========================================================================

def run_feature_pipeline(
    export: bool = True,
    samples_only: bool = False,
) -> list:
    """
    Execute the full feature generation pipeline.

    Parameters
    ----------
    export : bool
        Start Drive export tasks.
    samples_only : bool
        If True, skip exporting the feature stack raster and only export
        the training samples table.

    Returns
    -------
    list[ee.batch.Task]
        Started export tasks.
    """
    region = get_study_area_geometry()
    tasks = []

    # Build feature stack
    feature_stack = stack_all_features(region)

    if export and not samples_only:
        log.info("Exporting feature stack raster ...")
        task = export_to_drive(
            image=feature_stack,
            description="narino_flood_feature_stack",
            region=region,
            scale=SAMPLE_SCALE,
        )
        tasks.append(task)

    # Generate and export training samples
    log.info("Generating and exporting training samples ...")
    samples = generate_training_samples(feature_stack, region)

    if export:
        task = export_table_to_drive(
            collection=samples,
            description="narino_flood_training_samples",
            file_format="CSV",
        )
        tasks.append(task)

        # Also export as GeoJSON for spatial analysis
        task_geo = export_table_to_drive(
            collection=samples,
            description="narino_flood_training_samples_geo",
            file_format="GeoJSON",
        )
        tasks.append(task_geo)

    # Export individual feature groups for QC
    if export and not samples_only:
        dem = get_dem()
        feature_groups = {
            "topographic": compute_topographic_features(dem, region),
            "hydrologic": compute_hydrologic_features(region),
            "climate": compute_climate_features(region),
        }
        for group_name, img in feature_groups.items():
            task = export_to_drive(
                image=img.toFloat(),
                description=f"narino_features_{group_name}",
                region=region,
                scale=SAMPLE_SCALE,
            )
            tasks.append(task)

    log.info("Feature pipeline complete. Export tasks: %d", len(tasks))
    return tasks


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def main() -> None:
    """Command-line interface for feature generation."""
    parser = argparse.ArgumentParser(
        description="Generate flood susceptibility feature stack for Narino",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip exporting to Google Drive.",
    )
    parser.add_argument(
        "--samples-only",
        action="store_true",
        help="Only generate and export training samples (skip raster export).",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Block and monitor export tasks until completion.",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Flood Susceptibility Feature Generation Pipeline")
    log.info("=" * 60)

    tasks = run_feature_pipeline(
        export=not args.no_export,
        samples_only=args.samples_only,
    )

    if args.monitor and tasks:
        monitor_tasks(tasks)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
