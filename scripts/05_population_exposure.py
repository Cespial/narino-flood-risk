#!/usr/bin/env python3
"""
05_population_exposure.py
=========================
Population and infrastructure exposure analysis for Narino, Colombia.

This module overlays flood susceptibility maps with population density
and land cover data to quantify:
  1. Exposed population per municipality and risk class.
  2. Exposed area by land cover type.
  3. Zonal statistics (% municipal area per risk class).
  4. Municipal-level flood risk ranking.
  5. Temporal trends in exposed population (based on annual water extent).

Usage:
    python 05_population_exposure.py
    python 05_population_exposure.py --no-export
    python 05_population_exposure.py --monitor

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
    setup_logging,
    safe_getinfo,
    export_to_drive,
    export_table_to_drive,
    monitor_tasks,
)

log = setup_logging("05_exposure")

# Risk class probability thresholds (consistent with 04_ml module)
RISK_CLASSES = {
    1: {"label": "Very Low",  "range": (0.0, 0.2), "color": "#1a9850"},
    2: {"label": "Low",       "range": (0.2, 0.4), "color": "#91cf60"},
    3: {"label": "Moderate",  "range": (0.4, 0.6), "color": "#fee08b"},
    4: {"label": "High",      "range": (0.6, 0.8), "color": "#fc8d59"},
    5: {"label": "Very High", "range": (0.8, 1.01), "color": "#d73027"},
}

# ESA WorldCover class labels
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


# ===========================================================================
# SUSCEPTIBILITY MAP ACCESS
# ===========================================================================

def _load_susceptibility_map(region: ee.Geometry) -> ee.Image:
    """
    Load the ensemble flood susceptibility map.

    First tries to import from a GEE asset. If not available, generates
    the map dynamically using the 04_ml module's GEE-based ensemble.

    Parameters
    ----------
    region : ee.Geometry
        Analysis region.

    Returns
    -------
    ee.Image
        Probability image (0-1) named ``'susceptibility'``.
    """
    log.info("Loading ensemble susceptibility map ...")

    # Try asset-based approach first (post-export)
    try:
        asset_id = cfg.GEE_SUSCEPTIBILITY_ASSET
        susceptibility = ee.Image(asset_id).rename("susceptibility").clip(region)
        # Force server-side evaluation to verify asset exists.
        # ee.Image() is lazy: it never raises at construction time even
        # if the asset is missing.  Without this getInfo() call the
        # except block would never fire and the pipeline would later
        # fail with an opaque HttpError when a reducer tries to use the
        # non-existent image.
        susceptibility.bandNames().getInfo()
        log.info("Loaded susceptibility from asset: %s", asset_id)
        return susceptibility
    except Exception:
        pass

    # Fallback: generate dynamically via GEE classifiers
    log.info("Asset not found; generating susceptibility map via GEE classifiers ...")
    import importlib
    ml_module = importlib.import_module("scripts.04_ml_flood_susceptibility")
    generate_ensemble_map_gee = ml_module.generate_ensemble_map_gee

    feature_names = list(cfg.SUSCEPTIBILITY_FEATURES)
    ensemble = generate_ensemble_map_gee(region, feature_names)
    return ensemble.rename("susceptibility")


def _classify_risk(susceptibility: ee.Image) -> ee.Image:
    """
    Convert continuous susceptibility probability to discrete risk classes.

    Parameters
    ----------
    susceptibility : ee.Image
        Probability image (0-1).

    Returns
    -------
    ee.Image
        Integer-classified image (1-5) named ``'risk_class'``.
    """
    classified = ee.Image(0).rename("risk_class")
    for class_id, info in RISK_CLASSES.items():
        lo, hi = info["range"]
        mask = susceptibility.gte(lo).And(susceptibility.lt(hi))
        classified = classified.where(mask, class_id)
    return classified.selfMask()


# ===========================================================================
# POPULATION EXPOSURE
# ===========================================================================

def compute_population_exposure(
    susceptibility: ee.Image,
    region: ee.Geometry,
    pop_year: int = 2020,
) -> ee.FeatureCollection:
    """
    Compute exposed population per municipality per risk class.

    For each municipality and risk class, sums the WorldPop population
    density values falling within that class.

    Parameters
    ----------
    susceptibility : ee.Image
        Continuous probability image (0-1).
    region : ee.Geometry
        Analysis region.
    pop_year : int
        WorldPop data year.

    Returns
    -------
    ee.FeatureCollection
        Municipalities with population exposure properties:
        ``pop_total``, ``pop_very_low``, ``pop_low``, ``pop_moderate``,
        ``pop_high``, ``pop_very_high``, ``pop_high_plus``.
    """
    log.info("Computing population exposure (WorldPop %d) ...", pop_year)

    # Load WorldPop
    pop = (
        ee.ImageCollection(cfg.WORLDPOP)
        .filterBounds(region)
        .filter(ee.Filter.eq("year", pop_year))
        .filter(ee.Filter.eq("country", "COL"))
        .first()
        .rename("population")
        .clip(region)
    )

    risk_class = _classify_risk(susceptibility)
    municipalities = get_municipalities()

    # Create per-class population images
    class_pop_bands = {}
    for class_id, info in RISK_CLASSES.items():
        label = info["label"].lower().replace(" ", "_")
        band_name = f"pop_{label}"
        class_mask = risk_class.eq(class_id)
        class_pop = pop.updateMask(class_mask).rename(band_name)
        class_pop_bands[band_name] = class_pop

    # Stack all population bands
    pop_stack = pop.rename("pop_total")
    for band_name, class_pop in class_pop_bands.items():
        pop_stack = pop_stack.addBands(class_pop)

    # High-risk exposed population (classes 4 + 5)
    high_mask = risk_class.gte(4)
    pop_high_plus = pop.updateMask(high_mask).rename("pop_high_plus")
    pop_stack = pop_stack.addBands(pop_high_plus)

    # Reduce per municipality
    muni_exposure = pop_stack.reduceRegions(
        collection=municipalities,
        reducer=ee.Reducer.sum(),
        scale=100,  # WorldPop native resolution
    )

    log.info("Population exposure computed per municipality")
    return muni_exposure


# ===========================================================================
# AREA EXPOSURE BY LAND COVER
# ===========================================================================

def compute_area_exposure(
    susceptibility: ee.Image,
    region: ee.Geometry,
) -> ee.FeatureCollection:
    """
    Compute exposed area (km^2) by land cover type per risk class.

    Cross-tabulates ESA WorldCover 2021 classes with flood risk classes
    to quantify which land use types are most exposed.

    Parameters
    ----------
    susceptibility : ee.Image
        Continuous probability image (0-1).
    region : ee.Geometry
        Analysis region.

    Returns
    -------
    ee.FeatureCollection
        One feature per (land_cover, risk_class) combination with area.
    """
    log.info("Computing area exposure by land cover type ...")

    risk_class = _classify_risk(susceptibility)

    worldcover = (
        ee.ImageCollection(cfg.WORLDCOVER)
        .first()
        .select("Map")
        .rename("land_cover")
        .clip(region)
    )

    # Pixel area in km^2
    pixel_area_km2 = ee.Image.pixelArea().divide(1e6)

    # Cross-tabulation: group by land_cover then risk_class
    stacked = risk_class.addBands(worldcover).addBands(pixel_area_km2.rename("area_km2"))

    cross_tab = stacked.reduceRegion(
        reducer=ee.Reducer.sum().group(
            groupField=0, groupName="risk_class"
        ).group(
            groupField=1, groupName="land_cover"
        ),
        geometry=region,
        scale=cfg.EXPORT_SCALE,
        maxPixels=1e10,
        bestEffort=True,
    )

    log.info("Area exposure by land cover computed")

    # Convert to a FeatureCollection for export
    groups = ee.List(cross_tab.get("groups"))

    def _parse_group(group):
        group = ee.Dictionary(group)
        lc_code = group.get("land_cover")
        sub_groups = ee.List(group.get("groups"))

        def _parse_subgroup(sg):
            sg = ee.Dictionary(sg)
            return ee.Feature(None, {
                "land_cover_code": lc_code,
                "risk_class": sg.get("risk_class"),
                "area_km2": sg.get("sum"),
            })

        return sub_groups.map(_parse_subgroup)

    features = groups.map(_parse_group).flatten()
    return ee.FeatureCollection(features)


# ===========================================================================
# MUNICIPAL ZONAL STATISTICS
# ===========================================================================

def municipal_zonal_stats(
    susceptibility: ee.Image,
    region: ee.Geometry,
) -> ee.FeatureCollection:
    """
    Compute percentage of municipal area in each risk class.

    Parameters
    ----------
    susceptibility : ee.Image
        Continuous probability image (0-1).
    region : ee.Geometry
        Analysis region.

    Returns
    -------
    ee.FeatureCollection
        Municipalities with properties:
        ``pct_very_low``, ``pct_low``, ``pct_moderate``,
        ``pct_high``, ``pct_very_high``, ``mean_susceptibility``,
        ``total_area_km2``.
    """
    log.info("Computing municipal zonal statistics ...")

    risk_class = _classify_risk(susceptibility)
    municipalities = get_municipalities()
    pixel_area = ee.Image.pixelArea()

    def _compute_zonal(muni):
        muni = ee.Feature(muni)
        geom = muni.geometry()

        # Total area
        total_area = pixel_area.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geom,
            scale=cfg.EXPORT_SCALE,
            maxPixels=1e9,
            bestEffort=True,
        ).get("area")
        total_area_km2 = ee.Number(total_area).divide(1e6)

        # Area per risk class
        props = {"total_area_km2": total_area_km2}
        for class_id, info in RISK_CLASSES.items():
            label = info["label"].lower().replace(" ", "_")
            class_area = (
                pixel_area
                .updateMask(risk_class.eq(class_id))
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=geom,
                    scale=cfg.EXPORT_SCALE,
                    maxPixels=1e9,
                    bestEffort=True,
                )
                .get("area")
            )
            class_area_km2 = ee.Number(class_area).divide(1e6)
            pct = class_area_km2.divide(total_area_km2.max(0.001)).multiply(100)
            props[f"pct_{label}"] = pct
            props[f"area_km2_{label}"] = class_area_km2

        # Mean susceptibility
        mean_susc = susceptibility.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=cfg.EXPORT_SCALE,
            maxPixels=1e9,
            bestEffort=True,
        )
        props["mean_susceptibility"] = mean_susc.get(
            susceptibility.bandNames().get(0)
        )

        return muni.set(props)

    result = municipalities.map(_compute_zonal)
    log.info("Municipal zonal statistics computed")
    return result


# ===========================================================================
# MUNICIPAL RISK RANKING
# ===========================================================================

def municipal_risk_ranking(
    pop_exposure: ee.FeatureCollection,
    zonal_stats: ee.FeatureCollection,
) -> ee.FeatureCollection:
    """
    Generate a composite municipal flood risk ranking.

    Combines population exposure and area-based risk into a composite
    risk score per municipality.

    The composite score is:
        score = 0.4 * normalised(pop_high_plus)
              + 0.3 * normalised(pct_high + pct_very_high)
              + 0.3 * normalised(mean_susceptibility)

    Parameters
    ----------
    pop_exposure : ee.FeatureCollection
        Output of ``compute_population_exposure()``.
    zonal_stats : ee.FeatureCollection
        Output of ``municipal_zonal_stats()``.

    Returns
    -------
    ee.FeatureCollection
        Municipalities sorted by composite risk score (descending),
        with properties ``risk_score``, ``risk_rank``.
    """
    log.info("Computing municipal risk ranking ...")

    # Join exposure and zonal stats on municipality name
    join_filter = ee.Filter.equals(
        leftField="ADM2_NAME", rightField="ADM2_NAME"
    )
    inner_join = ee.Join.inner("exposure", "zonal")
    joined = inner_join.apply(pop_exposure, zonal_stats, join_filter)

    def _merge(feature):
        exp = ee.Feature(feature.get("exposure"))
        zon = ee.Feature(feature.get("zonal"))
        return exp.set(zon.toDictionary())

    merged = joined.map(_merge)

    # Compute normalisation bounds (server-side)
    pop_max = ee.Number(
        merged.aggregate_max("pop_high_plus")
    ).max(1)  # avoid zero division

    def _compute_score(feature):
        feature = ee.Feature(feature)

        # Normalised exposed population
        pop_norm = ee.Number(feature.get("pop_high_plus")).divide(pop_max)

        # Normalised area percentage (high + very high)
        pct_high = ee.Number(feature.get("pct_high")).add(
            ee.Number(feature.get("pct_very_high"))
        ).divide(100)

        # Mean susceptibility (already 0-1)
        mean_susc = ee.Number(feature.get("mean_susceptibility"))

        # Composite score
        score = (
            pop_norm.multiply(0.4)
            .add(pct_high.multiply(0.3))
            .add(mean_susc.multiply(0.3))
        )

        return feature.set("risk_score", score)

    ranked = merged.map(_compute_score).sort("risk_score", False)

    # Add rank number
    ranked_list = ranked.toList(ranked.size())
    n = ranked.size()

    def _add_rank(i):
        i = ee.Number(i)
        feat = ee.Feature(ranked_list.get(i))
        return feat.set("risk_rank", i.add(1))

    ranked_with_rank = ee.FeatureCollection(
        ee.List.sequence(0, n.subtract(1)).map(_add_rank)
    )

    log.info("Municipal risk ranking computed")
    return ranked_with_rank


# ===========================================================================
# TEMPORAL EXPOSURE ANALYSIS
# ===========================================================================

def temporal_exposure_analysis(
    region: ee.Geometry,
    start_year: int = 2015,
    end_year: int = 2025,
    pop_year: int = 2020,
) -> ee.FeatureCollection:
    """
    Analyse changes in exposed population over time using annual SAR
    water extent maps.

    For each year, overlays the annual maximum water extent with WorldPop
    to estimate the population exposed to flooding.

    Parameters
    ----------
    region : ee.Geometry
        Analysis region.
    start_year, end_year : int
        Year range.
    pop_year : int
        WorldPop reference year (held constant for comparability).

    Returns
    -------
    ee.FeatureCollection
        One feature per year with properties:
        ``year``, ``flood_area_km2``, ``exposed_population``,
        ``n_images_used``.
    """
    log.info("Computing temporal exposure analysis (%d-%d) ...", start_year, end_year)

    import importlib
    sar_module = importlib.import_module("scripts.01_sar_water_detection")
    annual_max_extent = sar_module.annual_max_extent

    # Load static population layer
    pop = (
        ee.ImageCollection(cfg.WORLDPOP)
        .filterBounds(region)
        .filter(ee.Filter.eq("year", pop_year))
        .filter(ee.Filter.eq("country", "COL"))
        .first()
        .rename("population")
        .clip(region)
    )

    pixel_area_km2 = ee.Image.pixelArea().divide(1e6)

    results = []
    for year in range(start_year, end_year + 1):
        water = annual_max_extent(year, region)

        # Flood area (km^2)
        flood_area = (
            pixel_area_km2
            .updateMask(water.select("water_annual_max"))
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=region,
                scale=cfg.EXPORT_SCALE,
                maxPixels=1e10,
                bestEffort=True,
            )
            .get("area")
        )

        # Exposed population
        exposed_pop = (
            pop
            .updateMask(water.select("water_annual_max"))
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=region,
                scale=100,
                maxPixels=1e10,
                bestEffort=True,
            )
            .get("population")
        )

        feature = ee.Feature(None, {
            "year": year,
            "flood_area_km2": flood_area,
            "exposed_population": exposed_pop,
        })
        results.append(feature)

    temporal_fc = ee.FeatureCollection(results)
    log.info("Temporal exposure analysis: %d years processed", end_year - start_year + 1)
    return temporal_fc


# ===========================================================================
# EXPORT PIPELINE
# ===========================================================================

def run_exposure_pipeline(export: bool = True) -> list:
    """
    Execute the full population / infrastructure exposure pipeline.

    Parameters
    ----------
    export : bool
        Start Drive export tasks.

    Returns
    -------
    list[ee.batch.Task]
        Started export tasks.
    """
    region = get_study_area_geometry()
    tasks = []

    # Load susceptibility map
    susceptibility = _load_susceptibility_map(region)

    # --- 1. Population exposure ---
    pop_exposure = compute_population_exposure(susceptibility, region)
    if export:
        task = export_table_to_drive(
            collection=pop_exposure,
            description="narino_population_exposure_by_municipality",
            file_format="CSV",
        )
        tasks.append(task)

    # --- 2. Area exposure by land cover ---
    area_exposure = compute_area_exposure(susceptibility, region)
    if export:
        task = export_table_to_drive(
            collection=area_exposure,
            description="narino_area_exposure_by_landcover",
            file_format="CSV",
        )
        tasks.append(task)

    # --- 3. Municipal zonal statistics ---
    zonal = municipal_zonal_stats(susceptibility, region)
    if export:
        task = export_table_to_drive(
            collection=zonal,
            description="narino_municipal_zonal_stats",
            file_format="CSV",
        )
        tasks.append(task)

    # --- 4. Municipal risk ranking ---
    ranking = municipal_risk_ranking(pop_exposure, zonal)
    if export:
        task = export_table_to_drive(
            collection=ranking,
            description="narino_municipal_risk_ranking",
            file_format="CSV",
        )
        tasks.append(task)

    # --- 5. Temporal exposure ---
    temporal = temporal_exposure_analysis(region, start_year=2015, end_year=2025)
    if export:
        task = export_table_to_drive(
            collection=temporal,
            description="narino_temporal_exposure_2015_2025",
            file_format="CSV",
        )
        tasks.append(task)

    # --- 6. Export classified risk map ---
    risk_class_map = _classify_risk(susceptibility)
    if export:
        task = export_to_drive(
            image=risk_class_map.toFloat(),
            description="narino_flood_risk_classified_map",
            region=region,
            scale=cfg.EXPORT_SCALE,
        )
        tasks.append(task)

    log.info("Exposure pipeline complete. Export tasks: %d", len(tasks))
    return tasks


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def main() -> None:
    """Command-line interface for population exposure analysis."""
    parser = argparse.ArgumentParser(
        description="Population and infrastructure exposure analysis for Narino",
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
    log.info("Population & Infrastructure Exposure Pipeline")
    log.info("=" * 60)

    tasks = run_exposure_pipeline(export=not args.no_export)

    if args.monitor and tasks:
        monitor_tasks(tasks)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
