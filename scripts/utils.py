#!/usr/bin/env python3
"""
utils.py
========
Shared utility functions for the Narino Flood Risk Assessment project.

Provides reusable helpers for:
  - Study area retrieval (department and municipal boundaries from FAO GAUL)
  - SAR speckle filtering
  - Terrain-derived indices (HAND, TWI, SPI)
  - Logging configuration
  - Safe GEE calls and export helpers

Usage:
    from utils import get_study_area, get_municipalities, setup_logging

Author : Narino Flood Risk Research Project
Date   : 2026-02-26
"""

import sys
import logging
import pathlib
import time
from typing import Optional, List, Dict, Any

import ee

# ---------------------------------------------------------------------------
# Resolve project root so we can import gee_config from any working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gee_config as cfg  # noqa: E402  (after path manipulation)


# ===========================================================================
# LOGGING
# ===========================================================================

def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[pathlib.Path] = None,
) -> logging.Logger:
    """
    Configure a logger that writes to both the console and a rotating log file.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).
    level : int
        Logging level (default ``logging.INFO``).
    log_dir : pathlib.Path, optional
        Directory for log files.  Defaults to ``<project_root>/logs/``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_dir = log_dir or cfg.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-28s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file = log_dir / f"{name.replace('.', '_')}.log"
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


_log = setup_logging("utils")


# ===========================================================================
# STUDY AREA HELPERS
# ===========================================================================

def get_study_area() -> ee.FeatureCollection:
    """
    Return the Narino department boundary from FAO GAUL Level 1.

    Uses the ``FAO/GAUL/2015/level1`` dataset and filters for
    Narino in Colombia.  The result is a single-feature
    ``ee.FeatureCollection`` suitable for clipping and region operations.

    Returns
    -------
    ee.FeatureCollection
        Narino department polygon.
    """
    narino = (
        ee.FeatureCollection(cfg.ADMIN_DATASET)
        .filter(ee.Filter.eq("ADM0_NAME", cfg.COUNTRY_NAME))
        .filter(ee.Filter.eq("ADM1_NAME", cfg.DEPARTMENT_NAME))
    )
    _log.info("Study area loaded: %s from %s", cfg.DEPARTMENT_NAME, cfg.ADMIN_DATASET)
    return narino


def get_study_area_geometry() -> ee.Geometry:
    """
    Return the dissolved geometry of Narino as a single ``ee.Geometry``.

    Useful for spatial filters and region parameters where a geometry
    (not a FeatureCollection) is required.

    Returns
    -------
    ee.Geometry
        Dissolved Narino boundary.
    """
    return get_study_area().geometry().dissolve()


def get_municipalities() -> ee.FeatureCollection:
    """
    Return all municipalities within Narino from FAO GAUL Level 2.

    Returns
    -------
    ee.FeatureCollection
        ~64 municipality polygons with properties including ``ADM2_NAME``.
    """
    municipalities = (
        ee.FeatureCollection(cfg.MUNICIPAL_DATASET)
        .filter(ee.Filter.eq("ADM0_NAME", cfg.COUNTRY_NAME))
        .filter(ee.Filter.eq("ADM1_NAME", cfg.DEPARTMENT_NAME))
    )
    _log.info("Municipalities loaded from %s", cfg.MUNICIPAL_DATASET)
    return municipalities


def get_subregion_municipalities(subregion_name: str) -> ee.FeatureCollection:
    """
    Filter municipalities belonging to a specific Narino subregion.

    Parameters
    ----------
    subregion_name : str
        One of the 13 official Narino subregions defined in
        ``gee_config.SUBREGIONS`` (e.g. ``'Centro'``).

    Returns
    -------
    ee.FeatureCollection
        Municipalities in the requested subregion.

    Raises
    ------
    ValueError
        If *subregion_name* is not found in ``gee_config.SUBREGIONS``.
    """
    if subregion_name not in cfg.SUBREGIONS:
        valid = list(cfg.SUBREGIONS.keys())
        raise ValueError(
            f"Unknown subregion '{subregion_name}'. Valid options: {valid}"
        )

    mun_names = cfg.SUBREGIONS[subregion_name]
    all_municipalities = get_municipalities()
    subregion_fc = all_municipalities.filter(
        ee.Filter.inList("ADM2_NAME", mun_names)
    )
    _log.info(
        "Subregion '%s': filtered %d municipalities",
        subregion_name,
        len(mun_names),
    )
    return subregion_fc


# ===========================================================================
# SAR SPECKLE FILTERING
# ===========================================================================

def apply_speckle_filter(image: ee.Image, radius_m: int = 50) -> ee.Image:
    """
    Apply a focal median speckle filter to a SAR image.

    A circular focal median is used rather than a boxcar mean because it
    better preserves edges around water bodies while suppressing salt-and-
    pepper noise typical of SAR imagery.

    Parameters
    ----------
    image : ee.Image
        Input SAR image (e.g. Sentinel-1 GRD in dB scale).
    radius_m : int
        Radius of the focal median kernel in **meters** (default 50 m).

    Returns
    -------
    ee.Image
        Speckle-filtered image with the same band names and properties.
    """
    filtered = image.focal_median(
        radius=radius_m,
        kernelType="circle",
        units="meters",
    )
    return filtered.copyProperties(image, image.propertyNames())


# ===========================================================================
# TERRAIN-DERIVED INDICES
# ===========================================================================

def _get_merit_hydro() -> ee.Image:
    """
    Load the MERIT Hydro dataset.

    MERIT Hydro provides pre-computed hydrological layers at ~90 m
    resolution, including flow direction, upstream area, and HAND.
    This avoids the need for ``ee.Terrain.fill`` / ``flowDirection`` /
    ``flowAccumulation`` which are not available in the current
    earthengine-api (1.7.x).

    Returns
    -------
    ee.Image
        MERIT Hydro multi-band image with bands:
        ``elv``, ``dir``, ``upa``, ``upg``, ``hnd``, ``wth``, ``wat``.
    """
    return ee.Image(cfg.MERIT_HYDRO)


def compute_hand(dem: ee.Image, region: ee.Geometry) -> ee.Image:
    """
    Retrieve Height Above Nearest Drainage (HAND) from MERIT Hydro.

    Uses the pre-computed ``hnd`` band from the MERIT Hydro dataset
    (~90 m resolution) rather than computing flow routing in GEE,
    since ``ee.Terrain.fill``, ``ee.Terrain.flowDirection``, and
    ``ee.Terrain.flowAccumulation`` are not available in the current
    earthengine-api.

    Parameters
    ----------
    dem : ee.Image
        Digital elevation model (unused; kept for API compatibility).
    region : ee.Geometry
        Region of interest for the computation.

    Returns
    -------
    ee.Image
        Single-band image named ``'hand'`` in meters.
    """
    merit = _get_merit_hydro()
    hand = merit.select("hnd").rename("hand").clip(region)
    _log.info("HAND loaded from MERIT Hydro (pre-computed)")
    return hand


def compute_twi(dem: ee.Image) -> ee.Image:
    """
    Compute the Topographic Wetness Index (TWI).

    TWI = ln(a / tan(beta))

    where *a* is the specific catchment area (upstream area) and *beta* is
    the local slope in radians.

    Upstream area is obtained from the MERIT Hydro ``upa`` band (km^2),
    converted to m^2. Slope is derived from the input DEM using
    ``ee.Terrain.slope``.

    Parameters
    ----------
    dem : ee.Image
        Digital elevation model.

    Returns
    -------
    ee.Image
        Single-band image named ``'twi'`` (dimensionless).
    """
    # Slope in radians
    slope_rad = ee.Terrain.slope(dem).multiply(3.14159265 / 180.0)
    # Clamp slope to avoid division by zero (minimum ~0.5 degrees)
    slope_clamped = slope_rad.max(0.00873)  # ~0.5 deg

    # Upstream area from MERIT Hydro (upa is in km^2)
    merit = _get_merit_hydro()
    upstream_area_km2 = merit.select("upa")

    # Convert upstream area from km^2 to m^2, then compute specific
    # catchment area by dividing by the MERIT cell width (~90 m).
    # a = upstream_area_m2 / cell_width
    cell_width = 90  # MERIT Hydro resolution in meters
    specific_area = upstream_area_km2.multiply(1e6).divide(cell_width).max(1)

    # TWI = ln(a / tan(slope))
    twi = specific_area.divide(slope_clamped.tan()).log().rename("twi")
    _log.info("TWI (Topographic Wetness Index) computed using MERIT Hydro upstream area")
    return twi


def compute_spi(dem: ee.Image) -> ee.Image:
    """
    Compute the Stream Power Index (SPI).

    SPI = a * tan(beta)

    where *a* is specific catchment area and *beta* is the local slope.
    SPI indicates the erosive power of flowing water.

    Upstream area is obtained from the MERIT Hydro ``upa`` band (km^2).
    Slope is derived from the input DEM using ``ee.Terrain.slope``.

    Parameters
    ----------
    dem : ee.Image
        Digital elevation model.

    Returns
    -------
    ee.Image
        Single-band image named ``'spi'`` (dimensionless).
    """
    slope_rad = ee.Terrain.slope(dem).multiply(3.14159265 / 180.0)

    # Upstream area from MERIT Hydro (upa is in km^2)
    merit = _get_merit_hydro()
    upstream_area_km2 = merit.select("upa")

    # Convert to specific catchment area (m)
    cell_width = 90  # MERIT Hydro resolution in meters
    specific_area = upstream_area_km2.multiply(1e6).divide(cell_width).max(1)

    spi = specific_area.multiply(slope_rad.tan()).rename("spi")
    _log.info("SPI (Stream Power Index) computed using MERIT Hydro upstream area")
    return spi


# ===========================================================================
# SAFE GEE CALLS
# ===========================================================================

def safe_getinfo(
    ee_obj: Any,
    label: str = "object",
    max_retries: int = 3,
    backoff_s: float = 5.0,
) -> Any:
    """
    Safely call ``.getInfo()`` on a GEE object with retries.

    GEE server calls can fail intermittently due to quotas or timeouts.
    This wrapper retries with exponential back-off.

    Parameters
    ----------
    ee_obj : ee.ComputedObject
        Any Earth Engine object that supports ``.getInfo()``.
    label : str
        Descriptive label for logging messages.
    max_retries : int
        Maximum number of attempts (default 3).
    backoff_s : float
        Initial back-off duration in seconds; doubles each retry.

    Returns
    -------
    Any
        The Python-side result of ``.getInfo()``.

    Raises
    ------
    ee.EEException
        If all retries are exhausted.
    """
    wait = backoff_s
    for attempt in range(1, max_retries + 1):
        try:
            result = ee_obj.getInfo()
            _log.debug("getInfo succeeded for '%s' on attempt %d", label, attempt)
            return result
        except ee.EEException as exc:
            _log.warning(
                "getInfo failed for '%s' (attempt %d/%d): %s",
                label,
                attempt,
                max_retries,
                exc,
            )
            if attempt == max_retries:
                _log.error("All %d retries exhausted for '%s'", max_retries, label)
                raise
            time.sleep(wait)
            wait *= 2


# ===========================================================================
# EXPORT HELPERS
# ===========================================================================

def export_to_drive(
    image: ee.Image,
    description: str,
    region: ee.Geometry,
    scale: int = cfg.EXPORT_SCALE,
    folder: str = "narino_flood_risk",
    crs: str = "EPSG:4326",
    max_pixels: int = 1e13,
) -> ee.batch.Task:
    """
    Export an ``ee.Image`` to Google Drive as a GeoTIFF.

    Parameters
    ----------
    image : ee.Image
        Image to export.
    description : str
        Task description (also used as filename prefix).
    region : ee.Geometry
        Export region.
    scale : int
        Spatial resolution in meters (default from ``gee_config.EXPORT_SCALE``).
    folder : str
        Google Drive folder name.
    crs : str
        Coordinate reference system (default WGS 84).
    max_pixels : int
        Maximum number of pixels per export.

    Returns
    -------
    ee.batch.Task
        The started export task.
    """
    # Sanitise description: GEE allows only alphanumeric, hyphens, underscores
    safe_desc = (
        description.replace(" ", "_")
        .replace(".", "_")
        .replace("/", "_")
    )

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=safe_desc,
        folder=folder,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=int(max_pixels),
        fileFormat="GeoTIFF",
    )
    task.start()
    _log.info("Export task started: '%s' (scale=%d m, folder='%s')", safe_desc, scale, folder)
    return task


def export_table_to_drive(
    collection: ee.FeatureCollection,
    description: str,
    folder: str = "narino_flood_risk",
    file_format: str = "CSV",
) -> ee.batch.Task:
    """
    Export an ``ee.FeatureCollection`` to Google Drive.

    Parameters
    ----------
    collection : ee.FeatureCollection
        Feature collection to export.
    description : str
        Task description / filename prefix.
    folder : str
        Google Drive folder name.
    file_format : str
        ``'CSV'``, ``'GeoJSON'``, or ``'SHP'``.

    Returns
    -------
    ee.batch.Task
        The started export task.
    """
    safe_desc = description.replace(" ", "_").replace(".", "_").replace("/", "_")
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=safe_desc,
        folder=folder,
        fileFormat=file_format,
    )
    task.start()
    _log.info("Table export started: '%s' (format=%s)", safe_desc, file_format)
    return task


# ===========================================================================
# MISCELLANEOUS HELPERS
# ===========================================================================

def get_dem() -> ee.Image:
    """
    Load the SRTM 30 m DEM clipped to Narino.

    Returns
    -------
    ee.Image
        Elevation image (band ``'elevation'``) in meters.
    """
    region = get_study_area_geometry()
    dem = ee.Image(cfg.SRTM).clip(region)
    return dem


def classify_by_thresholds(
    image: ee.Image,
    band: str,
    thresholds: Dict[str, Dict],
) -> ee.Image:
    """
    Reclassify a continuous image into discrete classes.

    Parameters
    ----------
    image : ee.Image
        Source image.
    band : str
        Band name to classify.
    thresholds : dict
        Mapping of class names to ``{'range': (low, high), ...}`` dicts,
        following the convention in ``gee_config`` (e.g. ``HAND_CLASSES``).

    Returns
    -------
    ee.Image
        Integer-classified image where pixel values correspond to class
        indices (1-based).
    """
    src = image.select(band)
    classified = ee.Image(0).rename("class")
    for idx, (_, cls) in enumerate(thresholds.items(), start=1):
        low, high = cls["range"]
        mask = src.gte(low).And(src.lt(high))
        classified = classified.where(mask, idx)
    return classified.selfMask()


def monitor_tasks(tasks: List[ee.batch.Task], poll_interval_s: int = 30) -> None:
    """
    Block and print progress until all export tasks complete or fail.

    Parameters
    ----------
    tasks : list[ee.batch.Task]
        List of active GEE export tasks.
    poll_interval_s : int
        Seconds between status checks.
    """
    remaining = list(tasks)
    while remaining:
        still_running = []
        for task in remaining:
            status = task.status()
            state = status.get("state", "UNKNOWN")
            desc = status.get("description", "?")
            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                _log.info("Task '%s' finished with state: %s", desc, state)
            else:
                still_running.append(task)
        remaining = still_running
        if remaining:
            _log.info("%d task(s) still running ...", len(remaining))
            time.sleep(poll_interval_s)
    _log.info("All tasks completed.")


# ===========================================================================
# MODULE SELF-TEST
# ===========================================================================

def _self_test() -> None:
    """Quick smoke test: verify GEE connectivity and study area loading."""
    _log.info("Running utils self-test ...")

    # Study area
    narino = get_study_area()
    n_features = safe_getinfo(narino.size(), label="narino.size()")
    _log.info("Narino feature count: %s", n_features)

    # Municipalities
    munis = get_municipalities()
    n_munis = safe_getinfo(munis.size(), label="municipalities.size()")
    _log.info("Municipality count: %s", n_munis)

    # Subregion filter
    va = get_subregion_municipalities("Centro")
    n_va = safe_getinfo(va.size(), label="centro.size()")
    _log.info("Centro municipalities: %s", n_va)

    # DEM
    dem = get_dem()
    _log.info("DEM loaded: %s", safe_getinfo(dem.bandNames(), label="dem.bandNames()"))

    _log.info("Self-test passed.")


if __name__ == "__main__":
    _self_test()
