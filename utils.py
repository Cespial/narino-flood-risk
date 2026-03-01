"""
Shared utility functions for the Narino Flood Risk Assessment project.

Provides common I/O helpers, boundary loaders, logging setup, and
reusable geospatial processing routines used across all analysis scripts.
"""

import os
import sys
import logging
import pathlib
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Project paths (mirrors gee_config.py but without GEE dependency)
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
BOUNDARIES_DIR = DATA_DIR / "boundaries"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
OVERLEAF_DIR = PROJECT_ROOT / "overleaf"
OVERLEAF_FIGURES = OVERLEAF_DIR / "figures"
OVERLEAF_TABLES = OVERLEAF_DIR / "tables"
LOGS_DIR = PROJECT_ROOT / "logs"

# Expected Narino area in km2 (DANE official)
NARINO_AREA_KM2 = 33_268.0
NARINO_AREA_TOLERANCE = 0.05  # 5% tolerance for area validation

# EPSG codes
CRS_WGS84 = "EPSG:4326"
CRS_COLOMBIA = "EPSG:3116"  # MAGNA-SIRGAS / Colombia Bogota zone


def setup_logging(
    script_name: str,
    log_dir: Optional[pathlib.Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure a logger that writes to both console and a timestamped log file.

    Parameters
    ----------
    script_name : str
        Name of the calling script (used for log file name).
    log_dir : Path, optional
        Directory for log files. Defaults to PROJECT_ROOT/logs.
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
    """
    if log_dir is None:
        log_dir = LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # File handler
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    logger.info("Log started: %s", log_file)
    return logger


# ---------------------------------------------------------------------------
# Boundary loaders
# ---------------------------------------------------------------------------

def load_narino_boundary(source: str = "gadm") -> gpd.GeoDataFrame:
    """
    Load the Narino department boundary polygon.

    Parameters
    ----------
    source : str
        One of 'gadm', 'geoboundaries', or 'naturalearth'.

    Returns
    -------
    gpd.GeoDataFrame with a single polygon row.
    """
    paths = {
        "gadm": BOUNDARIES_DIR / "narino_department_boundary_GADM41.geojson",
        "geoboundaries": BOUNDARIES_DIR / "narino_department_boundary_geoBoundaries.geojson",
        "naturalearth": BOUNDARIES_DIR / "narino_department_naturalearth.geojson",
    }
    path = paths.get(source)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Narino boundary not found for source='{source}'. "
            f"Expected: {path}. Run scripts/download_boundaries.py first."
        )
    gdf = gpd.read_file(path)
    return gdf


def load_municipalities(source: str = "gadm") -> gpd.GeoDataFrame:
    """
    Load Narino municipality polygons (64 municipalities).

    Parameters
    ----------
    source : str
        'gadm' or 'geoboundaries'.

    Returns
    -------
    gpd.GeoDataFrame
    """
    paths = {
        "gadm": BOUNDARIES_DIR / "narino_municipalities_64_GADM41.geojson",
        "geoboundaries": BOUNDARIES_DIR / "narino_municipalities_geoBoundaries_simplified.geojson",
    }
    path = paths.get(source)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Municipalities file not found for source='{source}'. "
            f"Expected: {path}."
        )
    return gpd.read_file(path)


def load_subregions() -> gpd.GeoDataFrame:
    """Load the 13 official subregions of Narino."""
    path = BOUNDARIES_DIR / "narino_13_subregions.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Subregions file not found: {path}")
    return gpd.read_file(path)


def load_river_basins(level: int = 5) -> gpd.GeoDataFrame:
    """
    Load HydroSHEDS river basins clipped to Narino.

    Parameters
    ----------
    level : int
        HydroBASINS level (5 or 7).
    """
    path = BOUNDARIES_DIR / f"narino_river_basins_HydroSHEDS_L{level}.geojson"
    if not path.exists():
        raise FileNotFoundError(f"River basins file not found: {path}")
    return gpd.read_file(path)


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create all required output directories if they do not exist."""
    for d in [
        FIGURES_DIR, TABLES_DIR,
        OVERLEAF_FIGURES, OVERLEAF_TABLES,
        LOGS_DIR,
        OUTPUTS_DIR / "phase1_water_maps",
        OUTPUTS_DIR / "phase2_flood_frequency",
        OUTPUTS_DIR / "phase3_risk_model",
        OUTPUTS_DIR / "phase4_municipal_stats",
        OUTPUTS_DIR / "phase5_qc",
    ]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Area and geometry utilities
# ---------------------------------------------------------------------------

def compute_area_km2(gdf: gpd.GeoDataFrame) -> float:
    """
    Compute total area in km2 by projecting to MAGNA-SIRGAS / Colombia.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with polygon geometries.

    Returns
    -------
    float : Total area in square kilometres.
    """
    projected = gdf.to_crs(CRS_COLOMBIA)
    return projected.geometry.area.sum() / 1e6


def validate_narino_area(gdf: gpd.GeoDataFrame, tolerance: float = 0.05) -> bool:
    """
    Check whether the total area of gdf is close to the expected
    Narino area (33,268 km2) within the given fractional tolerance.
    """
    area = compute_area_km2(gdf)
    expected = NARINO_AREA_KM2
    diff_frac = abs(area - expected) / expected
    return diff_frac <= tolerance


# ---------------------------------------------------------------------------
# Data I/O helpers
# ---------------------------------------------------------------------------

def save_dataframe(
    df: pd.DataFrame,
    name: str,
    csv_dir: Optional[pathlib.Path] = None,
    latex_dir: Optional[pathlib.Path] = None,
    index: bool = False,
) -> None:
    """
    Save a DataFrame as both CSV and LaTeX (.tex) table fragment.

    Parameters
    ----------
    df : pd.DataFrame
    name : str
        Base filename (without extension).
    csv_dir : Path, optional
        Directory for CSV output (defaults to TABLES_DIR).
    latex_dir : Path, optional
        Directory for LaTeX output (defaults to OVERLEAF_TABLES).
    index : bool
        Whether to include the DataFrame index.
    """
    if csv_dir is None:
        csv_dir = TABLES_DIR
    if latex_dir is None:
        latex_dir = OVERLEAF_TABLES

    csv_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / f"{name}.csv"
    tex_path = latex_dir / f"{name}.tex"

    df.to_csv(csv_path, index=index)
    df.to_latex(tex_path, index=index, escape=True, longtable=False)

    return csv_path, tex_path


def load_results(phase: str, filename: str) -> pd.DataFrame:
    """
    Load a CSV results file from the specified phase output directory.

    Parameters
    ----------
    phase : str
        Phase directory name, e.g. 'phase1_water_maps'.
    filename : str
        CSV filename.

    Returns
    -------
    pd.DataFrame
    """
    path = OUTPUTS_DIR / phase / filename
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Matplotlib publication style helper
# ---------------------------------------------------------------------------

def set_publication_style() -> None:
    """
    Configure matplotlib rcParams for publication-quality figures.
    Uses Times New Roman (or Liberation Serif fallback), 10pt font,
    and colorblind-safe defaults.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Try Times New Roman, fall back to Liberation Serif or serif generic
    available_fonts = [f.name for f in mpl.font_manager.fontManager.ttflist]
    if "Times New Roman" in available_fonts:
        font_family = "Times New Roman"
    elif "Liberation Serif" in available_fonts:
        font_family = "Liberation Serif"
    else:
        font_family = "serif"

    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": [font_family, "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Lines
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        # Axes
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Legend
        "legend.frameon": False,
        # PDF / SVG
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Figure size helpers (mm to inches for matplotlib)
# ---------------------------------------------------------------------------

# Journal standard widths
SINGLE_COL_MM = 89.0
DOUBLE_COL_MM = 183.0
MM_TO_INCH = 1.0 / 25.4


def figsize_single(aspect: float = 0.75) -> tuple:
    """Return (width, height) in inches for a single-column figure."""
    w = SINGLE_COL_MM * MM_TO_INCH
    return (w, w * aspect)


def figsize_double(aspect: float = 0.5) -> tuple:
    """Return (width, height) in inches for a double-column figure."""
    w = DOUBLE_COL_MM * MM_TO_INCH
    return (w, w * aspect)


def save_figure(
    fig,
    name: str,
    formats: Optional[list] = None,
    figures_dir: Optional[pathlib.Path] = None,
    overleaf_dir: Optional[pathlib.Path] = None,
    dpi: int = 600,
) -> None:
    """
    Save a matplotlib figure to outputs/figures/ and overleaf/figures/
    in the specified formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Base filename (without extension).
    formats : list of str
        File formats (default: ['pdf', 'png']).
    figures_dir : Path, optional
    overleaf_dir : Path, optional
    dpi : int
        DPI for raster formats.
    """
    if formats is None:
        formats = ["pdf", "png"]
    if figures_dir is None:
        figures_dir = FIGURES_DIR
    if overleaf_dir is None:
        overleaf_dir = OVERLEAF_FIGURES

    figures_dir.mkdir(parents=True, exist_ok=True)
    overleaf_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        for d in [figures_dir, overleaf_dir]:
            out = d / f"{name}.{fmt}"
            fig.savefig(out, format=fmt, dpi=dpi, bbox_inches="tight")
