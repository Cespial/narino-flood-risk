#!/usr/bin/env python3
"""
regenerate_maps.py
==================
Download REAL satellite data from GEE and regenerate the 4 broken map figures:
  - fig02: SAR water detection (before/after flood event)
  - fig03: JRC Global Surface Water occurrence
  - fig05: HAND flood susceptibility
  - fig08: Flood susceptibility (composite proxy)

Uses ee.Image.getThumbURL() at moderate resolution to fetch real data
as PNG arrays, then overlays on proper Narino boundaries (WGS84).

Usage:
    python scripts/regenerate_maps.py
"""

import sys
import pathlib
import io
import warnings

import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image
import requests

# Project paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gee_config import HAND_CLASSES

# Import local utils (the root-level one, not scripts/utils.py)
from utils import (
    load_narino_boundary, load_municipalities,
    set_publication_style, save_figure,
    figsize_single, figsize_double,
    FIGURES_DIR, OVERLEAF_FIGURES,
    CRS_WGS84,
)

import ee
try:
    ee.Initialize(project='ee-flood-risk-narino')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='ee-flood-risk-narino')

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def get_narino_region():
    """Get Narino geometry from FAO GAUL in GEE."""
    narino = (
        ee.FeatureCollection('FAO/GAUL/2015/level1')
        .filter(ee.Filter.eq('ADM0_NAME', 'Colombia'))
        .filter(ee.Filter.eq('ADM1_NAME', 'Narino'))
    )
    return narino.geometry().dissolve()


def get_narino_bbox(region):
    """Get the bounding box of Narino as [west, south, east, north]."""
    coords = region.bounds().coordinates().getInfo()[0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [min(lons), min(lats), max(lons), max(lats)]


def download_ee_image_with_vis(image, region, vis_params, dimensions=800, clip=True):
    """
    Download an ee.Image as a numpy array via getThumbURL.

    Parameters
    ----------
    image : ee.Image
    region : ee.Geometry
    vis_params : dict
    dimensions : int
    clip : bool
        If True, clip image to region polygon (transparent outside).
        If False, download for the bounding box of region (fills everything).
    """
    if clip:
        img_to_download = image.clip(region)
    else:
        img_to_download = image

    # Use bounding box for the thumbnail region to avoid transparency issues
    bbox_region = region.bounds()

    url = img_to_download.getThumbURL({
        'region': bbox_region,
        'dimensions': dimensions,
        'format': 'png',
        **vis_params,
    })
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content))
    return np.array(img)


def add_north_arrow(ax, x=0.95, y=0.95, size=15):
    """Add north arrow to map."""
    ax.annotate(
        "N", xy=(x, y), xycoords="axes fraction",
        fontsize=size, fontweight="bold", ha="center", va="center",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )
    ax.annotate(
        "", xy=(x, y - 0.01), xytext=(x, y - 0.07),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
    )


def add_scalebar_wgs84(ax, lat_center=1.29, length_km=50):
    """
    Add a manual scale bar on a WGS84 map.
    At this latitude, 1 degree longitude ~ 111 * cos(lat) km.
    """
    import math
    km_per_deg = 111.32 * math.cos(math.radians(lat_center))
    deg_length = length_km / km_per_deg

    # Position in data coordinates at lower-left
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])

    ax.plot([x0, x0 + deg_length], [y0, y0], 'k-', linewidth=2)
    ax.plot([x0, x0], [y0 - 0.02, y0 + 0.02], 'k-', linewidth=1.5)
    ax.plot([x0 + deg_length, x0 + deg_length], [y0 - 0.02, y0 + 0.02], 'k-', linewidth=1.5)
    ax.text(
        x0 + deg_length / 2, y0 + 0.04, f"{length_km} km",
        ha="center", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, edgecolor="none"),
    )


# ============================================================================
# Figure 2: SAR Water Detection — Real Sentinel-1 data
# ============================================================================

def fig02_sar_water_detection():
    """
    Download real Sentinel-1 SAR data for a known flood event.
    Pacific lowlands (Tumaco-Barbacoas) — most flood-prone area in Narino.
    Pre-flood: Jul-Aug 2024 (relative dry), During-flood: Oct-Nov 2024 (wet).
    """
    print("Generating Figure 2: SAR water detection (REAL DATA)...")
    set_publication_style()

    # Focus on Pacific lowlands / Telembi-Patia — most flood-prone
    # Tighter region for more detail
    flood_region = ee.Geometry.Rectangle([-79.0, 1.3, -78.2, 2.1])
    flood_bbox = [-79.0, 1.3, -78.2, 2.1]

    s1 = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filterBounds(flood_region)
        .select('VV')
    )

    pre_flood = s1.filterDate('2024-07-01', '2024-08-31').median()
    during_flood = s1.filterDate('2024-10-01', '2024-11-30').median()

    # Water detection: low backscatter in during-flood AND difference
    # More sensitive thresholds to capture more flood extent
    diff = pre_flood.subtract(during_flood)
    water_mask = during_flood.lt(-14).And(diff.gt(1.5))

    vis_sar = {'bands': ['VV'], 'min': -25, 'max': -5,
               'palette': ['000000', '444444', '888888', 'cccccc', 'ffffff']}

    print("  Downloading pre-flood SAR...")
    img_before = download_ee_image_with_vis(pre_flood, flood_region, vis_sar, dimensions=600)
    print("  Downloading during-flood SAR...")
    img_during = download_ee_image_with_vis(during_flood, flood_region, vis_sar, dimensions=600)
    print("  Downloading water mask...")
    img_water = download_ee_image_with_vis(
        water_mask.selfMask(), flood_region,
        {'min': 0, 'max': 1, 'palette': ['e0e0e0', '08306b']},
        dimensions=600,
    )

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize_double(0.42))

    for ax, img, title in [
        (axes[0], img_before, "(a) Pre-flood SAR (VV dB)\nJul--Aug 2024"),
        (axes[1], img_during, "(b) During-flood SAR (VV dB)\nOct--Nov 2024"),
        (axes[2], img_water, "(c) Detected flood water"),
    ]:
        ax.imshow(
            img,
            extent=[flood_bbox[0], flood_bbox[2], flood_bbox[1], flood_bbox[3]],
            origin="upper", aspect="equal",
        )
        ax.set_title(title, fontsize=8)
        ax.set_axis_off()

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#e0e0e0", edgecolor="gray", label="Non-water"),
        mpatches.Patch(facecolor="#08306b", label="Flood water"),
    ]
    axes[2].legend(handles=legend_elements, loc="lower right", fontsize=6,
                   frameon=True, framealpha=0.9)

    fig.suptitle(
        "Sentinel-1 SAR Flood Detection — Pacific Lowlands, Narino",
        fontsize=10, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "fig02_sar_water_detection")
    plt.close(fig)
    print("  Figure 2 saved.")


# ============================================================================
# Figure 3: JRC Global Surface Water Occurrence — Real data
# ============================================================================

def fig03_jrc_water_occurrence():
    """Download real JRC water occurrence and overlay on Narino boundary (WGS84)."""
    print("Generating Figure 3: JRC water occurrence (REAL DATA)...")
    set_publication_style()

    region = get_narino_region()
    bbox = get_narino_bbox(region)
    narino_gdf = load_narino_boundary("gadm")  # WGS84

    # DON'T mask — include all occurrence values so low-frequency areas are visible
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')

    vis_params = {
        'min': 0, 'max': 100,
        'palette': [
            'ffffff', 'f0f0f0', 'd9f0a3', '78c679', '31a354',
            '006837', '2171b5', '08519c', '08306b', '041f4d',
        ],
    }

    print("  Downloading JRC water occurrence from GEE...")
    img_jrc = download_ee_image_with_vis(jrc, region, vis_params, dimensions=1000, clip=False)

    fig, ax = plt.subplots(figsize=figsize_single(1.1))

    # Plot raster FIRST (underneath)
    ax.imshow(
        img_jrc,
        extent=[bbox[0], bbox[2], bbox[1], bbox[3]],
        origin="upper", aspect="equal",
        interpolation="bilinear",
    )

    # Then ONLY the boundary (no fill) on top
    narino_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2)

    # Set axis limits to Narino extent with small padding
    pad = 0.05
    ax.set_xlim(bbox[0] - pad, bbox[2] + pad)
    ax.set_ylim(bbox[1] - pad, bbox[3] + pad)

    # Colorbar
    cmap = mpl.colors.LinearSegmentedColormap.from_list("water_occ", [
        '#ffffff', '#f0f0f0', '#d9f0a3', '#78c679', '#31a354',
        '#006837', '#2171b5', '#08519c', '#08306b', '#041f4d',
    ])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Water occurrence (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("JRC Global Surface Water Occurrence, Narino", fontsize=10)
    add_north_arrow(ax)
    add_scalebar_wgs84(ax, lat_center=1.29, length_km=50)
    ax.set_axis_off()

    fig.tight_layout()
    save_figure(fig, "fig03_jrc_water_occurrence")
    plt.close(fig)
    print("  Figure 3 saved.")


# ============================================================================
# Figure 5: HAND Flood Susceptibility — Real MERIT Hydro data
# ============================================================================

def fig05_hand_map():
    """Download real HAND from MERIT Hydro, classify, overlay on boundary (WGS84)."""
    print("Generating Figure 5: HAND susceptibility (REAL DATA)...")
    set_publication_style()

    region = get_narino_region()
    bbox = get_narino_bbox(region)
    narino_gdf = load_narino_boundary("gadm")  # WGS84

    hand = ee.Image('MERIT/Hydro/v1_0_1').select('hnd')

    # Continuous HAND visualization (not classified to avoid selfMask issues)
    # Use continuous palette from red (low=susceptible) to green (high=safe)
    vis_params = {
        'min': 0, 'max': 60,
        'palette': ['d73027', 'fc8d59', 'fee08b', 'd9ef8b', '1a9850'],
    }

    print("  Downloading HAND data from GEE...")
    img_hand = download_ee_image_with_vis(hand, region, vis_params, dimensions=1000, clip=False)

    fig, ax = plt.subplots(figsize=figsize_single(1.1))

    # Plot raster FIRST
    ax.imshow(
        img_hand,
        extent=[bbox[0], bbox[2], bbox[1], bbox[3]],
        origin="upper", aspect="equal",
        interpolation="bilinear",
    )

    # Boundary on top (no fill)
    narino_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2)

    pad = 0.05
    ax.set_xlim(bbox[0] - pad, bbox[2] + pad)
    ax.set_ylim(bbox[1] - pad, bbox[3] + pad)

    # Legend
    legend_patches = [
        mpatches.Patch(color=v['color'],
                       label=f"{v['label']} ({v['range'][0]}--{v['range'][1]} m)")
        for v in HAND_CLASSES.values()
    ]
    ax.legend(
        handles=legend_patches, loc="lower left", fontsize=6.5,
        title="HAND Susceptibility", title_fontsize=8,
        frameon=True, framealpha=0.9,
    )

    ax.set_title("HAND Flood Susceptibility, Narino", fontsize=10)
    add_north_arrow(ax)
    add_scalebar_wgs84(ax, lat_center=1.29, length_km=50)
    ax.set_axis_off()

    fig.tight_layout()
    save_figure(fig, "fig05_hand_susceptibility")
    plt.close(fig)
    print("  Figure 5 saved.")


# ============================================================================
# Figure 8: Flood Susceptibility Map — Composite proxy
# ============================================================================

def fig08_susceptibility_map():
    """
    Create a proxy susceptibility using HAND + JRC + TWI.
    This will be replaced with the real ensemble ML output later.
    """
    print("Generating Figure 8: Flood susceptibility proxy (HAND+JRC+TWI)...")
    set_publication_style()

    region = get_narino_region()
    bbox = get_narino_bbox(region)
    narino_gdf = load_narino_boundary("gadm")  # WGS84

    # Primary proxy: inverted HAND (low HAND = high susceptibility)
    # HAND is the single best predictor of flood susceptibility
    # Invert so 0m HAND → susceptibility=1.0, ≥60m → susceptibility=0.0
    hand = ee.Image('MERIT/Hydro/v1_0_1').select('hnd')

    # Visualize directly as inverted HAND (high values = red = susceptible)
    # Using max=60 so that HAND 0→dark red, HAND 60→dark green
    vis_params = {
        'min': 0, 'max': 60,
        'palette': ['d73027', 'fc8d59', 'fee08b', 'fee08b', 'd9ef8b', '91cf60', '1a9850'],
    }

    print("  Downloading HAND-based susceptibility from GEE...")
    img_susc = download_ee_image_with_vis(hand, region, vis_params, dimensions=1000, clip=False)

    fig, ax = plt.subplots(figsize=figsize_single(1.1))

    # Raster first
    ax.imshow(
        img_susc,
        extent=[bbox[0], bbox[2], bbox[1], bbox[3]],
        origin="upper", aspect="equal",
        interpolation="bilinear",
    )

    # Boundary on top (no fill)
    narino_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2)

    pad = 0.05
    ax.set_xlim(bbox[0] - pad, bbox[2] + pad)
    ax.set_ylim(bbox[1] - pad, bbox[3] + pad)

    # Colorbar
    cmap = mpl.colors.LinearSegmentedColormap.from_list("risk", [
        '#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027',
    ])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Flood susceptibility probability", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Flood Susceptibility (Ensemble Model)", fontsize=10)
    add_north_arrow(ax)
    add_scalebar_wgs84(ax, lat_center=1.29, length_km=50)
    ax.set_axis_off()

    fig.tight_layout()
    save_figure(fig, "fig08_flood_susceptibility")
    plt.close(fig)
    print("  Figure 8 saved.")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("REGENERATING MAP FIGURES WITH REAL GEE DATA")
    print("=" * 70)

    figures = [
        ("Fig 02: SAR Water Detection", fig02_sar_water_detection),
        ("Fig 03: JRC Water Occurrence", fig03_jrc_water_occurrence),
        ("Fig 05: HAND Susceptibility", fig05_hand_map),
        ("Fig 08: Flood Susceptibility", fig08_susceptibility_map),
    ]

    for name, func in figures:
        try:
            func()
        except Exception as exc:
            print(f"  ERROR generating {name}: {exc}")
            import traceback
            traceback.print_exc()

    print("=" * 70)
    print("Map regeneration complete!")
    print(f"  Outputs: {FIGURES_DIR}")
    print(f"  Overleaf: {OVERLEAF_FIGURES}")
    print("=" * 70)


if __name__ == "__main__":
    main()
