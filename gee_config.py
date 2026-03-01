"""
Central configuration for the Narino Flood Risk Assessment project.
All parameters, constants, and GEE asset paths are defined here.

Replicated from the Antioquia framework — adapted for the Department of Narino.
"""

import os
import ee
from dotenv import load_dotenv

load_dotenv()

# --- GEE Initialization ---
try:
    ee.Initialize(project=os.getenv('GEE_PROJECT_ID', 'ee-flood-risk-narino'))
except Exception:
    ee.Authenticate()
    ee.Initialize(project=os.getenv('GEE_PROJECT_ID', 'ee-flood-risk-narino'))

# ============================================================================
# STUDY AREA: Department of Narino, Colombia
# ============================================================================

# Administrative boundaries from FAO GAUL Level 1
ADMIN_DATASET = 'FAO/GAUL/2015/level1'
DEPARTMENT_NAME = 'Narino'  # ASCII name as used in GAUL (ADM1_CODE = 955)
COUNTRY_NAME = 'Colombia'

# Municipal boundaries from FAO GAUL Level 2
MUNICIPAL_DATASET = 'FAO/GAUL/2015/level2'

# HydroSHEDS basin boundaries
HYDROBASINS_L5 = 'WWF/HydroSHEDS/v1/Basins/hybas_sa_lev05_v1c'
HYDROBASINS_L7 = 'WWF/HydroSHEDS/v1/Basins/hybas_sa_lev07_v1c'

# Department statistics
DEPARTMENT_AREA_KM2 = 33268  # Official area
EXPECTED_MUNICIPALITY_COUNT = 64

# ============================================================================
# 13 Subregions of Narino with their municipalities
# Municipality names use GAUL ASCII spelling (no accents/tildes)
# ============================================================================

SUBREGIONS = {
    'Centro': [
        'Pasto', 'Chachagui', 'La Florida', 'Narino', 'Tangua', 'Yacuanquer'
    ],
    'Guambuyaco': [
        'El Penol', 'El Tambo', 'La Llanada', 'Los Andes'
    ],
    'Juanambu': [
        'Arboleda', 'Buesaco', 'La Union', 'San Lorenzo',
        'San Pedro De Cartago'
    ],
    'La Cordillera': [
        'Cumbitara', 'El Rosario', 'Leiva', 'Policarpa', 'Taminango'
    ],
    'La Sabana': [
        'Guaitarilla', 'Imues', 'Ospina', 'Sapuyes', 'Tuquerres'
    ],
    'Los Abades': [
        'Providencia', 'Samaniego', 'Santa Cruz'
    ],
    'Obando': [
        'Aldana', 'Contadero', 'Cordoba', 'Cuaspud', 'Cumbal',
        'Funes', 'Guachucal', 'Gualmatan', 'Iles', 'Ipiales',
        'Potosi', 'Puerres', 'Pupiales'
    ],
    'Occidente': [
        'Ancuya', 'Consaca', 'Linares', 'Sandona'
    ],
    'Pacifico Sur': [
        'Francisco Pizarro', 'Tumaco'
    ],
    'Piedemonte Costero': [
        'Mallama', 'Ricaurte'
    ],
    'Rio Mayo': [
        'Alban', 'Belen', 'Colon', 'El Tablon', 'La Cruz',
        'San Bernardo', 'San Pablo'
    ],
    'Sanquianga': [
        'El Charco', 'La Tola', 'Mosquera', 'Olaya Herrera',
        'Santa Barbara'
    ],
    'Telembi': [
        'Barbacoas', 'Magui', 'Roberto Payan'
    ],
}

# ============================================================================
# SPATIAL CV FOLD GROUPINGS
# Group 13 subregions into 5 folds for spatial cross-validation
# Grouped by geographic proximity to ensure spatial independence
# ============================================================================

SPATIAL_CV_FOLDS = {
    0: ['Centro', 'Occidente'],                     # Central Andean
    1: ['La Sabana', 'Obando'],                     # Southern highlands (border)
    2: ['Juanambu', 'Rio Mayo', 'La Cordillera'],   # Northeastern Andean valleys
    3: ['Guambuyaco', 'Los Abades', 'Piedemonte Costero'],  # Western slopes
    4: ['Pacifico Sur', 'Sanquianga', 'Telembi'],   # Pacific lowlands
}

# ============================================================================
# TEMPORAL CONFIGURATION
# ============================================================================

# Full analysis period: 2015-2025 (Sentinel-1 era)
ANALYSIS_START = '2015-01-01'
ANALYSIS_END = '2025-12-31'

# Sentinel-1 availability
S1_START = '2014-10-03'  # Sentinel-1A launch
S1B_FAILURE = '2021-12-23'  # Sentinel-1B failure
S1C_LAUNCH = '2024-04-25'  # Sentinel-1C launch

# Seasonal periods for Narino
# NOTE: Narino has TWO distinct precipitation regimes:
#   - Andean region: Bimodal (similar to standard Colombian pattern)
#   - Pacific coast: Unimodal (wet most of year, slight dip Aug-Sep)
# We use the bimodal standard for consistency with the Antioquia framework,
# but this should be noted in the manuscript as a limitation for Pacific zones.
SEASONS = {
    'DJF': {'months': [12, 1, 2], 'label': 'Dry season 1'},
    'MAM': {'months': [3, 4, 5], 'label': 'Wet season 1 (first rains)'},
    'JJA': {'months': [6, 7, 8], 'label': 'Dry season 2 (veranillo)'},
    'SON': {'months': [9, 10, 11], 'label': 'Wet season 2 (peak floods)'},
}

# Pacific-specific seasons (for subregional analysis in climate script)
SEASONS_PACIFIC = {
    'Wet_peak': {'months': [1, 2, 3, 4, 5, 6], 'label': 'Peak wet season'},
    'Wet_mod': {'months': [7, 10, 11, 12], 'label': 'Moderate wet season'},
    'Relative_dry': {'months': [8, 9], 'label': 'Relative dry (Aug-Sep)'},
}

# Annual analysis windows
ANNUAL_WINDOWS = {year: {
    'start': f'{year}-01-01',
    'end': f'{year}-12-31',
} for year in range(2015, 2026)}

# ENSO classification for study period (2015-2025)
# Source: NOAA ONI (Oceanic Nino Index)
ENSO_YEARS = {
    'El Nino': [2015, 2016, 2023, 2024],
    'La Nina': [2020, 2021, 2022],
    'Neutral': [2017, 2018, 2019, 2025],
}

# ============================================================================
# SATELLITE DATA SOURCES
# ============================================================================

# Sentinel-1 SAR GRD (Ground Range Detected)
S1_COLLECTION = 'COPERNICUS/S1_GRD'
S1_PARAMS = {
    'instrumentMode': 'IW',  # Interferometric Wide swath
    'resolution': 10,  # meters
    'polarization': ['VV', 'VH'],  # Dual-pol
    # NOTE: For Narino's mountainous terrain, consider using BOTH orbits
    # to reduce shadow/layover effects. Primary = DESCENDING.
    'orbitProperties_pass': 'DESCENDING',
    'resolution_meters': 10,
}

# Water detection thresholds for SAR (VV polarization)
# NOTE: For Narino, the Pacific lowlands have dense vegetation cover
# that attenuates C-band SAR. The default thresholds are kept but
# may need adjustment based on initial results.
SAR_WATER_THRESHOLDS = {
    'otsu_adaptive': True,  # Use Otsu automatic thresholding
    'vv_default': -15.0,  # dB, fallback threshold
    'vv_range': (-20.0, -12.0),  # Valid range for water detection
    'vh_default': -22.0,  # dB, fallback for VH
    'min_water_area_ha': 1.0,  # Minimum mappable water body
    'speckle_filter_radius': 50,  # meters, focal median radius
}

# JRC Global Surface Water v1.4
JRC_GSW = 'JRC/GSW1_4/GlobalSurfaceWater'
JRC_GSW_MONTHLY = 'JRC/GSW1_4/MonthlyHistory'
JRC_GSW_YEARLY = 'JRC/GSW1_4/YearlyHistory'

# JRC GLOFAS Flood Hazard
GLOFAS_FLOOD_HAZARD = 'JRC/CEMS_GLOFAS/FloodHazard/v1'

# SRTM Digital Elevation Model (30m)
SRTM = 'USGS/SRTMGL1_003'

# MERIT Hydro (90m) - pre-computed hydrological layers
MERIT_HYDRO = 'MERIT/Hydro/v1_0_1'

# HAND (Height Above Nearest Drainage) - derived from SRTM
HAND_DATASET = 'users/gaborimbre/HAND_30m_SA'  # South America HAND

# CHIRPS Daily Precipitation (5.5km)
CHIRPS = 'UCSB-CHG/CHIRPS/DAILY'

# ERA5-Land Monthly (11km)
ERA5_LAND = 'ECMWF/ERA5_LAND/MONTHLY_AGGR'

# MODIS LST (1km)
MODIS_LST = 'MODIS/061/MOD11A2'

# WorldPop Population Density (100m)
WORLDPOP = 'WorldPop/GP/100m/pop'

# ESA WorldCover (10m land cover)
WORLDCOVER = 'ESA/WorldCover/v200'

# Sentinel-2 Surface Reflectance (for NDVI)
S2_SR = 'COPERNICUS/S2_SR_HARMONIZED'

# Friction surface (accessibility / distance to roads proxy)
FRICTION_SURFACE = 'projects/malariaatlasproject/assets/accessibility/friction_surface/2019_v5_1'

# ============================================================================
# FLOOD RISK MODEL PARAMETERS
# ============================================================================

# HAND thresholds for flood susceptibility
# NOTE: For Narino's mountainous terrain, HAND < 5m threshold is strict
# and appropriate. The steep terrain means most susceptible areas are
# truly confined to narrow floodplains.
HAND_CLASSES = {
    'very_high': {'range': (0, 5), 'label': 'Very High', 'color': '#d73027'},
    'high': {'range': (5, 15), 'label': 'High', 'color': '#fc8d59'},
    'moderate': {'range': (15, 30), 'label': 'Moderate', 'color': '#fee08b'},
    'low': {'range': (30, 60), 'label': 'Low', 'color': '#d9ef8b'},
    'very_low': {'range': (60, 9999), 'label': 'Very Low', 'color': '#1a9850'},
}

# Flood frequency classes (based on JRC occurrence)
FLOOD_FREQUENCY_CLASSES = {
    'permanent': {'range': (75, 100), 'label': 'Permanent water', 'color': '#08306b'},
    'very_frequent': {'range': (50, 75), 'label': 'Very frequent', 'color': '#2171b5'},
    'frequent': {'range': (25, 50), 'label': 'Frequent flooding', 'color': '#6baed6'},
    'occasional': {'range': (10, 25), 'label': 'Occasional flooding', 'color': '#bdd7e7'},
    'rare': {'range': (1, 10), 'label': 'Rare flooding', 'color': '#eff3ff'},
}

# Risk class thresholds (susceptibility probability)
RISK_CLASSES = {
    'Very Low':  (0.0, 0.2),
    'Low':       (0.2, 0.4),
    'Moderate':  (0.4, 0.6),
    'High':      (0.6, 0.8),
    'Very High': (0.8, 1.0),
}

# ML Model parameters
ML_PARAMS = {
    'random_forest': {
        'n_estimators': 500,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42,
        'n_jobs': -1,
    },
    'xgboost': {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'auc',
    },
    'lightgbm': {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'metric': 'auc',
        'verbose': -1,
    },
}

# Training sample configuration
# NOTE: Narino (33,268 km2) is about half the size of Antioquia (63,612 km2)
# but has 64 municipalities. 5,000 samples per class is still appropriate.
SAMPLES_PER_CLASS = 5000

# Flood susceptibility features (predictors) - same 18 features as Antioquia
SUSCEPTIBILITY_FEATURES = [
    'elevation',            # SRTM 30m
    'slope',                # Derived from SRTM
    'aspect',               # Derived from SRTM
    'curvature',            # Plan curvature
    'hand',                 # Height Above Nearest Drainage
    'twi',                  # Topographic Wetness Index
    'spi',                  # Stream Power Index
    'dist_rivers',          # Distance to nearest river
    'dist_roads',           # Distance to nearest road
    'drainage_density',     # River density per unit area
    'rainfall_annual',      # Mean annual precipitation (CHIRPS)
    'rainfall_max_monthly', # Max monthly precipitation
    'land_cover',           # ESA WorldCover 2021
    'ndvi_mean',            # Mean annual NDVI
    'soil_moisture',        # ERA5-Land soil moisture
    'pop_density',          # WorldPop 100m
    'flood_frequency_jrc',  # JRC water occurrence
    'sar_water_frequency',  # Sentinel-1 derived water frequency
]

# Training label criteria
FLOOD_LABEL_CRITERIA = {
    'positive': {
        'jrc_occurrence_min': 25,  # JRC occurrence >= 25%
        'hand_max': 5,             # OR HAND < 5 m
    },
    'negative': {
        'jrc_occurrence_max': 5,   # JRC occurrence < 5%
        'hand_min': 30,            # AND HAND >= 30 m
        'slope_min': 10,           # AND slope > 10 degrees
    },
}

# Cross-validation parameters
CV_PARAMS = {
    'n_splits': 5,  # Stratified spatial K-fold
    'test_size': 0.3,
    'random_state': 42,
}

# Flood Risk Index (FRI) weights
FRI_WEIGHTS = {
    'pop_normalized': 0.4,
    'pct_high_area': 0.3,
    'mean_susceptibility': 0.3,
}

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Color palettes
WATER_PALETTE = ['#ffffff', '#bdd7e7', '#6baed6', '#2171b5', '#08306b']
RISK_PALETTE = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b',
                '#fc8d59', '#d73027']
SUBREGION_PALETTE = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
    '#fc8d62', '#8da0cb', '#e78ac3'
]  # 13 colors for 13 subregions

# Map center for Narino
MAP_CENTER = {'lat': 1.29, 'lon': -77.36}
MAP_ZOOM = 8

# Bounding box (approximate)
BBOX = {
    'north': 2.35,
    'south': 0.40,
    'west': -79.10,
    'east': -76.80,
}

# Figure DPI for publication
FIGURE_DPI = 600
FIGURE_FORMAT = ['pdf', 'png']

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Export scale (meters)
EXPORT_SCALE = 30

# Google Drive export folder
DRIVE_EXPORT_FOLDER = 'narino_flood_risk'

# GEE asset path for susceptibility map
GEE_SUSCEPTIBILITY_ASSET = 'projects/ee-maestria-tesis/assets/narino_flood_susceptibility_ensemble'

# Project paths
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
TABLES_DIR = OUTPUTS_DIR / 'tables'
OVERLEAF_DIR = PROJECT_ROOT / 'overleaf'
LOGS_DIR = PROJECT_ROOT / 'logs'

# ============================================================================
# NARINO-SPECIFIC NOTES
# ============================================================================
"""
Key differences from Antioquia that affect the analysis:

1. TERRAIN: Narino has extreme elevation range (0-4,764 m). The Pacific
   lowlands are flat and suitable for SAR flood detection, but the Andean
   highlands will have significant radar shadow/layover issues.

2. PRECIPITATION: Two distinct regimes - bimodal Andean (800-1,600 mm/yr)
   and unimodal Pacific (2,600-5,000+ mm/yr). The Pacific coast is one of
   the wettest places in the world.

3. ENSO: Unlike most of Colombia, Narino tends to receive above-normal
   rainfall during BOTH El Nino and La Nina phases. This is unusual and
   should be discussed in the manuscript.

4. FLOOD-PRONE AREAS: The most flood-vulnerable municipalities are in the
   Pacific lowlands: Tumaco, Barbacoas, Olaya Herrera, Roberto Payan,
   Magui, El Charco, Santa Barbara.

5. SAR CONSIDERATIONS:
   - Dense tropical forest in Pacific zone attenuates C-band SAR
   - Steep Andean terrain causes shadow/layover
   - Consider using ASCENDING + DESCENDING orbits
   - SAR detection accuracy will be lower in steep valleys (~60-70%)

6. GAUL NOTES:
   - 'El Tablon' in GAUL = 'El Tablon de Gomez' officially
   - 'Magui' in GAUL = 'Magui Payan' officially
   - 'Narino' (municipality) and 'El Penol' may be absent from GAUL Level 2
"""
