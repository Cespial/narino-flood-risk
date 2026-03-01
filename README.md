# Municipality-Scale Flood Risk Mapping in Narino, Colombia

**Using Sentinel-1 SAR and Ensemble Machine Learning (2015--2025)**

Cristian Espinal Maya [![ORCID](https://img.shields.io/badge/ORCID-0009--0000--1009--8388-green)](https://orcid.org/0009-0000-1009-8388) · Santiago Jimenez Londono [![ORCID](https://img.shields.io/badge/ORCID-0009--0007--9862--7133-green)](https://orcid.org/0009-0007-9862-7133)

School of Applied Sciences and Engineering, Universidad EAFIT, Medellin, Colombia

[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow)](LICENSE) · [![License: CC BY 4.0](https://img.shields.io/badge/Manuscript-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

## About

This project replicates the [Antioquia flood risk mapping framework](https://github.com/Cespial/antioquia-flood-risk) for the **Department of Narino**, Colombia. Narino (33,268 km^2; 64 municipalities; ~1.9 million inhabitants) presents unique challenges for flood risk assessment due to its extreme topographic variability (0--4,764 m elevation), dual precipitation regimes (bimodal Andean vs. unimodal Pacific), and high vulnerability in Pacific lowland communities.

## Study Area Characteristics

| Feature | Description |
|---------|-------------|
| **Area** | 33,268 km^2 |
| **Municipalities** | 64, organized in 13 subregions |
| **Capital** | San Juan de Pasto (2,527 m asl) |
| **Elevation range** | 0 m (Pacific coast) to 4,764 m (Volcan Cumbal) |
| **Major rivers** | Patia, Guaitara, Telembi, Mira, Juanambu, Mayo, Sanquianga |
| **Climate zones** | Pacific lowlands (2,600--5,000+ mm/yr), Andean highlands (800--1,600 mm/yr) |
| **Most flood-prone areas** | Tumaco, Barbacoas, Olaya Herrera, Roberto Payan, Magui, El Charco |

## Repository Structure

```
.
├── gee_config.py              # Central configuration (Narino-specific)
├── scripts/                   # Processing and analysis pipeline
│   ├── 01_sar_water_detection.py
│   ├── 02_jrc_water_analysis.py
│   ├── 03_flood_susceptibility_features.py
│   ├── 04_ml_flood_susceptibility.py
│   ├── 05_population_exposure.py
│   ├── 06_climate_analysis.py
│   ├── 07_visualization.py
│   ├── 08_generate_tables.py
│   └── 09_quality_control.py
├── overleaf/                  # Manuscript (preprint format)
├── data/                      # Downloaded data (auto-created)
├── outputs/                   # Results (auto-created)
├── BIBLIOGRAPHY.md            # Research context and references
└── README.md
```

## Data Sources

All data are open-access and processed via [Google Earth Engine](https://earthengine.google.com/):

- **Sentinel-1 GRD** (ESA/Copernicus) -- 10 m SAR flood detection
- **JRC Global Surface Water** -- 38-year water dynamics
- **SRTM DEM v3** -- Topographic features (30 m)
- **MERIT Hydro** -- HAND computation (90 m)
- **CHIRPS / ERA5-Land** -- Precipitation and soil moisture
- **ESA WorldCover / Sentinel-2** -- Land cover and NDVI
- **WorldPop** -- Population density (100 m)
- **FAO GAUL 2015** -- Administrative boundaries

## Reproducing the Analysis

### Requirements

- Python 3.10+
- Google Earth Engine account ([sign up](https://signup.earthengine.google.com/))
- Libraries: see `requirements.txt`

### Setup

```bash
# 1. Clone and setup
cd narino_flood_risk_research
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure GEE credentials
earthengine authenticate
echo "GEE_PROJECT_ID=ee-flood-risk-narino" > .env

# 3. Run pipeline
python scripts/01_sar_water_detection.py   # ~3-4 hours (GEE)
python scripts/02_jrc_water_analysis.py    # ~30 min (GEE)
python scripts/03_flood_susceptibility_features.py  # ~2 hours (GEE)
python scripts/04_ml_flood_susceptibility.py        # <30 min (local)
python scripts/05_population_exposure.py   # ~1 hour
python scripts/06_climate_analysis.py      # ~1 hour
python scripts/07_visualization.py         # ~20 min
python scripts/08_generate_tables.py       # ~5 min
python scripts/09_quality_control.py       # ~10 min
```

## Key Differences from Antioquia Framework

1. **Terrain**: Extreme elevation range (0--4,764 m) causes SAR shadow/layover in Andean zones. Detection accuracy is highest in Pacific lowlands.
2. **Precipitation**: Dual regime -- bimodal Andean vs. unimodal Pacific (one of the wettest regions globally).
3. **ENSO Response**: Narino receives above-normal rainfall during **both** El Nino and La Nina, unlike most Colombian departments.
4. **Subregions**: 13 subregions (vs. 9 in Antioquia) used for spatial cross-validation.
5. **SAR Orbit**: Consider using both ASCENDING and DESCENDING passes to reduce terrain distortion effects.

## Citation

```bibtex
@article{EspinalMaya2026Narino,
  author  = {Espinal Maya, Cristian and Jim\'enez Londo\~no, Santiago},
  title   = {Municipality-Scale Flood Risk Mapping in {Nari\~no}, {Colombia},
             Using {Sentinel-1} {SAR} and Ensemble Machine Learning (2015--2025)},
  year    = {2026},
  note    = {Preprint}
}
```

## License

Source code: [MIT License](LICENSE). Manuscript and figures: CC BY 4.0.

## Acknowledgments

This research builds on the open-access framework developed for the Department of Antioquia. All data and computational resources are open-access (Google Earth Engine, Copernicus/ESA Sentinel-1).
