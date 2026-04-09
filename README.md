# Urban Resilience Engine — Nairobi

> Predicting extreme weather risk across Nairobi's urban infrastructure using a four-phase machine learning pipeline.

**Institution:** Strathmore University  
**Programme:** MSc Sustainable Energy Transitions  
**Course:** Data Science Concepts  
**Author:** George  
**Date:** April 2026

---

## Overview

The Urban Resilience Engine divides Nairobi into **873 hexagonal zones** (each ~0.46 km²) and scores every zone from 0 to 1 for extreme weather risk primarily **flooding** and **heat stress**. The tool is designed for city planners, NGOs, and government offices to identify high-risk neighbourhoods and prioritise infrastructure investment.

---

## Pipeline Phases

| Phase | Name | What it does |
|-------|------|--------------|
| 1 | **Data Engineering (ETL)** | Builds the H3 hex grid, collects OSM + urban data, outputs `nairobi_grid.parquet` |
| 2 | **Computer Vision** | Classifies satellite image tiles by land-use type (green / built / water) |
| 3 | **Predictive Modelling** | Trains XGBoost + Bayesian Neural Network; generates SHAP explanations |
| 4 | **Deployment & Ethics** | Streamlit interactive map, bias audit across income quintiles |

---

## Project Structure

```
urban-resilience-engine/
├── config.py                        # Shared paths, constants, Nairobi bbox
├── requirements-py314.txt           # Python dependencies
├── demo_ngrok.py                    # Public demo launcher
├── pytest.ini                       # Test configuration
│
├── src/
│   ├── phase1_etl/
│   │   ├── pipeline.py              # ETL orchestrator — run this first
│   │   ├── fetch_urban_data.py      # Spatially-realistic synthetic urban data
│   │   ├── fetch_osm.py             # OSM buildings, roads, drainage, hospitals
│   │   └── clean_merge.py           # Spatial join onto H3 grid
│   │
│   ├── phase2_cv/
│   │   ├── cnn_model.py             # ResNet18 transfer learning classifier
│   │   └── tile_images.py           # Satellite tile generator
│   │
│   ├── phase3_modeling/
│   │   ├── feature_eng.py           # Feature matrix + risk score target
│   │   ├── train_xgboost.py         # XGBoost classifier with SHAP
│   │   ├── train_bayesian.py        # MC-Dropout Bayesian Neural Network
│   │   └── evaluate.py              # ROC, calibration, confusion matrix
│   │
│   └── phase4_deploy/
│       ├── app.py                   # Streamlit dashboard (main UI)
│       └── bias_audit.py            # Fairness audit by income quintile
│
├── data/
│   ├── raw/                         # Source data (gitignored)
│   └── processed/                   # Model-ready parquet files (gitignored)
│
├── models/                          # Saved model artifacts (gitignored)
├── reports/
│   ├── audit_report.md
│   ├── model_comparison.md
│   └── figures/                     # SHAP plots, ROC curves, bias charts
│
├── tests/
│   ├── conftest.py                  # Shared in-memory fixtures
│   ├── test_phase1_etl.py           # 20 ETL tests
│   ├── test_phase2_cv.py            # 16 CV tests
│   ├── test_phase3_modeling.py      # 20 modelling tests
│   └── test_phase4_deploy.py        # 17 deployment + bias tests
│
└── notebooks/
    └── 01_data_exploration.ipynb
```

---

## Data Sources

| Source | What it provides | Access |
|--------|-----------------|--------|
| OpenStreetMap (Overpass API) | Building footprints, hospitals, roads, drainage | Free, no key required |
| NOAA GSOD | Daily temp, precipitation, wind — Wilson / JKIA stations | Free |
| Sentinel-2 via GEE | 10 m multispectral imagery → NDVI, urban index | Free (GEE account) |
| HDX / KNBS | Population density, poverty indices by sub-county | Free download |
| Kenya Met Department | Supplementary rainfall and flood records | Public PDFs |

---

## Risk Score

Each hexagonal zone is scored on four indicators:

| Indicator | Weight | Rationale |
|-----------|--------|-----------|
| Building density | 30% | More buildings → more surface runoff |
| Drainage coverage | 25% | Less drainage → higher flood risk |
| Vegetation (NDVI) | 25% | Green areas absorb water; bare land floods |
| Poverty index | 20% | Lower resilience, informal housing |

---

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements-py314.txt
```

**2. Run the pipeline in order**
```bash
# Phase 1 — build the city grid
py -m src.phase1_etl.pipeline

# Phase 3 — generate features and train the model
py src/phase3_modeling/feature_eng.py
py src/phase3_modeling/train_xgboost.py

# Phase 4 — launch the dashboard
py -m streamlit run src/phase4_deploy/app.py
```

Open `http://localhost:8501` in your browser.

---

## Dashboard Features

- **6 map layers** — Risk Probability, Poverty Index, Flood Risk, Heat Vulnerability, Building Density, NDVI
- **Risk threshold slider** — flag zones above any chosen probability
- **Hover tooltips** — per-zone risk score, poverty index, flood %, NDVI
- **Charts** — risk distribution histogram, risk vs building density scatter plot
- **Executive summary** — auto-generated policy brief

---

## Running the Tests

```bash
py -m pytest
```

73 tests across all four phases. To run a single phase:

```bash
py -m pytest tests/test_phase1_etl.py
py -m pytest tests/test_phase2_cv.py
py -m pytest tests/test_phase3_modeling.py
py -m pytest tests/test_phase4_deploy.py
```

---

## Bias Audit

```bash
py src/phase4_deploy/bias_audit.py
```

Checks model fairness across five income quintiles. Output saved to `reports/audit_report.md`. A disparity warning is raised if the false-positive rate difference between the richest and poorest quintiles exceeds 10%.

---

## Model Evaluation

```bash
py src/phase3_modeling/evaluate.py --model both
```

Generates ROC curves, precision-recall curves, calibration plots, confusion matrix, spatial error map, and a model comparison report to `reports/`.

---

## Demo

```bash
py demo_ngrok.py
```

Starts the Streamlit dashboard and optionally opens a public URL via ngrok for stakeholder presentations (requires `NGROK_AUTH_TOKEN` in `.env`).
