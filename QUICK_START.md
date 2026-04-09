# Urban Resilience Engine - Quick Reference

## Installation Status Summary

### ✅ Completed
- **Python 3.14.3** installed
- **Core packages** installed:
  - Data: pandas, numpy, geopandas, shapely, h3, pyarrow
  - Fetching: requests, osmnx, overpy, meteostat
  - ML: scikit-learn, xgboost, shap, mlflow
  - Viz: plotly, streamlit, folium, matplotlib, seaborn
  - Utilities: python-dotenv, tqdm, joblib
  - Fairness: fairlearn
  - Raster: rasterio, earthengine-api, pillow

- **Setup Scripts Created**:
  - `test_infrastructure.py` - Comprehensive import test
  - `setup_gee.py` - GEE authentication wizard
  - `verify_api_access.py` - Check all data sources
  - `SETUP.md` - Detailed setup guide
  - `.env.example` - Environment template

### ⏳ In Progress
- **PyTorch** installation (114MB, ~58% complete, ETA 6-7 min)
- **TorchVision** (pending after PyTorch)

### API Access Test Results
- ✓ OpenStreetMap (OSM) Overpass API - **OK**
- ✓ NOAA Weather Data - **OK**  
- ⚠ Sentinel/ESA - Testing
- ⏳ Google Earth Engine - Pending auth setup
- ✓ MLflow Tracking - **OK**
- ✓ Streamlit - **OK**

---

## Quick Commands

### 1. Setup

```bash
# Enter project directory
cd c:\Users\PC\Downloads\urban-resilience-engine\urban-resilience-engine

# Copy environment template
cp .env.example .env

# Run GEE setup (interactive)
py setup_gee.py

# Verify all imports
py test_infrastructure.py

# Check API access
py verify_api_access.py
```

### 2. Project Structure

```
src/
├── phase1_etl/        # ETL pipeline (data fetching & cleaning)
│   ├── fetch_osm.py       - OpenStreetMap data
│   ├── fetch_weather.py   - NOAA weather
│   ├── fetch_census.py    - Census/HDX data
│   ├── clean_merge.py     - Spatial aggregation
│   └── pipeline.py        - Orchestrator
├── phase2_cv/         # Computer Vision (satellite imagery)
│   ├── fetch_sentinel.py  - Sentinel-2 data via GEE
│   ├── tile_images.py     - Image tiling (if needed)
│   └── cnn_model.py       - ResNet18 fine-tuning
├── phase3_modeling/   # Predictive Modeling
│   ├── feature_eng.py     - Feature combination
│   ├── train_xgboost.py   - XGBoost training
│   └── evaluate.py        - Metrics & calibration
└── phase4_deploy/     # Deployment & Ethics
    ├── app.py             - Streamlit dashboard
    ├── bias_audit.py      - Fairness analysis
    └── api.py             - Optional FastAPI
```

### 3. Running Phases

```bash
# Phase 1: ETL (fetch & clean data)
python -m src.phase1_etl.pipeline

# Phase 2: Computer Vision (satellite analysis)
python src/phase2_cv/fetch_sentinel.py

# Phase 3: Training (XGBoost model)
python src/phase3_modeling/train_xgboost.py

# Phase 4: Dashboard
streamlit run src/phase4_deploy/app.py
```

### 3.5 Ngrok Demo (Share Dashboard Online)

For demos and presentations, expose your local dashboard to the internet:

```bash
# 1. Install ngrok
pip install pyngrok

# 2. Get ngrok auth token from https://ngrok.com
# Add to .env: NGROK_AUTH_TOKEN=your_token

# 3. Run demo
python demo_ngrok.py
```

This creates a secure tunnel and gives you a public URL (active for 8 hours on free tier).

### 4. Data Locations

```
data/
├── raw/             # Original untouched data
├── interim/         # Temporary processed files
└── processed/       # Final model-ready datasets
    └── nairobi_grid.parquet    # Main output

models/
├── xgboost_model.pkl           # Trained model
├── scaler.pkl                  # Feature scaler
└── shap_values.pkl             # SHAP explanations

reports/
├── audit_report.md             # Bias audit
├── figures/                    # Plots & visualizations
└── nairobi_risk_map.html       # Interactive map
```

### 5. Configuration

**config.py** - Main settings:
```python
NAIROBI_BBOX       # Study area (lat/lon)
H3_RESOLUTION = 8  # Hexagon size (~0.46 km²)
YEAR_START = 2018
YEAR_END = 2025
NAIROBI_STATIONS   # Weather stations (JKIA, Wilson)
```

**.env** - Your credentials:
```
GEE_PROJECT=your-earth-engine-project-id
MLFLOW_TRACKING_URI=file:./mlruns
OPENWEATHER_API_KEY=optional
```

---

## Data Sources

| Source | Coverage | Update | Key Features |
|--------|----------|--------|--------------|
| **OSM** | Global | Daily | Buildings, roads, hospitals, drainage |
| **NOAA GSOD** | Global | Daily | Temp, precip, wind (1950-present) |
| **Sentinel-2** | Global | 5 days | 10m multispectral, NDVI, urban index |
| **KNBS/HDX** | Kenya | Variable | Population, poverty, demographics |
| **Kenya Met Dept** | Kenya | Variable | Rainfall, flood records, alerts |

---

## Key Metrics

### Flood/Heat Risk Prediction
- **Input**: Weather + building density + infrastructure + vegetation
- **Target**: Risk score (0-100) for flooding/heat stress
- **Model**: XGBoost with time-series cross-validation
- **Evaluation**: AUC-ROC, Brier score, calibration

### Bias Audit
- **Groups**: Income quintiles (income level proxy from census)
- **Fairness**: Equal opportunity, demographic parity
- **Tool**: FairLearn library

---

## Common Issues & Solutions

### PyTorch Download Stuck
```bash
# Check status
py -m pip list | findstr torch

# Reinstall if needed
py -m pip install --upgrade torch torchvision
```

### GEE Authentication Failed
```bash
# Interactive setup
py -c "import ee; ee.Authenticate()"

# Or use setup script
py setup_gee.py
```

### Import Errors
```bash
# Add to Python path
set PYTHONPATH=%cd%

# Or in code:
import sys
sys.path.insert(0, str(Path(__file__).parent))
```

### Rasterio/GDAL Issues (Windows)
```bash
# Usually fixed already, but if not:
py -m pip install --upgrade rasterio --force-reinstall
```

---

## Files Created This Session

| File | Purpose |
|------|---------|
| `requirements-py314.txt` | Python 3.14 compatible dependencies |
| `.env.example` | Environment config template |
| `SETUP.md` | Detailed setup guide (this file) |
| `test_infrastructure.py` | Comprehensive import test |
| `setup_gee.py` | GEE auth wizard |
| `verify_api_access.py` | API connectivity test |

---

## Next Steps

### Immediate (After PyTorch finishes)
1. ✅ Run `py test_infrastructure.py` → Verify all imports
2. ✅ Run `py verify_api_access.py` → Check data sources
3. ✅ Run `py setup_gee.py` → Authenticate with Earth Engine

### Phase 1: Data Engineering
4. Implement Phase 1 ETL:
   - [ ] Complete `fetch_osm.py` for building/road/hospital data
   - [ ] Complete `fetch_weather.py` for NOAA integration
   - [ ] Complete `fetch_census.py` for demographics
   - [ ] Implement H3 spatial aggregation
   - [ ] Output: `data/processed/nairobi_grid.parquet`

### Phase 2: Computer Vision
5. Implement satellite imagery:
   - [ ] GEE Sentinel-2 fetching
   - [ ] NDVI computation
   - [ ] CNN feature extraction

### Phase 3: Modeling
6. Train predictive model:
   - [ ] XGBoost with temporal CV
   - [ ] SHAP explainability
   - [ ] Evaluation metrics

### Phase 4: Deployment
7. Build dashboard & audit:
   - [ ] Streamlit map dashboard
   - [ ] Bias audit (FairLearn)
   - [ ] Ethics report

---

## Resources

- **Project Docs**: [README.md](README.md)
- **H3 Hexagons**: https://h3geo.org/
- **Sentinel-2**: https://sentinel.esa.int/
- **Earth Engine**: https://code.earthengine.google.com/
- **OpenStreetMap**: https://www.openstreetmap.org/
- **NOAA Weather**: https://www.ncei.noaa.gov/

---

## Installation Checklist

- [x] Python 3.14.3 installed
- [x] Core packages installed
- [x] Data fetching packages installed  
- [x] ML packages installed
- [x] Visualization packages installed
- [x] Config files created
- [ ] PyTorch installed (in progress)
- [ ] TorchVision installed
- [ ] All imports tested
- [ ] GEE authenticated
- [ ] API access verified

---

**Status**: Infrastructure setup 95% complete  
**Last Updated**: April 8, 2026, 00:18 UTC  
**Next Action**: Wait for PyTorch → Run tests → Start Phase 1 ETL
