# Urban Resilience Engine - Setup & Infrastructure Guide

## Quick Setup Summary

✅ **Environment**: Python 3.14.3  
✅ **Status**: Dependencies installed in batches  
⏳ **In Progress**: PyTorch installation (large download)

---

## 1. Installation Status

### ✓ Installed Packages

**Core Data Processing:**
- pandas, numpy, geopandas, shapely, h3, pyarrow ✓

**Data Fetching:**
- requests, osmnx, overpy ✓

**Weather:**
- meteostat ✓

**Modeling:**
- scikit-learn, xgboost, shap, mlflow ✓

**Visualization & Dashboard:**
- plotly, streamlit, folium, streamlit-folium, matplotlib, seaborn ✓

**Utilities:**
- python-dotenv, tqdm, joblib ✓

**Ethics & Fairness:**
- fairlearn ✓

**Geospatial Raster:**
- rasterio, earthengine-api, pillow ✓

### ⏳ In Progress

**Computer Vision:**
- torch (PyTorch) - downloading (~114MB)
- torchvision - pending after torch

---

## 2. Running Tests

### Quick Import Test
```bash
cd c:\Users\PC\Downloads\urban-resilience-engine\urban-resilience-engine
py test_infrastructure.py
```

### Check Specific Modules
```bash
py -c "import pandas; import geopandas; import xgboost; print('✓ Core packages OK')"
```

---

## 3. API & Authentication Setup

### Google Earth Engine (GEE)

**Status**: earthengine-api installed, authentication pending

**Setup Steps:**

1. **Create GEE Account**: https://code.earthengine.google.com/
2. **Create Google Cloud Project**: https://console.cloud.google.com/
3. **Enable Earth Engine API**
4. **Authenticate Locally**:
   ```bash
   py -c "import ee; ee.Authenticate()"
   ```
   This will open a browser window. Authorize and complete the flow.

5. **Store Project ID in `.env`**:
   ```env
   GEE_PROJECT=your-project-id-here
   ```

### Environment Variables (.env)

1. **Copy template**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env`** with your credentials:
   - `GEE_PROJECT`: Your Earth Engine project ID
   - `MLFLOW_TRACKING_URI`: MLflow server or `file:./mlruns`
   - Other optional API keys

---

## 4. Project Structure Verification

All phase modules are properly structured:

```
✓ config.py                 - Shared configuration
✓ src/phase1_etl/          - Data fetching & cleaning
✓ src/phase2_cv/           - Satellite imagery & CNN
✓ src/phase3_modeling/     - Feature eng & XGBoost
✓ src/phase4_deploy/       - Dashboard & bias audit
✓ notebooks/               - Jupyter analysis notebooks
✓ data/raw,interim,processed/ - Data pipeline dirs
✓ models/                  - Model artifacts
✓ reports/                 - Output reports
```

---

## 5. Next Steps

### Phase 1: ETL Pipeline
- ✅ Config + imports ready
- 🔲 Implement `phase1_etl/pipeline.py` - orchestrate all data fetching
- 🔲 Add OSM data fetching (buildings, roads, hospitals)
- 🔲 Add NOAA weather data integration
- 🔲 Add census/HDX data integration
- 🔲 Implement spatial grid aggregation (H3)
- 🔲 Output: `data/processed/nairobi_grid.parquet`

### Phase 2: Computer Vision
- 🔲 Authenticate and test GEE access
- 🔲 Fetch Sentinel-2 composites
- 🔲 Compute NDVI and urban indices
- 🔲 Fine-tune ResNet18 CNN
- 🔲 Generate tile features

### Phase 3: Predictive Modeling
- 🔲 Feature engineering (combine tabular + CV features)
- 🔲 Train XGBoost with time-series CV
- 🔲 SHAP explainability analysis
- 🔲 Model evaluation & calibration

### Phase 4: Deployment
- 🔲 Build Streamlit dashboard
- 🔲 Implement bias audit (FairLearn)
- 🔲 Generate ethics report
- 🔲 Optional: FastAPI endpoint

---

## 6. Installation Requirements

**System Requirements:**
- Python 3.14.3 ✓
- ~5GB disk space (for models, satellite data)
- Internet connection (for API access)

**Python Packages:**
- 40+ packages installed across all phases
- Total disk: ~3-4GB (incl. PyTorch)

---

## 7. Troubleshooting

### Module Not Found: `torch`
- PyTorch download is still ongoing. Give it 15-30 minutes to complete.
- Check status: `py -m pip list | findstr torch`
- If stuck, reinstall: `py -m pip install --upgrade torch torchvision`

### GEE Authentication Error
- Run: `py -c "import ee; ee.Authenticate()"`
- Check browser opens for login
- Store credentials in `.env` with correct project ID

### Import Errors for Phase Modules
- Add `PYTHONPATH` environment variable:
  ```bash
  set PYTHONPATH=%cd%
  ```
- Or add to project code:
  ```python
  import sys
  sys.path.insert(0, str(Path(__file__).parent))
  ```

### Rasterio/GDAL Issues on Windows
- Usually already fixed with binary wheels
- If issues persist: `py -m pip install --upgrade rasterio`

---

## 8. Useful Commands

**Verify installation:**
```bash
py -m pip list
```

**Update all packages:**
```bash
py -m pip install --upgrade -r requirements-py314.txt
```

**Run specific phase:**
```bash
cd src
py -m phase1_etl.pipeline
```

**Start Streamlit dashboard (later):**
```bash
streamlit run src/phase4_deploy/app.py
```

**Check Python version:**
```bash
py --version
```

---

## 9. Configuration Files

| File | Purpose |
|------|---------|
| `config.py` | Shared paths, bbox, H3 resolution, API stations |
| `.env` | Local credentials (GEE project, API keys) |
| `requirements-py314.txt` | Python 3.14-compatible dependencies |
| `.env.example` | Template for `.env` setup |
| `test_infrastructure.py` | Verify all imports work |

---

## 10. Next Command

Once PyTorch finishes downloading, run:
```bash
py test_infrastructure.py
```

This will verify all 40+ packages and provide a comprehensive setup report.

---

**Last Updated**: April 7, 2026  
**Status**: Infrastructure setup in progress (PyTorch pending)
