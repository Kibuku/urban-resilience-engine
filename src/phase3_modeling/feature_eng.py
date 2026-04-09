"""
Phase 3 — Feature engineering: combine all data sources into model-ready features.
Creates the target variable (flood/heat risk) and final feature matrix.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import DATA_RAW, DATA_PROCESSED, SEED


def load_and_merge_features() -> gpd.GeoDataFrame:
    """Load grid + NDVI + weather features → unified dataset."""
    print("⏳ Phase 3: Building feature matrix...")

    # Base grid (from Phase 1)
    grid = gpd.read_parquet(DATA_PROCESSED / "nairobi_grid.parquet")

    # NDVI features (from Phase 2)
    ndvi_path = DATA_RAW / "ndvi_per_hex.parquet"
    if ndvi_path.exists():
        ndvi = pd.read_parquet(ndvi_path)
        grid = grid.merge(ndvi, on="h3_id", how="left")
        print(f"  ✅ NDVI features merged")
    else:
        print("  ⚠️  No NDVI data — deriving spatially-structured proxy from green_space_percent")
        rng = np.random.default_rng(SEED)
        # green_space_percent is spatially realistic (high in Karen/parks, low in settlements)
        # Scale it to a realistic NDVI range [0.05, 0.85] with small noise
        if "green_space_percent" in grid.columns:
            base = grid["green_space_percent"] / 100.0   # 0–1
            grid["ndvi_wet"]  = (0.05 + 0.80 * base + rng.normal(0, 0.04, len(grid))).clip(0.05, 0.85).round(3)
            grid["ndvi_dry"]  = (grid["ndvi_wet"] - rng.uniform(0.05, 0.20, len(grid))).clip(0.02, 0.75).round(3)
        else:
            grid["ndvi_wet"]  = rng.uniform(0.1, 0.8, len(grid)).round(3)
            grid["ndvi_dry"]  = rng.uniform(0.05, 0.6, len(grid)).round(3)
        grid["ndvi_change"]     = (grid["ndvi_wet"] - grid["ndvi_dry"]).round(3)
        grid["vegetation_loss"] = (grid["ndvi_change"] < -0.1).astype(int)

    return grid


def create_risk_target(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Derive a composite flood/heat risk score as the target variable.

    Methodology:
      - Higher building density + low drainage + low vegetation = higher flood risk
      - Lower NDVI + higher poverty = higher heat vulnerability
      - Combined into a normalised 0-1 risk score, then binary classification at median

    In production, use actual flood event records from Kenya Red Cross / NDMA.
    """
    print("  🎯 Creating target variable (risk score)...")
    df = gdf.copy()

    def norm(s: pd.Series) -> pd.Series:
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    # Building density — prefer raw OSM count, fall back to density proxy
    if "building_count" in df.columns:
        df["norm_buildings"] = norm(df["building_count"])
    elif "building_density_per_km2" in df.columns:
        df["norm_buildings"] = norm(df["building_density_per_km2"])
    else:
        print("  ⚠️  No building column found — using 0.5 placeholder")
        df["norm_buildings"] = 0.5

    # Drainage — inverted so less drainage = higher risk
    if "drainage_count" in df.columns:
        df["norm_drainage"] = 1 - norm(df["drainage_count"])
    else:
        print("  ⚠️  No drainage column found — using 0.5 placeholder")
        df["norm_drainage"] = 0.5

    # Vegetation loss from NDVI
    df["norm_veg_loss"] = norm(1 - df["ndvi_wet"])

    # Poverty — prefer normalised index, fall back to raw percentage
    if "poverty_index" in df.columns:
        df["norm_poverty"] = norm(df["poverty_index"])
    elif "poverty_rate_percent" in df.columns:
        df["norm_poverty"] = norm(df["poverty_rate_percent"])
    else:
        print("  ⚠️  No poverty column found — using 0.5 placeholder")
        df["norm_poverty"] = 0.5

    # Weighted composite risk
    df["risk_score"] = (
        0.30 * df["norm_buildings"] +
        0.25 * df["norm_drainage"]  +
        0.25 * df["norm_veg_loss"]  +
        0.20 * df["norm_poverty"]
    ).round(4)

    # Binary target at median split
    df["risk_class"] = (df["risk_score"] >= df["risk_score"].median()).astype(int)

    print(f"  ✅ Risk score — mean: {df['risk_score'].mean():.3f}, "
          f"high-risk cells: {df['risk_class'].sum()}/{len(df)}")

    return df


def prepare_model_data(gdf: gpd.GeoDataFrame):
    """Split into features (X) and target (y)."""
    FEATURE_COLS = [
        "building_count", "hospital_count", "road_length_m", "drainage_count",
        "poverty_index", "hex_lat", "hex_lon",
        "ndvi_wet", "ndvi_dry", "ndvi_change", "vegetation_loss",
        # Fallback aliases (present when OSM counts are missing)
        "building_density_per_km2", "poverty_rate_percent",
    ]
    # Keep only columns that exist — pipeline is resilient to partial data
    available = [c for c in FEATURE_COLS if c in gdf.columns]
    if not available:
        raise ValueError("No usable feature columns found in grid. Run Phase 1 first.")
    X = gdf[available].copy()
    y = gdf["risk_class"].copy()

    # Fill remaining NaNs
    X = X.fillna(X.median())

    print(f"  ✅ Feature matrix: {X.shape}, Target balance: {y.value_counts().to_dict()}")
    return X, y


if __name__ == "__main__":
    grid = load_and_merge_features()
    grid = create_risk_target(grid)
    X, y = prepare_model_data(grid)

    # Save
    grid.to_parquet(DATA_PROCESSED / "nairobi_grid_full.parquet")
    X.to_parquet(DATA_PROCESSED / "X_features.parquet")
    y.to_frame().to_parquet(DATA_PROCESSED / "y_target.parquet")
    print("  ✅ Saved to data/processed/")
