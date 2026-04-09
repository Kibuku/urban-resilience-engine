"""
Shared pytest fixtures for the Urban Resilience Engine test suite.

All fixtures use in-memory / temporary data so the tests run without
network access, GEE authentication, or pre-existing pipeline outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from pathlib import Path
from shapely.geometry import Point, Polygon


# ── Tiny Nairobi bounding box (subset — keeps tests fast) ────────────

BBOX = {"south": -1.31, "north": -1.29, "west": 36.81, "east": 36.83}
CENTER = (-1.30, 36.82)


# ── Minimal H3 hex grid ───────────────────────────────────────────────

@pytest.fixture(scope="session")
def tiny_hex_grid() -> gpd.GeoDataFrame:
    """Four manually defined hexagon-like polygons for spatial tests."""
    hexes = []
    for i, (lat, lon) in enumerate([
        (-1.30, 36.815), (-1.30, 36.825),
        (-1.295, 36.815), (-1.295, 36.825),
    ]):
        d = 0.004
        poly = Polygon([
            (lon - d, lat), (lon - d/2, lat + d),
            (lon + d/2, lat + d), (lon + d, lat),
            (lon + d/2, lat - d), (lon - d/2, lat - d),
        ])
        hexes.append({"h3_id": f"hex_{i:03d}", "geometry": poly})
    return gpd.GeoDataFrame(hexes, crs="EPSG:4326")


# ── Synthetic urban indicator points ─────────────────────────────────

@pytest.fixture(scope="session")
def urban_points() -> gpd.GeoDataFrame:
    """100 synthetic urban indicator points inside BBOX."""
    rng = np.random.default_rng(42)
    n = 100
    lats = rng.uniform(BBOX["south"], BBOX["north"], n)
    lons = rng.uniform(BBOX["west"],  BBOX["east"],  n)
    df = pd.DataFrame({
        "pop_density_per_km2":     rng.uniform(1000, 20000, n),
        "building_density_per_km2": rng.uniform(500,  8000,  n),
        "road_density_km_per_km2":  rng.uniform(10,   100,   n),
        "hospital_access_within_5km": rng.uniform(0.5, 15,   n),
        "flood_risk_percent":       rng.uniform(0,    80,    n),
        "heat_vulnerability_index": rng.uniform(30,   90,    n),
        "poverty_rate_percent":     rng.uniform(5,    70,    n),
        "median_income_kes":        rng.uniform(8000, 80000, n),
        "green_space_percent":      rng.uniform(0,    50,    n),
        "air_quality_index":        rng.uniform(20,   80,    n),
        "distance_from_cbd_km":     rng.uniform(0.5,  20,    n),
    })
    geoms = [Point(lon, lat) for lat, lon in zip(lats, lons)]
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")


# ── Synthetic OSM point/line features ────────────────────────────────

@pytest.fixture(scope="session")
def osm_buildings() -> gpd.GeoDataFrame:
    rng = np.random.default_rng(42)
    lats = rng.uniform(BBOX["south"], BBOX["north"], 40)
    lons = rng.uniform(BBOX["west"],  BBOX["east"],  40)
    geoms = [Point(lon, lat) for lat, lon in zip(lats, lons)]
    return gpd.GeoDataFrame({"geometry": geoms}, crs="EPSG:4326")


@pytest.fixture(scope="session")
def osm_hospitals() -> gpd.GeoDataFrame:
    rng = np.random.default_rng(7)
    lats = rng.uniform(BBOX["south"], BBOX["north"], 5)
    lons = rng.uniform(BBOX["west"],  BBOX["east"],  5)
    geoms = [Point(lon, lat) for lat, lon in zip(lats, lons)]
    return gpd.GeoDataFrame({"geometry": geoms, "name": [f"H{i}" for i in range(5)]},
                            crs="EPSG:4326")


# ── Minimal grid with all Phase-3-expected columns ───────────────────

@pytest.fixture(scope="session")
def full_grid(tiny_hex_grid) -> gpd.GeoDataFrame:
    """
    Tiny_hex_grid augmented with the columns that feature_eng.py expects,
    mirroring what clean_merge + fetch_sentinel produce.
    """
    rng = np.random.default_rng(42)
    n = len(tiny_hex_grid)
    g = tiny_hex_grid.copy()
    g["building_count"]          = rng.integers(0, 50, n).astype(float)
    g["hospital_count"]          = rng.integers(0,  5, n).astype(float)
    g["road_length_m"]           = rng.uniform(0, 5000, n)
    g["drainage_count"]          = rng.integers(0, 10, n).astype(float)
    g["poverty_rate_percent"]    = rng.uniform(5, 70, n)
    g["poverty_index"]           = (g["poverty_rate_percent"] - g["poverty_rate_percent"].min()) / \
                                    (g["poverty_rate_percent"].max() - g["poverty_rate_percent"].min() + 1e-9)
    g["ndvi_wet"]                = rng.uniform(0.1, 0.8, n)
    g["ndvi_dry"]                = rng.uniform(0.05, 0.6, n)
    g["ndvi_change"]             = g["ndvi_wet"] - g["ndvi_dry"]
    g["vegetation_loss"]         = (g["ndvi_change"] < -0.1).astype(int)
    g["hex_lat"]                 = g.geometry.centroid.y
    g["hex_lon"]                 = g.geometry.centroid.x
    return g


# ── Feature matrix + target ───────────────────────────────────────────

@pytest.fixture(scope="session")
def model_data(full_grid):
    """X, y pair — no file I/O."""
    from src.phase3_modeling.feature_eng import create_risk_target, prepare_model_data
    grid_with_target = create_risk_target(full_grid)
    X, y = prepare_model_data(grid_with_target)
    return X, y
