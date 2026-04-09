"""
Phase 1 — Clean, merge, and spatially join all datasets onto an H3 hexagonal grid.

This is the core Data Fusion step: making weather, OSM, census, and
(later) CV features speak the same spatial language via H3 hexagons.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import h3
from shapely.geometry import Polygon
from tqdm import tqdm
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import (
    NAIROBI_BBOX, NAIROBI_CENTER, H3_RESOLUTION,
    DATA_RAW, DATA_INTERIM, DATA_PROCESSED, CRS
)


# ── H3 Grid Generation ─────────────────────────────────────────────

def create_h3_grid() -> gpd.GeoDataFrame:
    """Generate H3 hexagonal grid covering Nairobi bounding box."""
    print("⏳ Generating H3 hex grid (resolution {})...".format(H3_RESOLUTION))

    # Get all H3 cells that cover the bounding box
    bbox_polygon = Polygon([
        (NAIROBI_BBOX["west"], NAIROBI_BBOX["south"]),
        (NAIROBI_BBOX["east"], NAIROBI_BBOX["south"]),
        (NAIROBI_BBOX["east"], NAIROBI_BBOX["north"]),
        (NAIROBI_BBOX["west"], NAIROBI_BBOX["north"]),
    ])

    # h3 v4 API: geo_to_cells accepts a GeoJSON dict
    geojson = {
        "type": "Polygon",
        "coordinates": [[
            [NAIROBI_BBOX["west"], NAIROBI_BBOX["south"]],
            [NAIROBI_BBOX["east"], NAIROBI_BBOX["south"]],
            [NAIROBI_BBOX["east"], NAIROBI_BBOX["north"]],
            [NAIROBI_BBOX["west"], NAIROBI_BBOX["north"]],
            [NAIROBI_BBOX["west"], NAIROBI_BBOX["south"]],
        ]]
    }
    hex_ids = h3.geo_to_cells(geojson, H3_RESOLUTION)

    # Build GeoDataFrame
    # cell_to_boundary returns [(lat, lon), ...] — swap to (lon, lat) for Shapely
    rows = []
    for hid in hex_ids:
        boundary = h3.cell_to_boundary(hid)          # [(lat, lon), ...]
        coords   = [(lon, lat) for lat, lon in boundary]
        rows.append({"h3_id": hid, "geometry": Polygon(coords)})

    gdf = gpd.GeoDataFrame(rows, crs=CRS)
    print(f"  ✅ {len(gdf)} hexagons generated")
    return gdf


# ── Spatial Aggregation Functions ───────────────────────────────────

def count_features_per_hex(hex_gdf: gpd.GeoDataFrame,
                           feature_gdf: gpd.GeoDataFrame,
                           col_name: str) -> gpd.GeoDataFrame:
    """Count how many point/polygon features fall within each hex cell."""
    # Ensure same CRS
    feature_gdf = feature_gdf.to_crs(CRS)
    # Represent each feature by its centroid for spatial join
    feat_pts = feature_gdf.copy()
    feat_pts["geometry"] = feat_pts.geometry.centroid

    joined = gpd.sjoin(feat_pts, hex_gdf[["h3_id", "geometry"]], how="inner", predicate="within")
    counts = joined.groupby("h3_id").size().rename(col_name)

    return hex_gdf.merge(counts, on="h3_id", how="left").fillna({col_name: 0})


def road_density_per_hex(hex_gdf: gpd.GeoDataFrame,
                         roads_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute total road length (m) within each hex cell."""
    roads_gdf = roads_gdf.to_crs(CRS)
    joined = gpd.sjoin(roads_gdf, hex_gdf[["h3_id", "geometry"]], how="inner", predicate="intersects")
    density = joined.groupby("h3_id")["length"].sum().rename("road_length_m")
    return hex_gdf.merge(density, on="h3_id", how="left").fillna({"road_length_m": 0})


# ── Main Merge Pipeline ────────────────────────────────────────────

def _aggregate_urban_indicators(grid: gpd.GeoDataFrame,
                                urban_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Aggregate urban indicator point data onto the H3 grid via spatial join."""
    print(f"  📊 Aggregating {len(urban_data)} urban indicator points onto {len(grid)} hexagons...")
    urban_data = urban_data.to_crs(CRS)

    aggregated_features = []
    URBAN_COLS = [
        "pop_density_per_km2", "building_density_per_km2", "road_density_km_per_km2",
        "hospital_access_within_5km", "flood_risk_percent", "heat_vulnerability_index",
        "poverty_rate_percent", "median_income_kes", "green_space_percent",
        "air_quality_index",
    ]

    for _, hex_row in tqdm(grid.iterrows(), total=len(grid), desc="Hex aggregation"):
        hex_id = hex_row.h3_id
        pts = urban_data[urban_data.geometry.within(hex_row.geometry)]

        if len(pts) > 0:
            feat = {"h3_id": hex_id, "n_urban_points": len(pts)}
            for col in URBAN_COLS:
                if col in pts.columns:
                    feat[col] = pts[col].mean()
            if "distance_from_cbd_km" in pts.columns:
                feat["avg_distance_from_cbd_km"] = pts["distance_from_cbd_km"].mean()
        else:
            feat = {"h3_id": hex_id, "n_urban_points": 0}
            for col in URBAN_COLS + ["avg_distance_from_cbd_km"]:
                feat[col] = np.nan

        aggregated_features.append(feat)

    features_df = pd.DataFrame(aggregated_features)
    grid = grid.merge(features_df, on="h3_id", how="left")

    # Fill NaNs with column medians
    fill_cols = URBAN_COLS + ["avg_distance_from_cbd_km"]
    for col in fill_cols:
        if col in grid.columns and grid[col].isna().any():
            grid[col] = grid[col].fillna(grid[col].median())

    return grid


def build_grid_dataset(urban_data: gpd.GeoDataFrame):
    """
    Merge urban indicators + OSM infrastructure features onto the H3 grid.
    Produces columns consumed by Phase 3: building_count, hospital_count,
    road_length_m, drainage_count, poverty_index.
    """
    from .fetch_osm import fetch_buildings, fetch_hospitals, fetch_roads, fetch_drainage

    grid = create_h3_grid()

    # ── Step 1: Urban indicators ──────────────────────────────────
    grid = _aggregate_urban_indicators(grid, urban_data)

    # ── Step 2: OSM infrastructure counts ────────────────────────
    print("  🏗️  Counting OSM infrastructure per hexagon...")
    buildings  = fetch_buildings()
    grid = count_features_per_hex(grid, buildings, "building_count")

    hospitals  = fetch_hospitals()
    grid = count_features_per_hex(grid, hospitals, "hospital_count")

    roads      = fetch_roads()
    grid = road_density_per_hex(grid, roads)

    drainage   = fetch_drainage()
    grid = count_features_per_hex(grid, drainage, "drainage_count")

    # ── Step 3: Derived / aliased columns ────────────────────────
    # poverty_index: normalised 0-1 version of poverty_rate_percent
    p = grid["poverty_rate_percent"]
    grid["poverty_index"] = ((p - p.min()) / (p.max() - p.min() + 1e-9)).round(4)

    # ── Step 4: Coordinates ───────────────────────────────────────
    grid["hex_lat"] = grid.geometry.centroid.y
    grid["hex_lon"] = grid.geometry.centroid.x

    # Save
    out = DATA_PROCESSED / "nairobi_grid.parquet"
    grid.to_parquet(out)
    print(f"\n🏁 Final grid: {len(grid)} hexagons × {len(grid.columns)} features → {out}")
    return grid


if __name__ == "__main__":
    from fetch_urban_data import fetch_nairobi_urban_dataset
    build_grid_dataset(fetch_nairobi_urban_dataset())
