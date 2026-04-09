"""
Phase 1 ETL tests — Data Engineering Pipeline.

What we test
------------
1. H3 grid generation produces non-empty output with required columns.
2. Urban indicator points aggregation lands in the right hexagons.
3. OSM mock data is generated with correct geometry types.
4. `count_features_per_hex` produces non-negative integer counts.
5. `road_density_per_hex` produces non-negative totals.
6. Final grid always contains the Phase-3-expected columns.
7. NaN handling — no NaNs in critical columns after the merge.
8. `poverty_index` is strictly in [0, 1].
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from src.phase1_etl.clean_merge import (
    create_h3_grid,
    count_features_per_hex,
    road_density_per_hex,
    _aggregate_urban_indicators,
)
from src.phase1_etl.fetch_osm import (
    fetch_buildings, fetch_hospitals, fetch_roads, fetch_drainage,
)
from src.phase1_etl.fetch_urban_data import fetch_nairobi_urban_dataset


# ─────────────────────────────────────────────────────────────────────
# 1. H3 Grid
# ─────────────────────────────────────────────────────────────────────

class TestH3Grid:
    def test_grid_is_not_empty(self):
        grid = create_h3_grid()
        assert len(grid) > 0, "H3 grid must contain at least one hexagon"

    def test_grid_has_required_columns(self):
        grid = create_h3_grid()
        assert "h3_id" in grid.columns
        assert "geometry" in grid.columns

    def test_all_h3_ids_are_unique(self):
        grid = create_h3_grid()
        assert grid["h3_id"].nunique() == len(grid), "Duplicate H3 IDs found"

    def test_geometries_are_polygons(self):
        from shapely.geometry import Polygon, MultiPolygon
        grid = create_h3_grid()
        types = grid.geometry.geom_type.unique()
        assert set(types).issubset({"Polygon", "MultiPolygon"}), \
            f"Unexpected geometry types: {types}"

    def test_grid_crs_is_wgs84(self):
        grid = create_h3_grid()
        assert grid.crs.to_epsg() == 4326


# ─────────────────────────────────────────────────────────────────────
# 2. Urban Indicator Aggregation
# ─────────────────────────────────────────────────────────────────────

class TestUrbanAggregation:
    def test_aggregation_preserves_hex_count(self, tiny_hex_grid, urban_points):
        result = _aggregate_urban_indicators(tiny_hex_grid.copy(), urban_points)
        assert len(result) == len(tiny_hex_grid)

    def test_aggregation_adds_urban_columns(self, tiny_hex_grid, urban_points):
        result = _aggregate_urban_indicators(tiny_hex_grid.copy(), urban_points)
        expected = ["pop_density_per_km2", "poverty_rate_percent", "n_urban_points"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nans_in_critical_columns(self, tiny_hex_grid, urban_points):
        """After fill-with-median, no NaN should remain in urban indicator cols."""
        result = _aggregate_urban_indicators(tiny_hex_grid.copy(), urban_points)
        cols = ["pop_density_per_km2", "poverty_rate_percent"]
        for col in cols:
            if col in result.columns:
                assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_n_urban_points_non_negative(self, tiny_hex_grid, urban_points):
        result = _aggregate_urban_indicators(tiny_hex_grid.copy(), urban_points)
        assert (result["n_urban_points"] >= 0).all()


# ─────────────────────────────────────────────────────────────────────
# 3. OSM Mock Data
# ─────────────────────────────────────────────────────────────────────

class TestOSMMockData:
    def test_buildings_returns_geodataframe(self):
        gdf = fetch_buildings()
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0

    def test_buildings_has_point_geometry(self):
        gdf = fetch_buildings()
        assert gdf.geometry.geom_type.isin(["Point"]).all()

    def test_hospitals_has_name_column(self):
        gdf = fetch_hospitals()
        assert "name" in gdf.columns

    def test_roads_has_length_column(self):
        gdf = fetch_roads()
        assert "length" in gdf.columns
        assert (gdf["length"] > 0).all()

    def test_drainage_returns_non_empty(self):
        gdf = fetch_drainage()
        assert len(gdf) > 0

    def test_all_osm_crs_is_wgs84(self):
        for fn in (fetch_buildings, fetch_hospitals, fetch_roads, fetch_drainage):
            gdf = fn()
            assert gdf.crs.to_epsg() == 4326, f"{fn.__name__} CRS mismatch"


# ─────────────────────────────────────────────────────────────────────
# 4. count_features_per_hex
# ─────────────────────────────────────────────────────────────────────

class TestCountFeaturesPerHex:
    def test_output_has_count_column(self, tiny_hex_grid, osm_buildings):
        result = count_features_per_hex(tiny_hex_grid.copy(), osm_buildings, "building_count")
        assert "building_count" in result.columns

    def test_counts_are_non_negative(self, tiny_hex_grid, osm_buildings):
        result = count_features_per_hex(tiny_hex_grid.copy(), osm_buildings, "building_count")
        assert (result["building_count"] >= 0).all()

    def test_total_count_does_not_exceed_input(self, tiny_hex_grid, osm_buildings):
        result = count_features_per_hex(tiny_hex_grid.copy(), osm_buildings, "building_count")
        # Due to spatial join, total ≤ number of input features (features outside bbox dropped)
        assert result["building_count"].sum() <= len(osm_buildings)

    def test_no_nans_after_count(self, tiny_hex_grid, osm_buildings):
        result = count_features_per_hex(tiny_hex_grid.copy(), osm_buildings, "building_count")
        assert result["building_count"].isna().sum() == 0

    def test_hospital_count_column(self, tiny_hex_grid, osm_hospitals):
        result = count_features_per_hex(tiny_hex_grid.copy(), osm_hospitals, "hospital_count")
        assert "hospital_count" in result.columns
        assert (result["hospital_count"] >= 0).all()


# ─────────────────────────────────────────────────────────────────────
# 5. road_density_per_hex
# ─────────────────────────────────────────────────────────────────────

class TestRoadDensityPerHex:
    def test_output_has_road_length_column(self, tiny_hex_grid):
        roads = fetch_roads()
        result = road_density_per_hex(tiny_hex_grid.copy(), roads)
        assert "road_length_m" in result.columns

    def test_road_lengths_non_negative(self, tiny_hex_grid):
        roads = fetch_roads()
        result = road_density_per_hex(tiny_hex_grid.copy(), roads)
        assert (result["road_length_m"] >= 0).all()


# ─────────────────────────────────────────────────────────────────────
# 6. Derived columns on final grid
# ─────────────────────────────────────────────────────────────────────

class TestDerivedColumns:
    def test_poverty_index_in_unit_range(self, full_grid):
        assert full_grid["poverty_index"].between(0, 1).all(), \
            "poverty_index must be in [0, 1]"

    def test_full_grid_has_phase3_columns(self, full_grid):
        required = ["building_count", "hospital_count", "road_length_m",
                    "drainage_count", "poverty_index", "hex_lat", "hex_lon"]
        for col in required:
            assert col in full_grid.columns, f"Missing required column: {col}"

    def test_hex_coords_are_in_nairobi(self, full_grid):
        """Centroids should be within the Nairobi BBOX (roughly)."""
        assert full_grid["hex_lat"].between(-1.5, -1.0).all()
        assert full_grid["hex_lon"].between(36.6, 37.2).all()


# ─────────────────────────────────────────────────────────────────────
# 7. fetch_nairobi_urban_dataset
# ─────────────────────────────────────────────────────────────────────

class TestFetchUrbanDataset:
    def test_returns_geodataframe(self):
        gdf = fetch_nairobi_urban_dataset()
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_expected_columns_present(self):
        gdf = fetch_nairobi_urban_dataset()
        expected = [
            "pop_density_per_km2", "building_density_per_km2",
            "poverty_rate_percent", "flood_risk_percent",
        ]
        for col in expected:
            assert col in gdf.columns

    def test_no_null_geometry(self):
        gdf = fetch_nairobi_urban_dataset()
        assert gdf.geometry.isna().sum() == 0

    def test_all_points_within_nairobi_bbox(self):
        from config import NAIROBI_BBOX
        gdf = fetch_nairobi_urban_dataset()
        lats = gdf.geometry.y
        lons = gdf.geometry.x
        assert lats.between(NAIROBI_BBOX["south"], NAIROBI_BBOX["north"]).all()
        assert lons.between(NAIROBI_BBOX["west"],  NAIROBI_BBOX["east"]).all()
