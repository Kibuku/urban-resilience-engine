"""
Phase 1 — Fetch comprehensive urban indicators dataset for Nairobi.
Uses a single dataset approach with multiple urban resilience features.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import requests
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import NAIROBI_BBOX, DATA_RAW, CRS


def fetch_nairobi_urban_dataset():
    """
    Fetch or create a comprehensive urban indicators dataset for Nairobi.
    Includes: population density, building density, infrastructure, vulnerability factors.

    Returns GeoDataFrame with point locations and multiple urban indicators.
    """
    print("⏳ Fetching Nairobi urban indicators dataset...")

    # For now, use synthetic data (can be replaced with real dataset later)
    return _create_synthetic_dataset()


def _download_real_dataset():
    """
    Try to download a real urban indicators dataset.
    Options: WorldPop, HDX urban indicators, or similar.
    """
    # Try WorldPop population density for Nairobi
    worldpop_url = (
        "https://data.worldpop.org/GIS/Population/Global_2020_2025_1km_UNadj/"
        "ken/ken_ppp_2020_1km_Aggregated_UNadj.tif"
    )

    # For now, let's use a simpler approach - create synthetic but realistic data
    # In production, this would download and process real geospatial data
    raise NotImplementedError("Real dataset integration pending")


def _create_synthetic_dataset():
    """
    Create spatially-realistic synthetic urban indicators for Nairobi.

    Each point's features are derived from its proximity to named geographic
    zones (informal settlements, wealthy suburbs, river corridors, green spaces).
    This produces genuine spatial clustering on the risk map rather than noise.

    Zone sources: OpenStreetMap / WorldPop / KNBS 2019 census estimates.
    """
    print("  📊 Creating spatially-realistic urban indicators dataset...")

    rng = np.random.default_rng(42)
    n_points = 5000

    lats = rng.uniform(NAIROBI_BBOX["south"], NAIROBI_BBOX["north"], n_points)
    lons = rng.uniform(NAIROBI_BBOX["west"],  NAIROBI_BBOX["east"],  n_points)

    # ── Named geographic zones ────────────────────────────────────────
    # Each entry: lat, lon, influence radius (km), zone characteristics
    # poverty ∈ [0,1], flood ∈ [0,100], heat ∈ [0,100], green ∈ [0,100]
    # building_density (per km²), income_kes (monthly), hospital_access (count)

    ZONES = [
        # ── High-risk informal settlements ──────────────────────────
        # Dense, poor, low drainage, flood-prone, near rivers
        dict(name="kibera",    lat=-1.313, lon=36.784, r=2.0,
             poverty=0.80, flood=78, heat=82, green=4,
             bld=7500, income=9000,  hosp=1.0),
        dict(name="mathare",   lat=-1.261, lon=36.864, r=1.5,
             poverty=0.78, flood=85, heat=84, green=3,
             bld=8000, income=8500,  hosp=0.8),
        dict(name="korogocho", lat=-1.247, lon=36.884, r=1.2,
             poverty=0.74, flood=80, heat=81, green=3,
             bld=7200, income=9000,  hosp=0.5),
        dict(name="mukuru",    lat=-1.312, lon=36.864, r=1.5,
             poverty=0.71, flood=72, heat=78, green=4,
             bld=6500, income=10000, hosp=1.0),
        dict(name="dandora",   lat=-1.252, lon=36.894, r=1.0,
             poverty=0.68, flood=65, heat=75, green=5,
             bld=6000, income=11000, hosp=1.0),
        dict(name="kayole",    lat=-1.272, lon=36.894, r=1.2,
             poverty=0.65, flood=60, heat=73, green=5,
             bld=5500, income=12000, hosp=1.2),

        # ── CBD & commercial ────────────────────────────────────────
        dict(name="cbd",       lat=-1.286, lon=36.817, r=1.5,
             poverty=0.22, flood=28, heat=68, green=8,
             bld=9000, income=50000, hosp=5.0),
        dict(name="westlands", lat=-1.264, lon=36.804, r=1.5,
             poverty=0.18, flood=18, heat=55, green=22,
             bld=4500, income=65000, hosp=8.0),
        dict(name="eastleigh", lat=-1.272, lon=36.854, r=1.2,
             poverty=0.45, flood=42, heat=72, green=5,
             bld=7000, income=25000, hosp=3.0),
        dict(name="industrial", lat=-1.309, lon=36.843, r=1.5,
             poverty=0.40, flood=38, heat=74, green=6,
             bld=3000, income=20000, hosp=1.5),

        # ── Wealthy / low-risk suburbs ───────────────────────────────
        dict(name="karen",     lat=-1.322, lon=36.709, r=3.0,
             poverty=0.05, flood=5,  heat=28, green=62,
             bld=700,  income=130000, hosp=10.0),
        dict(name="gigiri",    lat=-1.223, lon=36.803, r=2.0,
             poverty=0.06, flood=7,  heat=30, green=58,
             bld=550,  income=140000, hosp=9.0),
        dict(name="lavington", lat=-1.284, lon=36.769, r=1.5,
             poverty=0.09, flood=10, heat=36, green=46,
             bld=1100, income=100000, hosp=9.0),
        dict(name="langata",   lat=-1.340, lon=36.750, r=2.0,
             poverty=0.11, flood=11, heat=34, green=52,
             bld=900,  income=90000,  hosp=7.5),
        dict(name="parklands", lat=-1.265, lon=36.813, r=1.2,
             poverty=0.15, flood=15, heat=45, green=28,
             bld=2000, income=70000,  hosp=8.0),

        # ── Green / conservation zones ───────────────────────────────
        dict(name="nbi_park",  lat=-1.370, lon=36.840, r=3.5,
             poverty=0.02, flood=5,  heat=18, green=92,
             bld=30,   income=80000,  hosp=0.0),
        dict(name="karura",    lat=-1.235, lon=36.800, r=1.5,
             poverty=0.04, flood=8,  heat=22, green=88,
             bld=80,   income=90000,  hosp=0.0),
    ]

    # River corridors: (centre_lat, width_km) — flood risk spikes near rivers
    RIVERS = [
        (-1.285, 0.8),   # Nairobi River — main channel through city centre
        (-1.261, 0.6),   # Mathare River — runs through Mathare valley
        (-1.300, 0.7),   # Ngong River tributary — flows through Kibera / Langata
    ]

    def _zone_weights(lat, lon):
        """
        Return a dict of softmax-normalised proximity weights to each zone,
        plus a background weight so no point is fully uninfluenced.
        """
        weights = {}
        for z in ZONES:
            dist_km = np.sqrt((lat - z["lat"])**2 + (lon - z["lon"])**2) * 111
            weights[z["name"]] = np.exp(-dist_km / z["r"])
        # Background (suburbs / unclassified)
        weights["background"] = 0.15
        total = sum(weights.values()) + 1e-9
        return {k: v / total for k, v in weights.items()}

    def _river_flood_boost(lat):
        """Extra flood probability from proximity to rivers."""
        boost = 0.0
        for r_lat, r_width in RIVERS:
            dist_km = abs(lat - r_lat) * 111
            boost = max(boost, 35 * np.exp(-dist_km / r_width))
        return boost

    zone_lookup = {z["name"]: z for z in ZONES}
    zone_lookup["background"] = dict(
        poverty=0.35, flood=30, heat=55, green=15,
        bld=2500, income=28000, hosp=3.0
    )

    data = []
    for lat, lon in zip(lats, lons):
        w = _zone_weights(lat, lon)

        # Weighted average of zone characteristics
        def blend(key):
            return sum(w[z["name"]] * zone_lookup[z["name"]][key] for z in ZONES) \
                   + w["background"] * zone_lookup["background"][key]

        poverty_base    = blend("poverty")
        flood_base      = blend("flood")
        heat_base       = blend("heat")
        green_base      = blend("green")
        bld_base        = blend("bld")
        income_base     = blend("income")
        hosp_base       = blend("hosp")

        # River proximity adds flood risk independent of zone
        river_boost = _river_flood_boost(lat)

        # Final feature values with small Gaussian noise
        flood_risk      = np.clip(flood_base + river_boost + rng.normal(0, 4), 0, 98)
        poverty_rate    = np.clip(poverty_base * 100 + rng.normal(0, 5), 2, 92)
        heat_vuln       = np.clip(heat_base + rng.normal(0, 4), 15, 98)
        green_space     = np.clip(green_base + rng.normal(0, 4), 0, 95)
        building_density = np.clip(bld_base + rng.normal(0, 400), 50, 12000)
        income          = np.clip(income_base + rng.normal(0, 4000), 6000, 180000)
        hospital_access = np.clip(hosp_base + rng.normal(0, 0.5), 0, 20)

        # Road density: good in formal areas (low poverty), poor in informal
        road_density = np.clip(120 * (1 - poverty_base) + rng.normal(0, 8), 5, 150)

        # Population density correlated with building density + poverty
        pop_density  = np.clip(building_density * 3.5 * (1 + 0.5 * poverty_base)
                               + rng.normal(0, 800), 500, 40000)

        # Air quality: worse in industrial / dense-poor areas
        aqi = np.clip(35 + 30 * poverty_base + 20 * (building_density / 10000)
                      - 15 * (green_space / 100) + rng.normal(0, 5), 15, 95)

        dist_cbd_km = np.sqrt((lat - (-1.2864))**2 + (lon - 36.8172)**2) * 111

        data.append({
            "latitude":                   lat,
            "longitude":                  lon,
            "pop_density_per_km2":        round(pop_density, 1),
            "building_density_per_km2":   round(building_density, 1),
            "road_density_km_per_km2":    round(road_density, 1),
            "hospital_access_within_5km": round(hospital_access, 1),
            "flood_risk_percent":         round(flood_risk, 1),
            "heat_vulnerability_index":   round(heat_vuln, 1),
            "poverty_rate_percent":       round(poverty_rate, 1),
            "median_income_kes":          round(income, 0),
            "green_space_percent":        round(green_space, 1),
            "air_quality_index":          round(aqi, 1),
            "distance_from_cbd_km":       round(dist_cbd_km, 2),
        })

    df = pd.DataFrame(data)
    geometries = [Point(row["longitude"], row["latitude"]) for _, row in df.iterrows()]
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")

    out = DATA_RAW / "nairobi_urban_indicators.parquet"
    gdf.to_parquet(out)

    print(f"  ✅ {len(gdf)} spatially-realistic urban indicator points → {out}")
    print(f"     Features: {len(gdf.columns) - 1} urban indicators")
    print(f"     Poverty range: {gdf['poverty_rate_percent'].min():.1f}–"
          f"{gdf['poverty_rate_percent'].max():.1f}%")
    print(f"     Flood risk range: {gdf['flood_risk_percent'].min():.1f}–"
          f"{gdf['flood_risk_percent'].max():.1f}%")
    return gdf


if __name__ == "__main__":
    dataset = fetch_nairobi_urban_dataset()
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Sample of data:\n{dataset.head()}")