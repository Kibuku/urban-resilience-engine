"""
Phase 1 — Fetch OpenStreetMap data for Nairobi.
Pulls building footprints, hospitals, roads, and drainage via OSMnx / Overpass.
"""
import geopandas as gpd
import osmnx as ox
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import NAIROBI_BBOX, DATA_RAW, CRS

PLACE = "Nairobi, Kenya"

# ── Shared spatial helpers ───────────────────────────────────────────

import numpy as np
from shapely.geometry import Point, LineString

# Informal settlements: high building density, very low drainage coverage
_SETTLEMENTS = [
    (-1.313, 36.784),   # Kibera
    (-1.261, 36.864),   # Mathare
    (-1.247, 36.884),   # Korogocho
    (-1.312, 36.864),   # Mukuru
    (-1.252, 36.894),   # Dandora
    (-1.272, 36.894),   # Kayole
]

# Formal / wealthy zones: lower building density, good drainage, hospitals
_FORMAL = [
    (-1.286, 36.817),   # CBD
    (-1.264, 36.804),   # Westlands
    (-1.322, 36.709),   # Karen
    (-1.223, 36.803),   # Gigiri
    (-1.284, 36.769),   # Lavington
    (-1.265, 36.813),   # Parklands
]


def _nearest_zone_weight(lat, lon, zone_centres, radius_km=2.0):
    """Max Gaussian influence from any zone centre."""
    best = 0.0
    for z_lat, z_lon in zone_centres:
        d = np.sqrt((lat - z_lat)**2 + (lon - z_lon)**2) * 111
        best = max(best, np.exp(-d / radius_km))
    return best


def _biased_point(rng, n, settlement_weight, formal_weight):
    """
    Sample (lat, lon) pairs biased toward settlement / formal zones.
    Returns uniform background points blended with zone-attracted points.
    """
    lats, lons = [], []
    for _ in range(n):
        roll = rng.random()
        if roll < settlement_weight:
            centre = _SETTLEMENTS[rng.integers(len(_SETTLEMENTS))]
            lat = centre[0] + rng.normal(0, 0.012)
            lon = centre[1] + rng.normal(0, 0.012)
        elif roll < settlement_weight + formal_weight:
            centre = _FORMAL[rng.integers(len(_FORMAL))]
            lat = centre[0] + rng.normal(0, 0.015)
            lon = centre[1] + rng.normal(0, 0.015)
        else:
            lat = rng.uniform(NAIROBI_BBOX["south"], NAIROBI_BBOX["north"])
            lon = rng.uniform(NAIROBI_BBOX["west"],  NAIROBI_BBOX["east"])
        lats.append(np.clip(lat, NAIROBI_BBOX["south"], NAIROBI_BBOX["north"]))
        lons.append(np.clip(lon, NAIROBI_BBOX["west"],  NAIROBI_BBOX["east"]))
    return lats, lons


# ── Fetch functions ──────────────────────────────────────────────────

def fetch_buildings():
    """
    Synthetic building footprints biased toward informal settlements and CBD.
    Kibera / Mathare / Korogocho get ~3× more buildings per km² than suburbs.
    """
    print("⏳ Fetching building footprints (spatially-realistic mock)...")
    rng = np.random.default_rng(42)

    # 60 % near settlements, 20 % near formal zones, 20 % background
    lats, lons = _biased_point(rng, 2000, settlement_weight=0.60, formal_weight=0.20)

    geometries = [Point(lon, lat) for lat, lon in zip(lats, lons)]
    gdf = gpd.GeoDataFrame({"geometry": geometries}, crs="EPSG:4326").to_crs(CRS)

    out = DATA_RAW / "osm_buildings.parquet"
    gdf[["geometry"]].to_parquet(out)
    print(f"  ✅ {len(gdf)} mock buildings (settlement-biased) → {out}")
    return gdf


def fetch_hospitals():
    """
    Synthetic health facilities concentrated in formal / wealthy zones.
    Informal settlements have fewer hospitals per km² — a key equity factor.
    """
    print("⏳ Fetching health facilities (spatially-realistic mock)...")
    rng = np.random.default_rng(7)

    # Known approximate hospital clusters in Nairobi
    known = [
        (-1.300, 36.808),   # Kenyatta National Hospital
        (-1.289, 36.825),   # Nairobi Hospital
        (-1.263, 36.808),   # MP Shah Hospital (Parklands)
        (-1.272, 36.810),   # Aga Khan Hospital
        (-1.257, 36.802),   # Gertrude's (Muthaiga)
        (-1.340, 36.748),   # Karen Hospital
        (-1.222, 36.798),   # Gigiri Medical Centre
        (-1.295, 36.770),   # Lavington Clinic
        (-1.283, 36.848),   # Eastleigh Health Centre
        (-1.313, 36.784),   # Kibera (1 clinic — contrast)
    ]
    names = [
        "Kenyatta National", "Nairobi Hospital", "MP Shah", "Aga Khan",
        "Gertrudes", "Karen Hospital", "Gigiri Medical", "Lavington Clinic",
        "Eastleigh HC", "Kibera Clinic",
    ]
    # Add ~10 more scattered smaller clinics
    extra_lats, extra_lons = _biased_point(rng, 10, settlement_weight=0.2, formal_weight=0.5)
    for i, (la, lo) in enumerate(zip(extra_lats, extra_lons)):
        known.append((la, lo))
        names.append(f"Clinic {i+1}")

    lats, lons = zip(*known)
    geometries = [Point(lon, lat) for lat, lon in zip(lats, lons)]
    gdf = gpd.GeoDataFrame({"geometry": geometries, "name": names}, crs="EPSG:4326").to_crs(CRS)

    out = DATA_RAW / "osm_hospitals.parquet"
    gdf[["geometry", "name"]].to_parquet(out)
    print(f"  ✅ {len(gdf)} mock facilities (formal-zone-biased) → {out}")
    return gdf


def fetch_roads():
    """
    Synthetic road network: denser and longer segments in formal zones,
    shorter/sparser in informal settlements.
    """
    print("⏳ Fetching road network (spatially-realistic mock)...")
    rng = np.random.default_rng(42)

    roads_data = []
    n_roads = 600

    for _ in range(n_roads):
        # Formal areas get primary/secondary roads; settlements get tertiary/tracks
        roll = rng.random()
        if roll < 0.35:
            centre = _FORMAL[rng.integers(len(_FORMAL))]
            start_lat = centre[0] + rng.normal(0, 0.018)
            start_lon = centre[1] + rng.normal(0, 0.018)
            length  = rng.uniform(300, 2500)
            highway = rng.choice(["primary", "secondary", "residential"])
        elif roll < 0.65:
            centre = _SETTLEMENTS[rng.integers(len(_SETTLEMENTS))]
            start_lat = centre[0] + rng.normal(0, 0.010)
            start_lon = centre[1] + rng.normal(0, 0.010)
            length  = rng.uniform(50, 400)
            highway = rng.choice(["residential", "tertiary", "track"])
        else:
            start_lat = rng.uniform(NAIROBI_BBOX["south"], NAIROBI_BBOX["north"])
            start_lon = rng.uniform(NAIROBI_BBOX["west"],  NAIROBI_BBOX["east"])
            length  = rng.uniform(100, 1000)
            highway = rng.choice(["secondary", "residential", "tertiary"])

        end_lat = start_lat + rng.uniform(-0.008, 0.008)
        end_lon = start_lon + rng.uniform(-0.008, 0.008)
        roads_data.append({
            "geometry": LineString([(start_lon, start_lat), (end_lon, end_lat)]),
            "highway":  highway,
            "length":   round(length, 1),
        })

    gdf = gpd.GeoDataFrame(roads_data, crs="EPSG:4326").to_crs(CRS)
    out = DATA_RAW / "osm_roads.parquet"
    gdf[["geometry", "highway", "length"]].to_parquet(out)
    print(f"  ✅ {len(gdf)} mock road segments (density-biased) → {out}")
    return gdf


def fetch_drainage():
    """
    Synthetic drainage network: formal zones have engineered channels;
    informal settlements have very little formal drainage — driving flood risk.
    """
    print("⏳ Fetching drainage/waterways (spatially-realistic mock)...")
    rng = np.random.default_rng(42)

    drainage_data = []

    # Natural rivers (apply everywhere along their latitude band)
    rivers = [
        (-1.285, "river",  0.025),   # Nairobi River
        (-1.261, "stream", 0.015),   # Mathare River
        (-1.300, "stream", 0.018),   # Ngong River tributary
    ]
    for r_lat, r_type, width in rivers:
        for seg in range(12):
            lon_start = NAIROBI_BBOX["west"] + seg * (NAIROBI_BBOX["east"] - NAIROBI_BBOX["west"]) / 12
            lon_end   = lon_start + (NAIROBI_BBOX["east"] - NAIROBI_BBOX["west"]) / 12
            lat_jitter = rng.normal(0, width / 3)
            drainage_data.append({
                "geometry": LineString([
                    (lon_start, r_lat + lat_jitter),
                    (lon_end,   r_lat + rng.normal(0, width / 3)),
                ]),
                "waterway": r_type,
            })

    # Engineered drains — concentrated in formal / wealthy zones only
    for _ in range(60):
        centre = _FORMAL[rng.integers(len(_FORMAL))]
        lat = centre[0] + rng.normal(0, 0.015)
        lon = centre[1] + rng.normal(0, 0.015)
        drainage_data.append({
            "geometry": LineString([
                (lon, lat),
                (lon + rng.uniform(-0.005, 0.005),
                 lat + rng.uniform(-0.005, 0.005)),
            ]),
            "waterway": rng.choice(["drain", "canal"]),
        })

    # Minimal drainage in informal settlements (reflects reality)
    for _ in range(8):
        centre = _SETTLEMENTS[rng.integers(len(_SETTLEMENTS))]
        lat = centre[0] + rng.normal(0, 0.008)
        lon = centre[1] + rng.normal(0, 0.008)
        drainage_data.append({
            "geometry": LineString([
                (lon, lat),
                (lon + rng.uniform(-0.003, 0.003),
                 lat + rng.uniform(-0.003, 0.003)),
            ]),
            "waterway": "drain",
        })

    gdf = gpd.GeoDataFrame(drainage_data, crs="EPSG:4326").to_crs(CRS)
    out = DATA_RAW / "osm_drainage.parquet"
    gdf[["geometry", "waterway"]].to_parquet(out)
    print(f"  ✅ {len(gdf)} drainage features "
          f"(formal-zone-biased, rivers + {60} engineered drains) → {out}")
    return gdf


if __name__ == "__main__":
    fetch_buildings()
    fetch_hospitals()
    fetch_roads()
    fetch_drainage()
    print("\n🏁 OSM fetch complete.")
