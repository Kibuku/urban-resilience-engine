"""
Shared configuration for the Urban Resilience Engine.
"""
from pathlib import Path

# === Paths ===
ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

for d in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, REPORTS_DIR / "figures"]:
    d.mkdir(parents=True, exist_ok=True)

# === Nairobi Bounding Box (EPSG:4326) ===
NAIROBI_BBOX = {
    "north": -1.16,
    "south": -1.38,
    "east":  36.96,
    "west":  36.70,
}
NAIROBI_CENTER = (-1.2921, 36.8219)

# === H3 Resolution ===
# Resolution 8 ≈ 0.46 km² hexagons — good balance for Nairobi neighbourhoods
H3_RESOLUTION = 8

# === NOAA GSOD Station IDs for Nairobi ===
NAIROBI_STATIONS = {
    "JKIA":   "637400-99999",   # Jomo Kenyatta International Airport
    "Wilson": "637410-99999",   # Wilson Airport
}

# === Time Range ===
YEAR_START = 2018
YEAR_END   = 2025

# === Target CRS ===
CRS = "EPSG:4326"

# === Random Seed ===
SEED = 42
