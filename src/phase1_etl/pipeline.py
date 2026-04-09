"""
Phase 1 — Full ETL Pipeline Orchestrator.
Run: python src/phase1_etl/pipeline.py
"""
from .fetch_urban_data import fetch_nairobi_urban_dataset
from .clean_merge import build_grid_dataset
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import DATA_RAW


def run_etl():
    print("=" * 60)
    print("PHASE 1 — DATA ENGINEERING PIPELINE")
    print("=" * 60)

    # Step 1: Fetch comprehensive urban dataset
    print("\n📥 Step 1/2: Fetching Nairobi urban indicators dataset")
    urban_data = fetch_nairobi_urban_dataset()

    print("\n🔗 Step 2/2: Spatial merge onto H3 grid")
    grid = build_grid_dataset(urban_data)

    print("\n" + "=" * 60)
    print("✅ PHASE 1 COMPLETE")
    print(f"   Grid shape: {grid.shape}")
    print(f"   Columns: {list(grid.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    run_etl()
