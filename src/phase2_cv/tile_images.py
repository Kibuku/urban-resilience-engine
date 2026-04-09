"""
Phase 2 — Satellite Image Tiling Utility.

Cuts GeoTIFF raster files (e.g. Sentinel-2 bands exported from GEE) into
fixed-size tiles suitable for the CNN classifier.  Also supports creating
a synthetic tile set for pipeline testing when real imagery is unavailable.

Usage:
    python src/phase2_cv/tile_images.py --input data/raw/sentinel_nairobi.tif \
        --output data/raw/tiles --size 224 --stride 112

Tile directory layout expected by SatelliteTileDataset in cnn_model.py:
    data/raw/tiles/
        urban_built/   tile_0000.npy  tile_0001.npy ...
        green_vegetation/
        water/
        bare_soil/
        unlabelled/    (tiles with no class label — for inference only)
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import DATA_RAW, SEED

CLASSES = ["urban_built", "green_vegetation", "water", "bare_soil"]
DEFAULT_TILE_SIZE = 224
DEFAULT_STRIDE    = 112   # 50 % overlap


# ── Tiling from a real GeoTIFF ───────────────────────────────────────

def tile_geotiff(
    src_path: Path,
    out_dir:  Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    stride:    int = DEFAULT_STRIDE,
    bands:     list[int] | None = None,
) -> int:
    """
    Slice a multi-band GeoTIFF into (tile_size × tile_size × C) .npy tiles.

    Parameters
    ----------
    src_path  : Path to input GeoTIFF (e.g. Sentinel-2 export from GEE).
    out_dir   : Root output directory.  Tiles go to out_dir/unlabelled/.
    tile_size : Pixel size of each square tile (default 224 for ResNet18).
    stride    : Step between tile origins.  stride < tile_size → overlap.
    bands     : 1-based band indices to use (e.g. [4, 3, 2] for RGB).
                None means use all bands.

    Returns
    -------
    Number of tiles written.
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("Install rasterio: pip install rasterio")

    tile_dir = out_dir / "unlabelled"
    tile_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with rasterio.open(src_path) as ds:
        n_bands = ds.count
        sel_bands = bands if bands else list(range(1, n_bands + 1))
        height, width = ds.height, ds.width

        print(f"⏳ Tiling {src_path.name}  "
              f"({height}×{width} px, {n_bands} bands, "
              f"tile={tile_size}, stride={stride})")

        for row in range(0, height - tile_size + 1, stride):
            for col in range(0, width - tile_size + 1, stride):
                window = rasterio.windows.Window(col, row, tile_size, tile_size)
                data = ds.read(sel_bands, window=window)   # (C, H, W)
                tile = data.transpose(1, 2, 0)             # (H, W, C)

                # Skip tiles that are mostly NoData
                if np.isnan(tile).mean() > 0.3:
                    continue

                # Normalise to uint8 range for storage
                tile_min, tile_max = np.nanmin(tile), np.nanmax(tile)
                if tile_max > tile_min:
                    tile = ((tile - tile_min) / (tile_max - tile_min) * 255).astype(np.uint8)
                else:
                    tile = np.zeros_like(tile, dtype=np.uint8)

                out_path = tile_dir / f"tile_{count:05d}.npy"
                np.save(out_path, tile)
                count += 1

    print(f"  ✅ {count} tiles → {tile_dir}")
    return count


# ── Move/copy tiles into labelled class folders ──────────────────────

def assign_labels_from_csv(tile_dir: Path, label_csv: Path) -> None:
    """
    Move unlabelled tiles into class sub-directories using a CSV label file.

    CSV format (no header):
        tile_00000.npy,urban_built
        tile_00001.npy,green_vegetation
        ...

    Unlabelled tiles remain in tile_dir/unlabelled/.
    """
    import shutil
    import csv

    for cls in CLASSES:
        (tile_dir / cls).mkdir(parents=True, exist_ok=True)

    moved = 0
    with open(label_csv, newline="") as fh:
        for filename, label in csv.reader(fh):
            label = label.strip()
            if label not in CLASSES:
                print(f"  ⚠️  Unknown class '{label}' for {filename} — skipping")
                continue
            src = tile_dir / "unlabelled" / filename
            dst = tile_dir / label / filename
            if src.exists():
                shutil.move(str(src), str(dst))
                moved += 1

    print(f"  ✅ Assigned {moved} tiles to class directories")


# ── Synthetic tile generator ─────────────────────────────────────────

def create_synthetic_tiles(
    out_dir: Path,
    n_per_class: int = 50,
    tile_size: int = DEFAULT_TILE_SIZE,
    seed: int = SEED,
) -> dict[str, int]:
    """
    Generate synthetic RGB tiles with class-specific colour statistics.

    Each class has a distinct mean/std so the CNN can learn a trivial
    separating boundary — useful for smoke-testing the training pipeline
    without real satellite imagery.

    Returns a dict mapping class name → number of tiles written.
    """
    rng = np.random.default_rng(seed)

    # Per-class (mean, std) for each of 3 channels (R, G, B)
    class_params: dict[str, tuple[list[float], list[float]]] = {
        "urban_built":       ([150, 140, 130], [25, 25, 25]),   # grey-ish
        "green_vegetation":  ([60,  130,  60], [20, 30, 20]),   # green
        "water":             ([60,   90, 160], [20, 20, 30]),   # blue
        "bare_soil":         ([180, 160, 120], [25, 25, 20]),   # tan
    }

    counts: dict[str, int] = {}
    for cls, (means, stds) in class_params.items():
        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            channels = [
                rng.normal(m, s, (tile_size, tile_size))
                for m, s in zip(means, stds)
            ]
            tile = np.stack(channels, axis=-1).clip(0, 255).astype(np.uint8)
            np.save(cls_dir / f"tile_{i:04d}.npy", tile)
        counts[cls] = n_per_class

    total = sum(counts.values())
    print(f"  ✅ {total} synthetic tiles → {out_dir}  ({n_per_class} per class)")
    return counts


# ── CLI entry point ──────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tile satellite images for CNN training")
    p.add_argument("--input",  type=Path, help="Input GeoTIFF path")
    p.add_argument("--output", type=Path, default=DATA_RAW / "tiles",
                   help="Output root directory (default: data/raw/tiles)")
    p.add_argument("--size",   type=int, default=DEFAULT_TILE_SIZE,
                   help="Tile size in pixels (default: 224)")
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                   help="Stride in pixels (default: 112)")
    p.add_argument("--bands",  type=int, nargs="+",
                   help="1-based band indices to extract (default: all)")
    p.add_argument("--labels", type=Path,
                   help="CSV file mapping tile filenames to class labels")
    p.add_argument("--synthetic", action="store_true",
                   help="Generate synthetic tiles instead of processing a GeoTIFF")
    p.add_argument("--n-per-class", type=int, default=50,
                   help="Tiles per class when --synthetic is set (default: 50)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        create_synthetic_tiles(out_dir, n_per_class=args.n_per_class, tile_size=args.size)
    elif args.input:
        tile_geotiff(args.input, out_dir, tile_size=args.size,
                     stride=args.stride, bands=args.bands)
        if args.labels:
            assign_labels_from_csv(out_dir, args.labels)
    else:
        print("Provide --input <geotiff> or --synthetic.  Use --help for options.")
