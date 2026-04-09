"""
Phase 2 Computer Vision tests — CNN & Tile Utilities.

What we test
------------
1. SyntheticTileDataset produces tensors of the right shape and valid labels.
2. SatelliteTileDataset loads .npy tiles correctly when present.
3. build_model() returns a nn.Module with the expected output dimension.
4. extract_features() produces embeddings of the right shape.
5. train_model() completes without error (1 epoch, synthetic data, no GPU).
6. tile_images.create_synthetic_tiles() writes the expected directory structure.
7. tile_images.tile_geotiff() handles missing rasterio gracefully.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# 1. SyntheticTileDataset
# ─────────────────────────────────────────────────────────────────────

class TestSyntheticTileDataset:
    def setup_method(self):
        from src.phase2_cv.cnn_model import SyntheticTileDataset, NUM_CLASSES, IMG_SIZE
        self.ds = SyntheticTileDataset(n=20)
        self.num_classes = NUM_CLASSES
        self.img_size = IMG_SIZE

    def test_dataset_length(self):
        assert len(self.ds) == 20

    def test_item_image_shape(self):
        img, label = self.ds[0]
        assert img.shape == (3, self.img_size, self.img_size), \
            f"Expected (3, {self.img_size}, {self.img_size}), got {img.shape}"

    def test_labels_are_valid(self):
        for i in range(len(self.ds)):
            _, label = self.ds[i]
            assert 0 <= label < self.num_classes

    def test_all_classes_represented(self):
        """All NUM_CLASSES labels should appear in a dataset of size ≥ NUM_CLASSES."""
        labels = {self.ds[i][1] for i in range(self.num_classes * 2)}
        assert labels == set(range(self.num_classes))


# ─────────────────────────────────────────────────────────────────────
# 2. SatelliteTileDataset (from .npy files)
# ─────────────────────────────────────────────────────────────────────

class TestSatelliteTileDataset:
    @pytest.fixture(autouse=True)
    def make_tile_files(self, tmp_path):
        """Write a few synthetic .npy tiles in the expected directory layout."""
        from src.phase2_cv.cnn_model import CLASSES, IMG_SIZE
        self.tile_root = tmp_path / "tiles"
        for cls in CLASSES[:2]:          # only two classes for speed
            cls_dir = self.tile_root / cls
            cls_dir.mkdir(parents=True)
            for i in range(3):
                tile = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                np.save(cls_dir / f"tile_{i:04d}.npy", tile)

    def test_dataset_loads_correct_count(self):
        from src.phase2_cv.cnn_model import SatelliteTileDataset
        ds = SatelliteTileDataset(self.tile_root)
        assert len(ds) == 6          # 2 classes × 3 tiles

    def test_item_image_is_float_tensor(self):
        from src.phase2_cv.cnn_model import SatelliteTileDataset
        ds = SatelliteTileDataset(self.tile_root)
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.dtype == torch.float32

    def test_empty_directory_gives_zero_length(self, tmp_path):
        from src.phase2_cv.cnn_model import SatelliteTileDataset
        empty = tmp_path / "empty_tiles"
        empty.mkdir()
        ds = SatelliteTileDataset(empty)
        assert len(ds) == 0


# ─────────────────────────────────────────────────────────────────────
# 3. build_model
# ─────────────────────────────────────────────────────────────────────

class TestBuildModel:
    def test_model_is_nn_module(self):
        import torch.nn as nn
        from src.phase2_cv.cnn_model import build_model
        model = build_model(pretrained=False)
        assert isinstance(model, nn.Module)

    def test_output_dimension_matches_num_classes(self):
        from src.phase2_cv.cnn_model import build_model, NUM_CLASSES, IMG_SIZE
        model = build_model(pretrained=False)
        dummy = torch.zeros(2, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, NUM_CLASSES)

    def test_only_last_layers_trainable(self):
        from src.phase2_cv.cnn_model import build_model
        model = build_model(pretrained=False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        all_params = list(model.parameters())
        # At least the FC head must be trainable but not all params
        assert 0 < len(trainable) < len(all_params)


# ─────────────────────────────────────────────────────────────────────
# 4. extract_features
# ─────────────────────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_embeddings_have_correct_shape(self):
        from torch.utils.data import DataLoader
        from src.phase2_cv.cnn_model import (
            build_model, extract_features, SyntheticTileDataset
        )
        model = build_model(pretrained=False)
        ds = SyntheticTileDataset(n=8)
        loader = DataLoader(ds, batch_size=4)
        feats = extract_features(model, loader)
        # ResNet18 avgpool output is 512-d; batch flattened
        assert feats.shape[0] == 8
        assert feats.shape[1] == 512

    def test_embeddings_are_finite(self):
        from torch.utils.data import DataLoader
        from src.phase2_cv.cnn_model import (
            build_model, extract_features, SyntheticTileDataset
        )
        model = build_model(pretrained=False)
        loader = DataLoader(SyntheticTileDataset(4), batch_size=4)
        feats = extract_features(model, loader)
        assert np.isfinite(feats).all(), "Embeddings contain NaN or Inf"


# ─────────────────────────────────────────────────────────────────────
# 5. train_model (1 epoch smoke test)
# ─────────────────────────────────────────────────────────────────────

class TestTrainModel:
    def test_train_completes_and_returns_model(self, tmp_path, monkeypatch):
        """Patch MODELS_DIR so the test doesn't pollute the real model directory."""
        import src.phase2_cv.cnn_model as cnn_module
        monkeypatch.setattr(cnn_module, "EPOCHS", 1)
        monkeypatch.setattr(cnn_module, "MODELS_DIR", tmp_path)

        model = cnn_module.train_model(use_synthetic=True)
        import torch.nn as nn
        assert isinstance(model, nn.Module)

    def test_model_file_is_saved(self, tmp_path, monkeypatch):
        import src.phase2_cv.cnn_model as cnn_module
        monkeypatch.setattr(cnn_module, "EPOCHS", 1)
        monkeypatch.setattr(cnn_module, "MODELS_DIR", tmp_path)
        cnn_module.train_model(use_synthetic=True)
        assert (tmp_path / "cnn_landcover.pth").exists()


# ─────────────────────────────────────────────────────────────────────
# 6. tile_images.create_synthetic_tiles
# ─────────────────────────────────────────────────────────────────────

class TestTileImages:
    def test_synthetic_tiles_directory_structure(self, tmp_path):
        from src.phase2_cv.tile_images import create_synthetic_tiles, CLASSES
        counts = create_synthetic_tiles(tmp_path, n_per_class=5, tile_size=32)
        for cls in CLASSES:
            cls_dir = tmp_path / cls
            assert cls_dir.exists(), f"Class directory missing: {cls}"
            npy_files = list(cls_dir.glob("*.npy"))
            assert len(npy_files) == 5, f"Expected 5 tiles in {cls}, found {len(npy_files)}"

    def test_synthetic_tile_shape(self, tmp_path):
        from src.phase2_cv.tile_images import create_synthetic_tiles, CLASSES
        create_synthetic_tiles(tmp_path, n_per_class=3, tile_size=32)
        first = list((tmp_path / CLASSES[0]).glob("*.npy"))[0]
        tile = np.load(first)
        assert tile.shape == (32, 32, 3), f"Unexpected tile shape: {tile.shape}"

    def test_synthetic_tile_dtype_is_uint8(self, tmp_path):
        from src.phase2_cv.tile_images import create_synthetic_tiles, CLASSES
        create_synthetic_tiles(tmp_path, n_per_class=2, tile_size=16)
        first = list((tmp_path / CLASSES[0]).glob("*.npy"))[0]
        tile = np.load(first)
        assert tile.dtype == np.uint8

    def test_returned_count_matches_files(self, tmp_path):
        from src.phase2_cv.tile_images import create_synthetic_tiles
        counts = create_synthetic_tiles(tmp_path, n_per_class=4, tile_size=16)
        assert all(v == 4 for v in counts.values())

    def test_tile_geotiff_raises_without_rasterio(self, tmp_path, monkeypatch):
        """If rasterio is not importable, tile_geotiff should raise ImportError."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "rasterio":
                raise ImportError("mocked missing rasterio")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from src.phase2_cv import tile_images
        with pytest.raises(ImportError, match="rasterio"):
            tile_images.tile_geotiff(tmp_path / "fake.tif", tmp_path)
