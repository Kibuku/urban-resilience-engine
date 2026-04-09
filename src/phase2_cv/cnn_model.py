"""
Phase 2 — CNN-based land cover classification from satellite tiles.

Uses a pre-trained ResNet18 with transfer learning to classify
Nairobi tiles into: urban_built, green_vegetation, water, bare_soil.

NOTE: If you don't have labelled satellite tiles, this module provides:
  1. A synthetic data generator for demonstration
  2. The full training pipeline you'd use with real tiles
  3. A feature extractor that outputs embeddings per tile/hex

For the project submission, the NDVI-based features from fetch_sentinel.py
may be sufficient. This CNN adds depth but is optional given the timeline.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import MODELS_DIR, SEED

torch.manual_seed(SEED)

CLASSES = ["urban_built", "green_vegetation", "water", "bare_soil"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4


# ── Dataset ─────────────────────────────────────────────────────────

class SatelliteTileDataset(Dataset):
    """
    Expects tiles as .npy arrays (H, W, 3) in class-named subdirectories:
      data/raw/tiles/urban_built/tile_001.npy
      data/raw/tiles/green_vegetation/tile_002.npy
      ...
    """
    def __init__(self, root_dir: Path, transform=None):
        self.samples = []
        self.transform = transform
        for label_idx, cls in enumerate(CLASSES):
            cls_dir = root_dir / cls
            if cls_dir.exists():
                for f in cls_dir.glob("*.npy"):
                    self.samples.append((f, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.load(path).astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)  # (C, H, W)
        if self.transform:
            img = self.transform(img)
        return img, label


class SyntheticTileDataset(Dataset):
    """Generate random tiles for pipeline testing."""
    def __init__(self, n=200):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = torch.randn(3, IMG_SIZE, IMG_SIZE)
        label = idx % NUM_CLASSES
        return img, label


# ── Model ───────────────────────────────────────────────────────────

def build_model(pretrained=True):
    """ResNet18 with custom classification head."""
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    # Freeze early layers
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    # Replace final FC
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    return model


def extract_features(model, dataloader, device="cpu"):
    """Extract penultimate-layer embeddings (512-d) for each tile."""
    model.eval()
    # Hook into avgpool output
    features = []
    def hook(module, inp, out):
        features.append(out.squeeze().detach().cpu().numpy())

    handle = model.avgpool.register_forward_hook(hook)

    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            _ = model(imgs)

    handle.remove()
    return np.vstack(features)


# ── Training ────────────────────────────────────────────────────────

def train_model(use_synthetic=True):
    """Train the CNN classifier."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⏳ Phase 2 CNN: Training on {device}...")

    # Data
    if use_synthetic:
        print("  ℹ️  Using synthetic data for demo. Replace with real tiles.")
        train_ds = SyntheticTileDataset(200)
        val_ds   = SyntheticTileDataset(50)
    else:
        from config import DATA_RAW
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_ds = SatelliteTileDataset(DATA_RAW / "tiles", transform)
        val_ds   = train_ds  # Split properly in production

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model
    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # Train loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.3f}")

    # Save
    model_path = MODELS_DIR / "cnn_landcover.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  ✅ Model saved → {model_path}")
    return model


if __name__ == "__main__":
    train_model(use_synthetic=True)
