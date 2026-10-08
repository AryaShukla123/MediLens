import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from app.modules.retinopathy.preprocessing import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    NUM_CLASSES,
    SUPPORTED_ARCHITECTURE,
    build_model,
    preprocess_retina_image,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "aptos")
CACHE_DIR = os.path.join(DATA_DIR, f"cache_{IMG_SIZE}")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

SPLITS = {
    "train": ("train.csv", "train_images"),
    "val": ("valid.csv", "val_images"),
    "test": ("test.csv", "test_images"),
}
IMAGE_EXT = ".png"

MODEL_PATH = os.path.join(BASE_DIR, "model.pt")
LATEST_CKPT = os.path.join(CHECKPOINT_DIR, "retinopathy_latest.pt")
BEST_CKPT = os.path.join(CHECKPOINT_DIR, "retinopathy_best.pt")

SEED = 42
LEARNING_RATE = 1e-3
PATIENCE = 3


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(split: str) -> pd.DataFrame:
    csv_name, img_dir_name = SPLITS[split]
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name), dtype={"id_code": str})
    total = len(df)

    valid_id = df["id_code"].str.fullmatch(r"[0-9a-f]{12}")
    df = df[valid_id]

    img_dir = os.path.join(DATA_DIR, img_dir_name)
    has_image = df["id_code"].map(lambda c: os.path.exists(os.path.join(img_dir, c + IMAGE_EXT)))
    df = df[has_image].reset_index(drop=True)

    print(f"[{split}] {len(df)} usable rows ({total - len(df)} dropped of {total})")
    return df


def ensure_cache(df: pd.DataFrame, split: str) -> None:
    """Preprocess each image once and store it as a PNG for fast reuse."""
    img_dir = os.path.join(DATA_DIR, SPLITS[split][1])
    out_dir = os.path.join(CACHE_DIR, split)
    os.makedirs(out_dir, exist_ok=True)

    missing = [c for c in df["id_code"] if not os.path.exists(os.path.join(out_dir, c + ".png"))]
    if not missing:
        print(f"[{split}] cache already complete")
        return

    print(f"[{split}] caching {len(missing)} images (one-time cost)...")
    for i, id_code in enumerate(missing, 1):
        processed = preprocess_retina_image(os.path.join(img_dir, id_code + IMAGE_EXT), IMG_SIZE)
        cv2.imwrite(os.path.join(out_dir, id_code + ".png"), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        if i % 250 == 0:
            print(f"    {i}/{len(missing)}")


class RetinopathyDataset(Dataset):
    """Reads the already-preprocessed PNGs from the cache."""

    def __init__(self, df: pd.DataFrame, cache_split_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.cache_split_dir = cache_split_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(os.path.join(self.cache_split_dir, row["id_code"] + ".png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        return img, torch.tensor(int(row["diagnosis"]), dtype=torch.long)


def build_transforms():
    normalize = A.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        normalize,
        ToTensorV2(),
    ])
    eval_tf = A.Compose([normalize, ToTensorV2()])
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return kappa, macro_f1, all_labels, all_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train the retinopathy model.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fresh", action="store_true", help="ignore any saved checkpoint")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_df, val_df, test_df = (load_split(s) for s in ("train", "val", "test"))
    for split, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        ensure_cache(df, split)

    train_tf, eval_tf = build_transforms()
    train_ds = RetinopathyDataset(train_df, os.path.join(CACHE_DIR, "train"), train_tf)
    val_ds = RetinopathyDataset(val_df, os.path.join(CACHE_DIR, "val"), eval_tf)
    test_ds = RetinopathyDataset(test_df, os.path.join(CACHE_DIR, "test"), eval_tf)

    # Imbalance: sample rare grades more often AND weight them more in the loss.
    class_counts = train_df["diagnosis"].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = train_df["diagnosis"].map(lambda c: class_weights[c]).values
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # num_workers=0: multiprocess DataLoaders can hang on Windows.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True).to(device)

    loss_weights = torch.tensor(class_weights / class_weights.sum() * len(class_weights), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # --- resume from the last finished epoch if a checkpoint exists ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start_epoch, best_kappa, epochs_without_improvement = 0, -1.0, 0
    if os.path.exists(LATEST_CKPT) and not args.fresh:
        ckpt = torch.load(LATEST_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_kappa = ckpt["best_kappa"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        print(f"Resuming from epoch {start_epoch + 1} (best val kappa so far: {best_kappa:.4f})")
    else:
        print("Starting fresh from epoch 1")

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)
        val_kappa, val_f1, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_kappa)
        print(f"Epoch {epoch + 1}/{args.epochs} | loss={train_loss:.4f} | "
              f"val_kappa={val_kappa:.4f} | val_macro_f1={val_f1:.4f} | {time.time() - started:.0f}s")

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_CKPT)
            print(f"  -> new best (kappa={best_kappa:.4f})")
        else:
            epochs_without_improvement += 1

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_kappa": best_kappa,
            "epochs_without_improvement": epochs_without_improvement,
        }, LATEST_CKPT)

        if epochs_without_improvement >= PATIENCE:
            print(f"  -> no improvement for {PATIENCE} epochs, stopping early")
            break

    # --- final evaluation with the best weights ---
    model.load_state_dict(torch.load(BEST_CKPT, map_location=device))
    test_kappa, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"\nBest val kappa: {best_kappa:.4f}")
    print(f"Test quadratic weighted kappa: {test_kappa:.4f} | macro F1: {test_f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(y_true, y_pred))

    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "architecture": SUPPORTED_ARCHITECTURE,
        "num_classes": NUM_CLASSES,
        "img_size": IMG_SIZE,
        "class_names": CLASS_NAMES,
    }, MODEL_PATH)
    print(f"\nSaved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
