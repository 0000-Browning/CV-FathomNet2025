#!/usr/bin/env python3
"""
Full training pipeline for the 2025 FathomNet Competition: hierarchical classification
of marine taxa using a Hugging Face SwinV2 model, with CSV-format annotations.

Requirements:
    torch, torchvision, numpy, pandas, scikit-learn, pillow, tqdm, transformers

Usage:
    Adjust Config as needed, then:
        python train_fathomnet.py

Outputs:
    - best_model.pth
    - submission.csv
"""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


# -------------------- Configuration --------------------
class Config:
    data_dir = "."  # root for all paths
    train_csv = "plz/annotations.csv"
    test_csv = "other/annotations.csv"
    model_name = "microsoft/swinv2-tiny-patch4-window8-256"
    input_size = 256
    num_classes = None  # set after reading data
    batch_size = 32
    epochs = 50
    lr = 1e-4
    weight_decay = 1e-4
    backbone_lr_mul = 0.1
    patience = 5
    lr_patience = 2
    lr_factor = 0.5
    seed = 42


# Set random seeds for reproducibility
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)
random.seed(Config.seed)

# -------------------- Processor & Augmentations --------------------
# Explicitly set processor input size
processor = AutoImageProcessor.from_pretrained(
    Config.model_name, size=Config.input_size
)
# Simple PIL-level augmentations for training
train_augment = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ]
)


# -------------------- CSV Dataset --------------------
class CSVDataset(Dataset):

    def __init__(self, df, class_to_idx, mode="train"):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        image = Image.open(img_path).convert("RGB")
        # augment for train
        if self.mode == "train":
            image = train_augment(image)
        # processor handles resize & normalize
        pixel_values = processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)
        if self.mode in ("train", "val"):
            label = self.class_to_idx[row["label"]]
            return pixel_values, label
        else:
            return pixel_values, row["annotation_id"]


# -------------------- Main Pipeline --------------------
def main():
    # 1) LOAD DATAFRAMES
    train_df = pd.read_csv(Config.train_csv)
    test_df = pd.read_csv(Config.test_csv)

    # ensure annotation_id for test
    if "annotation_id" not in test_df.columns:
        if "id" in test_df.columns:
            test_df.rename(columns={"id": "annotation_id"}, inplace=True)
        else:
            test_df["annotation_id"] = test_df.index

    # make paths absolute
    train_df["path"] = train_df["path"].apply(
        lambda p: os.path.join(Config.data_dir, p)
    )
    test_df["path"] = test_df["path"].apply(lambda p: os.path.join(Config.data_dir, p))

    # 2) CLASS MAPPINGS
    classes = sorted(train_df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    Config.num_classes = len(classes)

    # 3) TRAIN/VAL SPLIT
    train_idx, val_idx = train_test_split(
        train_df.index.to_list(),
        test_size=0.2,
        stratify=train_df["label"],
        random_state=Config.seed,
    )
    df_train = train_df.loc[train_idx].reset_index(drop=True)
    df_val = train_df.loc[val_idx].reset_index(drop=True)

    # 4) DATASETS & LOADERS
    train_ds = CSVDataset(df_train, class_to_idx, mode="train")
    val_ds = CSVDataset(df_val, class_to_idx, mode="val")
    test_ds = CSVDataset(test_df, class_to_idx, mode="test")

    train_loader = DataLoader(
        train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.batch_size, shuffle=False, num_workers=4
    )

    # 5) MODEL INITIALIZATION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model with correct num_labels
    model = AutoModelForImageClassification.from_pretrained(
        Config.model_name, num_labels=Config.num_classes
    )
    model.to(device)

    # 6) OPTIMIZER & SCHEDULER
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if "classifier" in name or "head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.lr * Config.backbone_lr_mul},
            {"params": head_params, "lr": Config.lr},
        ],
        weight_decay=Config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.lr_factor,
        patience=Config.lr_patience,
        verbose=True,
    )

    # 7) TRAINING LOOP WITH EARLY STOPPING
    best_acc = 0.0
    no_improve = 0
    for epoch in range(1, Config.epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        for pixel_values, labels in tqdm(
            train_loader, desc=f"Train Epoch {epoch}/{Config.epochs}", leave=False
        ):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_samples += labels.size(0)

        train_loss = epoch_loss / epoch_samples
        train_acc = epoch_correct / epoch_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for pixel_values, labels in tqdm(
                val_loader, desc=f" Val  Epoch {epoch}/{Config.epochs}", leave=False
            ):
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)
                outputs = model(pixel_values)
                logits = outputs.logits
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)

        val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}/{Config.epochs}  "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}  "
            f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best model at epoch {epoch} (Val Acc: {best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= Config.patience:
                print(f"Early stopping at epoch {epoch}. Best Val Acc: {best_acc:.4f}")
                break

    # Load best model for inference
    model.load_state_dict(torch.load("bestv2.pth"))
    model.eval()

    # 8) INFERENCE & SUBMISSION
    preds_list = []
    with torch.no_grad():
        for pixel_values, ann_ids in tqdm(
            test_loader, desc="Test Inference", leave=False
        ):
            pixel_values = pixel_values.to(device)
            outputs = model(pixel_values)
            logits = outputs.logits
            idxs = logits.argmax(dim=1).cpu().numpy()
            for aid, pidx in zip(ann_ids, idxs):
                preds_list.append((aid, idx_to_class[pidx]))

    sub_df = pd.DataFrame(preds_list, columns=["annotation_id", "concept_name"])
    sub_df.to_csv("bestv2.csv", index=False)
    print(sub_df.head())


if __name__ == "__main__":
    main()
