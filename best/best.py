#!/usr/bin/env python3
"""
Full training pipeline for the 2025 FathomNet Competition: hierarchical classification
of 79 marine taxa using CSV-format annotations, with tqdm progress bars.

Requirements:
    torch, torchvision, timm, numpy, pandas, scikit-learn, pillow, tqdm

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
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# -------------------- Configuration --------------------
class Config:
    data_dir = "."  # root for all paths
    train_csv = "train/annotations.csv"
    test_csv = "test/annotations.csv"
    model_name = "swin_base_patch4_window7_224"
    pretrained = True
    num_classes = None  # will be set from data
    input_size = 224
    batch_size = 32
    epochs = 50
    lr = 1e-4
    weight_decay = 1e-4
    backbone_lr_mul = 0.1
    patience = 5
    lr_patience = 2
    lr_factor = 0.5
    seed = 42


# reproducibility
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)
random.seed(Config.seed)


# -------------------- CSV Dataset --------------------
class CSVDataset(Dataset):

    def __init__(self, df, class_to_idx, transform=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        if self.mode in ("train", "val"):
            lbl = self.class_to_idx[row["label"]]
            return img, lbl
        else:
            return img, row["annotation_id"]


# -------------------- Main Pipeline --------------------
def main():
    # 1) LOAD CSVs
    train_df = pd.read_csv(Config.train_csv)
    test_df = pd.read_csv(Config.test_csv)

    # ensure annotation_id exists in test_df
    if "annotation_id" not in test_df.columns:
        if "id" in test_df.columns:
            test_df.rename(columns={"id": "annotation_id"}, inplace=True)
        else:
            test_df["annotation_id"] = test_df.index

    # prepend data_dir if CSV paths are relative
    train_df["path"] = train_df["path"].apply(
        lambda p: os.path.join(Config.data_dir, p)
    )
    test_df["path"] = test_df["path"].apply(lambda p: os.path.join(Config.data_dir, p))

    # 2) BUILD CLASS MAPPINGS
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
    df_train = train_df.loc[train_idx]
    df_val = train_df.loc[val_idx]

    # 4) TRANSFORMS & DATASETS
    train_tf = transforms.Compose(
        [
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(5),
            transforms.Resize((Config.input_size, Config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((Config.input_size, Config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = CSVDataset(df_train, class_to_idx, transform=train_tf, mode="train")
    val_ds = CSVDataset(df_val, class_to_idx, transform=val_tf, mode="val")
    test_ds = CSVDataset(test_df, class_to_idx, transform=val_tf, mode="test")

    train_loader = DataLoader(
        train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.batch_size, shuffle=False, num_workers=4
    )

    # 5) MODEL, OPTIMIZER, SCHEDULER
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        Config.model_name, pretrained=Config.pretrained, num_classes=Config.num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if "head" in name or "fc" in name:
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

    # 6) TRAINING LOOP
    best_acc = 0.0
    no_improve = 0
    for epoch in range(1, Config.epochs + 1):
        # train
        model.train()
        total_loss = total_correct = total_samples = 0
        for imgs, labs in tqdm(
            train_loader, desc=f"Train Epoch {epoch}/{Config.epochs}", leave=False
        ):
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = out.argmax(1)
            total_correct += (preds == labs).sum().item()
            total_samples += labs.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # validate
        model.eval()
        val_loss = val_correct = val_samples = 0
        for imgs, labs in tqdm(
            val_loader, desc=f" Val  Epoch {epoch}/{Config.epochs}", leave=False
        ):
            imgs, labs = imgs.to(device), labs.to(device)
            out = model(imgs)
            loss = criterion(out, labs)
            val_loss += loss.item() * imgs.size(0)
            preds = out.argmax(1)
            val_correct += (preds == labs).sum().item()
            val_samples += labs.size(0)

        val_loss /= val_samples
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
        else:
            no_improve += 1
            if no_improve >= Config.patience:
                print("Early stopping…")
                break

    print(f"Best val accuracy: {best_acc:.4f}")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    # 7) INFERENCE & SUBMISSION
    preds_list = []
    for imgs, ann_ids in tqdm(test_loader, desc="Test Inference", leave=False):
        imgs = imgs.to(device)
        out = model(imgs)
        idxs = out.argmax(1).cpu().numpy()
        for aid, pidx in zip(ann_ids, idxs):
            preds_list.append((aid, idx_to_class[pidx]))

    sub_df = pd.DataFrame(preds_list, columns=["annotation_id", "concept_name"])
    sub_df.to_csv("submission.csv", index=False)
    print(sub_df.head())


if __name__ == "__main__":
    main()
