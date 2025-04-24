# train_hierarchical.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm

# 1) taxonomic utils -------------------------------------------------------

LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def build_raw_paths(class_names: list[str]) -> dict[str, list[str]]:
    """Fetch each class’s 7‐level path (filling 'UNK' where missing)."""
    from fathomnet.api.worms import get_ancestors

    def extract_lineage(node):
        lineage = {}
        curr = node
        while curr:
            lineage[curr.rank.lower()] = curr.scientific_name
            curr = getattr(curr, "children", [None])[0]
        return lineage

    raw = {}
    for name in class_names:
        try:
            tree = get_ancestors(name)
            ranks = extract_lineage(tree)
        except Exception:
            ranks = {}
        raw[name] = [ranks.get(lvl, "UNK") for lvl in LEVELS]
    return raw


def build_rank2idx(raw_paths: dict[str, list[str]]) -> list[dict[str, int]]:
    """Build vocab → index for each of the 7 levels."""
    vocab = [set() for _ in LEVELS]
    for path in raw_paths.values():
        for i, nm in enumerate(path):
            vocab[i].add(nm)
    return [{nm: idx for idx, nm in enumerate(sorted(v))} for v in vocab]


# 2) dataset ----------------------------------------------------------------


class FathomNetDataset(Dataset):

    def __init__(self, csv_file, transform, class_path_idxs):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

        # original 79 classes
        self.classes = sorted(self.annotations["label"].unique())
        self.class_to_idx = {lab: i for i, lab in enumerate(self.classes)}

        # dict[label_str] → LongTensor([i0…i6])
        self.class_path_idxs = class_path_idxs

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, i):
        row = self.annotations.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        lbl_str = row["label"]
        lbl = self.class_to_idx[lbl_str]
        path_idxs = self.class_path_idxs[lbl_str]  # torch.LongTensor(7,)
        return img, lbl, path_idxs


# 3) model ------------------------------------------------------------------


class HierarchicalTaxonEncoder(nn.Module):

    def __init__(self, rank2idx, embed_dim=32):
        super().__init__()
        self.embeds = nn.ModuleList(
            [nn.Embedding(len(mapping), embed_dim) for mapping in rank2idx]
        )
        self.output_dim = embed_dim * len(rank2idx)

    def forward(self, paths):  # paths: (B,7) LongTensor
        lvl_emb = [emb(paths[:, i]) for i, emb in enumerate(self.embeds)]
        return torch.cat(lvl_emb, dim=1)  # (B, 7*embed_dim)


class HierarchyAwareModel(nn.Module):

    def __init__(self, num_classes, tax_encoder: HierarchicalTaxonEncoder):
        super().__init__()
        # load resnet backbone, drop final fc
        resnet = models.resnet50(pretrained=True)
        feat_dim = resnet.fc.in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.tax_encoder = tax_encoder

        # fuse image + taxon features
        fusion_dim = feat_dim + tax_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, img, paths):
        x = self.backbone(img).flatten(1)  # (B, feat_dim)
        t = self.tax_encoder(paths)  # (B, tax_dim)
        return self.classifier(torch.cat([x, t], dim=1))  # (B, num_classes)


# 4) training loop ----------------------------------------------------------


def train(args):
    # transforms
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # 1) read CSV once to get class names
    df = pd.read_csv(args.csv)
    classes = sorted(df["label"].unique())

    # 2) build paths & mappings
    raw_paths = build_raw_paths(classes)
    rank2idx = build_rank2idx(raw_paths)
    class_path_idxs = {
        cls: torch.tensor(
            [rank2idx[i][nm] for i, nm in enumerate(raw_paths[cls])], dtype=torch.long
        )
        for cls in classes
    }

    # 3) make dataset + splits
    full_ds = FathomNetDataset(args.csv, transform=tf, class_path_idxs=class_path_idxs)
    idxs = np.arange(len(full_ds))
    train_idxs, val_idxs = train_test_split(
        idxs, test_size=0.2, stratify=df["label"], random_state=42
    )
    train_ds, val_ds = Subset(full_ds, train_idxs), Subset(full_ds, val_idxs)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=4)

    # 4) model, loss, optim
    tax_encoder = HierarchicalTaxonEncoder(rank2idx, embed_dim=32)
    model = HierarchyAwareModel(len(classes), tax_encoder).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5) early-stopping setup
    best_acc = 0.0
    patience = 3
    epochs_no_improve = 0

    # 6) training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels, paths in tqdm(
            train_loader, desc=f"Train Epoch {epoch}/{args.epochs}"
        ):
            imgs, labels, paths = (
                imgs.to(args.device),
                labels.to(args.device),
                paths.to(args.device),
            )
            optimizer.zero_grad()
            outputs = model(imgs, paths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_acc = 100 * correct / total
        print(
            f"Epoch {epoch} | Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%"
        )

        # --- validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels, paths in val_loader:
                imgs, labels, paths = (
                    imgs.to(args.device),
                    labels.to(args.device),
                    paths.to(args.device),
                )
                outputs = model(imgs, paths)
                preds = outputs.argmax(1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"→ Val Acc: {val_acc:.2f}%")

        # early-stopping check
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_hier_model.pth")
            print("  Saved new best model.")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print("Early stopping triggered. Stopping training.")
                break

    print(f"Training complete. Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="train/annotations.csv")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    train(args)
