import argparse
import json
from pathlib import Path
import requests
import time

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


def build_raw_paths(class_names: list[str], retries=3, delay=1) -> dict[str, list[str]]:
    """
    For each name:
      1) match it to a GBIF usageKey
      2) fetch its ancestor list
      3) pull out scientificName at each of LEVELS
    """
    raw = {}
    failed = []

    for name in tqdm(class_names, desc="Resolving taxonomy"):
        # 1) match to get usageKey
        usage_key = None
        match_url = f"https://api.gbif.org/v1/species/match?name={name}"
        for attempt in range(retries):
            try:
                r = requests.get(match_url, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    usage_key = data.get("usageKey")
                    break
            except requests.RequestException:
                pass
            time.sleep(delay)

        if not usage_key:
            failed.append((name, "no usageKey"))
            rank_map = {}
        else:
            # 2) fetch parents (ancestors)
            parents_url = f"https://api.gbif.org/v1/species/{usage_key}/parents?limit=100"
            ancestors = []
            for attempt in range(retries):
                try:
                    r = requests.get(parents_url, timeout=5)
                    if r.status_code == 200:
                        resp = r.json()
                        if isinstance(resp, dict):
                            # wrapped under "results"
                            ancestors = resp.get("results", [])
                        elif isinstance(resp, list):
                            # returned as a bare list
                            ancestors = resp
                        else:
                            ancestors = []
                        break
                except requests.RequestException:
                    pass
                time.sleep(delay)

            if not ancestors:
                failed.append((name, f"no ancestors for key={usage_key}"))
            # 3) build rank→scientificName map
            rank_map = {
                node["rank"].lower(): node["scientificName"]
                for node in ancestors
                if node.get("rank") and node.get("scientificName")
            }

        # 4) compose the 7-level path
        raw[name] = [rank_map.get(lvl, "UNK") for lvl in LEVELS]

    if failed:
        print("⚠️ Some taxonomy lookups failed:")
        for label, reason in failed:
            print(f"  • {label}: {reason}")

    return raw


def build_rank2idx(raw_paths: dict[str, list[str]]) -> list[dict[str, int]]:
    """Build vocab → index for each of the 7 levels."""
    vocab = [set() for _ in LEVELS]
    for path in raw_paths.values():
        for i, nm in enumerate(path):
            vocab[i].add(nm)
    return [{nm: idx for idx, nm in enumerate(sorted(v))} for v in vocab]


def lca_distance(path_true: list[str], path_pred: list[str]) -> int:
    """
    Compute distance based on the Lowest Common Ancestor (LCA).
    Distance = total levels - depth of LCA.
    """
    for i, (t, p) in enumerate(zip(path_true, path_pred)):
        if t != p or t =="UNK" or p =="UNK":
            return len(path_true) - i
    return 0  # identical paths


# 2) dataset ----------------------------------------------------------------


class FathomNetDataset(Dataset):

    def __init__(self, csv_file, transform, class_path_idxs):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

        self.classes = sorted(self.annotations["label"].unique())
        self.class_to_idx = {lab: i for i, lab in enumerate(self.classes)}
        self.class_path_idxs = class_path_idxs

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        lbl_str = row["label"]
        lbl = self.class_to_idx[lbl_str]
        path_idxs = self.class_path_idxs[lbl_str]
        return img, lbl, path_idxs


# 3) model ------------------------------------------------------------------


class HierarchicalTaxonEncoder(nn.Module):

    def __init__(self, rank2idx, embed_dim=32):
        super().__init__()
        self.embeds = nn.ModuleList(
            [nn.Embedding(len(mapping), embed_dim) for mapping in rank2idx]
        )
        self.output_dim = embed_dim * len(rank2idx)

    def forward(self, paths):  # paths: (B,7)
        lvl_emb = [emb(paths[:, i]) for i, emb in enumerate(self.embeds)]
        return torch.cat(lvl_emb, dim=1)  # (B, 7*embed_dim)


class HierarchyAwareModel(nn.Module):

    def __init__(self, num_classes, tax_encoder: HierarchicalTaxonEncoder):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        feat_dim = resnet.fc.in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.tax_encoder = tax_encoder

        fusion_dim = feat_dim + tax_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, paths):
        x = self.backbone(img).flatten(1)
        t = self.tax_encoder(paths)
        return self.classifier(torch.cat([x, t], dim=1))


# 4) training loop ----------------------------------------------------------


def train(args):
    # transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # read annotations & class names
    df = pd.read_csv(args.csv)
    classes = sorted(df["label"].unique())

    # build taxonomy structures
    raw_paths = build_raw_paths(classes)
    import json

    print("=== raw_paths lineage dump ===")
    print(json.dumps(raw_paths, indent=2))
    rank2idx = build_rank2idx(raw_paths)
    class_path_idxs = {
        cls: torch.tensor(
            [rank2idx[i][nm] for i, nm in enumerate(raw_paths[cls])],
            dtype=torch.long,
        )
        for cls in classes
    }

    # dataset & split
    full_ds = FathomNetDataset(args.csv, transform, class_path_idxs)
    indices = np.arange(len(full_ds))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=df["label"], random_state=42
    )
    train_loader = DataLoader(
        Subset(full_ds, train_idx), batch_size=args.bs, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        Subset(full_ds, val_idx), batch_size=args.bs, shuffle=False, num_workers=4
    )

    # model, loss, optimizer
    tax_encoder = HierarchicalTaxonEncoder(rank2idx, embed_dim=32)
    model = HierarchyAwareModel(len(classes), tax_encoder).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    patience, no_improve = 3, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels, paths in tqdm(
            train_loader, desc=f"Train {epoch}/{args.epochs}"
        ):
            imgs, labels, paths = (
                imgs.to(args.device),
                labels.to(args.device),
                paths.to(args.device),
            )
            optimizer.zero_grad()

            outputs = model(imgs, paths)  # (B, num_classes)
            ce_loss = criterion(outputs, labels)

            # LCA‐based penalty
            preds = outputs.argmax(1).cpu().tolist()
            true_idxs = labels.cpu().tolist()
            dists = [
                lca_distance(raw_paths[classes[t]], raw_paths[classes[p]])
                for t, p in zip(true_idxs, preds)
            ]
            avg_dist = sum(dists) / len(dists)
            loss = ce_loss + args.penalty * avg_dist

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            correct += sum(t == p for t, p in zip(true_idxs, preds))

        train_acc = 100 * correct / total
        print(
            f"Epoch {epoch} | Loss: {running_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%"
        )

        # validation
        model.eval()
        val_correct, val_total = 0, 0
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

        # early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), "best_hier_model.pth")
            print("  Saved new best model.")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{patience} epochs.")
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="train/annotations.csv")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--penalty", type=float, default=0.1, help="Weight for LCA distance penalty"
    )
    args = parser.parse_args()
    train(args)

