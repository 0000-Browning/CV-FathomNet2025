#!/usr/bin/env python3
"""
Run inference with a Swin‐Base model and dump predictions to CSV,
plus optional histogram and sample‐image display.
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ----------------------------------
# Parse command‐line arguments
# ----------------------------------
parser = argparse.ArgumentParser(
    description="Run inference on test set with Swin‐Base and save predictions."
)
parser.add_argument(
    "--samples",
    type=int,
    default=1,
    help="Number of sample predictions to display (default: 1)",
)
parser.add_argument(
    "--model-path",
    type=str,
    default="../models/swin_b.pth",
    help="Path to your Swin checkpoint.",
)
parser.add_argument(
    "--csv-in",
    type=str,
    default="data/annotations.csv",
    help="Input CSV with a 'path' column for your test images.",
)
parser.add_argument(
    "--csv-out",
    type=str,
    default="output/swin_base_results.csv",
    help="Where to write the [annotation_id, concept_name] CSV.",
)
parser.add_argument(
    "--img-size",
    type=int,
    default=224,
    help="Image side length for resizing (default: 224).",
)
args = parser.parse_args()

# ----------------------------------
# Configuration
# ----------------------------------
MODEL_PATH = args.model_path
CSV_IN = args.csv_in
CSV_OUT = args.csv_out
IMG_SIZE = (args.img_size, args.img_size)

# ----------------------------------
# Load class names from your training set
# ----------------------------------
df_train = pd.read_csv("data/annotations_train.csv")
CLASS_NAMES = sorted(df_train["label"].unique())

# ----------------------------------
# Build & load Swin‐Base
# ----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model(
    "swin_base_patch4_window7_224",
    pretrained=False,
    num_classes=len(CLASS_NAMES),
)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

# ----------------------------------
# Preprocessing pipeline
# ----------------------------------
tf = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# ----------------------------------
# Inference loop
# ----------------------------------
df_test = pd.read_csv(CSV_IN)


# turn “/some/other/machine/path/.../rois/img123.jpg” → “data/rois/img123.jpg”
def fix_path(orig_path: str) -> str:
    # 1) Unify separators so we can split reliably
    norm = orig_path.replace("\\", "/")
    # 2) Try to grab everything after the real "data/rois" in that string
    if "data/rois" in norm:
        # split off the prefix, keep "/subdir/.../file.png"
        rel = norm.split("data/rois", 1)[1]
    else:
        # fallback to just the filename if it wasn't there
        rel = "/" + os.path.basename(norm)
    # 3) Strip any leading slash/backslash
    rel = rel.lstrip("/\\")
    # 4) Re-build with your local data/rois base
    return os.path.join("data", "rois", rel)


df_test["path"] = df_test["path"].apply(fix_path)
paths = df_test["path"].tolist()
preds = []

with torch.no_grad():
    for p in tqdm(paths, desc="Running inference", unit="img"):
        img = Image.open(p).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        out = model(x)
        idx = out.argmax(dim=1).item()
        preds.append(CLASS_NAMES[idx])

# ----------------------------------
# Optional: show prediction distribution
# ----------------------------------
counts = pd.Series(preds).value_counts(sort=False).sort_index()
plt.bar(counts.index, counts.values)
plt.xticks(rotation=90)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.title("Swin‐Base Predictions Distribution")
plt.tight_layout()
plt.show()

# ----------------------------------
# Save to CSV (annotation_id = 1..N)
# ----------------------------------
df_out = pd.DataFrame(
    {"annotation_id": list(range(1, len(preds) + 1)), "concept_name": preds}
)
Path(CSV_OUT).parent.mkdir(exist_ok=True, parents=True)
df_out.to_csv(CSV_OUT, index=False)
print(f"Wrote {len(preds)} predictions to {CSV_OUT!r}")

# ----------------------------------
# Show a few sample predictions
# ----------------------------------
preds_df = pd.DataFrame({"path": paths, "concept_name": preds})
for _, row in preds_df.sample(args.samples).iterrows():
    img = Image.open(row["path"])
    plt.imshow(img)
    plt.title(f"Prediction: {row['concept_name']}")
    plt.axis("off")
    plt.show()
