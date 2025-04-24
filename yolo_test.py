# yolo_inference.py

import glob
import os
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

# ───── CONFIG ────────────────────────────────────────────────────────────────
RUNS_DIR = Path("runs") / "train_fathomnet"
TEST_ANN_CSV = Path("test/annotations.csv")
TEST_IMG_DIR = Path("test/images")
CSV_OUT = "test/yolo.csv"
# ───────────────────────────────────────────────────────────────────────────────┘

# 1) locate the latest best.pt
pattern = str(RUNS_DIR / "yolov8n_fathomnet*" / "weights" / "best.pt")
ckpts = glob.glob(pattern)
if not ckpts:
    raise FileNotFoundError(f"No checkpoints found: {pattern}")
ckpts.sort(key=os.path.getmtime)
MODEL_PATH = ckpts[-1]
print(f"Using model: {MODEL_PATH}")

# 2) load the YOLO model
model = YOLO(MODEL_PATH)

# 3) read your annotations CSV to get the list of image filenames
df_imgs = pd.read_csv(TEST_ANN_CSV)
if "path" in df_imgs.columns:
    files = df_imgs["path"].tolist()
elif "image_name" in df_imgs.columns:
    files = df_imgs["image_name"].tolist()
else:
    raise ValueError("CSV must have a 'path' or 'image_name' column")

# build full paths
image_paths = []
for f in files:
    p = Path(f)
    if not p.exists():
        p = TEST_IMG_DIR / f
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {f}")
    image_paths.append(p)

print(f"Found {len(image_paths)} images.")

# 4) run inference with low confidence and one box max
records = []
for img_path in image_paths:
    res = model.predict(
        str(img_path),
        conf=0.001,  # threshold so almost every image yields at least one box
        max_det=1,  # keep only the highest-confidence box
        verbose=False,
    )[0]

    if res.boxes:
        cls_idx = int(res.boxes[0].cls[0])
        cls_name = model.names[cls_idx]  # look up the human-readable name
    else:
        cls_name = "none"  # or whatever default you prefer

    records.append(
        {
            "image_name": img_path.name,
            "category_name": cls_name,
        }
    )

# 5) save one-row-per-image CSV
df = pd.DataFrame(records)
df.to_csv(CSV_OUT, index=False)
print(f"Wrote {len(df)} guesses (with names) to {CSV_OUT}")
