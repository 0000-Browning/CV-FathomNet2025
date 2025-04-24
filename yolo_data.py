import json
import os
from collections import defaultdict
from glob import glob

from sklearn.model_selection import train_test_split

# Paths
JSON_PATH = "dataset_train.json"
IMG_DIR = "train/images"
LABELS_DIR = "data/labels"
TRAIN_DIR = os.path.join(LABELS_DIR, "train_yolo")
VAL_DIR = os.path.join(LABELS_DIR, "val")

# 1) Load COCO JSON
with open(JSON_PATH) as f:
    coco = json.load(f)

# 2) Build lookup and group boxes by COCO filename
images_meta = {img["id"]: img for img in coco["images"]}
bboxes_by_image = defaultdict(list)
for ann in coco["annotations"]:
    fn = images_meta[ann["image_id"]]["file_name"]
    bboxes_by_image[fn].append(
        {
            "bbox": ann["bbox"],
            "category_id": ann["category_id"],
            "img_size": (
                images_meta[ann["image_id"]]["width"],
                images_meta[ann["image_id"]]["height"],
            ),
        }
    )

# 3) Map COCO category_id → 0‑based class_idx
all_cat_ids = sorted({a["category_id"] for a in coco["annotations"]})
cat_to_idx = {cid: i for i, cid in enumerate(all_cat_ids)}

# 4) Grab your numeric images and split
img_paths = sorted(
    glob(os.path.join(IMG_DIR, "*.png")),
    key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
)
indices = list(range(len(img_paths)))
train_idxs, val_idxs = train_test_split(indices, test_size=0.2, random_state=42)

# 5) Make output dirs
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)


def write_yolo_numeric(idxs, out_dir):
    for idx in idxs:
        img_path = img_paths[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]  # e.g. "3"
        # Map back to the COCO JSON by position:
        if idx >= len(coco["images"]):
            print(f"[WARN] No JSON entry for image index {idx} → skipping {img_path}")
            continue

        coco_fn = coco["images"][idx]["file_name"]
        records = bboxes_by_image.get(coco_fn, [])
        if not records:
            print(
                f"[WARN] No annotations for JSON file '{coco_fn}' → writing empty {stem}.txt"
            )
            # Option A: Create an empty label file so YOLO sees it exists
            open(os.path.join(out_dir, f"{stem}.txt"), "w").close()
            continue

        w, h = records[0]["img_size"]
        out_file = os.path.join(out_dir, f"{stem}.txt")
        with open(out_file, "w") as f:
            for rec in records:
                x, y, bw, bh = rec["bbox"]
                x_c = (x + bw / 2) / w
                y_c = (y + bh / 2) / h
                w_n = bw / w
                h_n = bh / h
                cls = cat_to_idx[rec["category_id"]]
                f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")


# 6) Write train & val labels
write_yolo_numeric(train_idxs, TRAIN_DIR)
write_yolo_numeric(val_idxs, VAL_DIR)

print("Done.")
