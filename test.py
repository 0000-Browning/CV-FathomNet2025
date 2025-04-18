from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from base import FathomNetDataset

# CONFIG
MODEL_PATH = "resnet50_fathomnet.pth"
CSV_IN = "test/annotations.csv"
CSV_OUT = "test/res.csv"
IMG_SIZE = (224, 224)

# get class names
train_ds = FathomNetDataset(csv_file="train/annotations.csv")
CLASS_NAMES = train_ds.classes

# build & load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

# preprocessing
tf = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# inference…
df = pd.read_csv(CSV_IN)
paths = df["path"].tolist()
preds = []
with torch.no_grad():
    for p in paths:
        img = Image.open(p).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        out = model(x)
        idx = out.argmax(dim=1).item()
        preds.append(CLASS_NAMES[idx])

# save
nums = [n for n in range(1, len(preds) + 1)]
pd.DataFrame({"annotation_id": nums, "concept_name": preds}).to_csv(
    CSV_OUT, index=False
)
print(f"Wrote predictions to {CSV_OUT}")
