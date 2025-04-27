from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm  
import matplotlib.pyplot as plt


# MODEL_PATH = input("Enter model path: ")
# CONFIG
MODEL_PATH = "models/resnet50_fathomnet_SGD_optimizer.pth"
CSV_IN = "data/annotations.csv"
CSV_OUT = "resnet/resnet50_blur_sgd.csv"
IMG_SIZE = (224, 224)

# get class names
df_train    = pd.read_csv("data/annotations_train.csv")
CLASS_NAMES = sorted(df_train["label"].unique())


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
    for p in tqdm(paths, desc="Running inference", unit="image"):
        img = Image.open(p).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        out = model(x)
        idx = out.argmax(dim=1).item()
        preds.append(CLASS_NAMES[idx])

# Generate histogram of predictions
pred_counts = pd.Series(preds).value_counts(sort=False)

# Sort the predictions alphabetically by class name
pred_counts = pred_counts.sort_index()

# Plot the histogram
plt.bar(pred_counts.index, pred_counts.values, tick_label=pred_counts.index)
plt.xlabel("Class Names")
plt.ylabel("Frequency")
plt.title("Distribution of Predictions (ResNet-50 w/SGD + Blur)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# save
nums = [n for n in range(1, len(preds) + 1)]
pd.DataFrame({"annotation_id": nums, "concept_name": preds}).to_csv(
    CSV_OUT, index=False
)
print(f"Wrote predictions to {CSV_OUT}")