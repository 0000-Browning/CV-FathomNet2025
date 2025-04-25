import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm  
import matplotlib.pyplot as plt

from hierarchyEncoding.hiearchy import (
    build_raw_paths,
    build_rank2idx,
    HierarchicalTaxonEncoder,
    HierarchyAwareModel,
)

# 1) Config
MODEL_PATH = "models/best_hier_model.pth"
CSV_IN     = "testdata/annotations.csv"
CSV_OUT    = "hierarchyEncoding/hier.csv"
IMG_SIZE   = (224, 224)
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Recover class names directly from your training CSV
df_train    = pd.read_csv("traindata/annotations.csv")
CLASS_NAMES = sorted(df_train["label"].unique())

# 3) Rebuild taxonomy mappings exactly as you did during training
raw_paths = build_raw_paths(CLASS_NAMES)      # <— you must call this!
rank2idx  = build_rank2idx(raw_paths)         # now raw_paths is defined

# 4) Instantiate and load your model
tax_encoder = HierarchicalTaxonEncoder(rank2idx, embed_dim=32)
model       = HierarchyAwareModel(len(CLASS_NAMES), tax_encoder)
state       = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model = model.to(device).eval()

# 5) Prepare a dummy all-'UNK' path tensor for inference
unk_path = torch.tensor(
    [[ rank2idx[i]["UNK"] for i in range(len(rank2idx)) ]],
    dtype=torch.long,
    device=device
)

# 6) Image preprocessor
tf = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# 7) Run inference with a loading bar
df    = pd.read_csv(CSV_IN)
preds = []
with torch.no_grad():
    for p in tqdm(df["path"], desc="Running inference", unit="image"):
        img = Image.open(p).convert("RGB")
        x   = tf(img).unsqueeze(0).to(device)  # shape (1,3,224,224)
        out = model(x, unk_path)              # shape (1, num_classes)
        idx = out.argmax(dim=1).item()
        preds.append(CLASS_NAMES[idx])


# Generate histogram of predictions
pred_counts = pd.Series(preds).value_counts(sort=False)
plt.bar(pred_counts.index, pred_counts.values, tick_label=pred_counts.index)
plt.xlabel("Class Names")
plt.ylabel("Frequency")
plt.title("Histogram of Predictions")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 8) Save results
pd.DataFrame({
    "annotation_id": range(1, len(preds)+1),
    "concept_name":  preds
}).to_csv(CSV_OUT, index=False)

print(f"Wrote predictions to {CSV_OUT}")

