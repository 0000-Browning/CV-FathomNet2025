import argparse

import open_clip
import pandas as pd
import torch
from PIL import Image

from base import FathomNetDataset

# use this to run
# python bioclip_test.py \
#   --model-path bioclip_finetuned.pth \
#   --train-csv train/annotations.csv \
#   --csv-in test/annotations.csv \
#   --csv-out test/bioclip.csv
#


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path", type=str, required=True, help="Path to bioclip_finetuned.pth"
    )
    p.add_argument(
        "--train-csv", type=str, required=True, help="Train CSV for FathomNetDataset"
    )
    p.add_argument(
        "--csv-in", type=str, required=True, help="Test CSV (must have a 'path' column)"
    )
    p.add_argument(
        "--csv-out", type=str, required=True, help="Where to write submission CSV"
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="hf-hub:imageomics/bioclip",
        help="open_clip model spec you trained",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load the BioCLIP model + transforms
    model, _, preprocess_val = open_clip.create_model_and_transforms(args.model_name)
    model = model.to(device)
    # load your fine‑tuned weights (this updates visual.proj / text_projection)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 2) Get your 79 class names exactly as at train time
    train_ds = FathomNetDataset(csv_file=args.train_csv)
    class_names = train_ds.classes  # list of 79 strings, in sorted order

    # 3) Tokenize & encode all class names (once!)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    text_tokens = tokenizer(class_names).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # 4) Run zero‑shot inference over your test set
    df = pd.read_csv(args.csv_in)
    preds = []
    logit_scale = model.logit_scale.exp().item()  # CLIP’s learned temperature
    with torch.no_grad():
        for img_path in df["path"]:
            img = Image.open(img_path).convert("RGB")
            x = preprocess_val(img).unsqueeze(0).to(device)

            # encode & normalize image
            img_feats = model.encode_image(x)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            # compute similarity & pick highest
            logits = logit_scale * img_feats @ text_feats.T
            idx = logits.argmax(dim=-1).item()
            preds.append(class_names[idx])

    # 5) Write out Kaggle‐style CSV
    submission = pd.DataFrame(
        {"annotation_id": range(1, len(preds) + 1), "concept_name": preds}
    )
    submission.to_csv(args.csv_out, index=False)
    print(f"Wrote predictions to {args.csv_out}")


if __name__ == "__main__":
    main()
