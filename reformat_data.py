import os
import pandas as pd
from shutil import move

# Paths
data_dir = "c:/Users/jake3/Desktop/CV Repo/CV-FathomNet2025/data"
csv_file = os.path.join(data_dir, "annotations.csv")
images_dir = os.path.join(data_dir, "images")
rois_dir = os.path.join(data_dir, "rois")  # Add the rois directory

# Read annotations.csv
annotations = pd.read_csv(csv_file)

# Ensure the 'path' column contains only filenames
annotations["path"] = annotations["path"].apply(os.path.basename)

# Create subdirectories for each class
for label in annotations["label"].unique():
    class_dir = os.path.join(data_dir, label)
    os.makedirs(class_dir, exist_ok=True)

# Move images to their respective class directories
for _, row in annotations.iterrows():
    image_name = row["path"]
    label = row["label"]
    src_path_images = os.path.join(images_dir, image_name)
    src_path_rois = os.path.join(rois_dir, image_name)
    dst_path = os.path.join(data_dir, label, image_name)

    # Check if the image exists in either the images or rois directory
    if os.path.exists(src_path_images):
        move(src_path_images, dst_path)
    elif os.path.exists(src_path_rois):
        move(src_path_rois, dst_path)
    else:
        print(f"Image not found in either directory: {image_name}")

print("Data reorganized into ImageFolder format.")