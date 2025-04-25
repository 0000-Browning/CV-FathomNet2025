from sklearn.model_selection import train_test_split

# Load COCO annotations from JSON
with open(Config.train_ann_file, "r") as f:
    train_ann_data = json.load(f)
with open(Config.test_ann_file, "r") as f:
    test_ann_data = json.load(f)

# Create category ID to name mapping, and vice versa
cat_id_to_name = {cat["id"]: cat["name"] for cat in train_ann_data["categories"]}
cat_name_to_id = {name: cid for cid, name in cat_id_to_name.items()}

# Prepare list of training samples (annotation-centric).
train_samples = []  # each element: (image_path, bbox, label_id, annotation_id)
for ann in train_ann_data["annotations"]:
    img_id = ann["image_id"]
    # Find corresponding image info (COCO structure might require a lookup)
    # Let's create an image_id -> file_name dict for efficiency.
    # We do this once outside loop for all images:
train_images_info = {img["id"]: img for img in train_ann_data["images"]}
train_samples = []
for ann in train_ann_data["annotations"]:
    img_info = train_images_info[ann["image_id"]]
    file_name = img_info["file_name"]
    # Construct full path to image file
    img_path = os.path.join(Config.images_dir, file_name)
    bbox = ann["bbox"]  # format [x, y, width, height]
    cat_id = ann["category_id"]
    # Use 0-indexed class label for PyTorch
    label = cat_id - 1  # assuming category ids are 1..79
    train_samples.append((img_path, bbox, label, ann["id"]))

# Similar preparation for test samples (without labels)
test_images_info = {img["id"]: img for img in test_ann_data["images"]}
test_samples = []
for ann in test_ann_data["annotations"]:
    img_info = test_images_info[ann["image_id"]]
    file_name = img_info["file_name"]
    img_path = os.path.join(Config.images_dir, file_name)
    bbox = ann["bbox"]
    ann_id = ann["id"]
    test_samples.append((img_path, bbox, ann_id))

# Split train_samples into train and val (stratified by label)
train_labels = [s[2] for s in train_samples]
train_indices, val_indices = train_test_split(
    list(range(len(train_samples))),
    test_size=0.2,
    stratify=train_labels,
    random_state=Config.seed,
)
train_indices = set(train_indices)
val_indices = set(val_indices)

train_data = [train_samples[i] for i in train_indices]
val_data = [train_samples[i] for i in val_indices]

print(f"Total training samples: {len(train_data)}, validation samples: {len(val_data)}")
