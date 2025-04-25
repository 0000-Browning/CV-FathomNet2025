import random

from torchvision.transforms import functional as F


class MarineDataset(Dataset):

    def __init__(self, samples, mode="train", transform=None):
        """
        samples: list of tuples (img_path, bbox, label, ann_id) for train/val;
                 for test, label can be None and ann_id is provided.
        mode: 'train', 'val', or 'test' to control augmentations.
        transform: optional external transform pipeline (if provided, overrides internal transforms).
        """
        self.samples = samples
        self.mode = mode
        self.transform = transform
        # Define transforms for each mode
        if transform is None:
            if mode == "train":
                # Training transforms: augmentations
                self.transform = transforms.Compose(
                    [
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                        ),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(degrees=5),
                        transforms.Resize((Config.input_size, Config.input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],  # Using ImageNet mean/std
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                )
                # Note: We could add more custom transforms like blur or noise by extending this pipeline.
            else:
                # Validation or Test: no random augmentation, just resize and normalize
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((Config.input_size, Config.input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox = self.samples[idx][0], self.samples[idx][1]
        # Load image
        img = Image.open(img_path).convert("RGB")
        # Crop ROI from image
        # Bbox is [x, y, width, height] in COCO (probably in pixel coordinates)
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        img = img.crop((x, y, x + w, y + h))
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        # Prepare output
        if self.mode in ["train", "val"]:
            # label and ann_id are present in samples for train/val
            label = self.samples[idx][2]
            return img, label
        else:  # test
            ann_id = self.samples[idx][2]  # In test_samples, third element is ann_id
            return img, ann_id


# Create dataset and dataloaders
train_dataset = MarineDataset(train_data, mode="train")
val_dataset = MarineDataset(val_data, mode="val")
test_dataset = MarineDataset(test_samples, mode="test")

train_loader = DataLoader(
    train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4
)
test_loader = DataLoader(
    test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4
)
