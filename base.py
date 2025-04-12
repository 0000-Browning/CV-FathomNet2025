import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split

class FathomNetDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

        # Create a label-to-index mapping for classification
        self.classes = sorted(self.annotations["label"].unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations.iloc[idx]["path"]
        image = Image.open(img_path).convert("RGB")

        label_str = self.annotations.iloc[idx]["label"]
        label = self.class_to_idx[label_str]  # convert label to integer

        if self.transform:
            image = self.transform(image)

        return image, label


# add main protection
if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = FathomNetDataset(
        csv_file="data/annotations.csv",
        transform=transform,
    )

    # Split the dataset into training and validation sets
    train_indices, val_indices = train_test_split(
        np.arange(len(full_dataset)), test_size=0.2, random_state=42, stratify=full_dataset.annotations["label"]
    )
    # make sure subsets are same format as full dataset
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=4)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # exit if no GPU is available
    if device.type == "cpu":
        print("No GPU available. Exiting.")
        exit()

    resnet50 = models.resnet50()
    num_classes = len(set(full_dataset.annotations["label"]))
    resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)
    resnet50 = resnet50.to(device)

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet50.parameters(), lr=0.001)

    best_acc = 0.0
    patience = 3
    epochs_no_improve = 0
    early_stop = False
    epochs = 1

    for epoch in range(epochs):  # adjust # of epochs
        resnet50.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = resnet50(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    torch.save(resnet50.state_dict(), "resnet50_fathomnet.pth")

