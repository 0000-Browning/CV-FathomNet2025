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
    # Define the transformation pipeline for preprocessing images
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize using ImageNet stats
        ]
    )

    # Load the full dataset using the custom FathomNetDataset class
    full_dataset = FathomNetDataset(
        csv_file="data/annotations.csv",  # Path to the CSV file containing annotations
        transform=transform,  # Apply the transformation pipeline
    )

    # Split the dataset into training and validation sets
    train_indices, val_indices = train_test_split(
        np.arange(len(full_dataset)),  # Indices of the dataset
        test_size=0.2,  # 20% of the data will be used for validation
        random_state=42,  # Ensure reproducibility
        stratify=full_dataset.annotations["label"],  # Stratify by label to maintain class distribution
    )

    # Create subsets for training and validation
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    # Create DataLoader for training data
    train_loader = DataLoader(
        train_dataset,  # Training dataset
        batch_size=50,  # Number of samples per batch
        shuffle=True,  # Shuffle the data for each epoch
        num_workers=4,  # Number of worker threads for data loading
    )

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Exit the program if no GPU is available
    if device.type == "cpu":
        print("No GPU available. Exiting.")
        exit()

    # Load the ResNet-50 model
    resnet50 = models.resnet50()

    # Modify the final fully connected layer to match the number of classes
    num_classes = len(set(full_dataset.annotations["label"]))  # Number of unique labels
    resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)  # Replace the FC layer
    resnet50 = resnet50.to(device)  # Move the model to the selected device

    # Define the loss function and optimizer
    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = optim.Adam(resnet50.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

    # Initialize variables for early stopping
    best_acc = 0.0
    patience = 3 # Number of epochs with no improvement before stopping
    epochs_no_improve = 0
    early_stop = False
    epochs = 20  # Number of epochs to train

    # Training loop
    for epoch in range(epochs):
        resnet50.train()  # Set the model to training mode
        running_loss = 0.0  # Track the cumulative loss
        correct = 0  # Track the number of correct predictions
        total = 0  # Track the total number of samples

        # Iterate over batches of training data
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
        for images, labels in loop:
            # Move images and labels to the selected device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = resnet50(images)
            loss = criterion(outputs, labels)  # Compute the loss

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

            # Update running loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class
            total += labels.size(0)  # Update total samples
            correct += (predicted == labels).sum().item()  # Update correct predictions

        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        # Print epoch metrics
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # Save the trained model's weights
    torch.save(resnet50.state_dict(), "resnet50_fathomnet.pth")

    # validation loop
    resnet50.eval()  # Set the model to evaluation mode
    val_loader = DataLoader(
        val_dataset,  # Validation dataset
        batch_size=50,  # Number of samples per batch
        shuffle=False,  # Do not shuffle validation data
        num_workers=4,  # Number of worker threads for data loading
    )

    val_loss = 0.0  # Track the cumulative validation loss
    val_correct = 0  # Track the number of correct predictions in validation

    val_total = 0  # Track the total number of validation samples

    with torch.no_grad():  # Disable gradient calculation for validation
        for images, labels in val_loader:
            # Move images and labels to the selected device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = resnet50(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()  # Update validation loss
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
        
        #calculate validation accuracy
        val_acc = 100 * val_correct / val_total
        # calculate validation loss
        val_loss /= len(val_loader)
        # Print validation metrics
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")



