import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm

# Logging setup
log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("zero_shot_training")

# Configuration
MODEL_NAME = "hf-hub:imageomics/bioclip"
MODEL_SAVE_PATH = "bioclip_finetuned.pth"

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Path to the dataset directory in ImageFolder format.",
    )
    parser.add_argument("--logs", type=str, default="./logs", help="Directory to save logs and results.")
    parser.add_argument("--exp", type=str, default="bioclip-training", help="Experiment name.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args(args)

def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load BioCLIP model and preprocessing pipeline
    logger.info("Loading BioCLIP model...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)
    model = model.to(device)

    # Freeze all layers except the classification head and visual projection layers
    for name, param in model.named_parameters():
        if "logit_scale" in name or "text_projection" in name or "visual.proj" in name:
            param.requires_grad = True  # Allow gradients for these layers
        else:
            param.requires_grad = False  # Freeze all other layers

    # Log trainable parameters
    logger.info("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name}: {param.shape}")

    # Load dataset using ImageFolder
    logger.info("Loading dataset...")
    dataset = datasets.ImageFolder(args.datasets, transform=preprocess_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()  # Ensure the model is in training mode
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            if isinstance(outputs, tuple):  # Ensure outputs are handled correctly
                logits = outputs[0]
            else:
                logits = outputs
            loss = criterion(logits, labels)

            # # Debugging: Check the logits tensor
            # logger.info(f"logits.requires_grad: {logits.requires_grad}")
            # logger.info(f"logits.grad_fn: {logits.grad_fn}")

            # # Debugging: Check the loss tensor
            # logger.info(f"loss.requires_grad: {loss.requires_grad}")
            # logger.info(f"loss.grad_fn: {loss.grad_fn}")

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        logger.info(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # Save the fine-tuned model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Fine-tuned model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
