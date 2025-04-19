import torch
import open_clip

# Configuration
MODEL_NAME = "hf-hub:imageomics/bioclip"

def print_model_layers():
    # Load the model
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
    model.eval()  # Set the model to evaluation mode

    # Print model layers, names, and sizes
    print("Model Layers, Names, and Sizes:")
    for name, param in model.named_parameters():
        print(f"Layer Name: {name}")
        print(f"Size: {param.size()}")
        print(f"Requires Grad: {param.requires_grad}")
        print("-" * 50)

if __name__ == "__main__":
    print_model_layers()