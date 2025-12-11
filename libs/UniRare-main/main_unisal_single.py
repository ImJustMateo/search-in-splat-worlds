import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model import load_model, load_dataloader, run_model
from src.UniRare import unirare


# -------------------------------
# Device
# -------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# -------------------------------
# Utilities
# -------------------------------

def to_numpy_map(tensor, target_shape):
    """Convert saliency/rarity maps to a normalized RGB heatmap."""
    arr = tensor.permute(1, 2, 0).detach().cpu().numpy()
    arr = cv2.resize(arr, target_shape)

    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    arr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    return arr


# -------------------------------
# Main processing
# -------------------------------

def run_unisal_on_image(image_path: str):
    device = get_device()
    print("➡️ Using device:", device)

    # Arguments mocks (to reuse existing model loaders)
    class Args:
        model = "Unisal"
        threshold = None
        layers = "3,4,5"

    args = Args()

    # Parse layers index
    layers_index = [[int(x)] for x in args.layers.split(",")]

    # Load rarity network
    rarity_model = unirare.RarityNetwork(threshold=args.threshold).to(device)

    # Load UniSal model + dataloader
    model = load_model(args, device).to(device)
    file_opener = load_dataloader(args)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Run saliency model
    saliency, layers = run_model(args, model, file_opener, image_path, device)

    # Run rarity model
    rarity_map, _ = rarity_model(layers_input=layers, layers_index=layers_index)

    # Fusion maps
    prod_map = rarity_model.prod_rarity(saliency, rarity_map)
    itti_map = rarity_model.fuse_rarity(saliency, rarity_map)

    # Convert maps to numpy for display
    target_shape = (img.shape[1], img.shape[0])  # width, height

    sal = to_numpy_map(saliency, target_shape)
    rar = to_numpy_map(rarity_map, target_shape)
    prod = to_numpy_map(prod_map, target_shape)
    itti = to_numpy_map(itti_map, target_shape)

    # Display results
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(sal)
    axs[0].set_title("Saliency (UniSal)")
    axs[0].axis("off")

    axs[1].imshow(rar)
    axs[1].set_title("Rarity Map")
    axs[1].axis("off")

    axs[2].imshow(prod)
    axs[2].set_title("Saliency × Rarity")
    axs[2].axis("off")

    axs[3].imshow(itti)
    axs[3].set_title("Itti Fusion")
    axs[3].axis("off")

    plt.show()


# -------------------------------
# Standalone main
# -------------------------------
if __name__ == "__main__":
    IMAGE_PATH = "images/"
    run_unisal_on_image(IMAGE_PATH)
