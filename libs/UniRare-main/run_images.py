
import os
import cv2
import numpy as np
import argparse
import time
from typing import Dict, List, Tuple, Any

import torch
import matplotlib.pyplot as plt

from src.utils import metrics
from src.UniRare import unirare
from src.utils import helper
from src.model import load_model
from src.model import load_dataloader
from src.model import run_model


def get_default_device() -> torch.device:
    """Get the default device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Parameters test unisal video/image")
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=None, 
        help="Threshold for torch rare 2021"
    )
    parser.add_argument(
        "--directory", 
        type=str, 
        default="./images/",
        help="path directory images"
    ) 
    parser.add_argument(
        "--layers",
        type=str,
        default="3,4,5",
        help="layers to use, comma separated"
    )
    return parser


def get_image_files(directory: str) -> List[str]:
    """Get list of image files from directory."""
    files = os.listdir(directory)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return [f for f in files if os.path.splitext(f.lower())[1] in image_extensions]


def parse_layers_index(layers_str: str) -> List[List[int]]:
    """Parse layers string into list of layer indices."""
    return [[int(x)] for x in layers_str.split(',')]


def process_single_model(
    model_name: str, 
    args: argparse.Namespace, 
    rarity_model: Any, 
    image_path: str, 
    layers_index: List[List[int]], 
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Process a single model and return saliency maps."""
    args.model = model_name
    
    # Load model
    model = load_model(args, device)
    file_opener = load_dataloader(args)
    
    # Move models to device
    rarity_model = rarity_model.to(device)
    model = model.to(device)
    
    # Run model
    start_time = time.time()
    saliency, layers = run_model(args, model, file_opener, image_path, device)
    
    # Run rarity network
    rarity_map, groups = rarity_model(
        layers_input=layers,
        layers_index=layers_index
    )
    process_time = time.time() - start_time
    
    # Create maps dictionary
    maps = {
        'saliency': saliency,
        'rarity': rarity_map,
        'saliency_Add': rarity_model.add_rarity(saliency, rarity_map),
        'saliency_Sub': rarity_model.sub_rarity(saliency, rarity_map),
        'saliency_Prod': rarity_model.prod_rarity(saliency, rarity_map),
        'saliency_Itti': rarity_model.fuse_rarity(saliency, rarity_map)
    }
    
    return maps


def process_saliency_map(
    saliency_map: torch.Tensor, 
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """Process a saliency map for visualization."""
    # Convert to numpy and resize
    processed_map = saliency_map.permute(1, 2, 0).detach().cpu().numpy()
    processed_map = cv2.resize(processed_map, target_shape)
    
    # Normalize and convert to uint8
    processed_map = cv2.normalize(processed_map, None, 0, 255, cv2.NORM_MINMAX)
    processed_map = processed_map.astype(np.uint8)
    
    # Apply colormap and convert color space
    processed_map = cv2.applyColorMap(processed_map, cv2.COLORMAP_JET)
    processed_map = cv2.cvtColor(processed_map, cv2.COLOR_BGR2RGB)
    
    return processed_map


def blend_with_image(
    img: np.ndarray, 
    saliency_map: np.ndarray, 
    alpha: float = 0.4
) -> np.ndarray:
    """Blend saliency map with original image."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, alpha, saliency_map, 1 - alpha, 0)


def visualize_results(
    img: np.ndarray, 
    all_maps: Dict[str, Dict[str, torch.Tensor]], 
    models: List[str]
) -> None:
    """Create and display visualization of all results."""
    fig, axs = plt.subplots(len(models), 5, figsize=(20, 10))
    alpha = 0.4
    target_shape = (img.shape[1], img.shape[0])
    
    for i, model_name in enumerate(models):
        maps = all_maps[model_name]
        
        # Original image
        axs[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i, 0].set_title(f'{model_name} Original Image')
        axs[i, 0].axis('off')
        
        # Process each map type
        map_types = ['saliency', 'saliency_Add', 'saliency_Prod', 'saliency_Itti']
        titles = ['Saliency', 'Saliency + Rarity', 'Saliency * Rarity', 'Saliency Itti']
        
        for j, (map_type, title) in enumerate(zip(map_types, titles)):
            processed_map = process_saliency_map(maps[map_type], target_shape)
            blended = blend_with_image(img, processed_map, alpha)
            
            axs[i, j + 1].imshow(blended)
            axs[i, j + 1].set_title(f'{model_name} {title}')
            axs[i, j + 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def process_single_image(
    image_path: str,
    models: List[str],
    rarity_model: Any,
    args: argparse.Namespace,
    layers_index: List[List[int]],
    device: torch.device
) -> None:
    """Process a single image with all models."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    all_maps = {}
    
    # Process each model
    for model_name in models:
        maps = process_single_model(
            model_name, args, rarity_model, image_path, layers_index, device
        )
        all_maps[model_name] = maps
    
    # Visualize results
    visualize_results(img, all_maps, models)


def main() -> None:
    """Main function to run the image processing pipeline."""
    # Setup
    device = get_default_device()
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configuration
    models = ["Unisal", "TranSalNetDense", "TranSalNetRes", "TempSal"]
    layers_index = parse_layers_index(args.layers)
    
    # Initialize rarity network
    rarity_model = unirare.RarityNetwork(threshold=args.threshold)
    
    # Get image files
    files = get_image_files(args.directory)
    start_time_global = time.time()
    
    # Process each image
    for index, filename in enumerate(files):
        image_path = os.path.join(args.directory, filename)
        print(f"Processing image {index + 1}/{len(files)}: {filename}")
        
        process_single_image(
            image_path, models, rarity_model, args, layers_index, device
        )


if __name__ == "__main__":
    main()