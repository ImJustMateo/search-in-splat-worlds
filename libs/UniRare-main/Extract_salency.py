import os
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Importations depuis votre structure de projet
from src.model import load_model, load_dataloader, run_model
from src.UniRare import unirare


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def process_map_bw(tensor_map, target_shape):
    """
    Convertit un tenseur de saillance en image Noir & Blanc (Niveaux de gris).
    """
    # Gestion des dimensions (B, C, H, W) ou (C, H, W) -> (H, W)
    if tensor_map.ndim == 4:
        tensor_map = tensor_map.squeeze(0)
    if tensor_map.ndim == 3:
        tensor_map = tensor_map.squeeze(0)

    # Conversion en numpy
    smap = tensor_map.detach().cpu().numpy()

    # Redimensionnement à la taille de l'image d'origine (W, H)
    target_h, target_w = target_shape[:2]
    smap = cv2.resize(smap, (target_w, target_h))

    # Normalisation 0-255
    smap = cv2.normalize(smap, None, 0, 255, cv2.NORM_MINMAX)
    smap = smap.astype(np.uint8)

    return smap


def extract_maps(image_path, model_name="Unisal", layers_str="3,4,5", threshold=None, output_dir=None):
    # 0. Vérification du fichier
    if not os.path.exists(image_path):
        print(f"ERREUR: L'image n'existe pas au chemin : {image_path}")
        return

    device = get_default_device()
    print(f"Device : {device}")

    # --- Configuration ---
    class Args:
        def __init__(self):
            self.model = model_name
            self.directory = os.path.dirname(image_path)
            self.layers = layers_str
            self.threshold = threshold

    args = Args()

    # --- Chargement ---
    print(f"Chargement modèle : {model_name}")
    model = load_model(args, device)
    model.to(device)
    model.eval()

    file_opener = load_dataloader(args)

    print("Chargement RarityNetwork...")
    rarity_net = unirare.RarityNetwork(threshold=args.threshold)
    rarity_net.to(device)
    rarity_net.eval()

    # --- Inférence ---
    print(f"Traitement : {image_path}")

    # 1. Carte du Modèle (Unisal, etc.)
    # run_model renvoie (saliency, layers)
    saliency_model, layers = run_model(args, model, file_opener, image_path, device)

    # Ajustement dimensions pour compatibilité
    if saliency_model.ndim == 2:
        saliency_model = saliency_model.unsqueeze(0)  # Devient (1, H, W)

    # 2. Carte de Rareté
    # Parsing des indices de couches
    try:
        layers_list_int = [int(x) for x in args.layers.split(',')]
        # Format attendu par RarityNetwork : liste de listes [[3], [4], [5]]
        layers_index = [[x] for x in layers_list_int]
        print(f"Couches utilisées pour la rareté : {layers_list_int}")
    except ValueError:
        print("Erreur : Le format des couches doit être des entiers séparés par des virgules (ex: '3,4,5').")
        return

    rarity_map, _ = rarity_net(layers_input=layers, layers_index=layers_index)

    if rarity_map.ndim == 2:
        rarity_map = rarity_map.unsqueeze(0)

    # 3. Carte Fusionnée (Itti)
    itti_map = rarity_net.fuse_rarity(saliency_model, rarity_map)

    # --- Traitement Noir & Blanc ---
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("Erreur lecture image originale (cv2).")
        return

    shape = original_img.shape

    img_model = process_map_bw(saliency_model, shape)
    img_rarity = process_map_bw(rarity_map, shape)
    img_itti = process_map_bw(itti_map, shape)

    # --- Affichage / Sauvegarde ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        path_model = os.path.join(output_dir, f"{base_name}_1_model_{model_name}.png")
        path_rarity = os.path.join(output_dir, f"{base_name}_2_rarity_L{layers_str.replace(',', '-')}.png")
        path_itti = os.path.join(output_dir, f"{base_name}_3_itti.png")

        cv2.imwrite(path_model, img_model)
        cv2.imwrite(path_rarity, img_rarity)
        cv2.imwrite(path_itti, img_itti)

        print(f"Images sauvegardées dans {output_dir} :")
        print(f" - {path_model}")
        print(f" - {path_rarity}")
        print(f" - {path_itti}")
    else:
        # Affichage Matplotlib dans 3 figures distinctes

        plt.figure(f"Carte Modèle ({model_name})", figsize=(6, 6))
        plt.imshow(img_model, cmap='gray')
        plt.title(f"Saliency ({model_name})")
        plt.axis('off')

        plt.figure(f"Carte Rareté (Couches {layers_str})", figsize=(6, 6))
        plt.imshow(img_rarity, cmap='gray')
        plt.title("Rarity")
        plt.axis('off')

        plt.figure("Carte Itti (Fusion)", figsize=(6, 6))
        plt.imshow(img_itti, cmap='gray')
        plt.title("Fusion Itti")
        plt.axis('off')

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraction de cartes de saillance (Modèle, Rareté, Fusion Itti) en N&B.")

    parser.add_argument("--img", type=str, required=True,
                        help="Chemin de l'image d'entrée.")

    parser.add_argument("--out", type=str, default=None,
                        help="Dossier de sortie. Si non spécifié, affiche les résultats à l'écran.")

    parser.add_argument("--model", type=str, default="Unisal",
                        help="Modèle de saillance à utiliser. Options disponibles : 'Unisal', 'TranSalNetDense', 'TranSalNetRes', 'TempSal'.")

    parser.add_argument("--layers", type=str, default="3,4,5",
                        help="Indices des couches pour le calcul de la rareté, séparés par des virgules (ex: '3,4,5' ou '4,5').")

    args = parser.parse_args()

    extract_maps(
        image_path=args.img,
        model_name=args.model,
        layers_str=args.layers,
        output_dir=args.out
    )