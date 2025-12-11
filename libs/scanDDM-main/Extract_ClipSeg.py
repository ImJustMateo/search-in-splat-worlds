import argparse
import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

# Importation spécifique à ScanDDM
try:
    from zs_clip_seg import get_obj_map
except ImportError:
    print("ERREUR : Ce script doit être placé dans le dossier 'scanDDM-main/'.")
    exit(1)


def process_map_bw(activation_map, target_shape):
    """
    Normalise la carte d'activation en image 0-255 (Niveaux de gris).
    """
    # activation_map est généralement une matrice numpy de floats

    # Redimensionnement à la taille de l'image d'origine (W, H)
    # target_shape provient de img.shape -> (H, W, C)
    target_h, target_w = target_shape[:2]

    # Redimensionnement
    smap = cv2.resize(activation_map, (target_w, target_h))

    # Normalisation Min-Max pour bien utiliser toute la dynamique 0-255
    min_val = np.min(smap)
    max_val = np.max(smap)

    if max_val - min_val > 1e-5:
        smap = (smap - min_val) / (max_val - min_val)
    else:
        smap = np.zeros_like(smap)

    smap = (smap * 255).astype(np.uint8)

    return smap


def extract_clipseg_map(image_path, prompt, output_dir=None):
    if not os.path.exists(image_path):
        print(f"ERREUR: Image introuvable : {image_path}")
        return

    print(f"Extraction carte CLIPSeg pour le prompt : '{prompt}'")

    # --- 1. Chargement et Préparation Image ---
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Logique de redimensionnement identique à ScanDDM (main.py)
    # pour garantir que le modèle voit la même chose
    typical_shape = (768, 1024)
    h, w, c = img_rgb.shape

    # ScanDDM redimensionne souvent pour standardiser l'entrée de CLIP
    if h > w:
        img_model = cv2.resize(img_rgb, (typical_shape[0], typical_shape[1]))
    else:
        img_model = cv2.resize(img_rgb, (typical_shape[1], typical_shape[0]))

    # --- 2. Extraction de la carte (CLIPSeg) ---
    print("Calcul de l'attention (get_obj_map)...")
    # get_obj_map renvoie une matrice numpy brute des scores de similarité
    raw_map = get_obj_map(img_model, [prompt])  # prompt doit être une liste parfois, ou str selon version

    # --- 3. Post-traitement ---
    # On remet à la taille de l'image originale fournie en entrée
    final_bw_map = process_map_bw(raw_map, img_bgr.shape)

    # --- 4. Sauvegarde / Affichage ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        sanitized_prompt = prompt.replace(" ", "_")[:20]

        filename = f"{base_name}_CLIPSeg_{sanitized_prompt}.png"
        output_path = os.path.join(output_dir, filename)

        cv2.imwrite(output_path, final_bw_map)
        print(f"Carte CLIPSeg sauvegardée : {output_path}")
    else:
        plt.figure("CLIPSeg Attention", figsize=(6, 6))
        plt.imshow(final_bw_map, cmap='gray')
        plt.title(f"CLIPSeg : {prompt}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraction directe de la carte d'attention CLIPSeg (sans simulation)")
    parser.add_argument("--img", type=str, required=True, help="Chemin de l'image")
    parser.add_argument("--prompt", type=str, required=True, help="Description de l'objet (ex: 'cup', 'person')")
    parser.add_argument("--out", type=str, default=None, help="Dossier de sortie (optionnel)")

    args = parser.parse_args()

    extract_clipseg_map(
        image_path=args.img,
        prompt=args.prompt,
        output_dir=args.out
    )