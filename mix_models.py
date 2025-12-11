import sys
import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- CONFIGURATION DES CHEMINS ---
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'UniRare-main'))
sys.path.append(os.path.join(cwd, 'scanDDM-main'))

# --- IMPORTS UNI-RARE ---
try:
    from src.model import load_model, load_dataloader, run_model
    from src.UniRare import unirare
except ImportError:
    print("ERREUR : Impossible d'importer UniRare.")
    sys.exit(1)

# --- IMPORTS SCANDDM ---
try:
    from scanDDM import scanDDM
    from zs_clip_seg import get_obj_map
    from vis import compute_density_image
except ImportError:
    print("ERREUR : Impossible d'importer scanDDM.")
    sys.exit(1)

# --- IMPORT METRICS (Optionnel) ---
try:
    from metrics import ScanEvaluator
except ImportError:
    ScanEvaluator = None
    print("Note : 'metrics.py' introuvable. L'évaluation ne sera pas possible même si demandée.")


def get_device():
    """Détection automatique du meilleur device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print(">>> Accélération macOS Metal (MPS) activée.")
        return torch.device("mps")
    else:
        return torch.device("cpu")


def normalize_map(smap):
    smap_min = smap.min()
    smap_max = smap.max()
    if smap_max - smap_min < 1e-8:
        return smap
    return (smap - smap_min) / (smap_max - smap_min)


# ==========================================
# 1. FONCTION D'EXTRACTION UNI-RARE
# ==========================================
def get_unirare_map(image_path, model_name="Unisal", layers_str="3,4,5", threshold=None, device=None):
    print(f"--- [UniRare] Calcul... ---")

    class Args:
        def __init__(self):
            self.model = model_name
            self.directory = os.path.dirname(image_path)
            self.layers = layers_str
            self.threshold = threshold

    args = Args()

    # Chargement
    model = load_model(args, device)
    model.to(device).eval()
    file_opener = load_dataloader(args)

    rarity_net = unirare.RarityNetwork(threshold=args.threshold)
    rarity_net.to(device).eval()

    # Inférence
    saliency, layers = run_model(args, model, file_opener, image_path, device)

    # Gestion dimensions tenseur
    if saliency.ndim == 2: saliency = saliency.unsqueeze(0)

    # Rareté
    layers_index = [[int(x)] for x in args.layers.split(',')]
    rarity_map, _ = rarity_net(layers_input=layers, layers_index=layers_index)

    if rarity_map.ndim == 2: rarity_map = rarity_map.unsqueeze(0)

    # Fusion Itti
    final_map_tensor = rarity_net.fuse_rarity(saliency, rarity_map)

    if final_map_tensor.ndim == 3: final_map_tensor = final_map_tensor.squeeze(0)

    # Retourne la carte brute (numpy), le redimensionnement final se fera dans le main
    return normalize_map(final_map_tensor.detach().cpu().numpy())


# ==========================================
# 2. FONCTION D'EXTRACTION CLIPSEG
# ==========================================
def get_clipseg_map(image, prompt):
    print(f"--- [ScanDDM] Calcul CLIPSeg pour '{prompt}' ---")
    raw_map = get_obj_map(image, [prompt])
    return normalize_map(raw_map)


# ==========================================
# 3. PIPELINE PRINCIPAL
# ==========================================
def run_mixed_simulation(image_path, prompt, alpha, n_obs, out_dir, bbox_path=None, target_class_id=0):
    """
    Lance la simulation mixte.
    Si bbox_path est fourni, lance l'évaluation. Sinon, ignore cette étape.
    """
    # Gestion fichiers cachés
    if not os.path.exists(image_path) or os.path.basename(image_path).startswith('.'):
        print(f"Fichier ignoré ou introuvable : {image_path}")
        return

    device = get_device()

    # --- A. Chargement Image ---
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Erreur lecture image.")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]
    print(f"Dimensions image : {orig_w}x{orig_h} px")

    # --- B. Calcul des cartes (GPU/MPS si dispo) ---
    map_u_raw = get_unirare_map(image_path, model_name="Unisal", device=device)
    map_u = cv2.resize(map_u_raw, (orig_w, orig_h))

    map_s_raw = get_clipseg_map(img_rgb, prompt)
    map_s = cv2.resize(map_s_raw, (orig_w, orig_h))

    # --- C. Fusion ---
    print(f"--- Fusion (Alpha = {alpha}) ---")
    mixed_map = alpha * map_u + (1 - alpha) * map_s
    mixed_map = normalize_map(mixed_map)

    # --- D. Préparation Simulation ---
    # Force CPU pour ScanDDM (stabilité macOS)
    simulation_device = "cpu"
    print(f"--- Simulation ScanDDM sur {simulation_device.upper()} ---")

    # Création du tenseur sur le CPU directement
    saliency_tensor = torch.tensor(mixed_map, dtype=torch.float32, device=simulation_device)
    saliency_tensor = saliency_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # --- E. Lancement Simulation ---
    model = scanDDM(
        experiment_dur=2.0, fps=25, threshold=1.0, noise=7, kappa=10, eta=17,
        device=simulation_device
    )

    scans, _ = model.simulate_scanpaths(
        n_observers=n_obs,
        image=img_rgb,
        saliency_map=saliency_tensor,
        prompt=[prompt]
    )

    # --- F. Vérification des résultats ---
    if not scans or len(scans) == 0:
        print("ERREUR CRITIQUE : Aucun scanpath généré !")
        return

    avg_fix = np.mean([len(s) for s in scans])
    print(f"Succès : {len(scans)} scanpaths générés (Moyenne de {avg_fix:.1f} fixations/scan).")

    # --- G. Évaluation (Optionnelle) ---
    accuracy_text = ""
    if bbox_path is not None:
        if ScanEvaluator is not None:
            print(f"\n--- Lancement de l'évaluation avec {bbox_path} ---")
            evaluator = ScanEvaluator(
                yolo_file_path=bbox_path,
                img_shape=(orig_h, orig_w),
                target_class_id=target_class_id
            )
            results = evaluator.evaluate(scans)
            evaluator.print_report(results)

            if results:
                accuracy_text = f" (Acc: {results['accuracy']:.1f}%)"
        else:
            print("\nATTENTION : Vous avez demandé une évaluation mais 'metrics.py' est manquant.")

    # --- H. Densité & Sauvegarde ---
    all_scans = np.vstack(scans)
    density_map = compute_density_image(all_scans[:, :2], size=(orig_h, orig_w))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(image_path).split('.')[0]

        cv2.imwrite(os.path.join(out_dir, f"{base}_1_UniRare.png"), (map_u * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(out_dir, f"{base}_2_CLIPSeg.png"), (map_s * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(out_dir, f"{base}_3_Mix_a{alpha}.png"), (mixed_map * 255).astype(np.uint8))

        d_norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{base}_4_Result.png"), d_norm)

        print(f"Résultats sauvegardés dans : {out_dir}")
    else:
        # --- NOUVEAU PLOT COMPLET ---
        plt.figure(figsize=(18, 10))

        # Titre global avec le PROMPT et éventuellement l'Accuracy
        plt.suptitle(f"Résultats pour le prompt : '{prompt}' (Alpha={alpha}, Obs={n_obs}){accuracy_text}", fontsize=16,
                     fontweight='bold')

        # 1. Image originale avec Alpha
        plt.subplot(2, 3, 1)
        plt.imshow(img_rgb)
        plt.title(f"1. Image Initiale")
        plt.axis('off')

        # 2. UniRare Map
        plt.subplot(2, 3, 2)
        plt.imshow(map_u, cmap='gray')
        plt.title("2. UniRare (Bottom-Up)")
        plt.axis('off')

        # 3. CLIPSeg Map
        plt.subplot(2, 3, 3)
        plt.imshow(map_s, cmap='gray')
        plt.title("3. CLIPSeg (Top-Down)")
        plt.axis('off')

        # 4. Map Résultante (Input Mix)
        plt.subplot(2, 3, 4)
        plt.imshow(mixed_map, cmap='gray')
        plt.title(f"4. Mixed Map (Input, Alpha={alpha})")
        plt.axis('off')

        # 5. Output Density (Carte de saillance finale)
        plt.subplot(2, 3, 5)
        plt.imshow(density_map, cmap='jet')
        plt.title("5. ScanDDM Output Density")
        plt.axis('off')

        # 6. Scanpaths sur image
        plt.subplot(2, 3, 6)
        plt.imshow(img_rgb)
        for s in scans:
            plt.plot(s[:, 0], s[:, 1], 'y-', alpha=0.3)  # Scanpaths jaunes transparents
        plt.plot(scans[0][:, 0], scans[0][:, 1], 'r-o', alpha=0.8, markersize=3)  # Premier scan rouge
        plt.title(f"6. Scanpaths")
        plt.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Laisse de la place pour le suptitle
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Chemin de l'image")
    parser.add_argument("--prompt", type=str, required=True, help="Objet à chercher")
    parser.add_argument("--alpha", type=float, default=0.5, help="0.0=ScanDDM pur, 1.0=UniRare pur")
    parser.add_argument("--obs", type=int, default=30, help="Nombre d'observateurs")
    parser.add_argument("--out", type=str, default=None, help="Dossier de sortie")

    # Nouveaux arguments optionnels pour l'évaluation
    parser.add_argument("--bbox", type=str, default=None,
                        help="(Optionnel) Chemin du fichier YOLO .txt pour évaluation")
    parser.add_argument("--target_id", type=int, default=0, help="(Optionnel) ID de la classe cible YOLO (défaut: 0)")

    args = parser.parse_args()

    run_mixed_simulation(
        image_path=args.img,
        prompt=args.prompt,
        alpha=args.alpha,
        n_obs=args.obs,
        out_dir=args.out,
        bbox_path=args.bbox,  # Passé uniquement si spécifié
        target_class_id=args.target_id
    )