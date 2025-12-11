import numpy as np
import matplotlib.pyplot as plt
from scanDDM import scanDDM
from vis import draw_scanpath, compute_density_image
import seaborn as sns
import cv2
import pandas as pd
import os

# --- IMPORT DU NOUVEAU MODULE ---
# Assurez-vous que metrics.py est dans le même dossier ou dans scanDDM-main
from metrics import ScanEvaluator

sns.set_context("talk")

# --------------------------------------------------------------------
# CONFIGURATION GLOBALE
# --------------------------------------------------------------------
# Chemins
IMG_PATH = "data/1.jpg"
# Le fichier YOLO original (normalisé, pas besoin de le convertir avant)
YOLO_PATH = "database_annoted_YOLO_normalisé/1.txt"
OUTPUT_DIR = "scanpaths_by_observer"

# Paramètres de l'expérience
TARGET_PROMPT = ["Duck"]
TARGET_CLASS_ID = 0  # L'ID qui correspond à "Duck" dans votre fichier YOLO (ex: 0)

# Paramètres Modèle
FPS = 25
EXP_DUR = 2.0
N_OBS = 30     # Réduit pour tester rapidement, remettez 300 pour le final
DEVICE = "cpu" # ou "cuda"

# --------------------------------------------------------------------
# 1. CHARGEMENT IMAGE
# --------------------------------------------------------------------
img = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)
typical_shape = (768, 1024, 3)

# Redimensionnement (Maintient la logique originale)
if img.shape != typical_shape:
    if img.shape[0] > img.shape[1]:
        img = cv2.resize(img, (768, 1024))
    else:
        img = cv2.resize(img, (1024, 768))

print(f"Image chargée : {IMG_PATH} | Dimensions : {img.shape}")

# --------------------------------------------------------------------
# 2. SIMULATION SCANDDM
# --------------------------------------------------------------------
model = scanDDM(
    experiment_dur=EXP_DUR,
    fps=FPS,
    threshold=1.0,
    noise=7,
    kappa=10,
    eta=17,
    device=DEVICE,
)

print(f"Lancement de la simulation pour '{TARGET_PROMPT}' avec {N_OBS} observateurs...")
scans, prior_map = model.simulate_scanpaths(
    image=img, prompt=TARGET_PROMPT, n_observers=N_OBS
)
all_scans = np.vstack(scans)
prompt_str = ", ".join(TARGET_PROMPT)

# --------------------------------------------------------------------
# 3. SAUVEGARDE TXT (Optionnel mais conservé pour votre usage)
# --------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
for obs_idx, scanpath in enumerate(scans):
    # scanpath est déjà [x, y, duration]
    df = pd.DataFrame(scanpath, columns=["x", "y", "time"])
    file_path = os.path.join(OUTPUT_DIR, f"observer_{obs_idx+1}.txt")
    df.to_csv(file_path, sep="\t", index=False, header=True)

print(f"Fichiers TXT générés dans : {OUTPUT_DIR}")

# --------------------------------------------------------------------
# 4. ÉVALUATION AUTOMATIQUE (C'est ici que ça devient propre !)
# --------------------------------------------------------------------
# On initialise l'évaluateur avec le fichier YOLO brut et les dimensions de l'image
evaluator = ScanEvaluator(
    yolo_file_path=YOLO_PATH,
    img_shape=img.shape,
    target_class_id=TARGET_CLASS_ID
)

# On lance l'évaluation directement sur les données 'scans' en mémoire
results = evaluator.evaluate(scans)

# On affiche le rapport
evaluator.print_report(results)

# --------------------------------------------------------------------
# 5. VISUALISATION (Code original conservé)
# --------------------------------------------------------------------
sp_to_plot = 1

fig = plt.figure(tight_layout=True, figsize=(15,10))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.axis("off")
plt.title("Original image")

plt.subplot(1, 3, 2)
plt.imshow(img)
if len(scans) > sp_to_plot:
    draw_scanpath(
        scans[sp_to_plot][:, 0], scans[sp_to_plot][:, 1], scans[sp_to_plot][:, 2] * 1000
    )
plt.axis("off")
plt.title("Simulated Scan")

plt.subplot(1, 3, 3)
sal = compute_density_image(all_scans[:, :2], img.shape[:2])
if np.max(sal) > 0:
    res = np.multiply(img, np.repeat(sal[:,:,None]/np.max(sal),3, axis=2))
    res = res/np.max(res)
    plt.imshow(res)
else:
    plt.imshow(img) # Fallback si saliency vide
plt.axis("off")
plt.title("Generated Saliency ("+str(N_OBS)+" scanpaths)")

fig.suptitle(prompt_str + f" (Acc: {results['accuracy']:.1f}%)", fontsize=20)
plt.show()