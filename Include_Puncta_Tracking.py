# -*- coding: utf-8 -*-
"""
Include_Puncta_Tracking.py
========================
Bibliothèque de fonctions pour le suivi automatisé de puncta en microscopie
de fluorescence temps-réel.

Pipeline général :
    1. Prétraitement de l'image (flou gaussien + CLAHE)
    2. Détection de candidats (blob_log)
    3. Classification CNN des somas → extraction de patches
    4. Clustering et fusion des ROI
    5. Segmentation Cellpose (GPU) des puncta dans chaque ROI
    6. Suivi temporel via TrackPy
    7. Analyse et export des graphiques (MSD, vitesse, densité, trajectoires)

Auteur  : huber
Créé le : Wed Mar 11 15:24:26 2026
"""

# ── Librairies standard ───────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

# ── Traitement d'images (scikit-image) ───────────────────────────────────
from skimage import io, measure, filters, exposure, feature, morphology
from skimage.draw import disk
from skimage.morphology import white_tophat, disk   # white_tophat : rehausse les petites structures brillantes
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu

# ── Affichage ─────────────────────────────────────────────────────────────
from matplotlib.patches import Circle, Rectangle
from matplotlib import cm, colors
import matplotlib.colors as mcolors

# ── Calcul scientifique ───────────────────────────────────────────────────
from scipy import ndimage as ndi
from scipy.stats import linregress, gaussian_kde
from scipy.spatial import cKDTree          # structure KD-tree pour recherche de voisins rapide

# ── Données / I/O ─────────────────────────────────────────────────────────
import pandas as pd
import os
from collections import defaultdict

# ── Deep Learning (PyTorch) ───────────────────────────────────────────────
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# ── Segmentation neuronale ────────────────────────────────────────────────
from cellpose import models                # Cellpose : segmentation de cellules/organites

# ── Utilitaires ───────────────────────────────────────────────────────────
from pathlib import Path
from tqdm import tqdm                      # barre de progression dans les boucles

# ── Tracking de particules ────────────────────────────────────────────────
import trackpy as tp                       # TrackPy : lien des détections entre frames


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — UTILITAIRES MATÉRIEL
# ═══════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    """
    Détecte automatiquement le meilleur dispositif de calcul disponible.

    Retourne
    --------
    torch.device
        ``cuda`` si un GPU NVIDIA compatible CUDA est détecté,
        ``cpu`` sinon.

    Notes
    -----
    Affiche un message de statut dans la console pour confirmer
    quel dispositif sera utilisé.
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"✅ GPU détecté : {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("⚠️  GPU non disponible → CPU utilisé")
    return dev


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DÉTECTION DE CANDIDATS (LoG)
# ═══════════════════════════════════════════════════════════════════════════

def detect_candidates(image: np.ndarray) -> np.ndarray:
    """
    Détecte les régions candidates (somas) dans une image par la méthode
    Laplacian of Gaussian (LoG).

    L'image est d'abord normalisée (centrée-réduite) pour stabiliser la
    détection quel que soit le niveau de signal d'entrée, puis
    ``skimage.feature.blob_log`` est appliqué.

    Paramètres
    ----------
    image : np.ndarray, 2D
        Image en niveaux de gris (float ou entier).

    Retourne
    --------
    blobs : np.ndarray, shape (N, 3)
        Tableau de N blobs détectés. Chaque ligne est ``[y, x, sigma]`` où
        ``sigma`` est proportionnel au rayon du blob.

    Notes
    -----
    Plage de sigmas : 5 à 18 px (20 niveaux).
    Seuil de détection : 0.3 (relatif au maximum de la réponse normalisée).
    Diviser par zéro évité si l'écart-type de l'image est < 1e-8.
    """
    image = image.astype(np.float32)

    # Normalisation z-score : soustrait la moyenne, divise par l'écart-type
    mean = image.mean()
    std  = image.std()
    if std > 1e-8:
        image = (image - mean) / std
    else:
        image = image - mean   # image quasi uniforme → on soustrait juste la moyenne

    blobs = feature.blob_log(
        image,
        min_sigma=5,      # sigma minimum : petits somas
        max_sigma=18,     # sigma maximum : grands somas
        num_sigma=20,     # nombre de niveaux d'échelle explorés
        threshold=0.3     # seuil de réponse normalisée pour considérer un blob
    )

    return blobs


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — CONSTRUCTION DES PATCHES
# ═══════════════════════════════════════════════════════════════════════════

def build_blob_dataset(
    image: np.ndarray,
    blobs: np.ndarray,
    patch_size: int = 64,
    allow_partial: bool = False,
) -> tuple:
    """
    Extrait un patch carré centré sur chaque blob détecté.

    Paramètres
    ----------
    image : np.ndarray, 2D
        Image source normalisée.
    blobs : np.ndarray, shape (N, 3)
        Sorties de ``detect_candidates`` : colonnes ``[y, x, sigma]``.
    patch_size : int, optionnel (défaut 64)
        Taille du patch extrait en pixels (carré : patch_size × patch_size).
    allow_partial : bool, optionnel (défaut False)
        Si True, extrait des patches partiels pour les blobs proches des bords.
        Si False (défaut), ignore les blobs dont le patch sort de l'image.

    Retourne
    --------
    patches : np.ndarray, shape (M, patch_size, patch_size)
        Tableau des M patches valides, normalisés entre 0 et 1.
    valid_blobs : list of tuple (y, x, sigma)
        Blobs correspondant aux patches conservés (même ordre).

    Notes
    -----
    Chaque patch est normalisé min-max indépendamment.
    La valeur 1e-8 est ajoutée au dénominateur pour éviter la division par zéro
    dans le cas d'un patch uniforme.
    """
    H, W = image.shape
    half = patch_size // 2

    patches     = []
    valid_blobs = []

    for y, x, sigma in blobs:
        y, x = int(round(y)), int(round(x))

        if not allow_partial:
            # Rejeter les blobs dont le patch déborderait de l'image
            if (x - half < 0 or x + half >= W or y - half < 0 or y + half >= H):
                continue
            patch = image[y-half:y+half, x-half:x+half]
            x0 = x - half
            y0 = y - half
        else:
            # Clamp aux bords → patch potentiellement plus petit
            y0 = max(0, y - half)
            y1 = min(H, y + half)
            x0 = max(0, x - half)
            x1 = min(W, x + half)
            patch = image[y0:y1, x0:x1]

        # Normalisation min-max du patch
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)

        patches.append(patch)
        valid_blobs.append((y, x, sigma))

    return np.array(patches), valid_blobs


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — ARCHITECTURE CNN (SomaCNN)
# ═══════════════════════════════════════════════════════════════════════════

class SomaCNN(nn.Module):
    """
    Réseau de neurones convolutif (CNN) pour la classification binaire de patches :
    soma (1) vs non-soma (0).

    Architecture
    ------------
    Bloc convolutif (``features``) :
        Conv2d(1→32, 3×3)  → ReLU → MaxPool2d(2)
        Conv2d(32→64, 3×3) → ReLU → MaxPool2d(2)
        Conv2d(64→128, 3×3)→ ReLU → AdaptiveAvgPool2d(1×1)

    Tête de classification (``classifier``) :
        Linear(128→1) → Sigmoid  (sortie : probabilité d'être un soma)

    Notes
    -----
    - L'entrée est un tenseur de forme ``(B, 1, H, W)`` (batch, canal unique,
      hauteur, largeur). Le canal unique correspond à l'image en niveaux de gris.
    - ``AdaptiveAvgPool2d(1)`` rend le réseau invariant à la taille du patch
      en entrée (utile si ``patch_size`` varie).
    - La sortie est une probabilité ∈ [0, 1] par image du batch.
    """

    def __init__(self):
        super().__init__()

        # Extracteur de caractéristiques convolutif
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),     # 1 canal → 32 feature maps, 3×3, padding=1 (même taille)
            nn.ReLU(),
            nn.MaxPool2d(2),                     # sous-échantillonnage ×2 (divise H et W par 2)
            nn.Conv2d(32, 64, 3, padding=1),     # 32 → 64 feature maps
            nn.ReLU(),
            nn.MaxPool2d(2),                     # sous-échantillonnage ×2
            nn.Conv2d(64, 128, 3, padding=1),    # 64 → 128 feature maps
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)              # pool global → vecteur 128-dim, quelle que soit la taille d'entrée
        )

        # Couche de classification : vecteur 128-dim → logit unique
        self.classifier = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du réseau.

        Paramètres
        ----------
        x : torch.Tensor, shape (B, 1, H, W)
            Batch de B patches en niveaux de gris.

        Retourne
        --------
        torch.Tensor, shape (B, 1)
            Probabilité d'être un soma pour chaque patch du batch.
        """
        z = self.features(x)
        z = z.view(len(x), -1)          # aplatir (B, 128, 1, 1) → (B, 128)
        return torch.sigmoid(self.classifier(z))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — PRÉTRAITEMENT D'IMAGE
# ═══════════════════════════════════════════════════════════════════════════

def Pretraitement_image(img: np.ndarray, sigma: float, clip_limit: float) -> np.ndarray:
    """
    Applique un prétraitement standard à une image de microscopie.

    Étapes
    ------
    1. **Flou gaussien** (``sigma``) : atténue le bruit haute fréquence.
    2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) : améliore
       le contraste local tout en limitant l'amplification du bruit
       (paramètre ``clip_limit``).

    Paramètres
    ----------
    img : np.ndarray, 2D
        Image brute en niveaux de gris.
    sigma : float
        Écart-type du noyau gaussien (en pixels).
        Valeurs typiques : 1.0 à 2.0.
    clip_limit : float
        Limite de rehaussement du contraste CLAHE ∈ [0, 1].
        Valeurs typiques : 0.01 à 0.05.

    Retourne
    --------
    img_eq : np.ndarray, 2D, float64 ∈ [0, 1]
        Image prétraitée prête pour la détection.
    """
    img_blur = filters.gaussian(img, sigma=sigma)                          # lissage gaussien
    img_eq   = exposure.equalize_adapthist(img_blur, clip_limit=clip_limit) # CLAHE
    return img_eq


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — INFÉRENCE CNN PAR BATCH
# ═══════════════════════════════════════════════════════════════════════════

def CNN_Proba_Construction(
    blobs_valid: list,
    patches_s,
    model_cnn: nn.Module,
    cnn_threshold: float,
    half: int,
    batch_size: int = 256,
) -> tuple:
    """
    Exécute l'inférence CNN par batches sur GPU (ou CPU en repli automatique).

    Pour chaque patch, le CNN prédit une probabilité d'appartenir à un soma.
    Seuls les patches dont la probabilité est ≥ ``cnn_threshold`` sont conservés.

    Optimisations intégrées
    -----------------------
    - Passage avant groupé (``batch_size`` patches à la fois) plutôt qu'un appel
      par patch → gain majeur sur GPU.
    - ``pin_memory=True`` sur les tenseurs CPU pour accélérer le transfert DMA
      CPU → GPU.
    - Copie vers numpy uniquement une fois par batch, pas par patch.
    - Le dispositif suivi automatiquement depuis le modèle (``next(model.parameters()).device``).

    Paramètres
    ----------
    blobs_valid : list of tuple (y, x, sigma)
        Blobs retenus après ``build_blob_dataset``.
    patches_s : array-like, shape (N, H, W)
        Patches correspondants (normalisés 0-1).
    model_cnn : nn.Module
        Instance de ``SomaCNN`` chargée et placée sur le bon dispositif.
    cnn_threshold : float
        Seuil de probabilité CNN pour retenir un patch comme soma.
    half : int
        Demi-taille du patch (= patch_size // 2).  Utilisé pour recalculer
        les coins supérieurs gauche des patches.
    batch_size : int, optionnel (défaut 256)
        Nombre de patches par passe avant.  Réduire en cas d'erreur OOM GPU.

    Retourne
    --------
    all_probs : np.ndarray, shape (N,), float32
        Probabilité CNN pour tous les patches (y compris rejetés).
    soma_patches : list of tuple (x0, y0, prob, patch)
        Patches retenus avec leur coin supérieur gauche (x0, y0),
        leur probabilité CNN et le tableau numpy du patch.
    """
    device     = next(model_cnn.parameters()).device
    patches_arr = np.array(patches_s, dtype=np.float32)    # (N, H, W)
    all_probs   = np.empty(len(patches_arr), dtype=np.float32)

    model_cnn.eval()
    with torch.no_grad():
        for start in range(0, len(patches_arr), batch_size):
            end   = start + batch_size
            chunk = patches_arr[start:end]

            # Ajouter la dimension canal : (B, H, W) → (B, 1, H, W)
            batch_cpu = torch.from_numpy(chunk).unsqueeze(1)

            # Transfert vers GPU avec pin_memory pour les DMA rapides
            if device.type == "cuda":
                batch_gpu = batch_cpu.pin_memory().to(device, non_blocking=True)
            else:
                batch_gpu = batch_cpu

            probs = model_cnn(batch_gpu).squeeze(1)
            all_probs[start:end] = probs.cpu().numpy()

    # Filtrage : ne garder que les patches dépassant le seuil CNN
    soma_patches = [
        (int(x - half), int(y - half), float(prob), patch)
        for (y, x, _), patch, prob in zip(blobs_valid, patches_s, all_probs)
        if prob >= cnn_threshold
    ]
    return all_probs, soma_patches


def Patch_construction(img_eq: np.ndarray, patch_size: int) -> tuple:
    """
    Enchaîne la détection de blobs et l'extraction des patches.

    Paramètres
    ----------
    img_eq : np.ndarray
        Image prétraitée (sortie de ``Pretraitement_image``).
    patch_size : int
        Taille des patches carrés à extraire.

    Retourne
    --------
    patches_s : np.ndarray, shape (N, patch_size, patch_size)
        Patches extraits et normalisés.
    blobs_valid : list of tuple (y, x, sigma)
        Blobs correspondant aux patches valides.
    """
    blobs = detect_candidates(img_eq)
    patches_s, blobs_valid = build_blob_dataset(img_eq, blobs, patch_size=patch_size)
    return patches_s, blobs_valid


def CNN_Patches_Construction(
    image: np.ndarray,
    pretraitement_sigma: float,
    clip_limit: float,
    patch_size: int,
    cnn_threshold: float,
    model_cnn: nn.Module,
    batch_size: int = 256,
) -> tuple:
    """
    Pipeline complet : prétraitement → détection → CNN.

    Combine ``Pretraitement_image``, ``Patch_construction`` et
    ``CNN_Proba_Construction`` en une seule fonction de haut niveau.

    Paramètres
    ----------
    image : np.ndarray
        Image brute (une frame du stack).
    pretraitement_sigma : float
        Sigma gaussien pour le prétraitement.
    clip_limit : float
        Limite CLAHE pour le prétraitement.
    patch_size : int
        Taille des patches (pixels, carré).
    cnn_threshold : float
        Seuil de probabilité CNN pour retenir un soma.
    model_cnn : nn.Module
        CNN entraîné (``SomaCNN``).
    batch_size : int, optionnel (défaut 256)
        Taille des batches pour l'inférence CNN.

    Retourne
    --------
    all_probs : np.ndarray, shape (N,)
        Probabilités CNN de tous les candidats.
    soma_patches : list of tuple (x0, y0, prob, patch)
        Patches somas retenus.
    """
    half = patch_size // 2
    img_eq = Pretraitement_image(image, pretraitement_sigma, clip_limit)
    patches_s, blobs_valid = Patch_construction(img_eq, patch_size)
    all_probs, soma_patches = CNN_Proba_Construction(
        blobs_valid, patches_s, model_cnn, cnn_threshold, half, batch_size=batch_size
    )
    return all_probs, soma_patches


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — CLUSTERING DES PATCHES (KDTree + Union-Find)
# ═══════════════════════════════════════════════════════════════════════════

def Clusterization_detected_patches(
    img: np.ndarray,
    soma_patches: list,
    patch_size: int,
) -> list:
    """
    Fusionne les patches qui se chevauchent en ROI (Regions Of Interest).

    Algorithme
    ----------
    1. Calcul des centres de chaque patch.
    2. Recherche de toutes les paires de centres distants de moins de
       ``patch_size / 1.25`` px via un ``scipy.cKDTree`` (complexité O(n log n)).
    3. Fusion des paires connectées par Union-Find avec compression de chemin.
    4. Construction des bounding boxes englobantes pour chaque cluster.

    L'approche KDTree + Union-Find remplace un BFS en O(n²) précédent :
    pour n=500 patches, gain de vitesse d'environ ×100.

    Paramètres
    ----------
    img : np.ndarray
        Image source (utilisée pour récupérer les dimensions H, W).
    soma_patches : list of tuple (x0, y0, prob, patch)
        Liste des patches somas retenus par le CNN.
    patch_size : int
        Taille des patches en pixels (nécessaire pour calculer les centres
        et définir le seuil de distance).

    Retourne
    --------
    merged_rois : list of tuple (xmin, ymin, xmax, ymax)
        Bounding boxes des clusters, clampées dans les dimensions de l'image.

    Notes
    -----
    Si ``soma_patches`` est vide, retourne une liste vide immédiatement.
    """
    if not soma_patches:
        return []

    half             = patch_size // 2
    distance_thresh  = patch_size / 1.25   # distance max pour considérer deux patches comme voisins

    # Centre de chaque patch : (x0 + half, y0 + half)
    centers = np.array([(x0 + half, y0 + half) for x0, y0, *_ in soma_patches])

    # ── KDTree : trouver toutes les paires de centres proches ────────────
    tree  = cKDTree(centers)
    pairs = tree.query_pairs(distance_thresh)

    # ── Union-Find : fusionner les groupes connectés ─────────────────────
    parent = list(range(len(centers)))

    def find(i: int) -> int:
        """Trouve la racine de i avec compression de chemin."""
        root = i
        while parent[root] != root:
            root = parent[root]
        # Compression de chemin : tous les nœuds pointent directement vers la racine
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj   # fusion des deux composantes

    # Regrouper les indices par composante
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(centers)):
        groups[find(i)].append(i)

    # ── Bounding boxes par cluster ───────────────────────────────────────
    H, W = img.shape
    merged_rois = []
    for cluster in groups.values():
        xs, ys = [], []
        for idx in cluster:
            x0, y0, *_ = soma_patches[idx]
            xs += [x0, x0 + patch_size]   # coins gauche et droit
            ys += [y0, y0 + patch_size]   # coins haut et bas
        merged_rois.append((
            max(0, min(xs)),   # xmin clampé à 0
            max(0, min(ys)),   # ymin clampé à 0
            min(W, max(xs)),   # xmax clampé à W
            min(H, max(ys)),   # ymax clampé à H
        ))

    return merged_rois


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — FILTRAGE / FUSION FINALE DES ROI
# ═══════════════════════════════════════════════════════════════════════════

def Finale_Fusion_patches(
    img: np.ndarray,
    merged_rois: list,
    overlap_thresh: float,
) -> tuple:
    """
    Supprime les ROI largement incluses dans une ROI plus grande.

    Critère de suppression : si la surface d'intersection d'une ROI avec
    une autre représente ≥ ``overlap_thresh`` de sa propre surface, elle est
    considérée comme redondante et supprimée.

    Paramètres
    ----------
    img : np.ndarray
        Image source (utilisée pour extraire les patches finaux).
    merged_rois : list of tuple (xmin, ymin, xmax, ymax)
        ROI issues de ``Clusterization_detected_patches``.
    overlap_thresh : float ∈ [0, 1]
        Fraction minimale de chevauchement pour supprimer une ROI.
        Valeur typique : 0.1 (10 %).

    Retourne
    --------
    final_patches : list of np.ndarray
        Sous-images correspondant aux ROI conservées.
    filtered_rois : list of tuple (xmin, ymin, xmax, ymax)
        ROI conservées après filtrage.
    """

    def _area(r: tuple) -> float:
        """Surface d'une ROI (xmin, ymin, xmax, ymax)."""
        return (r[2] - r[0]) * (r[3] - r[1])

    def _overlap_fraction(small: tuple, big: tuple) -> float:
        """
        Fraction de la surface de ``small`` couverte par l'intersection
        avec ``big``.  Retourne 0 si pas d'intersection.
        """
        x0, y0 = max(small[0], big[0]), max(small[1], big[1])
        x1, y1 = min(small[2], big[2]), min(small[3], big[3])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return (x1 - x0) * (y1 - y0) / (_area(small) + 1e-8)

    # Conserver uniquement les ROI qui ne sont pas absorbées par une autre
    filtered_rois = [
        r for i, r in enumerate(merged_rois)
        if not any(
            _overlap_fraction(r, other) >= overlap_thresh
            for j, other in enumerate(merged_rois) if i != j
        )
    ]

    # Extraire les sous-images correspondantes
    final_patches = [img[ymin:ymax, xmin:xmax] for xmin, ymin, xmax, ymax in filtered_rois]
    return final_patches, filtered_rois


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — SEGMENTATION CELLPOSE + CONSTRUCTION DU DATAFRAME
# ═══════════════════════════════════════════════════════════════════════════

def Cellpose_Analyse_Track_GPU(
    model,
    final_patches: list,
    patch_origins: list,
    time_index: int,
    diameter_cellpose: float,
    cellprob_threshold: float,
    flow_threshold: float,
    min_area: int,
    max_area: int,
    min_circularity: float,
    max_axis_ratio: float,
    top_hat_radius: int,
    flag_visu: bool,
    cellpose_batch_size: int = 24,
) -> pd.DataFrame:
    """
    Segmente les puncta sur des patches via Cellpose (GPU) et retourne
    un DataFrame compatible avec TrackPy.

    Pipeline interne
    ----------------
    1. **Top-hat** : soustraction du fond local pour rehausser les puncta
       (structures brillantes de petite taille).
    2. **Inférence Cellpose** par batch : segmentation d'instances.
    3. **Filtrage morphologique** : suppression des objets hors critères
       (aire, circularité, rapport des axes).
    4. **Construction DataFrame** : ajout des coordonnées globales
       (patch_origin + centroïde local).
    5. **Visualisation optionnelle** : affichage des 3 étapes (original,
       segmentation brute, segmentation filtrée) si ``flag_visu=True``.

    Paramètres
    ----------
    model : cellpose.models.Cellpose
        Modèle Cellpose instancié (``cyto3`` recommandé).
    final_patches : list of np.ndarray
        Patches ROI à segmenter.
    patch_origins : list of tuple (x0, y0)
        Coordonnées du coin supérieur gauche de chaque patch dans l'image globale.
    time_index : int
        Indice temporel de la frame courante (colonne ``frame`` du DataFrame).
    diameter_cellpose : float
        Diamètre estimé des puncta en pixels (paramètre Cellpose).
    cellprob_threshold : float
        Seuil de probabilité de cellule Cellpose ∈ [-6, 6].
        Valeur basse → plus de détections (défaut Cellpose : 0.0).
    flow_threshold : float
        Seuil de cohérence des flux optiques Cellpose ∈ [0, 1].
        Valeur haute → segmentation plus stricte.
    min_area : int
        Aire minimale (px²) pour retenir un objet.
    max_area : int
        Aire maximale (px²) pour retenir un objet.
    min_circularity : float ∈ [0, 1]
        Circularité minimale : 4π·A / P² (1 = cercle parfait).
    max_axis_ratio : float
        Rapport maximal grand axe / petit axe.  Filtre les formes allongées
        (dendrites, artefacts).
    top_hat_radius : int
        Rayon du disque structural pour le top-hat (en pixels).
        Doit être supérieur au rayon des puncta.
    flag_visu : bool
        Si True, affiche une figure matplotlib par patch (debug).
    cellpose_batch_size : int, optionnel (défaut 24)
        Nombre de patches traités simultanément par Cellpose.

    Retourne
    --------
    df_puncta : pd.DataFrame
        Colonnes : ``frame``, ``patch``, ``x``, ``y``, ``area``, ``circularity``.
        Chaque ligne est un puncta détecté et filtré.
        Si aucun puncta n'est détecté, retourne un DataFrame vide.
    """
    use_gpu = torch.cuda.is_available()

    # ── 1. Prétraitement top-hat ──────────────────────────────────────────
    # Le top-hat blanc supprime le fond lentement variable et conserve uniquement
    # les structures plus petites que le disque structural (radius = top_hat_radius).
    preprocessed = []
    for patch in final_patches:
        p        = np.array(patch, dtype=np.float32)
        p_tophat = white_tophat(p, footprint=disk(top_hat_radius))
        if p_tophat.ndim == 2:
            p_tophat = p_tophat[..., np.newaxis]   # Cellpose attend (H, W, C)
        preprocessed.append(p_tophat)

    # ── 2. Inférence Cellpose par batch ───────────────────────────────────
    masks_list, _, _, _ = model.eval(
        preprocessed,
        diameter=diameter_cellpose,
        channels=[0, 0],                   # image en niveaux de gris (canal unique)
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        batch_size=cellpose_batch_size
    )
    if use_gpu:
        torch.cuda.empty_cache()           # libérer la VRAM entre les frames

    # ── 3. Filtrage morphologique + construction DataFrame ────────────────
    puncta_records = []

    for i, ((x0, y0), patch, masks) in enumerate(zip(patch_origins, final_patches, masks_list)):
        labels = masks.copy()
        props  = pd.DataFrame()

        if labels.max() > 0:   # au moins un objet segmenté
            props = pd.DataFrame(
                measure.regionprops_table(
                    labels,
                    properties=[
                        "label",
                        "area",
                        "perimeter",
                        "major_axis_length",
                        "minor_axis_length",
                        "centroid"
                    ]
                )
            )

            # Calcul des métriques de forme
            props["circularity"] = 4 * np.pi * props["area"] / (props["perimeter"]**2 + 1e-8)
            props["axis_ratio"]  = props["major_axis_length"] / (props["minor_axis_length"] + 1e-8)

            # Identification des labels à supprimer (hors critères)
            remove_labels = props.loc[
                (props["area"]         < min_area)         |   # trop petit
                (props["area"]         > max_area)         |   # trop grand
                (props["circularity"]  < min_circularity)  |   # trop allongé
                (props["axis_ratio"]   > max_axis_ratio),      # rapport axes > seuil
                "label"
            ].values

            if len(remove_labels) > 0:
                labels[np.isin(labels, remove_labels)] = 0   # effacer dans le masque

            props = props[~props["label"].isin(remove_labels)]

            # Conversion coordonnées locales → coordonnées globales
            for _, row in props.iterrows():
                puncta_records.append({
                    "frame":        time_index,
                    "patch":        i,
                    "x":            row["centroid-1"] + x0,   # centroïde colonne + offset X
                    "y":            row["centroid-0"] + y0,   # centroïde ligne   + offset Y
                    "area":         row["area"],
                    "circularity":  row["circularity"]
                })

        # ── 4. Visualisation optionnelle (mode debug) ─────────────────────
        if flag_visu:
            removed_mask = masks.copy()
            removed_mask[labels > 0] = 0   # masque des objets supprimés

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(np.array(patch).squeeze(), cmap="gray")
            axes[0].set_title("Patch original")

            axes[1].imshow(np.array(patch).squeeze(), cmap="gray")
            axes[1].imshow(masks, alpha=0.3)
            axes[1].set_title(f"Segmentation brute ({masks.max()} obj)")

            axes[2].imshow(np.array(patch).squeeze(), cmap="gray")
            axes[2].imshow(labels, alpha=0.3)
            axes[2].imshow(removed_mask, cmap="Reds", alpha=0.5)
            n_valid = len(props) if not props.empty else 0
            axes[2].set_title(f"Filtrée ({n_valid}) puncta")

            for ax in axes:
                ax.axis("off")
            plt.show()

    # ── 5. Assemblage du DataFrame final ──────────────────────────────────
    df_puncta = pd.DataFrame(puncta_records)
    return df_puncta


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — GRAPHIQUES & SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════════

def Plot_and_Save_Longueur_Trajectoires(output_dir: str, track_length, flag_CNN: bool):
    """
    Génère et sauvegarde l'histogramme des longueurs de trajectoires.

    L'axe Y est en échelle logarithmique pour visualiser les queues de distribution.

    Paramètres
    ----------
    output_dir : str
        Dossier de destination pour les figures.
    track_length : pd.Series
        Nombre de frames par particule (``tracks.groupby("particle").size()``).
    flag_CNN : bool
        Si True, ajoute le suffixe ``_CNN`` au nom du fichier.
    """
    plt.figure()
    plt.hist(track_length, bins=50)
    plt.xlabel("Longueur des trajectoires (frames)")
    plt.ylabel("Nombre")
    plt.yscale("log")
    plt.title("Histogramme longueurs trajectoires")
    suffix = "_CNN" if flag_CNN else ""
    plt.savefig(os.path.join(output_dir, f"histogramme_trajectoires{suffix}.png"))
    plt.close()


def Plot_and_Save_Coefficient_Diffusion(output_dir: str, tracks: pd.DataFrame, flag_CNN: bool):
    """
    Calcule et sauvegarde la courbe MSD avec ajustement de la loi de puissance.

    Le déplacement quadratique moyen (MSD) est calculé via ``trackpy.motion.imsd``
    puis ajusté sur le premier tiers des points en log-log :
        MSD(τ) ∝ τ^α

    - α ≈ 1 : diffusion normale (brownienne)
    - α < 1 : diffusion sous-diffusive (confinement)
    - α > 1 : diffusion supra-diffusive (transport actif)

    Paramètres
    ----------
    output_dir : str
        Dossier de destination.
    tracks : pd.DataFrame
        DataFrame TrackPy filtré (colonnes ``x``, ``y``, ``frame``, ``particle``).
    flag_CNN : bool
        Suffixe de fichier.
    """
    imsd      = tp.motion.imsd(tracks, mpp=1, fps=1)  # MSD individuel (pixels², par frame)
    imsd_mean = imsd.mean(axis=1)                      # MSD moyen sur toutes les particules

    lag_times  = imsd_mean.index.to_numpy()
    msd_values = imsd_mean.to_numpy()

    # Régression linéaire en log-log (premier tiers des points pour éviter le régime saturé)
    log_lag = np.log10(lag_times[1:])
    log_msd = np.log10(msd_values[1:])
    n_points = len(log_lag)
    end_idx  = n_points // 3   # seul le premier tiers est utilisé pour le fit

    slope, intercept, r_value, p_value, std_err = linregress(log_lag[:end_idx], log_msd[:end_idx])
    alpha = slope   # exposant de diffusion

    plt.figure(figsize=(6, 4))
    for col in imsd.columns:
        plt.plot(imsd.index, imsd[col], color="gray", alpha=0.5)  # MSD individuels
    plt.plot(imsd.index, imsd_mean, color="red", linewidth=2, label="MSD moyenne")
    fit_line = 10**intercept * lag_times**alpha
    plt.plot(lag_times, fit_line, color="blue", linestyle='--', linewidth=2,
             label=f"Fit: α={alpha:.2f}")
    plt.xlabel("Lag time (frames)")
    plt.ylabel("MSD (pixels²)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title("MSD & fit diffusion")
    suffix = "_CNN" if flag_CNN else ""
    plt.savefig(os.path.join(output_dir, f"msd_fit{suffix}.png"))
    plt.close()
    print(f"Exposant de diffusion α ≈ {alpha:.2f}")


def Plot_and_Save_Nombre_Puncta_per_Frame(output_dir: str, counts, flag_CNN: bool):
    """
    Trace et sauvegarde l'évolution du nombre de puncta détectés par frame.

    Paramètres
    ----------
    output_dir : str
        Dossier de destination.
    counts : pd.Series
        Nombre de puncta par frame (``tracks_df.groupby("frame").size()``).
    flag_CNN : bool
        Suffixe de fichier.
    """
    plt.figure()
    plt.plot(counts)
    plt.xlabel("Frame")
    plt.ylabel("Nombre de puncta")
    plt.title("puncta détectés par frame")
    plt.grid()
    suffix = "_CNN" if flag_CNN else ""
    plt.savefig(os.path.join(output_dir, f"nombre_puncta_par_frame{suffix}.png"))
    plt.close()


def Plot_and_Save_Trajectoires(mean_img: np.ndarray, output_dir: str, tracks: pd.DataFrame, flag_CNN: bool):
    """
    Superpose et sauvegarde les trajectoires filtrées sur l'image moyenne du stack.

    Paramètres
    ----------
    mean_img : np.ndarray
        Image moyenne temporelle du stack (projection Z moyenne).
    output_dir : str
        Dossier de destination.
    tracks : pd.DataFrame
        DataFrame TrackPy filtré.
    flag_CNN : bool
        Suffixe de fichier.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(mean_img, cmap='gray')
    for particle, traj in tracks.groupby("particle"):
        plt.plot(traj["x"], traj["y"], linewidth=4, alpha=0.6)
    plt.axis('off')
    plt.title("Trajectoires filtrées")
    suffix = "_CNN" if flag_CNN else ""
    plt.savefig(os.path.join(output_dir, f"trajectoires_moyenne{suffix}.png"))
    plt.close()


def Plot_and_Save_DensityMap(mean_img: np.ndarray, output_dir: str, tracks: pd.DataFrame, flag_CNN: bool):
    """
    Génère et sauvegarde une carte de densité spatiale des puncta.

    La densité est estimée par noyau gaussien (KDE) sur les positions (x, y)
    de toutes les détections agrégées sur l'ensemble des frames.

    Paramètres
    ----------
    mean_img : np.ndarray
        Image moyenne temporelle du stack.
    output_dir : str
        Dossier de destination.
    tracks : pd.DataFrame
        DataFrame TrackPy filtré.
    flag_CNN : bool
        Suffixe de fichier.
    """
    xy      = np.vstack([tracks["x"], tracks["y"]])
    density = gaussian_kde(xy)(xy)   # densité KDE aux positions des détections

    plt.figure(figsize=(6, 6))
    plt.imshow(mean_img, cmap='gray')
    plt.scatter(tracks["x"], tracks["y"], c=density, s=5, cmap='inferno')
    plt.colorbar(label="Densité")
    plt.axis('off')
    plt.title("Carte de densité des puncta")
    suffix = "_CNN" if flag_CNN else ""
    plt.savefig(os.path.join(output_dir, f"densite_puncta{suffix}.png"))
    plt.close()


def Plot_and_Save_Distribution_Vitesse(output_dir: str, tracks: pd.DataFrame, flag_CNN: bool):
    """
    Calcule et sauvegarde la distribution des vitesses instantanées.

    La vitesse est calculée frame par frame : √(Δx² + Δy²) en pixels/frame.
    L'axe Y est en échelle logarithmique.

    Paramètres
    ----------
    output_dir : str
        Dossier de destination.
    tracks : pd.DataFrame
        DataFrame TrackPy filtré.
    flag_CNN : bool
        Suffixe de fichier.
    """
    tracks["dx"]    = tracks.groupby("particle")["x"].diff()   # déplacement en X entre frames consécutives
    tracks["dy"]    = tracks.groupby("particle")["y"].diff()   # déplacement en Y
    tracks["speed"] = np.sqrt(tracks["dx"]**2 + tracks["dy"]**2)

    plt.figure()
    plt.hist(tracks["speed"].dropna(), bins=100)
    plt.yscale("log")
    plt.xlabel("Vitesse (pixels/frame)")
    plt.ylabel("Nombre")
    plt.title("Distribution des vitesses")
    suffix = "_CNN" if flag_CNN else ""
    plt.savefig(os.path.join(output_dir, f"distribution_vitesses{suffix}.png"))
    plt.close()


def Plot_and_Save_Trajectoire_Vitesse(mean_img: np.ndarray, output_dir: str, tracks: pd.DataFrame, flag_CNN: bool):
    """
    Superpose les trajectoires colorées par vitesse instantanée sur l'image moyenne.

    Paramètres
    ----------
    mean_img : np.ndarray
        Image moyenne temporelle du stack.
    output_dir : str
        Dossier de destination.
    tracks : pd.DataFrame
        DataFrame TrackPy filtré.
    flag_CNN : bool
        Suffixe de fichier.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(mean_img, cmap='gray')
    for particle, traj in tracks.groupby("particle"):
        dx    = traj["x"].diff()
        dy    = traj["y"].diff()
        speed = np.sqrt(dx**2 + dy**2)
        plt.scatter(traj["x"], traj["y"], c=speed, cmap='plasma', s=5)
    plt.colorbar(label="Vitesse (pixel/frame)")
    plt.axis('off')
    plt.title("Trajectoires colorées par vitesse")
    suffix = "_CNN" if flag_CNN else ""
    plt.savefig(os.path.join(output_dir, f"trajectoires_vitesse{suffix}.png"))
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11 — CONSTRUCTION DES TRACKS (TrackPy seul)
# ═══════════════════════════════════════════════════════════════════════════

def TrackPy_Construction(
    stack: np.ndarray,
    pretraitement_sigma: float,
    clip_limit: float,
    diameter_trackpy: int,
    search_range: float,
    memory: int,
) -> tuple:
    """
    Détecte et relie les particules frame par frame avec TrackPy uniquement.

    Stratégie de ``minmass`` adaptatif
    -----------------------------------
    Le seuil de masse minimum (``minmass``) est calculé automatiquement à
    partir du 99e percentile des masses détectées sur la première frame, puis
    ajusté proportionnellement à l'intensité moyenne de chaque frame.  Cela
    compense les variations de fluorescence au cours de l'acquisition.

    Paramètres
    ----------
    stack : np.ndarray, shape (T, H, W)
        Stack temporel d'images.
    pretraitement_sigma : float
        Sigma gaussien pour ``Pretraitement_image``.
    clip_limit : float
        Clip CLAHE pour ``Pretraitement_image``.
    diameter_trackpy : int
        Diamètre (impair) des particules pour ``tp.locate``.
    search_range : float
        Distance maximale (px) qu'une particule peut parcourir entre deux frames
        (paramètre ``tp.link_df``).
    memory : int
        Nombre de frames pendant lesquelles une particule disparue est mémorisée
        avant d'être abandonnée (paramètre ``tp.link_df``).

    Retourne
    --------
    tracks : pd.DataFrame
        Trajectoires liées (colonnes standard TrackPy : ``x``, ``y``, ``frame``,
        ``particle``, ``mass``, ...).
    counts : pd.Series
        Nombre de détections par frame (avant linking).
    """
    tp.quiet()

    # ── Calibration du minmass sur la première frame ──────────────────────
    img_0 = Pretraitement_image(stack[0], pretraitement_sigma, clip_limit)
    f_all = tp.locate(img_0, diameter=diameter_trackpy, minmass=0, invert=False)
    base_minmass = f_all['mass'].quantile(0.99)   # 99e percentile → seuil conservateur
    mean_intensity_frame0 = np.mean(img_0)

    # ── Détection frame par frame ─────────────────────────────────────────
    all_df = []
    for t, img in enumerate(stack):
        img_eq = Pretraitement_image(img, pretraitement_sigma, clip_limit)

        # Ajustement du minmass selon l'intensité relative de la frame courante
        mean_intensity_current = np.mean(img_eq)
        minmass_frame = base_minmass * (mean_intensity_current / mean_intensity_frame0)

        f = tp.locate(img_eq, diameter=diameter_trackpy, minmass=minmass_frame, invert=False)
        f['frame'] = t
        all_df.append(f)

    # ── Linking (association temporelle) ─────────────────────────────────
    tracks_df = pd.concat(all_df, ignore_index=True)
    counts    = tracks_df.groupby("frame").size()
    tracks    = tp.link_df(tracks_df, search_range=search_range, memory=memory)

    return tracks, counts


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12 — CONSTRUCTION DES TRACKS (CNN + Cellpose + TrackPy)
# ═══════════════════════════════════════════════════════════════════════════

def TrackPy_Construction_CNN_Cellpose(
    stack: np.ndarray,
    pretraitement_sigma: float,
    clip_limit: float,
    patch_size: int,
    cnn_threshold: float,
    model_cnn: nn.Module,
    cnn_batch_size: int,
    overlap_thresh: float,
    cellpose_model,
    cellprob_threshold: float,
    flow_threshold: float,
    min_area: int,
    max_area: int,
    min_circularity: float,
    max_axis_ratio: float,
    top_hat_radius: int,
    diameter_cellpose: float,
    search_range: float,
    memory: int,
) -> tuple:
    """
    Pipeline complet CNN + Cellpose + TrackPy pour le suivi des puncta.

    Stratégie
    ---------
    Les ROI sont détectées **une seule fois** sur la première frame puis
    réutilisées pour tout le stack.  Cela suppose que les somas (et donc les
    ROI contenant les puncta) ne se déplacent pas significativement au cours
    de l'acquisition.

    Étapes
    ------
    1. Détection des somas sur ``stack[0]`` via CNN (``CNN_Patches_Construction``).
    2. Clustering et fusion des patches somas (``Clusterization_detected_patches``,
       ``Finale_Fusion_patches``).
    3. Pour chaque frame, extraction des patches aux positions fixes et
       segmentation Cellpose (``Cellpose_Analyse_Track_GPU``).
    4. Concaténation des DataFrames et linking TrackPy.

    Paramètres
    ----------
    stack : np.ndarray, shape (T, H, W)
        Stack temporel.
    pretraitement_sigma : float
        Sigma gaussien pour le prétraitement CNN.
    clip_limit : float
        Clip CLAHE pour le prétraitement CNN.
    patch_size : int
        Taille des patches CNN (px, carré).
    cnn_threshold : float
        Seuil de probabilité CNN pour retenir un soma.
    model_cnn : nn.Module
        CNN soma entraîné (``SomaCNN``).
    cnn_batch_size : int
        Taille de batch pour l'inférence CNN.
    overlap_thresh : float
        Seuil de chevauchement pour la fusion des ROI.
    cellpose_model : cellpose.models.Cellpose
        Modèle Cellpose instancié.
    cellprob_threshold : float
        Seuil de probabilité Cellpose.
    flow_threshold : float
        Seuil de cohérence des flux Cellpose.
    min_area : int
        Aire minimale des puncta (px²).
    max_area : int
        Aire maximale des puncta (px²).
    min_circularity : float
        Circularité minimale.
    max_axis_ratio : float
        Rapport d'axes maximal.
    top_hat_radius : int
        Rayon du disque top-hat.
    diameter_cellpose : float
        Diamètre estimé des puncta (px) pour Cellpose.
    search_range : float
        Portée de recherche TrackPy (px).
    memory : int
        Mémoire TrackPy (frames).

    Retourne
    --------
    tracks : pd.DataFrame
        Trajectoires liées par TrackPy.
    counts : pd.Series
        Nombre de puncta détectés par frame.
    """
    # ── 1. Détection des somas et des ROI sur la 1re frame ────────────────
    img0 = stack[0]
    all_probs, soma_patches = CNN_Patches_Construction(
        img0, pretraitement_sigma, clip_limit,
        patch_size, cnn_threshold, model_cnn,
        batch_size=cnn_batch_size,
    )
    print(f"🔹 Nombre total de patchs détectés (soma_patches) : {len(soma_patches)}")

    merged_rois              = Clusterization_detected_patches(img0, soma_patches, patch_size)
    final_patches, filtered_rois = Finale_Fusion_patches(img0, merged_rois, overlap_thresh)
    print(f"🔹 Nombre total de clusters : {len(merged_rois)}")
    print(f"🔹 Nombre de patchs après fusion/filtrage : {len(filtered_rois)}")

    # Coins supérieurs gauche des ROI (coordonnées globales)
    patch_origins = [(roi[0], roi[1]) for roi in filtered_rois]

    # ── 2. Boucle temporelle : segmentation Cellpose sur tout le stack ────
    all_df = []
    for t, img in enumerate(stack):
        # Extraire les patches aux positions fixes
        current_patches = [
            img[y0:y0 + patch_size, x0:x0 + patch_size]
            for (x0, y0) in patch_origins
        ]

        df = Cellpose_Analyse_Track_GPU(
            model=cellpose_model,
            final_patches=current_patches,
            patch_origins=patch_origins,
            time_index=t,
            diameter_cellpose=diameter_cellpose,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_area=min_area,
            max_area=max_area,
            min_circularity=min_circularity,
            max_axis_ratio=max_axis_ratio,
            top_hat_radius=top_hat_radius,
            flag_visu=False
        )
        all_df.append(df)

    # ── 3. Linking TrackPy ────────────────────────────────────────────────
    tracks_df = pd.concat(all_df, ignore_index=True)
    counts    = tracks_df.groupby("frame").size()
    tracks    = tp.link_df(tracks_df, search_range=search_range, memory=memory)

    return tracks, counts


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13 — FONCTION PRINCIPALE DE SUIVI
# ═══════════════════════════════════════════════════════════════════════════

def Tracking_Puncta_TrackPy(
    output_dir: str,
    path: str,
    stack: np.ndarray,
    times: list,
    pretraitement_sigma: float,
    clip_limit: float,
    patch_size: int,
    cnn_threshold: float,
    model_cnn: nn.Module,
    cnn_batch_size: int,
    overlap_thresh: float,
    cellpose_model,
    cellprob_threshold: float,
    flow_threshold: float,
    min_area: int,
    max_area: int,
    min_circularity: float,
    max_axis_ratio: float,
    top_hat_radius: int,
    diameter_cellpose: float,
    diameter_trackpy: int,
    search_range: float,
    memory: int,
    threshold_filtered: int,
    flag_CNN: bool,
):
    """
    Point d'entrée principal pour l'analyse complète d'un stack de puncta.

    Orchestre l'ensemble du pipeline de détection, suivi et visualisation :
    selon ``flag_CNN``, utilise soit TrackPy seul, soit le pipeline CNN +
    Cellpose + TrackPy.  Génère et sauvegarde 7 figures d'analyse.

    Paramètres
    ----------
    output_dir : str
        Dossier de sortie pour les figures.
    path : str
        Chemin du dossier source des images (non utilisé directement ici,
        conservé pour traçabilité).
    stack : np.ndarray, shape (T, H, W)
        Stack temporel chargé en mémoire.
    times : list of int
        Indices temporels extraits des noms de fichiers.
    pretraitement_sigma : float
        Sigma gaussien du prétraitement.
    clip_limit : float
        Clip CLAHE du prétraitement.
    patch_size : int
        Taille des patches CNN (px).
    cnn_threshold : float
        Seuil CNN pour retenir un soma.
    model_cnn : nn.Module
        CNN soma (``SomaCNN``).
    cnn_batch_size : int
        Batch size CNN.
    overlap_thresh : float
        Seuil de chevauchement des ROI.
    cellpose_model : cellpose.models.Cellpose
        Modèle Cellpose.
    cellprob_threshold : float
        Seuil Cellpose.
    flow_threshold : float
        Seuil de flux Cellpose.
    min_area : int
        Aire minimale des puncta.
    max_area : int
        Aire maximale des puncta.
    min_circularity : float
        Circularité minimale.
    max_axis_ratio : float
        Rapport d'axes maximal.
    top_hat_radius : int
        Rayon top-hat.
    diameter_cellpose : float
        Diamètre Cellpose.
    diameter_trackpy : int
        Diamètre TrackPy.
    search_range : float
        Portée de recherche TrackPy.
    memory : int
        Mémoire TrackPy.
    threshold_filtered : int
        Longueur minimale de trajectoire conservée (``tp.filter_stubs``).
    flag_CNN : bool
        True  → pipeline CNN + Cellpose + TrackPy.
        False → TrackPy seul.

    Sorties sauvegardées
    --------------------
    - ``histogramme_trajectoires[_CNN].png``
    - ``msd_fit[_CNN].png``
    - ``nombre_puncta_par_frame[_CNN].png``
    - ``trajectoires_moyenne[_CNN].png``
    - ``densite_puncta[_CNN].png``
    - ``distribution_vitesses[_CNN].png``
    - ``trajectoires_vitesse[_CNN].png``
    """
    tp.quiet()
    mean_img = stack.mean(axis=0)   # image moyenne pour la visualisation des trajectoires

    # ── Choix du moteur de détection ──────────────────────────────────────
    if not flag_CNN:
        tracks, counts = TrackPy_Construction(
            stack, pretraitement_sigma, clip_limit,
            diameter_trackpy, search_range, memory
        )
    else:
        tracks, counts = TrackPy_Construction_CNN_Cellpose(
            stack, pretraitement_sigma, clip_limit,
            patch_size, cnn_threshold, model_cnn, cnn_batch_size,
            overlap_thresh, cellpose_model, cellprob_threshold,
            flow_threshold, min_area, max_area, min_circularity,
            max_axis_ratio, top_hat_radius, diameter_cellpose,
            search_range, memory
        )

    # ── Filtrage des trajectoires trop courtes ─────────────────────────────
    tracks_filtered = tp.filter_stubs(tracks, threshold=threshold_filtered)
    track_length    = tracks.groupby("particle").size()   # longueur AVANT filtrage

    # ── Génération des 7 figures d'analyse ───────────────────────────────
    Plot_and_Save_Longueur_Trajectoires(output_dir, track_length, flag_CNN)
    Plot_and_Save_Coefficient_Diffusion(output_dir, tracks_filtered, flag_CNN)
    Plot_and_Save_Nombre_Puncta_per_Frame(output_dir, counts, flag_CNN)
    Plot_and_Save_Trajectoires(mean_img, output_dir, tracks_filtered, flag_CNN)
    Plot_and_Save_DensityMap(mean_img, output_dir, tracks_filtered, flag_CNN)
    Plot_and_Save_Distribution_Vitesse(output_dir, tracks_filtered, flag_CNN)
    Plot_and_Save_Trajectoire_Vitesse(mean_img, output_dir, tracks_filtered, flag_CNN)