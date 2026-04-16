# Application Puncta Tracking

Suivi automatisé et analyse du mouvement des puncta dans les somas neuronaux, à partir d'images de microscopie de fluorescence en time-lapse.

Développé au LP2I Bordeaux — Arnaud HUBER, Ingénieur de Recherche  
Contact : huber@lp2ib.in2p3.fr

---

## Démarrage rapide — Récupérer le code

**GitHub :**
```bash
git clone https://github.com/ahuber33/Application_Puncta_Tracking.git
```

Puis se déplacer dans le dossier du projet :
```bash
cd Application_Puncta_Tracking
```

Une fois cloné, vous avez accès au fichier `Include_Puncta_Tracking.py` qui regroupe les différentes fonctions utilisés et qui sont détaillés plus loin dans ce README ainsi qu'au fichier `puncta_tracking_app.py` qui fait office de main et qui gère l'application ainsi que l'appel aux différentes fonctions.

Si vous désirez juste installer/créer l'executable, dirigez vous de suite vers la section [Installation](#4-installation).

---

## Table des matières

1. [Contexte scientifique](#1-contexte-scientifique)
2. [Vue d'ensemble du pipeline](#2-vue-densemble-du-pipeline)
3. [Structure du projet](#3-structure-du-projet)
4. [Installation](#4-installation)
5. [Utilisation](#5-utilisation)
6. [Référence des paramètres](#6-référence-des-paramètres)
7. [Référence des fonctions](#7-référence-des-fonctions)
8. [Fichiers de sortie et interprétation des résultats](#8-fichiers-de-sortie-et-interprétation-des-résultats)
9. [Performances et limitations](#9-performances-et-limitations)
10. [FAQ / Dépannage](#10-faq--dépannage)

---

## 1. Contexte scientifique

Cet outil a été développé pour étudier la **dynamique intracellulaire des puncta** dans les somas neuronaux dans différentes conditions expérimentales (ex. : exposition aux métaux lourds tels que le Cadmium). 

---

## 2. Vue d'ensemble du pipeline

```
Stack TIFF (T frames, plan Z unique)
        │
        ▼
 Prétraitement — flou gaussien + CLAHE
        │
        ▼
 Détection des somas — CNN (SomaCNN) + détection de blobs (blob_log)
        │
        ▼
 Clustering des patches (KDTree + Union-Find) + fusion des ROI
        │                         [calculé une seule fois sur la frame 0]
        ▼
 ┌─── Pour chaque frame t ──────────────────────────────────────┐
 │  Filtrage top-hat blanc                                      │
 │  Segmentation des puncta — Cellpose (cyto3)               │
 │  Filtrage morphologique (aire, circularité, forme)           │
 └──────────────────────────────────────────────────────────────┘
        │
        ▼
 Liaison temporelle — TrackPy
        │
        ▼
 Export des figures (MSD, trajectoires, vitesse, densité)
```

Deux modes de détection sont disponibles, contrôlés par `flag_CNN` dans `puncta_tracking_app.py` :

| Mode | Description |
|---|---|
| `flag_CNN = True` | **CNN + Cellpose + TrackPy** (recommandé) — détection des ROI guidée par le soma, segmentation précise des puncta via Cellpose. |
| `flag_CNN = False` | **TrackPy seul** — plus rapide, détection basée sur les maxima d'intensité. Adapté aux images peu denses. |

---

## 3. Structure du projet

Le dépôt contient uniquement le code. Les données d'images et les résultats sont stockés où vous le souhaitez sur votre machine — le chemin est configurable via l'application.

```
Application_Puncta_Tracking_ICS/
│
├── Include_Puncta_Tracking.py  # Toutes les fonctions d'analyse (bibliothèque)
├── puncta_tracking_app.py      # Script principal de l'application — paramètres et boucle d'analyse
└── soma_cnn_test.pth           # Poids CNN pré-entraînés (détection des somas)
```

Lors de l'analyse, les figures de sortie sont sauvegardées dans le dossier défini par `output_dir` dans `puncta_tracking_app.py` et défini via l'application.

### Convention de nommage des fichiers d'entrée

Chaque fichier `.tif` est **une frame temporelle unique** (plan Z unique). Les fichiers sont identifiés et triés par l'indice temporel dans leur nom :

```
CTRL 1 _ mitotracker_lysotracker_002_t001.tif
CTRL 1 _ mitotracker_lysotracker_002_t002.tif
...
CTRL 1 _ mitotracker_lysotracker_002_t150.tif
```

Le motif `_t(\d+)` est utilisé pour extraire et trier les frames numériquement. Adapter le motif glob dans `puncta_tracking_app.py` selon la convention de nommage de votre acquisition.

---

## 4. Installation

### 4.1 Prérequis

Il est fortement recommandé d'utiliser un environnement virtuel via conda permettant de garantir que l'ensemble des dépendances sont présentes et en accord avec ce que veut faire l'application. Pour cela un fichier `environment.yml` est dispo afin de pouvoir le créer sur votre machine via un Anaconda Prompt (IMPORTANT)

```bash
conda env create -f environment.yml
conda activate lyso_tracking
```

Un fichier requirements.txt a ensuite été crée via 

```bash
pip list > requirements.txt
```

afin de recenser l'intégralité des bibliothèques utilisés ainsi que leur versions dans le cas d'une installation externe/autre.

Si PyInstaller n'est pas installé, il faut également le faire pour pouvoir générer l'exectuable :

```bash
pip install pyinstaller
```

Ensuite pour créer l'executable :

```bash
pyinstaller ^
--noconfirm ^
--windowed ^
--name PunctaTracking ^
--clean ^
--collect-all torch ^
--collect-all PyQt5 ^
--collect-all cellpose ^
--collect-all tifffile ^
--collect-all skimage ^
--hidden-import=trackpy ^
--icon=logo.ico ^
puncta_tracking_app.py
```

Vous aurez en sortie un executable ainsi qu'un dossier `_internal` qu'il sera nécessaire de copier avec l'executable !!!!

### 4.2 Vérifier l'activation GPU

Exécuter ce code Python pour confirmer que tout est correctement installé :
```python
import torch
from cellpose import models

print("Version PyTorch :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nom du GPU      :", torch.cuda.get_device_name(0))

model = models.Cellpose(gpu=torch.cuda.is_available(), model_type='cyto3')
print("Cellpose GPU    :", model.gpu)
```

**Sortie attendue (avec GPU) :**
```
Version PyTorch : 2.6.0+cu124
CUDA disponible : True
Nom du GPU      : <nom de votre GPU>
Cellpose GPU    : True
```

**Sortie attendue (CPU uniquement) :**
```
Version PyTorch : 2.6.0+cpu
CUDA disponible : False
Cellpose GPU    : False
```

> Si `CUDA disponible` affiche `False` malgré un GPU NVIDIA, consulter la [FAQ](#10-faq--dépannage).

---

## 5. Utilisation

TO DO 
---

## 6. Référence des paramètres

### Prétraitement

| Paramètre | Valeur défaut | Description |
|---|---|---|
| `pretraitement_sigma` | `1.5` | Écart-type du flou gaussien (px) appliqué avant la détection de blobs. |
| `clip_limit` | `0.02` | Limite CLAHE de rehaussement de contraste ∈ [0, 1]. |

### Détection des somas (CNN)

| Paramètre | Valeur défaut | Description |
|---|---|---|
| `cnn_threshold` | `0.23` | Probabilité CNN minimale pour classer un patch comme soma. Baisser → plus de détections (risque de faux positifs). |
| `patch_size` | `64` | Taille (px) du patch carré extrait autour de chaque candidat blob. |
| `cnn_batch_size` | `512` | Patches par passe avant CNN. Réduire en cas d'erreur OOM CUDA. |

### Fusion des ROI

| Paramètre | Valeur défaut | Description |
|---|---|---|
| `overlap_thresh` | `0.1` | Fraction de chevauchement au-delà de laquelle une ROI plus petite est supprimée (10 %). |

### Segmentation Cellpose

| Paramètre | Valeur défaut | Description |
|---|---|---|
| `diameter_cellpose` | `11` | Diamètre estimé des puncta en pixels. Doit correspondre à la taille réelle dans l'image. |
| `cellprob_threshold` | `0.1` | Seuil de probabilité Cellpose. Baisser → plus de détections. |
| `flow_threshold` | `0.5` | Seuil de cohérence des flux Cellpose ∈ [0, 1]. Plus élevé → segmentation plus stricte. |
| `min_area` | `5` | Aire minimale (px²) pour conserver un objet détecté. Filtre les artefacts sub-pixelliques. |
| `max_area` | `500` | Aire maximale (px²). Filtre les agrégats et les grands artefacts. |
| `min_circularity` | `0.5` | Circularité minimale (4πA/P²). Filtre les fragments de dendrites allongés. |
| `max_axis_ratio` | `2` | Rapport maximal grand axe / petit axe. Filtre supplémentaire pour les formes allongées. |
| `top_hat_radius` | `10` | Rayon du disque (px) pour le top-hat blanc. Doit être supérieur au rayon des puncta. |

### Liaison TrackPy

| Paramètre | Valeur défaut | Description |
|---|---|---|
| `diameter_trackpy` | `11` | Diamètre (entier impair) des particules pour `tp.locate` (mode TrackPy seul). |
| `search_range` | `4` | Déplacement maximal (px) entre deux frames. Utiliser `2` en mode CNN+Cellpose. |
| `memory` | `1` | Nombre de frames pendant lesquelles une particule disparue est mémorisée avant abandon. |
| `threshold_filtered` | `20` | Longueur minimale de trajectoire (frames) conservée après `tp.filter_stubs`. |

---

## 7. Référence des fonctions

### 7.0 Récapitulatif

| Fonction | Fichier | Rôle |
|---|---|---|
| `get_device()` | `Include` | Détecte et retourne le meilleur dispositif disponible (CUDA / CPU). |
| `detect_candidates(image)` | `Include` | Détection LoG des régions candidates (somas) dans une image normalisée. |
| `build_blob_dataset(...)` | `Include` | Extrait un patch carré normalisé centré sur chaque blob détecté. |
| `SomaCNN` | `Include` | Classe PyTorch : CNN 3 couches pour classification binaire soma / non-soma. |
| `Pretraitement_image(...)` | `Include` | Flou gaussien + CLAHE : prétraitement standard d'une image de microscopie. |
| `CNN_Proba_Construction(...)` | `Include` | Inférence CNN par batch GPU ; retourne les probabilités et les patches retenus. |
| `Patch_construction(...)` | `Include` | Wrapper : détection LoG → extraction de patches. |
| `CNN_Patches_Construction(...)` | `Include` | Pipeline haut-niveau : prétraitement → détection → CNN. |
| `Clusterization_detected_patches(...)` | `Include` | Clustering KDTree + Union-Find des patches → bounding boxes ROI. |
| `Finale_Fusion_patches(...)` | `Include` | Suppression des ROI largement incluses dans une ROI plus grande. |
| `Cellpose_Analyse_Track_GPU(...)` | `Include` | Top-hat + segmentation Cellpose + filtrage morphologique → DataFrame puncta. |
| `Plot_and_Save_*` | `Include` | 7 fonctions de visualisation : trajectoires, MSD, densité, vitesse, histogrammes. |
| `TrackPy_Construction(...)` | `Include` | Détection + liaison TrackPy seul (mode sans CNN). |
| `TrackPy_Construction_CNN_Cellpose(...)` | `Include` | Pipeline CNN + Cellpose + TrackPy complet sur tout le stack. |
| `Tracking_Lyso_TrackPy(...)` | `Include` | **Point d'entrée principal** : orchestre tout le pipeline et sauvegarde les 7 figures. |

---

### 7.1 `get_device()`

```python
device = Include_Puncta_Tracking.get_device()
```

Détecte automatiquement le meilleur dispositif PyTorch disponible.

- Si `torch.cuda.is_available()` → retourne `torch.device('cuda')` et affiche le nom du GPU.
- Sinon → retourne `torch.device('cpu')`.

> Appelée une seule fois au démarrage. Le dispositif retourné est ensuite transmis à tous les modèles.

---

### 7.2 `detect_candidates(image)`

```python
blobs = detect_candidates(image)   # retourne np.ndarray shape (N, 3)
```

Détecte les régions candidates (somas) par la méthode **Laplacian of Gaussian (LoG)**.

**Étapes de traitement :**
1. Normalisation z-score de l'image (moyenne nulle, variance unitaire) — stabilise la détection quel que soit le niveau de signal.
2. `skimage.feature.blob_log` avec sigma ∈ [5, 18] px (20 niveaux), seuil `0.3`.

**Retourne :** `blobs` — tableau de forme `(N, 3)`, colonnes `[y, x, sigma]`.

> La normalisation z-score est critique : sans elle, la même structure imagée à des niveaux d'intensité différents produirait des résultats incohérents entre frames.

---

### 7.3 `build_blob_dataset(image, blobs, patch_size=64, allow_partial=False)`

```python
patches, valid_blobs = build_blob_dataset(image, blobs, patch_size=64)
```

Extrait un patch carré normalisé centré sur chaque blob détecté.

- `allow_partial=False` (défaut) : les blobs dont le patch dépasserait les bords de l'image sont ignorés.
- `allow_partial=True` : les patches sont tronqués aux bords (taille variable).
- Chaque patch est normalisé min-max indépendamment ∈ [0, 1].

**Retourne :**
- `patches` — `np.ndarray` shape `(M, patch_size, patch_size)`
- `valid_blobs` — `list of tuple (y, x, sigma)`, même ordre que `patches`

---

### 7.4 `SomaCNN` (classe `nn.Module`)

```python
model = SomaCNN()
model.load_state_dict(torch.load("soma_cnn_test.pth", map_location=device))
model = model.to(device)
model.eval()
```

Réseau de neurones convolutif pour la **classification binaire** patch → soma (1) / non-soma (0).

**Architecture :**
```
Conv2d(1→32, 3×3)  + ReLU + MaxPool2d(2)
Conv2d(32→64, 3×3) + ReLU + MaxPool2d(2)
Conv2d(64→128, 3×3)+ ReLU + AdaptiveAvgPool2d(1×1)
Linear(128→1) + Sigmoid  →  probabilité ∈ [0, 1]
```

- **Entrée :** tenseur `(B, 1, H, W)` — batch de B patches en niveaux de gris.
- **Sortie :** tenseur `(B, 1)` — probabilité d'être un soma pour chaque patch.

> `AdaptiveAvgPool2d(1)` rend le réseau invariant à la taille du patch en entrée. Les poids sont chargés depuis `soma_cnn_test.pth`.

---

### 7.5 `Pretraitement_image(img, sigma, clip_limit)`

```python
img_eq = Pretraitement_image(img, sigma=1.5, clip_limit=0.02)
```

Prétraitement standard en deux étapes pour les images de microscopie de fluorescence :

1. **Flou gaussien** (`sigma`) : atténue le bruit haute fréquence sans dégrader les structures d'intérêt.
2. **CLAHE** (`clip_limit`) : rehaussement de contraste local adaptatif — améliore la visibilité des puncta sans amplifier excessivement le bruit.

Retourne une image `float64` ∈ [0, 1].

---

### 7.6 `CNN_Proba_Construction(...)`

```python
all_probs, soma_patches = CNN_Proba_Construction(
    blobs_valid, patches_s, model_cnn, cnn_threshold=0.23, half=32, batch_size=512
)
```

Exécute l'inférence CNN par batches et filtre les patches selon le seuil de probabilité.

**Optimisations GPU :**
- Passe avant groupée (`batch_size` patches simultanément) vs un appel par patch.
- `pin_memory=True` sur les tenseurs CPU pour transfert DMA rapide vers GPU.
- Copie numpy uniquement une fois par batch.
- Dispositif suivi automatiquement depuis `next(model.parameters()).device`.

**Retourne :**
- `all_probs` — `np.ndarray (N,)` : probabilité CNN pour chaque patch candidat.
- `soma_patches` — `list of tuple (x0, y0, prob, patch)` : patches retenus (`prob ≥ cnn_threshold`).

---

### 7.7 `Clusterization_detected_patches(img, soma_patches, patch_size)`

```python
merged_rois = Clusterization_detected_patches(img, soma_patches, patch_size=64)
```

Fusionne les patches somas qui se chevauchent en ROI via **KDTree + Union-Find**.

**Algorithme :**
1. Calcul du centre de chaque patch.
2. Recherche de toutes les paires de centres distants de moins de `patch_size / 1.25` px via `scipy.cKDTree.query_pairs()` — complexité **O(n log n)**.
3. Fusion des paires connectées par Union-Find avec compression de chemin.
4. Construction des bounding boxes `(xmin, ymin, xmax, ymax)` par cluster, clampées dans les dimensions de l'image.

> Cette implémentation remplace un BFS O(n²) précédent : pour n=500 patches, le gain de vitesse est d'environ **×100**.

---

### 7.8 `Finale_Fusion_patches(img, merged_rois, overlap_thresh)`

```python
final_patches, filtered_rois = Finale_Fusion_patches(img, merged_rois, overlap_thresh=0.1)
```

Supprime les ROI largement contenues dans une ROI plus grande.

- **Critère :** si l'intersection d'une ROI avec une autre couvre ≥ `overlap_thresh` (10 %) de sa propre surface, elle est supprimée.
- Retourne les sous-images (`final_patches`) et les coordonnées (`filtered_rois`) des ROI conservées.

---

### 7.9 `Cellpose_Analyse_Track_GPU(...)`

```python
df = Cellpose_Analyse_Track_GPU(
    model, final_patches, patch_origins, time_index=t,
    diameter_cellpose=11, cellprob_threshold=0.1, flow_threshold=0.5,
    min_area=5, max_area=500, min_circularity=0.5, max_axis_ratio=2,
    top_hat_radius=10, flag_visu=False, cellpose_batch_size=24
)
```

Segmente les puncta dans chaque ROI et retourne un DataFrame compatible TrackPy.

**Pipeline interne :**

| Étape | Description |
|---|---|
| **1 — Top-hat** | Filtre top-hat blanc morphologique (disque de rayon `top_hat_radius`) : supprime le fond à variation lente et rehausse les petites structures brillantes. |
| **2 — Cellpose** | Segmentation d'instances par batch (`cellpose_batch_size` patches). Cache CUDA vidée après chaque frame. |
| **3 — Filtrage morphologique** | Suppression des objets hors critères (aire, circularité, rapport d'axes) du masque de segmentation. |
| **4 — Conversion de coordonnées** | Centroïdes locaux → coordonnées globales (centroïde + offset `patch_origin`). |
| **5 — Visualisation optionnelle** | Si `flag_visu=True` : figure à 3 panneaux par patch (original / segmentation brute / filtrée). |

**Colonnes du DataFrame retourné :**

| Colonne | Description |
|---|---|
| `frame` | Indice temporel de la frame courante. |
| `patch` | Indice du patch ROI source. |
| `x` | Coordonnée X globale du centroïde (px). |
| `y` | Coordonnée Y globale du centroïde (px). |
| `area` | Aire du puncta (px²). |
| `circularity` | Circularité = 4πA/P² ∈ [0, 1]. |

---

### 7.10 `TrackPy_Construction(...)`

```python
tracks, counts = TrackPy_Construction(stack, sigma=1.5, clip_limit=0.02,
                                       diameter=11, search_range=4, memory=4)
```

Détection et liaison **TrackPy seul** (sans CNN, sans Cellpose).

**Stratégie `minmass` adaptatif :**
- Le seuil `minmass` est calibré sur le **99e percentile** des masses détectées sur la frame 0.
- Il est ensuite ajusté proportionnellement à l'intensité moyenne de chaque frame, compensant les variations de fluorescence au cours de l'acquisition.

---

### 7.11 `TrackPy_Construction_CNN_Cellpose(...)`

```python
tracks, counts = TrackPy_Construction_CNN_Cellpose(stack, ...)
```

Pipeline complet **CNN + Cellpose + TrackPy**.

- Les ROI somas sont calculées **une seule fois sur `stack[0]`** et réutilisées pour toutes les frames.
- Pour chaque frame, les patches sont extraits aux positions fixes puis segmentés par Cellpose.
- La liaison temporelle est effectuée par `tp.link_df()` sur la concaténation des DataFrames par frame.

> ⚠️ **Hypothèse de ROI fixes** : cette approche suppose que les corps cellulaires ne se déplacent pas significativement au cours de l'acquisition. Dans le cas contraire, les ROI doivent être recalculées périodiquement.

---

### 7.12 `Tracking_Lyso_TrackPy(...)` — Point d'entrée principal

```python
Include_Lyso_Tracking.Tracking_Lyso_TrackPy(
    output_dir, path, stack, times,
    pretraitement_sigma, clip_limit, patch_size, cnn_threshold,
    model_cnn, cnn_batch_size, overlap_thresh, cellpose_model,
    cellprob_threshold, flow_threshold, min_area, max_area,
    min_circularity, max_axis_ratio, top_hat_radius, diameter_cellpose,
    diameter_trackpy, search_range, memory, threshold_filtered,
    flag_CNN  # True ou False
)
```

Orchestre l'intégralité du pipeline et sauvegarde les 7 figures d'analyse. Appelle en interne soit `TrackPy_Construction`, soit `TrackPy_Construction_CNN_Cellpose` selon `flag_CNN`, puis applique `tp.filter_stubs` avant la génération des figures.

---

## 8. Fichiers de sortie et interprétation des résultats

### 8.1 Figures générées

Toutes les figures sont sauvegardées dans le dossier défini par `output_dir`. Le suffixe `_CNN` est ajouté lorsque `flag_CNN = True`.

| Fichier | Description |
|---|---|
| `histogramme_trajectoires[_CNN].png` | Distribution des longueurs de trajectoires (échelle Y logarithmique). |
| `msd_fit[_CNN].png` | Courbes MSD individuelles (gris) + MSD moyen (rouge) + ajustement loi de puissance (bleu, exposant α). |
| `nombre_puncta_par_frame[_CNN].png` | Nombre de puncta détectés par frame au cours du temps. |
| `trajectoires_moyenne[_CNN].png` | Toutes les trajectoires filtrées superposées à l'image moyenne du stack. |
| `densite_puncta[_CNN].png` | Carte de densité KDE des positions des puncta sur l'image moyenne. |
| `distribution_vitesses[_CNN].png` | Distribution des vitesses instantanées (px/frame, échelle Y log). |
| `trajectoires_vitesse[_CNN].png` | Trajectoires colorées par vitesse instantanée (colormap `plasma`). |

---

### 8.2 Exposant de diffusion α (ajustement MSD)

Le MSD moyen est ajusté par une loi de puissance **MSD(τ) ∝ τ^α** par régression linéaire en log-log sur le **premier tiers** des points de lag-time (pour éviter le régime de saturation aux temps longs).

| Valeur de α | Interprétation |
|---|---|
| α ≈ 1 | Diffusion normale (brownienne) — mouvement aléatoire sans contrainte. |
| α < 1 | Sous-diffusion — puncta confiné ou freiné (cage cytosquelette, encombrement). |
| α > 1 | Supra-diffusion — transport actif (moteurs moléculaires kinésine/dynéine). |
| α ≈ 2 | Mouvement balistique — transport dirigé le long des microtubules. |

> **Important :** les valeurs de MSD sont calculées en **pixels²/frame** (unités normalisées : `mpp=1`, `fps=1`). Pour convertir en unités physiques (µm²/s), multiplier les MSD par `taille_pixel²` et diviser les lag-times par la fréquence d'acquisition (images/s).

---

### 8.3 Histogramme des longueurs de trajectoires

Cet histogramme (axe Y log) révèle la distribution des longueurs de pistes sur l'ensemble des particules détectées. Une bonne segmentation produit typiquement une décroissance approximativement exponentielle — la plupart des pistes courtes correspondent à des détections spurieuses ou des événements de clignotement, tandis que les pistes longues correspondent à des puncta réellement suivis sur de nombreuses frames.

Utiliser cette figure pour choisir `threshold_filtered` : le définir au-delà du coude de la distribution pour ne conserver que les puncta bien suivis.

---

### 8.4 Nombre de puncta par frame

Ce graphe temporel montre le nombre total de puncta détectés par frame. Il est utile pour :
- Détecter le **photoblanchiment** (diminution systématique au cours du temps).
- Vérifier la **cohérence de la détection** (fortes fluctuations frame à frame → paramètres `cnn_threshold` ou `diameter_cellpose` mal réglés).
- Identifier des **artefacts d'acquisition** (dérive du focus, vibrations).

---

### 8.5 Carte de densité spatiale

La carte KDE agrège toutes les positions des puncta sur l'ensemble des frames. Les zones chaudes (régions claires sur la colormap `inferno`) indiquent des zones de concentration préférentielle ou de temps de résidence élevé. Cela peut refléter :
- Un **regroupement périnucléaire** (typique des puncta sous stress).
- Un **transport actif** le long de branches dendritiques spécifiques.
- Des **zones de confinement** induites par les structures du cytosquelette.

---

### 8.6 Distribution des vitesses

La vitesse instantanée est calculée frame par frame : `v = √(Δx² + Δy²)` en px/frame. L'histogramme en log-Y peut révéler **deux populations distinctes** :
- Un pic lent (v faible) correspondant à un mouvement diffusif ou confiné.
- Une queue rapide (v élevée) correspondant à des événements de transport actif par moteurs moléculaires.

Un déplacement de la population rapide sous conditions expérimentales (ex. : exposition au Cd) peut indiquer une perturbation du transport basé sur les microtubules.

---

### 8.7 Trajectoires colorées par vitesse

Les trajectoires sont tracées sur l'image moyenne, chaque point étant coloré selon la vitesse instantanée (colormap `plasma`, violet foncé = lent, jaune = rapide). Cette figure combine informations spatiales et dynamiques, et permet d'identifier visuellement les corridors de transport ou les zones de confinement.

---

## 9. Performances et limitations

### 9.1 Optimisations implémentées

| Optimisation | Gain |
|---|---|
| **Inférence CNN GPU par batch** avec `pin_memory` | ×10–50 vs. un appel par patch |
| **Clustering KDTree + Union-Find** O(n log n) | ×100 pour n=500 patches vs. BFS O(n²) précédent |
| **ROI somas fixes** calculées une fois sur la frame 0 | Cellpose appelé uniquement sur les zones pertinentes pour toutes les T frames |
| **Inférence Cellpose par batch** (`cellpose_batch_size`) | Amélioration significative de l'utilisation GPU |
| `torch.cuda.empty_cache()` après chaque frame | Prévient la saturation de la VRAM sur les stacks longs |

---

### 9.2 Limitations connues

**Hypothèse de ROI fixes**  
Les ROI somas sont calculées sur la **première frame uniquement** et maintenues fixes sur tout le stack. Si les cellules dérivent significativement au cours de l'acquisition (instabilité mécanique, photoblanchiment intense), les ROI peuvent se désaligner sur les frames tardives. Dans ce cas, diviser le stack en sous-stacks et relancer l'analyse indépendamment est recommandé.

**Spécificité du modèle CNN**  
Le modèle `SomaCNN` (`soma_cnn_test.pth`) a été entraîné sur un dispositif d'imagerie spécifique (grossissement, fluorophore, taille de pixel, niveau de bruit). Les performances peuvent se dégrader significativement si ces paramètres diffèrent. Un ré-entraînement sur des images représentatives du nouveau dispositif est fortement recommandé avant utilisation dans un nouveau contexte expérimental.

**Limitations du mode TrackPy seul**  
Le mode TrackPy seul (`flag_CNN=False`) repose sur la détection de maxima d'intensité et n'est pas bien adapté aux images denses où les puncta se chevauchent spatialement. Dans ce cas, le mode CNN+Cellpose offre une séparation d'instances significativement meilleure.

**MSD en unités normalisées**  
Les valeurs de MSD sont calculées avec `fps=1` et `mpp=1` (normalisées). L'exposant α est sans dimension et valide quelle que soit l'unité, mais les coefficients de diffusion absolus nécessitent une conversion utilisant la taille réelle du pixel (µm/px) et la fréquence d'acquisition (images/s).

**Régression MSD sur le premier tiers seulement**  
L'ajustement loi de puissance utilise uniquement le premier tiers des points de lag-time, choix heuristique pour éviter le régime de saturation. Pour des trajectoires très courtes (proches de `threshold_filtered`), cela peut laisser trop peu de points pour un ajustement fiable, surestimant ou sous-estimant potentiellement α.

**Utilisation mémoire sur les grands stacks**  
Le stack complet est chargé en RAM d'un seul coup (`np.array([tiff.imread(f) for f in files])`). Pour les acquisitions très longues (> 500 frames à haute résolution), cela peut dépasser la mémoire système disponible. Dans ce cas, charger les frames par blocs et concaténer les DataFrames résultants avant la liaison est conseillé.

---

## 10. FAQ / Dépannage

**`torch.cuda.is_available()` retourne `False`**  
→ L'installation PyTorch ne correspond pas au pilote CUDA. Exécuter `nvidia-smi` et vérifier la CUDA Version. Puis réinstaller PyTorch :
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

**Erreur CUDA out of memory**  
→ Réduire les tailles de batch dans `Lyso_Tracking.py` :
```python
cnn_batch_size      = 128
cellpose_batch_size = 8
```

---

**Aucun soma détecté sur une image**  
→ Essayer de baisser `cnn_threshold` (ex. de `0.23` à `0.15`). Le CNN a été entraîné sur un dispositif d'imagerie spécifique — si les conditions d'acquisition diffèrent significativement (grossissement, fluorophore, niveau de bruit), le modèle peut nécessiter un ré-entraînement.

---

**Trop ou trop peu de puncta comptés**  
→ Ajuster les paramètres Cellpose : augmenter `min_area` pour supprimer le bruit, diminuer `diameter_cellpose` si les puncta paraissent plus petits qu'attendu, ou affiner `min_circularity` pour filtrer plus agressivement les fragments de dendrites.

---

**`FileNotFoundError: soma_cnn_test.pth`**  
→ Le fichier de poids CNN doit être placé dans le **même dossier** que `Lyso_Tracking.py`. Ne pas le renommer.

---

**Téléchargement du modèle Cellpose `cyto3`**  
→ Le modèle Cellpose `cyto3` (~200 Mo) est téléchargé automatiquement au premier lancement. Une connexion internet n'est requise que pour ce téléchargement initial. Il est ensuite mis en cache localement dans `C:\Users\<votre_nom>\.cellpose\models\` et réutilisé pour tous les lancements suivants.

---

**`HTTPError: HTTP Error 500` lors du chargement de Cellpose**  
→ Problème connu de Cellpose 2.x. Deux options pour le résoudre :

**Option 1 — Téléchargement direct depuis cellpose.org (le plus rapide)**  
Télécharger directement dans le navigateur :
- https://www.cellpose.org/models/cyto3
- https://www.cellpose.org/models/size_cyto3.npy

**Option 2 — Téléchargement depuis BioImage.IO Model Zoo**
1. Aller sur https://bioimage.io/#/artifacts/famous-fish
2. Cliquer sur l'icône de téléchargement de la fiche modèle
3. Sélectionner *Download by Weight Format* → *Pytorch State Dict*

Après téléchargement (les deux options), placer les fichiers dans le dossier cache Cellpose :
- **Windows** : `C:\Users\<votre_nom>\.cellpose\models\`
- **Linux / macOS** : `/home/<votre_nom>/.cellpose/models/`

Les fichiers doivent être nommés exactement `cyto3` et `size_cyto3.npy` (sans extension pour le premier).

> Sous Windows, le dossier `.cellpose` peut être masqué. Dans l'Explorateur de fichiers, activer **Affichage > Éléments masqués**.

Relancer le script — Cellpose trouvera les fichiers localement et ignorera le téléchargement.

---

**Spyder utilise le mauvais environnement Python**  
→ Toujours lancer Spyder depuis l'Anaconda Prompt après activation de l'environnement :
```bash
conda activate lyso_tracking
spyder
```
Dans Spyder, vérifier via **Outils > Préférences > Interpréteur Python** que le chemin pointe vers l'environnement `lyso_tracking`.

---

**Les trajectoires sont trop courtes ou fragmentées**  
→ Augmenter `search_range` si les puncta bougent vite entre frames, augmenter `memory` s'ils disparaissent temporairement, ou baisser `threshold_filtered` pour conserver des trajectoires plus courtes.

---

**Les positions des ROI dérivent au cours du temps**  
→ Les ROI somas sont calculées **une seule fois sur la première frame** et maintenues fixes sur tout le stack. Si les cellules dérivent significativement au cours de l'acquisition (instabilité de la platine, photoblanchiment), les ROI peuvent se désaligner sur les frames tardives. Dans ce cas, diviser le stack et relancer l'analyse sur chaque sous-stack indépendamment.

---

**L'exposant α du MSD semble peu fiable**  
→ Vérifier d'abord l'histogramme des longueurs de trajectoires. Si `threshold_filtered` est trop bas, de nombreuses pistes courtes seront incluses dans le calcul du MSD, produisant une courbe moyenne bruitée. Augmenter `threshold_filtered` (ex. à 30–50) et relancer pour obtenir une estimation d'α plus robuste.

---

*ICS Puncta Tracking — LP2I Bordeaux — Arnaud HUBER*