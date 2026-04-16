# -*- coding: utf-8 -*-
"""
puncta_tracking_app.py
====================
Interface graphique PyQt5 pour le pipeline de suivi de puncta.
Développé au LP2I Bordeaux — Arnaud HUBER

Lancement :
    python puncta_tracking_app.py
"""

import sys
import os
import glob
import re
import time
import subprocess
import traceback
from pathlib import Path

import numpy as np
import tifffile as tiff

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QLineEdit, QFileDialog,
    QGroupBox, QSlider, QDoubleSpinBox, QSpinBox, QComboBox,
    QProgressBar, QTextEdit, QTabWidget, QScrollArea, QFrame,
    QSizePolicy, QSplitter, QCheckBox, QMessageBox, QToolButton,
    QStatusBar
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QPropertyAnimation,
    QEasingCurve
)
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QPixmap, QIcon, QPainter,
    QLinearGradient, QBrush, QPen, QFontDatabase
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════
# PALETTE DE COULEURS — thème scientifique sombre
# ═══════════════════════════════════════════════════════════════════════

COLORS = {
    "bg_dark":      "#0D1117",
    "bg_panel":     "#161B22",
    "bg_card":      "#1C2333",
    "bg_input":     "#0D1117",
    "accent":       "#58A6FF",
    "accent_green": "#3FB950",
    "accent_orange":"#F78166",
    "accent_yellow":"#E3B341",
    "border":       "#30363D",
    "border_light": "#484F58",
    "text_primary": "#E6EDF3",
    "text_secondary":"#8B949E",
    "text_muted":   "#484F58",
    "success":      "#238636",
    "warning":      "#9E6A03",
    "error":        "#DA3633",
    "hover":        "#21262D",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_primary']};
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
    font-size: 13px;
}}

/* ── Groupbox ─────────────────────────────────────────────────────── */
QGroupBox {{
    background-color: {COLORS['bg_panel']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-size: 12px;
    font-weight: 600;
    color: {COLORS['text_secondary']};
    letter-spacing: 0.8px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: -1px;
    padding: 0 6px;
    background-color: {COLORS['bg_panel']};
    color: {COLORS['accent']};
}}

/* ── Labels ───────────────────────────────────────────────────────── */
QLabel {{
    color: {COLORS['text_primary']};
    background: transparent;
}}
QLabel#section_title {{
    font-size: 11px;
    font-weight: 600;
    color: {COLORS['text_secondary']};
    letter-spacing: 0.6px;
    text-transform: uppercase;
}}
QLabel#value_display {{
    color: {COLORS['accent']};
    font-weight: 700;
    font-size: 13px;
    min-width: 45px;
}}

/* ── Inputs ───────────────────────────────────────────────────────── */
QLineEdit {{
    background-color: {COLORS['bg_input']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 7px 10px;
    color: {COLORS['text_primary']};
    font-size: 13px;
    selection-background-color: {COLORS['accent']};
}}
QLineEdit:focus {{
    border: 1px solid {COLORS['accent']};
    background-color: {COLORS['bg_card']};
}}
QLineEdit:hover {{
    border: 1px solid {COLORS['border_light']};
}}

QDoubleSpinBox, QSpinBox {{
    background-color: {COLORS['bg_input']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 5px 8px;
    color: {COLORS['text_primary']};
    font-size: 13px;
    min-width: 80px;
}}
QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 1px solid {COLORS['accent']};
}}
QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    background-color: {COLORS['bg_card']};
    border: none;
    width: 18px;
}}
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {COLORS['text_secondary']};
    width: 0; height: 0;
}}
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {COLORS['text_secondary']};
    width: 0; height: 0;
}}

/* ── Boutons principaux ───────────────────────────────────────────── */
QPushButton#btn_primary {{
    background-color: {COLORS['accent']};
    color: #0D1117;
    border: none;
    border-radius: 7px;
    padding: 10px 22px;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.3px;
}}
QPushButton#btn_primary:hover {{
    background-color: #79C0FF;
}}
QPushButton#btn_primary:pressed {{
    background-color: #388BFD;
}}
QPushButton#btn_primary:disabled {{
    background-color: {COLORS['border']};
    color: {COLORS['text_muted']};
}}

QPushButton#btn_secondary {{
    background-color: transparent;
    color: {COLORS['accent']};
    border: 1px solid {COLORS['accent']};
    border-radius: 7px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton#btn_secondary:hover {{
    background-color: rgba(88, 166, 255, 0.1);
}}
QPushButton#btn_secondary:disabled {{
    color: {COLORS['text_muted']};
    border-color: {COLORS['border']};
}}

QPushButton#btn_danger {{
    background-color: transparent;
    color: {COLORS['accent_orange']};
    border: 1px solid {COLORS['accent_orange']};
    border-radius: 7px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton#btn_danger:hover {{
    background-color: rgba(247, 129, 102, 0.1);
}}

QPushButton#btn_browse {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_secondary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 500;
    min-width: 80px;
}}
QPushButton#btn_browse:hover {{
    background-color: {COLORS['hover']};
    color: {COLORS['text_primary']};
    border-color: {COLORS['border_light']};
}}

/* ── Barre de progression ─────────────────────────────────────────── */
QProgressBar {{
    background-color: {COLORS['bg_input']};
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    height: 10px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS['accent']}, stop:1 #A5D6FF);
    border-radius: 4px;
}}

/* ── Console de log ───────────────────────────────────────────────── */
QTextEdit#console {{
    background-color: #010409;
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    color: #A8D8A8;
    font-family: 'Consolas', 'Cascadia Code', 'Courier New', monospace;
    font-size: 12px;
    padding: 8px;
    selection-background-color: {COLORS['accent']};
}}

/* ── Onglets ──────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    background-color: {COLORS['bg_panel']};
    top: -1px;
}}
QTabBar::tab {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_secondary']};
    border: 1px solid {COLORS['border']};
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    padding: 8px 18px;
    font-size: 12px;
    font-weight: 500;
    margin-right: 3px;
}}
QTabBar::tab:selected {{
    background-color: {COLORS['bg_panel']};
    color: {COLORS['text_primary']};
    border-color: {COLORS['border']};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    background-color: {COLORS['hover']};
    color: {COLORS['text_primary']};
}}

/* ── Scrollbars ───────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {COLORS['bg_dark']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['border_light']};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['text_secondary']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

QScrollBar:horizontal {{
    background: {COLORS['bg_dark']};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {COLORS['border_light']};
    border-radius: 4px;
    min-width: 20px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {COLORS['text_secondary']};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── CheckBox ─────────────────────────────────────────────────────── */
QCheckBox {{
    spacing: 8px;
    color: {COLORS['text_primary']};
    font-size: 13px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {COLORS['border_light']};
    border-radius: 4px;
    background-color: {COLORS['bg_input']};
}}
QCheckBox::indicator:checked {{
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
    image: none;
}}
QCheckBox::indicator:hover {{
    border-color: {COLORS['accent']};
}}

/* ── Séparateur ───────────────────────────────────────────────────── */
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {COLORS['border']};
}}

/* ── StatusBar ────────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {COLORS['bg_panel']};
    border-top: 1px solid {COLORS['border']};
    color: {COLORS['text_secondary']};
    font-size: 12px;
    padding: 3px 10px;
}}

/* ── Splitter ─────────────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {COLORS['border']};
    width: 1px;
    height: 1px;
}}
"""


# ═══════════════════════════════════════════════════════════════════════
# THREAD D'ANALYSE (pour ne pas geler l'interface)
# ═══════════════════════════════════════════════════════════════════════

class AnalysisThread(QThread):
    """Thread dédié à l'analyse pour maintenir l'interface réactive."""

    # Signaux émis vers le thread principal
    progress     = pyqtSignal(int)          # pourcentage 0-100
    log_message  = pyqtSignal(str)          # ligne de log
    frame_done   = pyqtSignal(int, int)     # (frame_courante, total)
    finished     = pyqtSignal(bool, str)    # (succès, message)
    figures_ready = pyqtSignal(list)        # liste des chemins de figures

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            self._run_analysis()
        except Exception as e:
            tb = traceback.format_exc()
            self.log_message.emit(f"[ERREUR] {e}")
            self.log_message.emit(tb)
            self.finished.emit(False, str(e))

    def _run_analysis(self):
        p = self.params

        # ── Import du module d'analyse ─────────────────────────────
        self.log_message.emit("[INFO] Chargement du module d'analyse...")
        try:
            import Include_Puncta_Tracking as ILT
            import torch
            from cellpose import models as cp_models
        except ImportError as e:
            self.finished.emit(False, f"Module manquant : {e}")
            return

        self.progress.emit(5)

        # ── Détection du device ────────────────────────────────────
        self.log_message.emit("[INFO] Détection du dispositif de calcul...")
        device = ILT.get_device()
        self.log_message.emit(f"[INFO] Dispositif : {device}")
        self.progress.emit(10)

        if self._abort:
            self.finished.emit(False, "Analyse annulée.")
            return

        # ── Chargement du CNN ──────────────────────────────────────
        if p["flag_cnn"]:
            self.log_message.emit("[INFO] Chargement du modèle CNN...")
            cnn_path = p["cnn_weights"]
            if not os.path.exists(cnn_path):
                self.finished.emit(False, f"Fichier CNN introuvable : {cnn_path}")
                return
            model_cnn = ILT.SomaCNN()
            model_cnn.load_state_dict(torch.load(cnn_path, map_location=device))
            model_cnn = model_cnn.to(device)
            model_cnn.eval()
            self.log_message.emit("[OK]   Modèle CNN chargé.")

            # ── Chargement Cellpose ────────────────────────────────
            self.log_message.emit("[INFO] Chargement de Cellpose cyto3...")
            cellpose_model = cp_models.Cellpose(
                gpu=torch.cuda.is_available(), model_type='cyto3'
            )
            self.log_message.emit("[OK]   Cellpose prêt.")
        else:
            model_cnn      = None
            cellpose_model = None

        self.progress.emit(20)

        # ── Chargement des fichiers TIFF ───────────────────────────
        self.log_message.emit("[INFO] Recherche des fichiers TIFF...")
        pattern = os.path.join(p["input_dir"], p["file_pattern"])
        files = glob.glob(pattern)
        if not files:
            self.finished.emit(False, f"Aucun fichier trouvé avec le motif :\n{pattern}")
            return

        files = sorted(files, key=lambda x: int(re.search(r"_t(\d+)", x).group(1)))
        self.log_message.emit(f"[INFO] {len(files)} fichiers trouvés.")

        self.log_message.emit("[INFO] Chargement du stack en mémoire...")
        stack = np.array([tiff.imread(f) for f in files])
        times = [int(re.search(r"_t(\d+)", f).group(1)) for f in files]
        self.log_message.emit(f"[OK]   Stack chargé : {stack.shape}")
        self.progress.emit(35)

        os.makedirs(p["output_dir"], exist_ok=True)

        if self._abort:
            self.finished.emit(False, "Analyse annulée.")
            return

        # ── Lancement de l'analyse ─────────────────────────────────
        self.log_message.emit("[INFO] Démarrage de l'analyse...")
        start = time.time()

        # Redirection de la progression TrackPy via signal
        # On lance dans le même thread (déjà dans QThread)
        import trackpy as tp
        tp.quiet()

        ILT.Tracking_Puncta_TrackPy(
            output_dir          = p["output_dir"],
            path                = p["input_dir"],
            stack               = stack,
            times               = times,
            pretraitement_sigma = p["sigma"],
            clip_limit          = p["clip_limit"],
            patch_size          = p["patch_size"],
            cnn_threshold       = p["cnn_threshold"],
            model_cnn           = model_cnn,
            cnn_batch_size      = p["cnn_batch_size"],
            overlap_thresh      = p["overlap_thresh"],
            cellpose_model      = cellpose_model,
            cellprob_threshold  = p["cellprob_threshold"],
            flow_threshold      = p["flow_threshold"],
            min_area            = p["min_area"],
            max_area            = p["max_area"],
            min_circularity     = p["min_circularity"],
            max_axis_ratio      = p["max_axis_ratio"],
            top_hat_radius      = p["top_hat_radius"],
            diameter_cellpose   = p["diameter_cellpose"],
            diameter_trackpy    = p["diameter_trackpy"],
            search_range        = p["search_range"],
            memory              = p["memory"],
            threshold_filtered  = p["threshold_filtered"],
            flag_CNN            = p["flag_cnn"],
        )

        elapsed = time.time() - start
        self.progress.emit(95)
        self.log_message.emit(f"[OK]   Analyse terminée en {elapsed:.1f}s.")

        # ── Collecte des figures générées ──────────────────────────
        suffix = "_CNN" if p["flag_cnn"] else ""
        expected = [
            f"histogramme_trajectoires{suffix}.png",
            f"msd_fit{suffix}.png",
            f"nombre_puncta_par_frame{suffix}.png",
            f"trajectoires_moyenne{suffix}.png",
            f"densite_puncta{suffix}.png",
            f"distribution_vitesses{suffix}.png",
            f"trajectoires_vitesse{suffix}.png",
        ]
        figures = [
            os.path.join(p["output_dir"], f)
            for f in expected
            if os.path.exists(os.path.join(p["output_dir"], f))
        ]
        self.figures_ready.emit(figures)
        self.progress.emit(100)
        self.finished.emit(True, f"Analyse complète — {len(figures)} figures générées.")


# ═══════════════════════════════════════════════════════════════════════
# WIDGET : PARAMÈTRE AVEC LABEL + SPINBOX + TOOLTIP
# ═══════════════════════════════════════════════════════════════════════

class ParamRow(QWidget):
    """Une ligne : label | description | spinbox."""

    def __init__(self, label, description, widget, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(10)

        lbl = QLabel(label)
        lbl.setFixedWidth(160)
        lbl.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 500;")

        desc = QLabel(description)
        desc.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        desc.setWordWrap(True)

        layout.addWidget(lbl)
        layout.addWidget(desc, 1)
        layout.addWidget(widget)


def make_double_spin(val, lo, hi, step, decimals=2):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setValue(val)
    w.setFixedWidth(90)
    return w


def make_int_spin(val, lo, hi):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    w.setFixedWidth(90)
    return w


# ═══════════════════════════════════════════════════════════════════════
# WIDGET : VISIONNEUSE DE FIGURES AVEC NAVIGATION
# ═══════════════════════════════════════════════════════════════════════

class FigureViewer(QWidget):
    """Affiche les figures générées avec navigation précédent/suivant."""

    FIGURE_NAMES = [
        ("Histogramme trajectoires",   "histogramme_trajectoires"),
        ("MSD & diffusion",            "msd_fit"),
        ("puncta / frame",          "nombre_puncta_par_frame"),
        ("Trajectoires",               "trajectoires_moyenne"),
        ("Carte de densité",           "densite_puncta"),
        ("Distribution vitesses",      "distribution_vitesses"),
        ("Trajectoires × vitesse",     "trajectoires_vitesse"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figures  = []
        self.current  = 0
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # ── En-tête : titre + navigation ──────────────────────────
        header = QHBoxLayout()

        self.btn_prev = QPushButton("◀")
        self.btn_prev.setObjectName("btn_secondary")
        self.btn_prev.setFixedSize(36, 36)
        self.btn_prev.clicked.connect(self._prev)
        self.btn_prev.setEnabled(False)

        self.fig_title = QLabel("Aucune figure disponible")
        self.fig_title.setAlignment(Qt.AlignCenter)
        self.fig_title.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 14px; font-weight: 600;"
        )

        self.btn_next = QPushButton("▶")
        self.btn_next.setObjectName("btn_secondary")
        self.btn_next.setFixedSize(36, 36)
        self.btn_next.clicked.connect(self._next)
        self.btn_next.setEnabled(False)

        self.fig_counter = QLabel("")
        self.fig_counter.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        self.fig_counter.setFixedWidth(55)
        self.fig_counter.setAlignment(Qt.AlignCenter)

        header.addWidget(self.btn_prev)
        header.addWidget(self.fig_title, 1)
        header.addWidget(self.btn_next)
        header.addWidget(self.fig_counter)
        layout.addLayout(header)

        # ── Zone d'image matplotlib ────────────────────────────────
        self.figure  = Figure(facecolor=COLORS['bg_panel'])
        self.canvas  = FigureCanvas(self.figure)
        self.canvas.setStyleSheet(f"background-color: {COLORS['bg_panel']};")
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, 1)

        # ── Message placeholder ────────────────────────────────────
        self.placeholder = QLabel(
            "Les figures s'afficheront ici\naprès la fin de l'analyse."
        )
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 14px;"
            f"background-color: {COLORS['bg_panel']};"
            f"border: 1px dashed {COLORS['border']}; border-radius: 8px;"
        )
        layout.addWidget(self.placeholder, 1)
        self.canvas.hide()

    def load_figures(self, paths: list):
        self.figures = [p for p in paths if os.path.exists(p)]
        self.current = 0
        if self.figures:
            self.placeholder.hide()
            self.canvas.show()
            self._show(0)
            self.btn_next.setEnabled(len(self.figures) > 1)
        else:
            self.canvas.hide()
            self.placeholder.show()

    def _show(self, idx: int):
        if not self.figures:
            return
        self.current = idx % len(self.figures)
        path = self.figures[self.current]

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis('off')
        self.figure.tight_layout(pad=0.2)
        self.canvas.draw()

        # Titre depuis le nom de fichier
        basename = os.path.basename(path).replace(".png", "").replace("_CNN", "")
        label = next(
            (name for name, key in self.FIGURE_NAMES if key in basename),
            basename
        )
        suffix = " (CNN)" if "_CNN" in os.path.basename(path) else ""
        self.fig_title.setText(label + suffix)
        self.fig_counter.setText(f"{self.current + 1} / {len(self.figures)}")

        self.btn_prev.setEnabled(self.current > 0)
        self.btn_next.setEnabled(self.current < len(self.figures) - 1)

    def _prev(self):
        self._show(self.current - 1)

    def _next(self):
        self._show(self.current + 1)


# ═══════════════════════════════════════════════════════════════════════
# FENÊTRE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════

class PunctaTrackingApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ICS puncta Tracking  —  LP2I Bordeaux")
        self.setMinimumSize(1200, 780)
        self.resize(1400, 860)
        self._thread = None
        self._build_ui()
        self._apply_style()

    # ── Construction de l'interface ────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Panneau gauche : configuration ────────────────────────
        left_panel = QWidget()
        left_panel.setFixedWidth(420)
        left_panel.setStyleSheet(
            f"background-color: {COLORS['bg_panel']};"
            f"border-right: 1px solid {COLORS['border']};"
        )
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(10)

        # Logo / titre
        title_widget = self._make_header()
        left_layout.addWidget(title_widget)

        # Scroll area pour les paramètres
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 6, 0)
        scroll_layout.setSpacing(10)

        # Groupes de paramètres
        scroll_layout.addWidget(self._make_paths_group())
        scroll_layout.addWidget(self._make_mode_group())
        scroll_layout.addWidget(self._make_preproc_group())
        scroll_layout.addWidget(self._make_cnn_group())
        scroll_layout.addWidget(self._make_cellpose_group())
        scroll_layout.addWidget(self._make_trackpy_group())
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll, 1)

        # Boutons d'action
        left_layout.addWidget(self._make_action_bar())

        root.addWidget(left_panel)

        # ── Panneau droit : résultats ──────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(10)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        # Onglet figures
        self.figure_viewer = FigureViewer()
        tabs.addTab(self.figure_viewer, "📊  Figures")

        # Onglet console
        console_widget = QWidget()
        console_layout = QVBoxLayout(console_widget)
        console_layout.setContentsMargins(8, 8, 8, 8)
        self.console = QTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setPlaceholderText(
            "Les messages d'analyse s'afficheront ici..."
        )
        console_layout.addWidget(self.console)
        tabs.addTab(console_widget, "🖥  Console")

        right_layout.addWidget(tabs, 1)

        # Barre de progression
        prog_frame = QWidget()
        prog_layout = QVBoxLayout(prog_frame)
        prog_layout.setContentsMargins(0, 4, 0, 0)
        prog_layout.setSpacing(4)

        prog_header = QHBoxLayout()
        self.lbl_status = QLabel("Prêt")
        self.lbl_status.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        self.lbl_pct = QLabel("—")
        self.lbl_pct.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px; font-weight: 600;")
        prog_header.addWidget(self.lbl_status, 1)
        prog_header.addWidget(self.lbl_pct)
        prog_layout.addLayout(prog_header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(10)
        prog_layout.addWidget(self.progress_bar)

        right_layout.addWidget(prog_frame)

        root.addWidget(right_panel, 1)

        # ── Barre de statut ────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "ICS Puncta Tracking v.1.0.0  ·  LP2I Bordeaux  ·  Arnaud HUBER"
        )

    def _make_header(self):
        w = QWidget()
        w.setStyleSheet(
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
            f"stop:0 #1C2333, stop:1 #161B22);"
            f"border: 1px solid {COLORS['border']};"
            f"border-radius: 10px;"
        )
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        title = QLabel("ICS Puncta Tracking")
        title.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 17px; font-weight: 700;"
            f"background: transparent; border: none;"
        )
        subtitle = QLabel("Suivi automatisé de puncta en microscopie")
        subtitle.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
            f"background: transparent; border: none;"
        )
        institution = QLabel("LP2I Bordeaux  ·  Arnaud HUBER")
        institution.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px;"
            f"background: transparent; border: none;"
        )
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(institution)
        return w

    def _make_paths_group(self):
        grp = QGroupBox("Chemins")
        layout = QVBoxLayout(grp)
        layout.setSpacing(8)

        # Dossier images
        row1 = QHBoxLayout()
        lbl1 = QLabel("Dossier images :")
        lbl1.setFixedWidth(120)
        lbl1.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        self.input_dir = QLineEdit()
        self.input_dir.setPlaceholderText(r"\\serveur\images\CTRL1")
        btn1 = QPushButton("Parcourir")
        btn1.setObjectName("btn_browse")
        btn1.clicked.connect(self._browse_input)
        row1.addWidget(lbl1)
        row1.addWidget(self.input_dir, 1)
        row1.addWidget(btn1)

        # Motif de fichiers
        row2 = QHBoxLayout()
        lbl2 = QLabel("Motif fichiers :")
        lbl2.setFixedWidth(120)
        lbl2.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        self.file_pattern = QLineEdit("*_t*.tif")
        self.file_pattern.setPlaceholderText("ex: CTRL1_xxx_t*.tif")
        row2.addWidget(lbl2)
        row2.addWidget(self.file_pattern, 1)

        # Dossier sortie
        row3 = QHBoxLayout()
        lbl3 = QLabel("Dossier sortie :")
        lbl3.setFixedWidth(120)
        lbl3.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText(r"C:\Resultats\Analyse")
        btn3 = QPushButton("Parcourir")
        btn3.setObjectName("btn_browse")
        btn3.clicked.connect(self._browse_output)
        row3.addWidget(lbl3)
        row3.addWidget(self.output_dir, 1)
        row3.addWidget(btn3)

        # Poids CNN
        row4 = QHBoxLayout()
        lbl4 = QLabel("Poids CNN :")
        lbl4.setFixedWidth(120)
        lbl4.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        self.cnn_weights = QLineEdit("soma_cnn_test.pth")
        btn4 = QPushButton("Parcourir")
        btn4.setObjectName("btn_browse")
        btn4.clicked.connect(self._browse_cnn)
        row4.addWidget(lbl4)
        row4.addWidget(self.cnn_weights, 1)
        row4.addWidget(btn4)

        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)
        layout.addLayout(row4)
        return grp

    def _make_mode_group(self):
        grp = QGroupBox("Mode d'analyse")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        self.chk_cnn = QCheckBox(
            "CNN + Cellpose + TrackPy  (recommandé — plus précis)"
        )
        self.chk_cnn.setChecked(True)
        self.chk_cnn.setStyleSheet(
            f"color: {COLORS['accent_green']}; font-weight: 600; font-size: 13px;"
        )

        note = QLabel(
            "Décocher pour utiliser TrackPy seul (plus rapide, images peu denses)."
        )
        note.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; padding-left: 24px;"
        )
        note.setWordWrap(True)

        layout.addWidget(self.chk_cnn)
        layout.addWidget(note)
        return grp

    def _make_preproc_group(self):
        grp = QGroupBox("Prétraitement")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)

        self.sp_sigma = make_double_spin(1.5, 0.5, 5.0, 0.1)
        self.sp_clip  = make_double_spin(0.02, 0.001, 0.1, 0.005, decimals=3)

        layout.addWidget(ParamRow(
            "Sigma gaussien", "Lissage avant détection (px)", self.sp_sigma
        ))
        layout.addWidget(ParamRow(
            "Clip CLAHE", "Rehaussement contraste [0–1]", self.sp_clip
        ))
        return grp

    def _make_cnn_group(self):
        grp = QGroupBox("CNN — Détection des somas")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)

        self.sp_cnn_thresh  = make_double_spin(0.23, 0.05, 0.99, 0.01)
        self.sp_patch_size  = make_int_spin(64, 32, 256)
        self.sp_cnn_batch   = make_int_spin(512, 32, 2048)

        layout.addWidget(ParamRow(
            "Seuil CNN", "Probabilité min. pour retenir un soma", self.sp_cnn_thresh
        ))
        layout.addWidget(ParamRow(
            "Taille patch", "Patch carré autour de chaque candidat (px)", self.sp_patch_size
        ))
        layout.addWidget(ParamRow(
            "Batch CNN", "Réduire à 128 si erreur CUDA OOM", self.sp_cnn_batch
        ))
        return grp

    def _make_cellpose_group(self):
        grp = QGroupBox("Cellpose — Segmentation puncta")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)

        self.sp_diam_cp     = make_double_spin(11.0, 3.0, 50.0, 0.5)
        self.sp_cellprob    = make_double_spin(0.1, -6.0, 6.0, 0.1)
        self.sp_flow        = make_double_spin(0.5, 0.1, 1.0, 0.05)
        self.sp_min_area    = make_int_spin(5, 1, 500)
        self.sp_max_area    = make_int_spin(500, 10, 5000)
        self.sp_min_circ    = make_double_spin(0.5, 0.1, 1.0, 0.05)
        self.sp_max_axis    = make_double_spin(2.0, 1.0, 10.0, 0.1)
        self.sp_tophat      = make_int_spin(10, 3, 50)
        self.sp_cp_batch    = make_int_spin(24, 1, 128)
        self.sp_overlap     = make_double_spin(0.1, 0.0, 1.0, 0.05)

        layout.addWidget(ParamRow("Diamètre", "Diamètre puncta estimé (px)", self.sp_diam_cp))
        layout.addWidget(ParamRow("Prob. cellule", "Seuil probabilité Cellpose", self.sp_cellprob))
        layout.addWidget(ParamRow("Seuil flux", "Cohérence flux optiques", self.sp_flow))
        layout.addWidget(ParamRow("Aire min (px²)", "Filtre artefacts sub-pixel", self.sp_min_area))
        layout.addWidget(ParamRow("Aire max (px²)", "Filtre agrégats trop grands", self.sp_max_area))
        layout.addWidget(ParamRow("Circularité min", "Filtre formes allongées (4πA/P²)", self.sp_min_circ))
        layout.addWidget(ParamRow("Ratio axes max", "Filtre dendrites (grand/petit axe)", self.sp_max_axis))
        layout.addWidget(ParamRow("Rayon top-hat", "Soustraction fond local (px)", self.sp_tophat))
        layout.addWidget(ParamRow("Batch Cellpose", "Réduire à 8 si erreur CUDA OOM", self.sp_cp_batch))
        layout.addWidget(ParamRow("Seuil overlap", "Fraction chevauchement ROI", self.sp_overlap))
        return grp

    def _make_trackpy_group(self):
        grp = QGroupBox("TrackPy — Suivi temporel")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)

        self.sp_diam_tp   = make_int_spin(11, 3, 51)
        self.sp_search    = make_double_spin(4.0, 1.0, 30.0, 0.5)
        self.sp_memory    = make_int_spin(1, 0, 20)
        self.sp_thresh_f  = make_int_spin(20, 2, 200)

        layout.addWidget(ParamRow(
            "Diamètre TP", "Diamètre particule tp.locate (impair, px)", self.sp_diam_tp
        ))
        layout.addWidget(ParamRow(
            "Portée (px)", "Déplacement max entre frames — utiliser 2 en mode CNN", self.sp_search
        ))
        layout.addWidget(ParamRow(
            "Mémoire", "Frames pendant lesquelles une particule disparue est mémorisée", self.sp_memory
        ))
        layout.addWidget(ParamRow(
            "Filtre trajectoires", "Longueur minimale conservée (frames)", self.sp_thresh_f
        ))
        return grp

    def _make_action_bar(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(6)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {COLORS['border']};")
        layout.addWidget(sep)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_run = QPushButton("▶  Lancer l'analyse")
        self.btn_run.setObjectName("btn_primary")
        self.btn_run.setMinimumHeight(42)
        self.btn_run.clicked.connect(self._run_analysis)

        self.btn_abort = QPushButton("⏹  Arrêter")
        self.btn_abort.setObjectName("btn_danger")
        self.btn_abort.setMinimumHeight(42)
        self.btn_abort.clicked.connect(self._abort_analysis)
        self.btn_abort.setEnabled(False)

        self.btn_open = QPushButton("📂  Ouvrir le dossier")
        self.btn_open.setObjectName("btn_secondary")
        self.btn_open.setMinimumHeight(42)
        self.btn_open.clicked.connect(self._open_output_dir)

        btn_row.addWidget(self.btn_run, 2)
        btn_row.addWidget(self.btn_abort, 1)
        btn_row.addWidget(self.btn_open, 1)
        layout.addLayout(btn_row)
        return w

    # ── Style ──────────────────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet(STYLESHEET)

    # ── Actions ────────────────────────────────────────────────────────

    def _browse_input(self):
        path = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier d'images")
        if path:
            self.input_dir.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier de sortie")
        if path:
            self.output_dir.setText(path)

    def _browse_cnn(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner les poids CNN", "", "Fichiers PyTorch (*.pth *.pt)"
        )
        if path:
            self.cnn_weights.setText(path)

    def _open_output_dir(self):
        path = self.output_dir.text().strip()
        if path and os.path.isdir(path):
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])
        else:
            QMessageBox.warning(self, "Dossier introuvable",
                                "Le dossier de sortie n'existe pas encore.")

    def _validate_inputs(self) -> bool:
        errors = []
        if not self.input_dir.text().strip():
            errors.append("• Le dossier d'images est requis.")
        if not self.output_dir.text().strip():
            errors.append("• Le dossier de sortie est requis.")
        if not self.file_pattern.text().strip():
            errors.append("• Le motif de fichiers est requis.")
        if self.chk_cnn.isChecked() and not os.path.exists(self.cnn_weights.text().strip()):
            errors.append("• Le fichier de poids CNN est introuvable.")
        if errors:
            QMessageBox.critical(self, "Champs manquants", "\n".join(errors))
            return False
        return True

    def _collect_params(self) -> dict:
        return {
            "input_dir":         self.input_dir.text().strip(),
            "output_dir":        self.output_dir.text().strip(),
            "file_pattern":      self.file_pattern.text().strip(),
            "cnn_weights":       self.cnn_weights.text().strip(),
            "flag_cnn":          self.chk_cnn.isChecked(),
            # prétraitement
            "sigma":             self.sp_sigma.value(),
            "clip_limit":        self.sp_clip.value(),
            # CNN
            "cnn_threshold":     self.sp_cnn_thresh.value(),
            "patch_size":        self.sp_patch_size.value(),
            "cnn_batch_size":    self.sp_cnn_batch.value(),
            # ROI
            "overlap_thresh":    self.sp_overlap.value(),
            # Cellpose
            "diameter_cellpose": self.sp_diam_cp.value(),
            "cellprob_threshold":self.sp_cellprob.value(),
            "flow_threshold":    self.sp_flow.value(),
            "min_area":          self.sp_min_area.value(),
            "max_area":          self.sp_max_area.value(),
            "min_circularity":   self.sp_min_circ.value(),
            "max_axis_ratio":    self.sp_max_axis.value(),
            "top_hat_radius":    self.sp_tophat.value(),
            # TrackPy
            "diameter_trackpy":  self.sp_diam_tp.value(),
            "search_range":      self.sp_search.value(),
            "memory":            self.sp_memory.value(),
            "threshold_filtered":self.sp_thresh_f.value(),
        }

    def _run_analysis(self):
        if not self._validate_inputs():
            return

        self.console.clear()
        self.progress_bar.setValue(0)
        self.figure_viewer.load_figures([])
        self._log("[INFO] Démarrage de l'analyse...")

        params = self._collect_params()

        # Résumé des paramètres dans la console
        mode = "CNN + Cellpose + TrackPy" if params["flag_cnn"] else "TrackPy seul"
        self._log(f"[INFO] Mode       : {mode}")
        self._log(f"[INFO] Entrée     : {params['input_dir']}")
        self._log(f"[INFO] Motif      : {params['file_pattern']}")
        self._log(f"[INFO] Sortie     : {params['output_dir']}")
        self._log("─" * 55)

        self.btn_run.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.lbl_status.setText("Analyse en cours...")

        self._thread = AnalysisThread(params)
        self._thread.log_message.connect(self._log)
        self._thread.progress.connect(self._update_progress)
        self._thread.figures_ready.connect(self.figure_viewer.load_figures)
        self._thread.finished.connect(self._on_finished)
        self._thread.start()

    def _abort_analysis(self):
        if self._thread and self._thread.isRunning():
            self._thread.abort()
            self._log("[WARN]  Arrêt demandé...")

    def _on_finished(self, success: bool, message: str):
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)

        if success:
            self.lbl_status.setText("✔  Analyse terminée avec succès")
            self._log("─" * 55)
            self._log(f"[OK]   {message}")
            self.status_bar.showMessage(f"✔  {message}", 8000)
        else:
            self.lbl_status.setText("✘  Erreur lors de l'analyse")
            self._log(f"[ERREUR] {message}")
            QMessageBox.critical(self, "Erreur d'analyse", message)

    def _log(self, msg: str):
        # Coloration selon le préfixe
        color = COLORS['text_primary']
        if "[OK]" in msg or "[INFO]" in msg:
            color = "#A8D8A8"
        elif "[WARN]" in msg:
            color = COLORS['accent_yellow']
        elif "[ERREUR]" in msg or "Error" in msg or "Traceback" in msg:
            color = COLORS['accent_orange']
        elif "─" in msg:
            color = COLORS['text_muted']

        self.console.append(
            f'<span style="color:{color}; font-family:Consolas,monospace;">'
            f'{msg}</span>'
        )
        # Auto-scroll
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _update_progress(self, value: int):
        self.progress_bar.setValue(value)
        self.lbl_pct.setText(f"{value} %")


# ═══════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ICS Puncta Tracking")
    app.setOrganizationName("LP2I Bordeaux")

    # Activer le rendu haute résolution sur les écrans 4K
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    window = PunctaTrackingApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
