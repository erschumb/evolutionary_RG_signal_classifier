"""
Global plotting configuration and utilities for publication-quality figures.

Import and call setup_plotting() once at the top of any analysis module to
apply consistent styling. Other modules should import constants (FIGSIZE_*,
GROUP_COLORS) and helpers (save_figure, significance_stars) from here.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# ── Paths ───────────────────────────────────────────────────────────────────
FIGURE_DIR = Path("/mnt/d/phd/scripts/16_ev_signature_predictor/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ── Publication defaults ────────────────────────────────────────────────────
_RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 10,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "pdf.fonttype": 42,     # embed fonts as text, not outlines
    "svg.fonttype": "none", # text stays editable in SVG
}


# ── Figure sizes (inches) ───────────────────────────────────────────────────
# Nature/Science typical widths: single-col ≈ 89 mm, double-col ≈ 183 mm
FIGSIZE_SINGLE = (3.5, 2.8)   # single-column
FIGSIZE_DOUBLE = (7.2, 2.8)   # double-column
FIGSIZE_LARGE  = (7.2, 5.0)   # large panel


# ── Colors ──────────────────────────────────────────────────────────────────
GROUP_COLORS = {"pos": "#6FBF55", "neg": "#C97B7B"}   # colorblind-safe


# ── Initialisation ──────────────────────────────────────────────────────────
def setup_plotting() -> None:
    """Apply publication defaults. Idempotent — safe to call multiple times."""
    mpl.rcParams.update(_RC_PARAMS)
    sns.set_context("paper")
    sns.set_style("ticks")


# Apply on import so any module using this package gets the defaults
setup_plotting()


# ── Helpers ─────────────────────────────────────────────────────────────────
def save_figure(fig, name: str, dataset: str = "gnomad") -> None:
    """
    Save a figure in multiple formats:
      - SVG: vector, editable in Illustrator/Inkscape
      - PDF: vector, best for LaTeX/print
      - PNG: raster, 600 DPI, safe for PowerPoint
    """
    formats = {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 600},
    }
    for ext, kwargs in formats.items():
        path = FIGURE_DIR / f"{name}_{dataset}.{ext}"
        fig.savefig(path, format=ext, **kwargs)
    print(f"Saved: {name}_{dataset}.svg / .pdf / .png (600 dpi)")


def significance_stars(p: float) -> str:
    """Standard APA-style significance notation."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."