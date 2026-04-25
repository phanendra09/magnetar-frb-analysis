"""
style.py - matplotlib styling for paper plots
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = {
    "primary": "#0072B2",
    "secondary": "#D55E00",
    "accent": "#009E73",
    "warning": "#E69F00",
    "highlight": "#CC79A7",
    "dark": "#332288",
    "light": "#88CCEE",
    "neutral": "#555555",
    "background": "#FAFAFA",
    "frb": "#FF2400",
    "grid": "#E0E0E0",
}

PALETTE = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["accent"],
    COLORS["warning"],
    COLORS["highlight"],
    COLORS["dark"],
    COLORS["light"],
]

FIGURE_SIZES = {
    "single_column": (3.5, 2.8),
    "single_column_tall": (3.5, 4.0),
    "double_column": (7.0, 4.5),
    "double_column_tall": (7.0, 7.0),
    "three_panel": (7.0, 2.5),
    "four_panel": (7.0, 6.0),
    "square": (5.0, 5.0),
    "presentation": (10.0, 6.0),
}

SCIENCE_STYLE = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.labelpad": 6,
    "axes.prop_cycle": plt.cycler("color", PALETTE),
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": False,
    "xtick.major.size": 4,
    "xtick.minor.size": 2,
    "xtick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "xtick.labelsize": 9,
    "xtick.direction": "in",
    "xtick.top": True,
    "ytick.major.size": 4,
    "ytick.minor.size": 2,
    "ytick.major.width": 0.8,
    "ytick.minor.width": 0.5,
    "ytick.labelsize": 9,
    "ytick.direction": "in",
    "ytick.right": True,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
    "legend.fancybox": True,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "grid.color": COLORS["grid"],
}


def apply_style() -> None:
    """Apply the plotting style globally."""
    mpl.rcParams.update(SCIENCE_STYLE)


def get_figure(name: str = "double_column", nrows: int = 1, ncols: int = 1, **kwargs):
    """Create a figure using a named size preset."""
    apply_style()
    figsize = FIGURE_SIZES.get(name, FIGURE_SIZES["double_column"])
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


def annotate_frb(ax, x, y, text="FRB 200428", **kwargs) -> None:
    """Annotate the FRB burst on a plot."""
    default_kwargs = dict(
        fontsize=9,
        fontweight="bold",
        color=COLORS["frb"],
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color=COLORS["frb"], lw=1.5),
        xytext=(0, 25),
        textcoords="offset points",
    )
    default_kwargs.update(kwargs)
    ax.annotate(text, (x, y), **default_kwargs)
