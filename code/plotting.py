"""Shared IEEE plot style (spec §7): one colour and marker per method across every figure."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_AO = {"color": "black", "marker": "s"}
STYLE = {
    "FP-AO": dict(_AO),
    "FP-AO (explicit)": dict(_AO),
    "FP-AO (efficient)": {"color": "grey", "marker": "D"},
    "FONR": {"color": "C0", "marker": "o"},
    "MASO": {"color": "C1", "marker": "^"},
    "TIAS": {"color": "C2", "marker": "v"},
}
NET_LABEL = {"fonr": "FONR", "maso": "MASO", "tias": "TIAS"}
FIG_WIDTH_IN = 3.5
FONT_SIZE = 8


def setup_ieee() -> None:
    """IEEE single-column defaults: 3.5 in wide, 8 pt fonts, grid on."""
    plt.rcParams.update({
        "figure.figsize": (FIG_WIDTH_IN, FIG_WIDTH_IN * 0.75),
        "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 1, "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
        "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": ":",
        "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def new_fig():
    setup_ieee()
    return plt.subplots()


def save_fig(fig, name: str, paper_dir) -> tuple:
    """Write ``{name}.png`` (300 dpi) and ``{name}.pdf`` into ``paper_dir``; returns both paths."""
    os.makedirs(paper_dir, exist_ok=True)
    png = os.path.join(str(paper_dir), f"{name}.png")
    pdf = os.path.join(str(paper_dir), f"{name}.pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    return png, pdf
