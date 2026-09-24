"""snptx.viz.theme — GitHub-dark visualization palette and helpers.

Extracted (verbatim, with attribution) from ``csci104/viz_utils.py`` so that
notebooks consuming the snptx package do not need a sibling clone of the
coursework repo. Only the constants and helpers used by EXP-01 are mirrored
here; the original module remains the source of truth for the broader
CSCI-E-104 assignment set.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -- Color palette (GitHub-Dark) ----------------------------------------------
DARK_BG        = "#0d1117"
CARD_BG        = "#161b22"
BORDER         = "#30363d"
TEXT_PRIMARY   = "#e6edf3"
TEXT_SECONDARY = "#8b949e"
ACCENT_BLUE    = "#58a6ff"
ACCENT_GREEN   = "#3fb950"
ACCENT_PURPLE  = "#bc8cff"
ACCENT_ORANGE  = "#d29922"
ACCENT_RED     = "#f85149"
ACCENT_CYAN    = "#39d353"
ACCENT_PINK    = "#f778ba"

# -- Neon outline palette -----------------------------------------------------
NEON_BLUE   = "#00f0ff"
NEON_GREEN  = "#39ff14"
NEON_PURPLE = "#bf00ff"
NEON_ORANGE = "#ff6e00"
NEON_RED    = "#ff003c"
NEON_PINK   = "#ff00e4"
NEON_YELLOW = "#e6ff00"
NEON_GREY   = "#c9d1d9"
NEON_CYCLE  = [NEON_BLUE, NEON_GREEN, NEON_PURPLE, NEON_ORANGE, NEON_RED, NEON_PINK, NEON_YELLOW]

# -- Preferred series palette (lines and bars) --------------------------------
# House style: bright neon blue / yellow / pink / grey, no orange. Ordered so the
# first three series read as three distinct saturated hues and the fourth (grey)
# reads as a muted baseline/reference series.
SERIES_CYCLE = [NEON_BLUE, NEON_YELLOW, NEON_PINK, NEON_GREY]

# -- Line styling defaults (thin and bright) ----------------------------------
LINE_WIDTH  = 1.1   # thin lines; series stay legible because the hues are bright
MARKER_SIZE = 4.0

# -- Bar styling defaults -----------------------------------------------------
BAR_FILL_ALPHA = 0.22   # translucent fill so the bright neon edge dominates
BAR_EDGE_WIDTH = 1.2    # thin, crisp neon outline
BAR_WIDTH      = 0.20   # narrow bars

GH_FONT = "DejaVu Sans"

# -- SNPTX heatmap colormap ---------------------------------------------------
# Used for confusion matrices and attention heatmaps. The palette stays within
# a medium-to-dark blue family so low values never wash out to white.
SNPTX_HEAT = LinearSegmentedColormap.from_list(
    "snptx_heat",
    ["#4fa3d9", "#2f7fbd", "#1b5f96", "#0f426f", "#06284a"],
    N=256,
)


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Convert a ``#rrggbb`` string to an RGBA tuple in the 0..1 range."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    return (r, g, b, alpha)


def apply_dark_theme(ax, title: str = "") -> None:
    """Apply the GitHub-dark theme to a matplotlib ``Axes``."""
    ax.set_facecolor(CARD_BG)
    ax.figure.set_facecolor(DARK_BG)
    ax.grid(True, linestyle="-", linewidth=0.3, color=hex_to_rgba(BORDER, 0.5))
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    if title:
        ax.set_title(title, fontsize=12, fontweight="semibold",
                     color=TEXT_PRIMARY, fontfamily=GH_FONT, pad=12)


def dark_legend(ax, **kwargs):
    """Add a dark-themed legend to ``ax``."""
    return ax.legend(facecolor=CARD_BG, edgecolor=BORDER,
                     labelcolor=TEXT_PRIMARY, framealpha=0.9, **kwargs)


def use_neon_rcparams(plt_module) -> None:
    """Set global matplotlib defaults to the SNPTX neon house style.

    Applies the dark canvas, the blue/yellow/pink/grey series cycle (no orange),
    and thin, bright line defaults so every live figure matches the committed
    ones without per-plot color bookkeeping. Pass the ``matplotlib.pyplot``
    module so this helper stays import-light.
    """
    from cycler import cycler

    plt_module.rcParams.update({
        "figure.facecolor": DARK_BG, "savefig.facecolor": DARK_BG,
        "axes.facecolor": CARD_BG, "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT_PRIMARY,
        "text.color": TEXT_PRIMARY, "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY, "grid.color": BORDER, "grid.alpha": 0.4,
        "axes.grid": True, "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=SERIES_CYCLE),
        "lines.linewidth": LINE_WIDTH, "lines.markersize": MARKER_SIZE,
        "font.size": 11, "font.family": GH_FONT,
    })


def dark_bar(ax, x, height, *, color=None, neon=None, width=None,
             fill_alpha=None, edge_width=None, label=None, bottom=None, **kw):
    """Draw thin transparent-fill bars with neon outlines."""
    width      = width      or BAR_WIDTH
    fill_alpha = fill_alpha if fill_alpha is not None else BAR_FILL_ALPHA
    edge_width = edge_width or BAR_EDGE_WIDTH
    neon       = neon or color or NEON_BLUE
    rgba_fill  = hex_to_rgba(neon, fill_alpha)
    return ax.bar(x, height, width=width, color=rgba_fill,
                  edgecolor=neon, linewidth=edge_width,
                  label=label, bottom=bottom, **kw)


def plot_loss_curve(losses, title: str = "Training Loss", skip: int = 0):
    """Plot a training loss curve with the dark theme applied."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(range(skip, len(losses)), losses[skip:],
            color=ACCENT_BLUE, linewidth=1.2, alpha=0.8)
    ax.fill_between(range(skip, len(losses)), losses[skip:],
                    alpha=0.07, color=ACCENT_BLUE)
    ax.set_xlabel("Step", fontsize=10, fontfamily=GH_FONT)
    ax.set_ylabel("Loss", fontsize=10, fontfamily=GH_FONT)
    apply_dark_theme(ax, title=title)
    plt.tight_layout()
    plt.show()
