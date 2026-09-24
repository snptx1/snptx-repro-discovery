"""Generate the revised system-architecture schematic (fig0 v2) for the DL-forward study.

Non-destructive companion to `architecture_figure.py`. This version re-centres the
diagram on the *learned* oracle (a multi-task GIN/GINE encoder plus a deep ensemble),
demotes the random forest to an explicit descriptor baseline, exposes the training
sub-workflow (graph featurizer -> multi-task GNN -> deep ensemble -> temperature
scaling / conformal), and adds an attention -> interpretability branch feeding the
abductive-discovery box. The decision engine, DuckDB lineage and feedback loop are
unchanged from the original.

CPU-only, no data. Figure -> preprint/figures/fig0_architecture_v2.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from snptx.viz.theme import (  # noqa: E402
    BORDER,
    CARD_BG,
    DARK_BG,
    NEON_BLUE,
    NEON_GREY,
    NEON_PINK,
    NEON_PURPLE,
    NEON_YELLOW,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200,
    "font.size": 11, "font.family": "DejaVu Sans",
    "figure.facecolor": DARK_BG, "savefig.facecolor": DARK_BG,
    "text.color": TEXT_PRIMARY,
})

# Neon house palette (no orange): the star oracle is yellow, the decision engine pink,
# the interpretability branch purple, and structural/reference elements bright grey.
C_SUBSTRATE = NEON_BLUE
C_ORACLE    = NEON_YELLOW
C_ENGINE    = NEON_PINK
C_SPRT      = NEON_BLUE
C_CONFORMAL = NEON_YELLOW
C_ABDUCTIVE = NEON_PURPLE
C_GONOGO    = NEON_PINK
C_ATTN      = NEON_PURPLE
C_STRUCT    = NEON_GREY


def stage(ax, x, y, w, h, title, accent, *, body=None, tag=None,
          face=CARD_BG, title_size=14, body_size=11, tag_size=9, title_pad=3.2):
    """Draw a rounded stage box: accent title, optional body lines and module tag."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.7,rounding_size=1.8",
        linewidth=1.5, edgecolor=accent, facecolor=face))
    cx = x + w / 2
    ax.text(cx, y + h - title_pad, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=accent)
    if body:
        ax.text(cx, y + h - title_pad - 4.6, "\n".join(body), ha="center", va="top",
                fontsize=body_size, color=TEXT_PRIMARY, linespacing=1.55)
    if tag:
        ax.text(cx, y + 2.2, tag, ha="center", va="bottom",
                fontsize=tag_size, style="italic", color=TEXT_SECONDARY)


def arrow(ax, p0, p1, color, *, ls="-", cs="arc3,rad=0", lw=1.4, ms=13):
    """Draw a themed flow arrow between two points."""
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=color, linestyle=ls, connectionstyle=cs))


def bracket(ax, pts, color, *, ls="--", lw=1.4, ms=14):
    """Draw an orthogonal multi-segment arrow through the given points (last = head)."""
    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1)
    ax.add_patch(FancyArrowPatch(
        path=MPath(pts, codes), arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color, linestyle=ls))


def build() -> Path:
    fig, ax = plt.subplots(figsize=(16.5, 9.6), constrained_layout=False)
    ax.set_xlim(-2, 172)
    ax.set_ylim(0, 110)
    ax.axis("off")
    ax.set_title("System architecture: a learned oracle driving defensible decisions",
                 fontsize=16, fontweight="bold", color=TEXT_PRIMARY, pad=12)

    # ------------------------------------------------------------------ main row
    # Substrate -> learned oracle -> decision engine -> GO/NO-GO, all on one band.
    stage(ax, 6, 62, 28, 20, "Substrate", C_SUBSTRATE,
          body=["TDC ADMET benchmarks", "6 ADMET endpoints", "molecular graphs"],
          tag="adapters/drugcomb.py")
    stage(ax, 48, 62, 28, 20, "Learned oracle", C_ORACLE,
          body=["multi-task GIN/GINE", "encoder + per-task heads", "K=5 deep ensemble"],
          tag="models/gnn.py")

    # Decision-engine container holds the three coupled decisions, stacked and spaced.
    stage(ax, 90, 50, 42, 38, "Decision engine", C_ENGINE, title_pad=3.0)
    stage(ax, 93, 72, 36, 8.5, "SPRT · when to stop", C_SPRT,
          tag="experiment_design.py", face=DARK_BG,
          title_size=11.5, tag_size=8.5, title_pad=2.4)
    stage(ax, 93, 62, 36, 8.5, "Conformal + selective", C_CONFORMAL,
          tag="uncertainty.py", face=DARK_BG,
          title_size=11.5, tag_size=8.5, title_pad=2.4)
    stage(ax, 93, 52, 36, 8.5, "Abductive discovery", C_ABDUCTIVE,
          tag="scientific_discovery.py", face=DARK_BG,
          title_size=11.5, tag_size=8.5, title_pad=2.4)

    stage(ax, 140, 62, 24, 20, "GO / NO-GO", C_GONOGO,
          body=["calibrated go/no-go", "fewer measurements", "full lineage"],
          tag="e8_campaign.py")

    # Left-to-right pipeline flow (each arrow sits in a clear inter-box gap at y=72).
    arrow(ax, (34, 72), (48, 72), TEXT_PRIMARY)
    arrow(ax, (76, 72), (90, 72), TEXT_PRIMARY)
    arrow(ax, (132, 72), (140, 72), TEXT_PRIMARY)

    # ---------------------------------------------- interpretability branch (top)
    # Sits on its own band above the oracle; connects oracle -> attention, then
    # attention -> abductive via an orthogonal bracket routed clear of the engine.
    stage(ax, 46, 90, 40, 9, "GAT attention -> interpretability", C_ATTN,
          tag="attention_attribution.py", face=DARK_BG,
          title_size=11, tag_size=8.5, title_pad=2.4)
    arrow(ax, (62, 82), (62, 90), C_ATTN)                      # oracle -> attention
    # attention -> abductive: route down the clear gap between the engine and GO/NO-GO
    # so it never crosses the SPRT / conformal sub-boxes, entering abductive from the right.
    bracket(ax, [(86, 94.5), (136, 94.5), (136, 56), (129, 56)], C_ATTN, ls="--")

    # ----------------------------------------------- active-learning loop (topmost)
    # A dashed bracket from GO/NO-GO back to the substrate, staggered above the
    # interpretability bracket so the two never cross.
    bracket(ax, [(152, 82), (152, 107), (20, 107), (20, 82)], C_SUBSTRATE, ls="--")
    ax.text(86, 107, "active-learning loop · future work",
            ha="center", va="center", fontsize=10, style="italic", color=C_SUBSTRATE,
            bbox=dict(facecolor=DARK_BG, edgecolor="none", pad=2))

    # ------------------------------------------------ descriptor baseline (comparator)
    # Demoted RF, parked below the substrate with a dashed comparator link to the
    # oracle. Kept out of every vertical corridor so nothing overlaps it.
    stage(ax, 6, 48, 28, 8.5, "Descriptor baseline (RF)", C_STRUCT,
          tag="Morgan/RDKit · comparator", face=DARK_BG,
          title_size=11, tag_size=8.5, title_pad=2.4)
    arrow(ax, (34, 52), (48, 66), C_STRUCT, ls="--", cs="arc3,rad=0.15", lw=1.3, ms=11)

    # ----------------------------------------------- training sub-workflow (bottom)
    # How the oracle is built. A left-to-right strip under the pipeline; one arrow
    # rises straight into the oracle in an otherwise empty corridor at x=62.
    sub_y, sub_h, sub_w = 26, 9, 24
    subs = [
        (6, "Graph featurizer", C_SUBSTRATE, "9-dim nodes · 3-dim edges"),
        (38, "Multi-task GNN", C_ORACLE, "shared trunk · mixed heads"),
        (70, "Deep ensemble", C_ORACLE, "K init-seed members"),
        (102, "Temp scale + conformal", C_ORACLE, "calibrate · 90% sets"),
    ]
    for x, title, accent, tag in subs:
        stage(ax, x, sub_y, sub_w, sub_h, title, accent, tag=tag, face=DARK_BG,
              title_size=10.5, tag_size=8.2, title_pad=2.4)
    for x0 in (30, 62, 94):  # horizontal flow between the four steps
        arrow(ax, (x0, sub_y + sub_h / 2), (x0 + 8, sub_y + sub_h / 2),
              C_STRUCT, lw=1.3, ms=11)
    ax.text(64, sub_y - 2.4, "training sub-workflow  ·  dl_forward/train_multitask_gnn.py",
            ha="center", va="top", fontsize=10, style="italic", color=TEXT_SECONDARY)
    # Strip -> oracle: straight up the empty x=62 corridor into the oracle's base.
    arrow(ax, (62, sub_y + sub_h), (62, 62), C_ORACLE, lw=1.5, ms=13)

    # ----------------------------------------------------------- DuckDB lineage bar
    ax.add_patch(FancyBboxPatch(
        (6, 4), 158, 9, boxstyle="round,pad=0.7,rounding_size=1.8",
        linewidth=1.4, edgecolor=C_STRUCT, facecolor=CARD_BG))
    ax.text(85, 8.5,
            "DuckDB lineage  ·  every decision traced from task to discovered rule  ·  intelligence/catalog.py",
            ha="center", va="center", fontsize=12.5, color=TEXT_PRIMARY)
    # Dotted drops into the lineage bar from clear corridors only (no box crossings).
    for x0, y0 in [(100, 50), (152, 62)]:
        arrow(ax, (x0, y0), (x0, 13), C_STRUCT, ls=":", lw=1.3, ms=11)

    out = FIGDIR / "fig0_architecture_v2.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
