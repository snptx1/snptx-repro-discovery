"""Generate the revised system-architecture schematic (fig0 v2) for the DL-forward study.

Non-destructive companion to `architecture_figure.py`. This version re-centres the
diagram on the *learned* oracle (a K=5 deep ensemble of temperature-scaled single-task GIN
members), demotes the random forest to an explicit descriptor baseline, exposes the
model-building sub-workflow (graph featurizer -> GIN encoder, trained single- and
multi-task for the transfer study -> deep ensemble -> temperature scaling / conformal),
and shows the GAT attention probe as a separate model trained on the same graphs. The
abductive-discovery component belongs to the companion engine and is not re-run here.

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
ROOT = next(p for p in HERE.parents if (p / "src").is_dir())  # repo root (public or private layout)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from snptx.viz.theme import (  # noqa: E402
    CARD_BG,
    DARK_BG,
    NEON_BLUE,
    NEON_GREY,
    NEON_GREEN,
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

# Color identifies a component's role: blue inputs/feedback, yellow learning/calibration,
# green decisions, purple interpretability, grey baseline/lineage.
C_SUBSTRATE = NEON_BLUE
C_ORACLE    = NEON_YELLOW
C_ENGINE    = NEON_GREEN
C_SPRT      = NEON_BLUE
C_CONFORMAL = NEON_YELLOW
C_ABDUCTIVE = NEON_PURPLE
C_GONOGO    = NEON_GREEN
# Subtle green fills so the decisions/outputs role reads clearly, matching the legend.
FILL_DECISION = (0.223, 1.0, 0.078, 0.07)
FILL_GONOGO   = (0.223, 1.0, 0.078, 0.16)
C_ATTN      = NEON_PURPLE
C_STRUCT    = NEON_GREY


def stage(ax, x, y, w, h, title, accent, *, body=None, tag=None,
          face=CARD_BG, title_size=14, body_size=11, tag_size=9,
          title_pad=3.2, tag_pad=2.2):
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
        ax.text(cx, y + tag_pad, tag, ha="center", va="bottom",
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
    ax.set_ylim(-8, 110)
    ax.axis("off")
    ax.set_title("System architecture: a learned oracle driving defensible decisions",
                 fontsize=16, fontweight="bold", color=TEXT_PRIMARY, pad=12)

    # ------------------------------------------------------------------ main row
    # Substrate -> learned oracle -> decision engine -> GO/NO-GO, all on one band.
    stage(ax, 6, 60.5, 28, 23, "Substrate", C_SUBSTRATE,
          body=["TDC ADMET benchmarks", "6 ADMET endpoints", "molecular graphs"],
          tag="adapters/drugcomb.py")
    stage(ax, 48, 60.5, 28, 23, "Learned oracle", C_ORACLE,
          body=["K=5 deep ensemble", "single-task GIN members", "temperature-scaled"],
          tag="models/gnn.py")

    # Decision-engine container holds the three coupled decisions, stacked and spaced.
    stage(ax, 90, 47.5, 42, 43.5, "Decision engine", C_ENGINE, title_pad=3.0,
          face=FILL_DECISION)
    stage(ax, 93, 73.5, 36, 8.5, "SPRT · when to stop", C_SPRT,
          tag="experiment_design.py", face=DARK_BG,
          title_size=11.5, tag_size=8.5, title_pad=1.8, tag_pad=1.1)
    stage(ax, 93, 62.5, 36, 8.5, "Conformal + selective", C_CONFORMAL,
          tag="uncertainty.py", face=DARK_BG,
          title_size=11.5, tag_size=8.5, title_pad=1.8, tag_pad=1.1)
    stage(ax, 93, 51.5, 36, 8.5, "Abductive discovery", C_ABDUCTIVE,
          tag="scientific_discovery.py · not re-run here", face=DARK_BG,
          title_size=11.5, tag_size=8.5, title_pad=1.8, tag_pad=1.1)

    stage(ax, 140, 60.5, 24, 23, "GO / NO-GO", C_GONOGO,
          body=["calibrated go/no-go", "sequential stopping", "full lineage"],
          tag="e8_campaign_learned.py", face=FILL_GONOGO)

    # Left-to-right pipeline flow (each arrow sits in a clear inter-box gap at y=72).
    arrow(ax, (34, 72), (48, 72), C_SUBSTRATE)
    arrow(ax, (76, 72), (90, 72), C_ORACLE)
    arrow(ax, (132, 72), (140, 72), C_ENGINE)

    # ---------------------------------------------- interpretability probe (top)
    # A separate GAT trained on the same molecular graphs; its attention is tested
    # against a descriptor atom-salience rule (a negative result). Fed from the substrate
    # through a corridor clear of the active-learning bracket at x=20.
    stage(ax, 46, 90, 40, 9, "GAT attention probe", C_ATTN,
          tag="attention_attribution.py · negative result", face=DARK_BG,
          title_size=11, tag_size=8.5, title_pad=1.8, tag_pad=1.1)
    bracket(ax, [(28, 83.5), (28, 94.5), (46, 94.5)], C_ATTN, ls="--")

    # ----------------------------------------------- active-learning loop (topmost)
    # A dashed bracket from GO/NO-GO back to the substrate, staggered above the
    # interpretability bracket so the two never cross.
    bracket(ax, [(152, 82), (152, 107), (20, 107), (20, 82)], C_SUBSTRATE, ls="--")
    ax.text(86, 107, "active-learning loop · future work",
            ha="center", va="center", fontsize=10, style="italic", color=C_SUBSTRATE,
            bbox=dict(facecolor=DARK_BG, edgecolor="none", pad=2))

    # ------------------------------------------------ descriptor baseline (comparator)
    # Demoted RF, parked below the substrate with an orthogonal comparator link.
    stage(ax, 6, 48, 28, 8.5, "Descriptor baseline", C_STRUCT,
          tag="RF · Morgan/RDKit", face=DARK_BG,
          title_size=10.5, tag_size=8.5, title_pad=1.8, tag_pad=1.1)
    bracket(ax, [(34.7, 52.25), (42, 52.25), (42, 67.5), (48, 67.5)],
            C_STRUCT, ls="--", lw=1.3, ms=11)

    # ----------------------------------------------- training sub-workflow (bottom)
    # How the oracle is built. A left-to-right strip under the pipeline; one arrow
    # rises straight into the oracle in an otherwise empty corridor at x=62.
    sub_y, sub_h, sub_w = 26, 9, 24
    subs = [
        (6, "Graph featurizer", C_SUBSTRATE, "9-dim atoms · bond graph"),
        (38, "GIN encoder", C_ORACLE, "single- and multi-task"),
        (70, "Deep ensemble", C_ORACLE, "K=5 single-task members"),
        (102, "Temp scale + conformal", C_ORACLE, "calibrate · 90% sets"),
    ]
    for x, title, accent, tag in subs:
        stage(ax, x, sub_y, sub_w, sub_h, title, accent, tag=tag, face=DARK_BG,
              title_size=9.0 if title == "Temp scale + conformal" else 10.5,
              tag_size=8.2, title_pad=1.8, tag_pad=1.1)
    for x0 in (30, 62, 94):  # horizontal flow between the four steps
        arrow(ax, (x0, sub_y + sub_h / 2), (x0 + 8, sub_y + sub_h / 2),
              C_STRUCT, lw=1.3, ms=11)
    ax.text(6.5, sub_y - 2.4, "model-building sub-workflow  ·  train_multitask_gnn.py  ·  make_ensemble_uncertainty.py",
            ha="left", va="top", fontsize=10, style="italic", color=TEXT_SECONDARY)
    # Strip -> oracle: straight up the empty x=62 corridor into the oracle's base.
    arrow(ax, (62, sub_y + sub_h), (62, 60.5), C_ORACLE, lw=1.5, ms=13)

    # ----------------------------------------------------------- DuckDB lineage bar
    ax.add_patch(FancyBboxPatch(
        (6, 4), 158, 9, boxstyle="round,pad=0.7,rounding_size=1.8",
        linewidth=1.4, edgecolor=C_STRUCT, facecolor=CARD_BG))
    ax.text(85, 8.5,
            "DuckDB lineage  ·  every decision traced from task to oracle to SPRT verdict  ·  intelligence/catalog.py",
            ha="center", va="center", fontsize=12.5, color=TEXT_PRIMARY)
    # Dotted drops into the lineage bar from clear corridors only (no box crossings).
    for x0, y0 in [(100, 47.5), (152, 62)]:
        arrow(ax, (x0, y0), (x0, 13), C_STRUCT, ls=":", lw=1.3, ms=11)

    # Keep the role key below the lineage bar, outside every connector route.
    for x, color, label in [
        (6, C_SUBSTRATE, "Input + feedback"),
        (38, C_ORACLE, "Model + calibration"),
        (70, C_ENGINE, "Decisions + outputs"),
        (102, C_ATTN, "Interpretability"),
        (134, C_STRUCT, "Baseline + lineage"),
    ]:
        ax.plot([x, x + 4], [-3, -3], color=color, lw=3, solid_capstyle="round")
        ax.text(x + 5.5, -3, label, ha="left", va="center",
                fontsize=9.2, color=TEXT_PRIMARY)

    out = FIGDIR / "fig0_architecture_v2.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
