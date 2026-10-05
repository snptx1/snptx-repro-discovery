"""Generate the revised system-architecture schematic (fig0 v2) for the DL-forward study.

Companion to `architecture_figure.py`. The four-endpoint classification oracle averages
five single-task GIN members after fitting each member's temperature. The RF is an
independent comparator. Conformal and selective metrics are evaluated alongside the
retrospective SPRT campaign and logged; they do not determine its stopping boundary.
The GAT score diagnostic is separate. The companion engine's abductive-discovery
component is not evaluated here.

CPU-only, no data. Figure -> preprint/dl_forward/figures/fig0_architecture_v2.png.
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
    ax.set_title("Study architecture: graph models and a retrospective sequential campaign",
                 fontsize=16, fontweight="bold", color=TEXT_PRIMARY, pad=12)

    # ------------------------------------------------------------------ main row
    # Six-endpoint transfer study; the classification ensemble uses four endpoints.
    stage(ax, 6, 60.5, 28, 23, "Substrate", C_SUBSTRATE,
          body=["TDC ADMET benchmarks", "6 ADMET endpoints", "molecular graphs"],
          tag="adapters/drugcomb.py")
    stage(ax, 48, 60.5, 28, 23, "Learned oracle", C_ORACLE,
          body=["K=5 single-task GINs", "4 binary endpoints", "mean scaled probabilities"],
          tag="models/gnn.py")

    # Evaluation metrics are logged alongside stopping; they do not drive the SPRT.
    stage(ax, 90, 47.5, 42, 43.5, "Retrospective campaign", C_ENGINE, title_pad=3.0,
          title_size=12.5,
          face=FILL_DECISION)
    stage(ax, 93, 73.5, 36, 8.5, "SPRT · when to stop", C_SPRT,
          tag="ranked labels · experiment_design.py", face=DARK_BG,
          title_size=11.5, tag_size=8.5, title_pad=1.8, tag_pad=1.1)
    stage(ax, 93, 62.5, 36, 8.5, "Conformal + selective", C_CONFORMAL,
          tag="evaluation metrics · logged only", face=DARK_BG,
          title_size=11.5, tag_size=8.5, title_pad=1.8, tag_pad=1.1)
    stage(ax, 93, 51.5, 36, 8.5, "Companion discovery", C_ABDUCTIVE,
          tag="abductive component · not evaluated here", face=DARK_BG,
          title_size=11.5, tag_size=8.5, title_pad=1.8, tag_pad=1.1)

    stage(ax, 140, 60.5, 24, 23, "SPRT outcome", C_GONOGO,
          body=["GO / NO-GO / undecided", "Gaussian working model", "endpoint record"],
          body_size=10, tag="e8_campaign_learned.py", face=FILL_GONOGO)

    # Probabilities rank the campaign subgroup; metrics form a separate evaluation.
    arrow(ax, (34, 72), (48, 72), C_SUBSTRATE)
    arrow(ax, (76, 78), (93, 78), C_ORACLE)
    arrow(ax, (76, 67), (93, 67), C_ORACLE, ls="--")
    arrow(ax, (129, 78), (140, 78), C_ENGINE)

    # ---------------------------------------------- interpretability probe (top)
    # The separate GAT diagnostic identifies a normalized constant atom-score sum.
    stage(ax, 46, 90, 40, 9, "GAT score diagnostic", C_ATTN,
          tag="incoming attention sum = 1 · Figure 6", face=DARK_BG,
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
    # Independent descriptor comparator, supplied by the molecular substrate.
    stage(ax, 6, 48, 28, 8.5, "RF comparator", C_STRUCT,
          tag="RDKit + Morgan fingerprints", face=DARK_BG,
          title_size=10.5, tag_size=8.5, title_pad=1.8, tag_pad=1.1)
    arrow(ax, (20, 59.8), (20, 57.2), C_STRUCT, ls="--", lw=1.3, ms=11)

    # ----------------------------------------------- training sub-workflow (bottom)
    # Per-member temperature fitting precedes the ensemble probability average.
    sub_y, sub_h, sub_w = 26, 9, 24
    subs = [
        (6, "Graph featurizer", C_SUBSTRATE, "9-dim atoms · bond graph"),
        (38, "Single-task GINs", C_ORACLE, "K=5 per classification endpoint"),
        (70, "Temperature / member", C_ORACLE, "fit on validation labels"),
        (102, "Average probabilities", C_ORACLE, "ensemble predictive distribution"),
    ]
    for x, title, accent, tag in subs:
        stage(ax, x, sub_y, sub_w, sub_h, title, accent, tag=tag, face=DARK_BG,
              title_size=9.0, tag_size=7.5, title_pad=1.8, tag_pad=1.1)
    for x0 in (30, 62, 94):  # horizontal flow between the four steps
        arrow(ax, (x0, sub_y + sub_h / 2), (x0 + 8, sub_y + sub_h / 2),
              C_STRUCT, lw=1.3, ms=11)
    ax.text(6.5, sub_y - 2.4,
            "Transfer study: single-/multi-task GINs on six endpoints  ·  Ensemble: single-task members on four classifiers",
            ha="left", va="top", fontsize=9, style="italic", color=TEXT_SECONDARY)
    # Only the final probability average feeds the learned-oracle box.
    bracket(ax, [(114, sub_y + sub_h), (114, 42), (62, 42), (62, 60.5)],
            C_ORACLE, ls="-", lw=1.5, ms=13)

    # ----------------------------------------------------------- DuckDB lineage bar
    ax.add_patch(FancyBboxPatch(
        (6, 4), 158, 9, boxstyle="round,pad=0.7,rounding_size=1.8",
        linewidth=1.4, edgecolor=C_STRUCT, facecolor=CARD_BG))
    ax.text(85, 8.5,
            "DuckDB records  ·  parameters, oracle summaries, SPRT outcomes, and evaluation metrics  ·  intelligence/catalog.py",
            ha="center", va="center", fontsize=10.8, color=TEXT_PRIMARY)
    # Dotted drops into the lineage bar from clear corridors only (no box crossings).
    for x0, y0 in [(131, 47.5), (152, 59.8)]:
        arrow(ax, (x0, y0), (x0, 13), C_STRUCT, ls=":", lw=1.3, ms=11)

    # Keep the role key below the lineage bar, outside every connector route.
    for x, color, label in [
        (6, C_SUBSTRATE, "Input + feedback"),
        (38, C_ORACLE, "Model + calibration"),
        (70, C_ENGINE, "Decisions + outputs"),
        (102, C_ATTN, "Score diagnostic"),
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
