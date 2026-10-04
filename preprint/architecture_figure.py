"""Generate the system-architecture schematic (fig0) for the walkthrough.

A static, wide block diagram of the decision engine in the SNPTX GitHub-dark
theme, matching the other committed figures. Molecules enter as the substrate
for a calibrated oracle; the decision engine then decides when to stop (SPRT),
what it does not know (conformal + selective prediction), and what it can read
off (abductive discovery); every stage is logged to a DuckDB lineage store; and
acquisitions can loop back (a regime-dependent benefit, Section 4.5).

CPU-only, no data. Figure -> preprint/figures/fig0_architecture.png and the
curated snapshot preprint/figures/sequential_preprint/fig0_architecture.png.
Section tags (§4.x) cite the manuscript Results subsections.
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
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from snptx.viz.theme import (  # noqa: E402
    CARD_BG,
    DARK_BG,
    NEON_BLUE,
    NEON_GREEN,
    NEON_GREY,
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

# Match the learned-oracle schematic's semantic color key, not its model.
C_INPUT = NEON_BLUE
C_MODEL = NEON_YELLOW
C_DECISION = NEON_GREEN
C_DISCOVERY = NEON_PURPLE
C_LINEAGE = NEON_GREY


def stage(ax, x, y, w, h, title, accent, *, body=None, tag=None,
          face=CARD_BG, title_size=14, body_size=11, tag_size=9,
          title_pad=3.2, tag_pad=2.2):
    """Draw a rounded stage box: accent title, optional body lines and module tag."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.7,rounding_size=1.8",
        linewidth=2.0, edgecolor=accent, facecolor=face))
    cx = x + w / 2
    ax.text(cx, y + h - title_pad, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=accent)
    if body:
        ax.text(cx, y + h - title_pad - 4.6, "\n".join(body), ha="center", va="top",
                fontsize=body_size, color=TEXT_PRIMARY, linespacing=1.55)
    if tag:
        ax.text(cx, y + tag_pad, tag, ha="center", va="bottom",
                fontsize=tag_size, style="italic", color=TEXT_SECONDARY)


def arrow(ax, p0, p1, color, *, ls="-", cs="arc3,rad=0", lw=2.0, ms=16):
    """Draw a themed flow arrow between two points."""
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=color, linestyle=ls, connectionstyle=cs))


def build() -> Path:
    fig, ax = plt.subplots(figsize=(15, 7.1), constrained_layout=False)
    ax.set_xlim(-2, 152)
    ax.set_ylim(-8, 66)
    ax.axis("off")
    ax.set_title("System architecture: from molecules to defensible decisions",
                 fontsize=16, fontweight="bold", color=TEXT_PRIMARY, pad=10)

    # --- main pipeline: substrate -> oracle -> decision engine -> GO/NO-GO ---
    stage(ax, 4, 24, 27, 24, "Substrate", C_INPUT,
          body=["TDC ADMET benchmarks", "6 ADMET endpoints", "RDKit + Morgan features"],
          tag="adapters/admet.py")
    stage(ax, 41, 24, 27, 24, "Calibrated oracle", C_MODEL,
          body=["RandomForest", "property model", "split-conformal calibration"],
          tag="safety/uncertainty.py")

    # decision engine container holding the three coupled decisions
    stage(ax, 78, 15, 38, 43, "Decision engine", C_DECISION, title_pad=3.2)
    stage(ax, 81, 41, 32, 9, "SPRT · when to stop", C_DECISION,
          tag="§4.1 · experiment_design.py", face=DARK_BG,
          title_size=12, tag_size=9, title_pad=1.8, tag_pad=1.1)
    stage(ax, 81, 29.5, 32, 9, "Conformal + selective", C_MODEL,
          tag="§4.2–4.3 · uncertainty.py", face=DARK_BG,
          title_size=12, tag_size=9, title_pad=1.8, tag_pad=1.1)
    stage(ax, 81, 18, 32, 9, "Abductive discovery", C_DISCOVERY,
          tag="§4.4 · scientific_discovery.py", face=DARK_BG,
          title_size=12, tag_size=9, title_pad=1.8, tag_pad=1.1)

    stage(ax, 122, 24, 24, 24, "GO / NO-GO", C_DECISION,
          body=["6 GO decisions", "140 vs 357 assays", "61% fewer (§4.6)"],
          tag="e8_campaign.py")

    # left-to-right pipeline flow
    for x0, x1, color in [(31, 41, C_INPUT), (68, 78, C_MODEL),
                          (116, 122, C_DECISION)]:
        arrow(ax, (x0, 36), (x1, 36), color)

    # active-learning feedback loop routed as a bracket over the top (§4.5)
    fb = FancyArrowPatch(
        path=MPath([(134, 48), (134, 61), (17.5, 61), (17.5, 48)],
                   [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO]),
        arrowstyle="-|>", mutation_scale=16, linewidth=2.0,
        color=C_INPUT, linestyle="--")
    ax.add_patch(fb)
    ax.text(75.75, 61, "active-learning loop · regime-dependent (§4.5)",
            ha="center", va="center", fontsize=10, style="italic", color=C_INPUT,
            bbox=dict(facecolor=DARK_BG, edgecolor="none", pad=2))

    # DuckDB lineage store logs every stage
    ax.add_patch(FancyBboxPatch(
        (4, 3), 142, 9, boxstyle="round,pad=0.7,rounding_size=1.8",
        linewidth=1.8, edgecolor=C_LINEAGE, facecolor=CARD_BG))
    ax.text(75, 7.5,
            "DuckDB lineage  ·  every decision traced from task to discovered rule  ·  intelligence/catalog.py",
            ha="center", va="center", fontsize=12.5, color=TEXT_PRIMARY)
    for x, y0 in [(17.5, 24), (54.5, 24), (97, 15), (134, 24)]:
        arrow(ax, (x, y0), (x, 12), C_LINEAGE, ls=":", lw=1.4, ms=11)

    for x, color, label in [
        (4, C_INPUT, "Input + feedback"),
        (34, C_MODEL, "Oracle + calibration"),
        (64, C_DECISION, "Decisions + outputs"),
        (94, C_DISCOVERY, "Discovery"),
        (124, C_LINEAGE, "Lineage"),
    ]:
        ax.plot([x, x + 4], [-4, -4], color=color, lw=3, solid_capstyle="round")
        ax.text(x + 5.5, -4, label, ha="left", va="center",
                fontsize=9, color=TEXT_PRIMARY)

    out = FIGDIR / "fig0_architecture.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.25)
    snap = FIGDIR / "sequential_preprint" / "fig0_architecture.png"
    if snap.parent.is_dir():
        fig.savefig(snap, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
