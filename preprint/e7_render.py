"""E7 manuscript rendering: rule cards + structure-property cliff panel.

Reads the confirmed discovery results (``e7_discovery_results.json``) and renders
two publication figures from them (no recompute, deterministic):

  fig7_rule_cards.png   the abductive driver table as human-readable rule cards,
                        one per ADMET endpoint (recovered structure-property driver,
                        symbolic-tree threshold, distilled meta-model R2). The
                        manuscript leads with BBB / solubility / HIA / Caco2; the
                        semi-circular lipophilicity card is drawn muted as a labelled
                        reference.
  fig8_cliff_panel.png  annotated structure-property cliffs: matched molecule pairs
                        (high Tanimoto, large property gap) drawn side by side with
                        the single change highlighted by the delta-property.

CPU-only, seconds. Figures -> pilot_phd/preprint/figures/.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)
RESULTS = HERE / "e7_discovery_results.json"

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402

from snptx.viz.theme import (  # noqa: E402
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    CARD_BG,
    DARK_BG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    hex_to_rgba,
)

RDLogger.DisableLog("rdApp.*")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
    "font.family": "DejaVu Sans",
    "figure.facecolor": DARK_BG, "savefig.facecolor": DARK_BG,
    "text.color": TEXT_PRIMARY,
})
BLUE, ORANGE, GREEN, GREY, INK = (
    ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, TEXT_SECONDARY, TEXT_PRIMARY)
REF_EDGE = ACCENT_PURPLE  # muted accent for the semi-circular reference card

# lead demos first, semi-circular lipophilicity reference last (muted)
CARD_ORDER = ["bbb", "solubility", "hia", "caco2", "lipophilicity"]
DRIVER_PLAIN = {
    "bbb": "low polar surface area / few H-bond donors cross the blood-brain barrier",
    "solubility": "higher lipophilicity (logP) lowers aqueous solubility",
    "hia": "low polar surface area improves intestinal absorption",
    "caco2": "fewer H-bond donors raise Caco-2 permeability",
    "lipophilicity": "calculated logP tracks measured lipophilicity (reference)",
}


def log(m: str) -> None:
    print(m, flush=True)


def _load() -> dict:
    with open(RESULTS) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# fig7: abductive rule cards
# ---------------------------------------------------------------------------
def render_rule_cards(res: dict) -> None:
    by_ep = {r["endpoint"]: r for r in res["rediscovery"]}
    cards = [by_ep[e] for e in CARD_ORDER if e in by_ep]

    fig, axes = plt.subplots(1, len(cards), figsize=(3.1 * len(cards), 4.2))
    if len(cards) == 1:
        axes = [axes]
    fig.suptitle("Abductive discovery recovers the known structure-property driver "
                 "on 5/5 ADMET endpoints", fontsize=12.5, y=1.02, color=TEXT_PRIMARY)

    for ax, r in zip(axes, cards):
        ep = r["endpoint"]
        is_ref = ep == "lipophilicity"
        face = CARD_BG
        edge = BLUE if not is_ref else REF_EDGE
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=2.0, edgecolor=edge, facecolor=face, mutation_aspect=1.0))

        title = ep.upper() if not is_ref else "LIPOPHILICITY*"
        ax.text(0.5, 0.90, title, ha="center", va="top", fontsize=14,
                fontweight="bold", color=INK)
        ax.text(0.5, 0.80, f"recovered driver: {r['tree_feature']}", ha="center",
                va="top", fontsize=11, color=edge, fontweight="bold")
        # wrapped plain-language rule
        plain = DRIVER_PLAIN.get(ep, r["note"])
        words, line, lines = plain.split(), "", []
        for w in words:
            if len(line + " " + w) > 26:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        lines.append(line)
        ax.text(0.5, 0.70, "\n".join(lines), ha="center", va="top", fontsize=9.3,
                color=INK)

        ax.text(0.5, 0.40, "symbolic rule", ha="center", fontsize=8.5, color=GREY)
        rule = r["tree_rule"].replace(" then ", "\n  -> ").replace(" else ", "\n  else ")
        ax.text(0.5, 0.355, rule, ha="center", va="top", fontsize=8.2,
                family="DejaVu Sans Mono", color=INK)

        ax.text(0.28, 0.12, "meta-R\u00b2", ha="center", fontsize=8.5, color=GREY)
        ax.text(0.28, 0.06, f"{r['meta_r2']:.2f}", ha="center", fontsize=13,
                fontweight="bold", color=GREEN if r["meta_r2"] >= 0.3 else ORANGE)
        ax.text(0.72, 0.12, "|corr|", ha="center", fontsize=8.5, color=GREY)
        ax.text(0.72, 0.06, f"{abs(r['top_corr']):.2f}", ha="center", fontsize=13,
                fontweight="bold", color=INK)

    fig.text(0.5, -0.02,
             "*semi-circular reference: the lipophilicity driver is calculated logP, "
             "itself a descriptor; shown muted.",
             ha="center", fontsize=8, color=GREY)
    fig.savefig(FIGDIR / "fig7_rule_cards.png", bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {FIGDIR / 'fig7_rule_cards.png'}")


# ---------------------------------------------------------------------------
# fig8: structure-property cliff panel
# ---------------------------------------------------------------------------
def _draw_mol(smiles: str, size: int = 320):
    mol = Chem.MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DCairo(size, size)
    rdMolDraw2D.SetDarkMode(d.drawOptions())
    d.drawOptions().setBackgroundColour(hex_to_rgba(DARK_BG))
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    import io

    from PIL import Image
    return Image.open(io.BytesIO(d.GetDrawingText()))


def render_cliff_panel(res: dict) -> None:
    # pick the two most striking cliffs per endpoint (skip ECFP collisions sim>=0.999)
    picks = []
    for block in res["cliffs"]:
        ep = block["endpoint"]
        exs = [e for e in block["examples"] if e["sim"] < 0.999]
        exs = sorted(exs, key=lambda e: abs(e["dY"]), reverse=True)[:1]
        for e in exs:
            picks.append((ep, e))
    picks = picks[:3]

    fig, axes = plt.subplots(len(picks), 2, figsize=(8.2, 3.4 * len(picks)),
                             gridspec_kw={"wspace": 0.5})
    if len(picks) == 1:
        axes = axes.reshape(1, 2)
    fig.suptitle("Structure-property cliffs: a single change flips the property",
                 fontsize=12.5, y=1.005, color=TEXT_PRIMARY)

    for row, (ep, e) in enumerate(picks):
        for col, key in enumerate(("smiles_a", "smiles_b")):
            ax = axes[row, col]
            ax.axis("off")
            try:
                ax.imshow(_draw_mol(e[key]))
            except Exception:
                ax.text(0.5, 0.5, "(draw failed)", ha="center")
        axes[row, 0].set_title(f"{ep}   y = {e['y_a']:+.2f}", fontsize=10, color=TEXT_PRIMARY)
        axes[row, 1].set_title(f"y = {e['y_b']:+.2f}", fontsize=10, color=TEXT_PRIMARY)
        # centered delta annotation between the two structures
        axes[row, 0].text(
            1.02, 0.5, f"\u0394Y = {e['dY']:+.2f}\nsim = {e['sim']:.2f}",
            transform=axes[row, 0].transAxes, ha="center", va="center",
            fontsize=11, fontweight="bold", color=ORANGE,
            bbox=dict(boxstyle="round,pad=0.35", fc=hex_to_rgba(ORANGE, 0.14), ec=ORANGE))

    fig.savefig(FIGDIR / "fig8_cliff_panel.png", bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {FIGDIR / 'fig8_cliff_panel.png'}")


def main() -> int:
    if not RESULTS.exists():
        log(f"ERROR: {RESULTS} not found; run e7_discovery_probe.py first.")
        return 1
    res = _load()
    log("E7 rendering: rule cards + cliff panel")
    render_rule_cards(res)
    render_cliff_panel(res)
    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
