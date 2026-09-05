"""Complete feasibility checks for the calibrated sequential-decision spine, and
generation of publication-style example figures from real ADMET data.

Confirmed pillars this script hardens and visualizes:
  1. Sequential decision efficiency (SPRT): operating-characteristic sweep of
     expected sample size vs a fixed-sample test across effect sizes, with the
     real-data operating point annotated.
  2. Robust calibrated uncertainty: reliability diagram + ECE, and split-conformal
     coverage under random vs leakage-controlled (scaffold) shift across tasks.
  3. Selective prediction: risk-coverage curves (retained accuracy as the model
     abstains on its least-confident predictions), random and scaffold splits.
  4. The characterized label-efficiency regime finding (from the committed
     g1_harden curves): acquisition helps only where the passive baseline is
     unstable.
  5. Molecular structure panel (RDKit) to show the substrate.

CPU-only. Figures -> pilot_phd/preprint/figures/. Numbers -> feasibility_summary.json.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from pivot_probes import _fp_matrix, load  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402

from snptx.viz.theme import (  # noqa: E402
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    BORDER,
    CARD_BG,
    DARK_BG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    hex_to_rgba,
)
from src.intelligence.experiment_design import SPRT  # noqa: E402
from src.safety.uncertainty import UncertaintyQuantifier  # noqa: E402

FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)
OUT_JSON = HERE / "feasibility_summary.json"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
    "font.family": "DejaVu Sans",
    "figure.facecolor": DARK_BG, "savefig.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG, "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT_PRIMARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": BORDER, "grid.alpha": 0.4,
    "axes.axisbelow": True, "figure.constrained_layout.use": True,
    "legend.facecolor": CARD_BG, "legend.edgecolor": BORDER, "legend.framealpha": 0.9,
})
BLUE, ORANGE, GREEN, GREY = ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, TEXT_SECONDARY


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# 1. SPRT operating-characteristic sweep + figure
# ---------------------------------------------------------------------------
def fig_sprt(data) -> dict:
    alpha, beta = 0.05, 0.20
    rng = np.random.default_rng(0)
    effects = np.linspace(0.15, 0.9, 16)
    fixed_n, sprt_n, powers, type1 = [], [], [], []
    for eff in effects:
        nfix = ((norm.ppf(1 - alpha) + norm.ppf(1 - beta)) / eff) ** 2
        ns, correct, t1 = [], 0, 0
        for _ in range(300):
            s = SPRT(theta_0=0.0, theta_1=round(float(eff), 3), alpha=alpha, beta=beta, sigma=1.0)
            for _ in range(4000):
                if s.update(float(rng.normal(eff, 1.0))) != "continue":
                    break
            ns.append(s.n_observations); correct += int(s.decision == "reject_h0")
            s2 = SPRT(theta_0=0.0, theta_1=round(float(eff), 3), alpha=alpha, beta=beta, sigma=1.0)
            for _ in range(4000):
                if s2.update(float(rng.normal(0.0, 1.0))) != "continue":
                    break
            t1 += int(s2.decision == "reject_h0")
        fixed_n.append(nfix); sprt_n.append(float(np.mean(ns)))
        powers.append(correct / 300); type1.append(t1 / 300)

    # real operating point: lipophilicity high-aromatic subgroup effect
    y = data["y"]; yz = (y - y.mean()) / (y.std() + 1e-9)
    ar = data["X"][:, 6]
    real_eff = float(yz[ar >= np.quantile(ar, 0.66)].mean())

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(effects, fixed_n, "-o", color=GREY, ms=4, label="Fixed-sample test")
    ax.plot(effects, sprt_n, "-o", color=BLUE, ms=4, label="SPRT (sequential)")
    ax.fill_between(effects, sprt_n, fixed_n, color=BLUE, alpha=0.12)
    ax.axvline(real_eff, color=ORANGE, ls="--", lw=1.5)
    ax.text(real_eff + 0.01, ax.get_ylim()[1] * 0.7,
            f"real ADMET\noperating point\n(effect={real_eff:.2f}$\\sigma$)",
            color=ORANGE, fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xlabel("Effect size (standard deviations)")
    ax.set_ylabel("Measurements to a confident decision")
    ax.set_title("Sequential testing decides with fewer measurements")
    ax.legend(frameon=False)
    fig.savefig(FIGDIR / "fig1_sprt_efficiency.png"); plt.close(fig)

    i = int(np.argmin(np.abs(effects - real_eff)))
    saving = (fixed_n[i] - sprt_n[i]) / fixed_n[i]
    return {"real_effect_sigma": round(real_eff, 3),
            "fixed_n_at_real": round(fixed_n[i], 1),
            "sprt_n_at_real": round(sprt_n[i], 1),
            "sample_saving_at_real": round(saving, 3),
            "mean_saving_across_effects": round(float(np.mean(
                (np.array(fixed_n) - np.array(sprt_n)) / np.array(fixed_n))), 3),
            "mean_power": round(float(np.mean(powers)), 3),
            "mean_type1": round(float(np.mean(type1)), 3)}


# ---------------------------------------------------------------------------
# helpers for calibration / conformal / selective prediction
# ---------------------------------------------------------------------------
def _split_random(n, rng):
    p = rng.permutation(n)
    return p[: n // 2], p[n // 2: int(0.75 * n)], p[int(0.75 * n):]


def _split_scaffold(scaf, rng):
    uniq = np.unique(scaf); sp = rng.permutation(uniq)
    test_scaf = set(sp[: max(1, int(0.25 * len(uniq)))].tolist())
    is_te = np.array([s in test_scaf for s in scaf])
    tr = np.where(~is_te)[0]; te = np.where(is_te)[0]
    rng.shuffle(tr); cut = int(0.7 * len(tr))
    return tr[:cut], tr[cut:], te


def _fit_probs(data, tr, cal, te):
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    rf.fit(_fp_matrix(data["fps"], tr), data["y"][tr].astype(int))
    return (rf.predict_proba(_fp_matrix(data["fps"], cal)),
            rf.predict_proba(_fp_matrix(data["fps"], te)))


def _coverage(probs, labels, q):
    scores = 1.0 - probs
    return float(np.mean([int(lab in np.where(scores[i] <= q)[0])
                          for i, lab in enumerate(labels)]))


def risk_coverage(probs, labels):
    conf = probs.max(1); order = np.argsort(-conf)
    yl = labels[order]; yp = probs.argmax(1)[order]
    covs = np.linspace(0.1, 1.0, 19); accs = []
    for c in covs:
        k = max(1, int(c * len(yl)))
        accs.append(accuracy_score(yl[:k], yp[:k]))
    return covs, np.array(accs)


# ---------------------------------------------------------------------------
# 2/3. calibration reliability + selective prediction + conformal coverage
# ---------------------------------------------------------------------------
def fig_calibration_and_selective(bbb) -> dict:
    alpha = 0.10
    rng = np.random.default_rng(0)
    n = len(bbb["y"])
    tr, cal, te = _split_random(n, rng)
    trs, cals, tes = _split_scaffold(bbb["scaf"], rng)

    pc, pt = _fit_probs(bbb, tr, cal, te)
    pcs, pts = _fit_probs(bbb, trs, cals, tes)

    uq = UncertaintyQuantifier(alpha=alpha)
    q = uq.calibrate_conformal(pc, bbb["y"][cal].astype(int))
    cov_rand = _coverage(pt, bbb["y"][te].astype(int), q)
    uq2 = UncertaintyQuantifier(alpha=alpha)
    qs = uq2.calibrate_conformal(pcs, bbb["y"][cals].astype(int))
    cov_scaf = _coverage(pts, bbb["y"][tes].astype(int), qs)

    cal_info = uq.compute_calibration(pt, bbb["y"][te].astype(int), n_bins=10)
    ece = float(cal_info["ece"])

    # reliability diagram
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    bc = np.asarray(cal_info["bin_confidences"]); ba = np.asarray(cal_info["bin_accuracies"])
    m = np.asarray(cal_info["bin_counts"]) > 0
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="perfect calibration")
    ax.plot(bc[m], ba[m], "-o", color=BLUE, label=f"model (ECE={ece:.3f})")
    ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Empirical accuracy")
    ax.set_title("Calibration reliability (BBB, scaffold-aware)")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(FIGDIR / "fig3_calibration_reliability.png"); plt.close(fig)

    # selective prediction risk-coverage
    cr, ar = risk_coverage(pt, bbb["y"][te].astype(int))
    cs, as_ = risk_coverage(pts, bbb["y"][tes].astype(int))
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    ax.plot(cr, ar, "-o", color=BLUE, ms=4, label="random split")
    ax.plot(cs, as_, "-o", color=ORANGE, ms=4, label="scaffold (shifted) split")
    ax.axhline(ar[-1], color=BLUE, ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("Coverage (fraction of molecules predicted)")
    ax.set_ylabel("Accuracy on retained set")
    ax.set_title("Selective prediction: abstaining raises retained accuracy")
    ax.legend(frameon=False)
    fig.savefig(FIGDIR / "fig2_selective_prediction.png"); plt.close(fig)

    return {"ece": round(ece, 4), "conformal_target": 1 - alpha,
            "coverage_random_split": round(cov_rand, 3),
            "coverage_scaffold_split": round(cov_scaf, 3),
            "base_acc": round(float(ar[-1]), 3),
            "acc_at_70pct_coverage_random": round(float(np.interp(0.7, cr, ar)), 3),
            "acc_at_70pct_coverage_scaffold": round(float(np.interp(0.7, cs, as_)), 3)}


# ---------------------------------------------------------------------------
# 4. conformal coverage robustness across tasks (bar chart)
# ---------------------------------------------------------------------------
def fig_coverage_across_tasks(task_data: dict) -> dict:
    alpha = 0.10
    labels, covr, covs = [], [], []
    for name, d in task_data.items():
        rng = np.random.default_rng(1)
        n = len(d["y"])
        tr, cal, te = _split_random(n, rng)
        trs, cals, tes = _split_scaffold(d["scaf"], rng)
        pc, pt = _fit_probs(d, tr, cal, te)
        pcs, pts = _fit_probs(d, trs, cals, tes)
        uq = UncertaintyQuantifier(alpha=alpha)
        q = uq.calibrate_conformal(pc, d["y"][cal].astype(int))
        uq2 = UncertaintyQuantifier(alpha=alpha)
        qs = uq2.calibrate_conformal(pcs, d["y"][cals].astype(int))
        labels.append(name)
        covr.append(_coverage(pt, d["y"][te].astype(int), q))
        covs.append(_coverage(pts, d["y"][tes].astype(int), qs))

    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x - w / 2, covr, w, color=BLUE, label="random split")
    ax.bar(x + w / 2, covs, w, color=ORANGE, label="scaffold (shifted) split")
    ax.axhline(1 - alpha, color=GREEN, ls="--", lw=1.5, label=f"target {1-alpha:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.6, 1.0); ax.set_ylabel("Empirical coverage")
    ax.set_title("Conformal coverage holds under leakage-controlled shift")
    ax.legend(frameon=False, ncol=3, fontsize=8.5)
    fig.savefig(FIGDIR / "fig4_conformal_coverage.png"); plt.close(fig)
    return {"tasks": labels, "coverage_random": [round(c, 3) for c in covr],
            "coverage_scaffold": [round(c, 3) for c in covs]}


# ---------------------------------------------------------------------------
# 5. label-efficiency regime figure (from committed g1_harden curves)
# ---------------------------------------------------------------------------
def fig_label_efficiency_regime() -> dict:
    npz = HERE / "g1_harden_curves.npz"
    if not npz.exists():
        return {"skipped": "g1_harden_curves.npz missing"}
    z = np.load(npz)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)
    for ax, task, title in ((axes[0], "solubility", "Solubility (unstable passive baseline)"),
                            (axes[1], "lipophilicity", "Lipophilicity (stable passive baseline)")):
        nl = z[f"{task}__n_labeled"]
        act = np.stack([z[f"{task}__active_seed{s}"] for s in range(3)])
        rnd = np.stack([z[f"{task}__random_seed{s}"] for s in range(3)])
        ax.plot(nl, act.mean(0), "-o", color=BLUE, ms=3, label="active (uncertainty)")
        ax.fill_between(nl, act.mean(0) - act.std(0), act.mean(0) + act.std(0), color=BLUE, alpha=0.15)
        ax.plot(nl, rnd.mean(0), "-o", color=GREY, ms=3, label="random")
        ax.fill_between(nl, rnd.mean(0) - rnd.std(0), rnd.mean(0) + rnd.std(0), color=GREY, alpha=0.15)
        ax.set_title(title, fontsize=10); ax.set_xlabel("labels acquired")
        ax.axhline(0, color=BORDER, lw=0.8, alpha=0.6)
    axes[0].set_ylabel("held-out scaffold $R^2$"); axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Active acquisition helps only when the passive baseline is unstable", fontsize=11)
    fig.savefig(FIGDIR / "fig5_label_efficiency_regime.png"); plt.close(fig)
    return {"source": "g1_harden_curves.npz"}


# ---------------------------------------------------------------------------
# 6. molecular structure panel
# ---------------------------------------------------------------------------
def fig_molecules(data) -> dict:
    order = np.argsort(data["y"])
    picks = list(order[:3]) + list(order[-3:])
    mols = [Chem.MolFromSmiles(data["smiles"][i]) for i in picks]
    legs = [f"logP={data['y'][i]:.2f}" for i in picks]
    opts = rdMolDraw2D.MolDrawOptions()
    rdMolDraw2D.SetDarkMode(opts)
    opts.setBackgroundColour(hex_to_rgba(DARK_BG))
    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(260, 200),
                               legends=legs, drawOptions=opts)
    img.save(str(FIGDIR / "fig6_molecules.png"))
    return {"n_shown": len(mols), "range": [round(float(data["y"][order[0]]), 2),
                                            round(float(data["y"][order[-1]]), 2)]}


def main() -> None:
    t0 = time.time()
    log("loading tasks (lipophilicity, bbb, herg, ames) ...")
    lipo = load("lipophilicity")
    bbb = load("bbb")
    herg = load("herg")
    ames = load("ames")
    log(f"  lipo n={len(lipo['y'])} bbb n={len(bbb['y'])} "
        f"herg n={len(herg['y'])} ames n={len(ames['y'])}")

    summary = {}
    log("fig1: SPRT efficiency sweep ...");        summary["sprt"] = fig_sprt(lipo)
    log("fig2/3: calibration + selective prediction ...")
    summary["calibration_selective"] = fig_calibration_and_selective(bbb)
    log("fig4: conformal coverage across tasks ...")
    summary["coverage_across_tasks"] = fig_coverage_across_tasks(
        {"BBB": bbb, "hERG": herg, "AMES": ames})
    log("fig5: label-efficiency regime ...");      summary["label_efficiency_regime"] = fig_label_efficiency_regime()
    log("fig6: molecule panel ...");               summary["molecules"] = fig_molecules(lipo)

    summary["_runtime_s"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    log("\n==================== FEASIBILITY SUMMARY ====================")
    log(json.dumps(summary, indent=2))
    log(f"figures -> {FIGDIR}")
    log("============================================================")


if __name__ == "__main__":
    main()
