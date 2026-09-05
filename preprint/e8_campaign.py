"""E8: the end-to-end autonomous discovery campaign (the wired engine).

This is the closed loop that composes the confirmed SNPTX intelligence-layer
primitives into a single autonomous campaign over real ADMET endpoints. For each
discovery target the engine:

  1. ORACLE. Trains a calibrated bootstrap-ensemble oracle on a leakage-controlled
     (Murcko scaffold) cold split. Classification endpoints emit conformal prediction
     sets + selective-prediction confidences (``src.safety.uncertainty``); regression
     endpoints emit an ensemble-variance uncertainty.
  2. SEQUENTIAL DECISION (SPRT-in-the-loop). Poses an a-priori go/no-go question -- is
     the oracle's top-predicted (novel-scaffold) subgroup shifted from the pool
     baseline by at least a meaningful effect? -- and feeds per-molecule z-scored
     measurements to Wald's SPRT (``src.intelligence.experiment_design.SPRT``), which
     stops as soon as it can decide. Records measurements-to-decision vs the
     fixed-sample requirement.
  3. SELECTIVE PREDICTION. On classification endpoints, reports conformal coverage and
     the retained accuracy at 70% coverage (abstain on the least-confident molecules).
  4. ABDUCTIVE DISCOVERY (E7 in-loop). Runs ``run_discovery_cycle`` on the endpoint's
     descriptors to distil a symbolic structure-property rule + meta-model R2.
  5. LINEAGE. Logs every experiment, its decision and its discovered rule to a DuckDB
     experiment catalog (``src.intelligence.catalog``), and grows a novelty archive.

Outputs (all under pilot_phd/preprint/):
  e8_campaign_results.json     the campaign summary (per-endpoint decisions + savings)
  e8_campaign_lineage.duckdb   the DuckDB provenance store
  figures/fig9_campaign_timeline.png   cumulative measurements + running savings
  figures/fig10_lineage_graph.png      task -> experiment -> decision -> rule graph

CPU-only, deterministic (fixed seeds). Reuses the confirmed oracle recipe from the
committed spine so the campaign reproduces from a clean checkout with no GPU.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, r2_score  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import _tdc_download_patch  # noqa: E402,F401
from pivot_probes import DESC_FNS, _fp_matrix, load  # noqa: E402

from snptx.viz.theme import (  # noqa: E402
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_ORANGE,
    ACCENT_PURPLE,
    ACCENT_RED,
    BORDER,
    CARD_BG,
    DARK_BG,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    hex_to_rgba,
)
from src.intelligence.catalog import ExperimentCatalog  # noqa: E402
from src.intelligence.experiment_design import SPRT  # noqa: E402
from src.intelligence.scientific_discovery import (  # noqa: E402
    NoveltyArchive,
    run_discovery_cycle,
)
from src.safety.uncertainty import UncertaintyQuantifier  # noqa: E402

DESC_NAMES = [n for n, _ in DESC_FNS]
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)
OUT_JSON = HERE / "e8_campaign_results.json"
DB_PATH = HERE / "e8_campaign_lineage.duckdb"

# endpoints in the campaign: (name, kind). Lead with the clean demos.
CAMPAIGN = [
    ("bbb", "classification"),
    ("ames", "classification"),
    ("herg", "classification"),
    ("solubility", "regression"),
    ("caco2", "regression"),
    ("hia", "classification"),
]

# a-priori campaign parameters (fixed before seeing data)
ALPHA, BETA = 0.05, 0.20          # SPRT error rates
MEANINGFUL_EFFECT = 0.30          # go/no-go threshold in sigma units
CONFORMAL_ALPHA = 0.10            # 90% target coverage
TOPFRAC = 0.30                    # "top-predicted" subgroup for the decision
DISCOVERY_CAP = 3000              # subsample for the abductive rule (novelty is O(n^2))
SEED = 20260905

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
BLUE, ORANGE, GREEN, GREY, RED, INK = (
    ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, TEXT_SECONDARY, ACCENT_RED, TEXT_PRIMARY)
PURPLE = ACCENT_PURPLE


def log(m: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {m}", flush=True)


def zscore(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd


def scaffold_split(scaf: np.ndarray, rng: np.random.Generator, test_frac=0.25):
    uniq = np.unique(scaf)
    sp = rng.permutation(uniq)
    test_scaf = set(sp[: max(1, int(test_frac * len(uniq)))].tolist())
    is_te = np.array([s in test_scaf for s in scaf])
    tr_all = np.where(~is_te)[0]
    te = np.where(is_te)[0]
    rng.shuffle(tr_all)
    cut = int(0.75 * len(tr_all))
    return tr_all[:cut], tr_all[cut:], te   # train, calibration, test(novel scaffolds)


# ---------------------------------------------------------------------------
# oracle: calibrated bootstrap ensemble on the leakage-controlled split
# ---------------------------------------------------------------------------
def classification_oracle(data, tr, cal, te):
    """Return dict: conformal coverage, retained acc@70%, test probs, calib info."""
    Xtr, ytr = _fp_matrix(data["fps"], tr), data["y"][tr].astype(int)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pcal = rf.predict_proba(_fp_matrix(data["fps"], cal))
    pte = rf.predict_proba(_fp_matrix(data["fps"], te))

    uq = UncertaintyQuantifier(alpha=CONFORMAL_ALPHA)
    q = uq.calibrate_conformal(pcal, data["y"][cal].astype(int))
    yte = data["y"][te].astype(int)
    scores = 1.0 - pte
    covered = [int(lab in np.where(scores[i] <= q)[0]) for i, lab in enumerate(yte)]
    coverage = float(np.mean(covered))
    cal_info = uq.compute_calibration(pte, yte, n_bins=10)

    # selective prediction: retained accuracy at 70% coverage
    conf = pte.max(1)
    order = np.argsort(-conf)
    yl, yp = yte[order], pte.argmax(1)[order]
    k = max(1, int(0.70 * len(yl)))
    ret_acc70 = float(accuracy_score(yl[:k], yp[:k]))
    base_acc = float(accuracy_score(yl, yp))
    return {
        "coverage": round(coverage, 3),
        "ece": round(float(cal_info["ece"]), 4),
        "base_acc": round(base_acc, 3),
        "retained_acc_at_70pct": round(ret_acc70, 3),
        "pred_signal": pte[:, 1],   # P(positive) as the ranking score
        "y_test": yte.astype(float),
    }


def regression_oracle(data, tr, cal, te, n_boot=8):
    """Bootstrap-ensemble RF regressor; returns preds + ensemble std on test."""
    Xtr, ytr = _fp_matrix(data["fps"], tr), data["y"][tr]
    Xte = _fp_matrix(data["fps"], te)
    rng = np.random.default_rng(SEED)
    preds = []
    for b in range(n_boot):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        rf = RandomForestRegressor(n_estimators=120, random_state=SEED + b, n_jobs=-1)
        rf.fit(Xtr[idx], ytr[idx])
        preds.append(rf.predict(Xte))
    stk = np.stack(preds, 0)
    mean, std = stk.mean(0), stk.std(0)
    yte = data["y"][te]
    r2 = float(r2_score(yte, mean))
    rho = float(spearmanr(yte, mean).correlation)
    return {
        "test_r2": round(r2, 3),
        "test_spearman": round(rho, 3),
        "mean_uncertainty": round(float(std.mean()), 3),
        "pred_signal": mean,
        "y_test": yte.astype(float),
    }


# ---------------------------------------------------------------------------
# SPRT-in-the-loop go/no-go decision on the top-predicted novel subgroup
# ---------------------------------------------------------------------------
def sprt_decision(pred_signal: np.ndarray, y_test: np.ndarray, rng):
    """Sequentially measure the oracle's top-predicted subgroup; SPRT decides whether
    its z-scored property is shifted from the pool baseline by >= MEANINGFUL_EFFECT."""
    yz = (y_test - y_test.mean()) / (y_test.std() + 1e-9)
    order = np.argsort(-pred_signal)
    k = max(8, int(TOPFRAC * len(order)))
    subgroup = order[:k]
    measurements = yz[subgroup]
    sign = 1.0 if measurements.mean() >= 0 else -1.0     # test the realized direction
    stream = sign * measurements
    perm = rng.permutation(len(stream))                  # random measurement order

    sprt = SPRT(theta_0=0.0, theta_1=MEANINGFUL_EFFECT, alpha=ALPHA, beta=BETA, sigma=1.0)
    decision = "continue"
    for i in perm:
        decision = sprt.update(float(stream[i]))
        if decision != "continue":
            break
    n_used = sprt.n_observations
    fixed_n = ((norm.ppf(1 - ALPHA) + norm.ppf(1 - BETA)) / MEANINGFUL_EFFECT) ** 2
    fixed_n = float(min(fixed_n, len(stream)))
    verdict = {"reject_h0": "GO (meaningful shift)",
               "accept_h0": "NO-GO (no meaningful shift)",
               "continue": "UNDECIDED (budget)"}[decision]
    return {
        "decision": decision,
        "verdict": verdict,
        "effect_observed": round(float(sign * measurements.mean()), 3),
        "n_measurements": int(n_used),
        "fixed_sample_n": round(fixed_n, 1),
        "measurements_saved": round(float(fixed_n - n_used), 1),
    }


# ---------------------------------------------------------------------------
# abductive discovery cycle (E7 in-loop)
# ---------------------------------------------------------------------------
def discovery_rule(data):
    y = data["y"]
    X = data["X"]
    # subsample for tractability: the distilled tree rule + meta-R2 are stable, and
    # the novelty search inside run_discovery_cycle is O(n^2).
    if len(y) > DISCOVERY_CAP:
        idx = np.random.default_rng(SEED).choice(len(y), DISCOVERY_CAP, replace=False)
        X, y = X[idx], y[idx]
    Xz, _, _ = zscore(X)
    meta = pd.DataFrame(Xz, columns=DESC_NAMES)
    results = pd.DataFrame({"experiment_id": [f"mol_{i}" for i in range(len(y))],
                            "property_value": y})
    rep = run_discovery_cycle(results, meta, metric_col="property_value",
                              z_threshold=2.5, n_novel=5)
    tree = rep.symbolic_rules[0]
    feat = tree.expression.split()[1] if tree.expression.startswith("if ") else "(leaf)"
    return {"tree_feature": feat, "tree_rule": tree.expression,
            "meta_r2": round(rep.meta_model_r2, 3), "n_surprises": len(rep.surprises)}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_timeline(records: list[dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.4), height_ratios=[2, 1])
    cum_used, cum_fixed = 0.0, 0.0
    xs, used_line, fixed_line, saved_line, labels = [0], [0], [0], [0], []
    for step, r in enumerate(records, start=1):
        s = r["sprt"]
        cum_used += s["n_measurements"]
        cum_fixed += s["fixed_sample_n"]
        xs.append(step)
        used_line.append(cum_used)
        fixed_line.append(cum_fixed)
        saved_line.append(cum_fixed - cum_used)
        labels.append(r["endpoint"])

    ax1.step(xs, fixed_line, where="post", color=GREY, lw=2,
             label="fixed-sample requirement")
    ax1.step(xs, used_line, where="post", color=BLUE, lw=2.4,
             label="SPRT (sequential) measurements")
    ax1.fill_between(xs, used_line, fixed_line, step="post", color=BLUE, alpha=0.12)
    for step, (r, u) in enumerate(zip(records, used_line[1:]), start=1):
        d = r["sprt"]["decision"]
        col = GREEN if d == "reject_h0" else (ORANGE if d == "accept_h0" else RED)
        ax1.scatter([step], [u], color=col, zorder=5, s=45,
                    edgecolor="white", linewidth=1)
    for step, lab in enumerate(labels, start=1):
        ax1.annotate(lab, (step, used_line[step]), textcoords="offset points",
                     xytext=(4, 8), fontsize=8.5, color=INK)
    ax1.set_ylabel("Cumulative measurements")
    ax1.set_title("Autonomous campaign: sequential testing keeps the measurement "
                  "budget below fixed-sample")
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_xticks(xs)

    ax2.step(xs, saved_line, where="post", color=GREEN, lw=2.2)
    ax2.fill_between(xs, 0, saved_line, step="post", color=GREEN, alpha=0.15)
    ax2.set_ylabel("Measurements\nsaved (cum.)")
    ax2.set_xlabel("Campaign decision (endpoint)")
    ax2.set_xticks(xs)
    ax2.axhline(0, color=GREY, lw=0.8)
    fig.savefig(FIGDIR / "fig9_campaign_timeline.png", bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {FIGDIR / 'fig9_campaign_timeline.png'}")


def fig_lineage(records: list[dict]) -> None:
    n = len(records)
    fig, ax = plt.subplots(figsize=(11.5, 1.35 * n + 1.4))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, n + 0.9)
    ax.axis("off")
    cols = ["TASK", "ORACLE (calibrated)", "DECISION (SPRT)", "RULE (abductive)"]
    colx = [0.5, 1.5, 2.5, 3.5]
    cfill = [hex_to_rgba(BLUE, 0.12), hex_to_rgba(GREEN, 0.12),
             hex_to_rgba(ORANGE, 0.12), hex_to_rgba(PURPLE, 0.12)]
    ax.text(2.0, n + 0.72,
            "Campaign lineage: every decision traced from task to discovered rule "
            "(DuckDB provenance)", ha="center", fontsize=12.5, fontweight="bold",
            color=INK)
    for x, title in zip(colx, cols):
        ax.text(x, n + 0.30, title, ha="center", fontsize=10.5,
                fontweight="bold", color=INK)

    for row, r in enumerate(records):
        y = n - row - 0.5
        ep = r["endpoint"]
        orc = r["oracle"]
        if "coverage" in orc:
            orc_txt = f"cover {orc['coverage']:.2f}\nret.acc70 {orc['retained_acc_at_70pct']:.2f}"
        else:
            orc_txt = f"R\u00b2 {orc['test_r2']:.2f}\n\u00b1{orc['mean_uncertainty']:.2f}"
        s = r["sprt"]
        dcol = GREEN if s["decision"] == "reject_h0" else (
            ORANGE if s["decision"] == "accept_h0" else RED)
        dec_txt = f"{s['verdict'].split(' ')[0]}\n{s['n_measurements']} vs {s['fixed_sample_n']:.0f}"
        rl = r["rule"]
        rule_txt = f"{rl['tree_feature']}\nmeta-R\u00b2 {rl['meta_r2']:.2f}"
        texts = [ep, orc_txt, dec_txt, rule_txt]
        edges = [BLUE, GREEN, dcol, PURPLE]
        for x, txt, fc, ec in zip(colx, texts, cfill, edges):
            ax.add_patch(FancyBboxPatch(
                (x - 0.44, y - 0.36), 0.88, 0.72,
                boxstyle="round,pad=0.01,rounding_size=0.06",
                linewidth=1.6, edgecolor=ec, facecolor=fc))
            ax.text(x, y, txt, ha="center", va="center", fontsize=8.6, color=INK)
        for xa, xb in zip(colx[:-1], colx[1:]):
            ax.add_patch(FancyArrowPatch(
                (xa + 0.45, y), (xb - 0.45, y), arrowstyle="-|>",
                mutation_scale=11, color=GREY, lw=1.2))

    fig.savefig(FIGDIR / "fig10_lineage_graph.png", bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {FIGDIR / 'fig10_lineage_graph.png'}")


# ---------------------------------------------------------------------------
# main campaign loop
# ---------------------------------------------------------------------------
def main() -> int:
    t_start = time.time()
    log("E8 autonomous discovery campaign starting")
    log(f"  endpoints={[e for e, _ in CAMPAIGN]}  seed={SEED}")
    log(f"  a-priori: meaningful_effect={MEANINGFUL_EFFECT}sigma  "
        f"SPRT(alpha={ALPHA}, beta={BETA})  conformal_target={1 - CONFORMAL_ALPHA}")

    if DB_PATH.exists():
        DB_PATH.unlink()
    catalog = ExperimentCatalog(DB_PATH)
    archive = NoveltyArchive(max_size=256)

    records: list[dict] = []
    rng = np.random.default_rng(SEED)
    ep_times: list[float] = []

    for step, (ep, kind) in enumerate(CAMPAIGN, start=1):
        t0 = time.time()
        log(f"[{step}/{len(CAMPAIGN)}] loading '{ep}' ({kind}) ...")
        data = load(ep)
        n = len(data["y"])
        tr, cal, te = scaffold_split(data["scaf"], rng)
        log(f"    n={n}  scaffolds={data['n_scaf']}  "
            f"split tr/cal/te={len(tr)}/{len(cal)}/{len(te)} (test = novel scaffolds)")

        # 1. calibrated oracle
        if kind == "classification":
            orc = classification_oracle(data, tr, cal, te)
            log(f"    oracle: conformal coverage={orc['coverage']} (target "
                f"{1 - CONFORMAL_ALPHA})  ECE={orc['ece']}  "
                f"base_acc={orc['base_acc']} -> retained@70%={orc['retained_acc_at_70pct']}")
            surr_pred, surr_unc = orc["base_acc"], orc["ece"]
        else:
            orc = regression_oracle(data, tr, cal, te)
            log(f"    oracle: test R2={orc['test_r2']}  rho={orc['test_spearman']}  "
                f"mean uncertainty(+/-)={orc['mean_uncertainty']}")
            surr_pred, surr_unc = orc["test_r2"], orc["mean_uncertainty"]

        # 2. SPRT-in-the-loop go/no-go
        dec = sprt_decision(orc["pred_signal"], orc["y_test"], rng)
        log(f"    SPRT: {dec['verdict']}  effect={dec['effect_observed']}sigma  "
            f"used {dec['n_measurements']} vs fixed {dec['fixed_sample_n']} "
            f"(saved {dec['measurements_saved']})")

        # 3. abductive discovery rule (E7 in-loop)
        rule = discovery_rule(data)
        log(f"    rule: driver={rule['tree_feature']}  meta-R2={rule['meta_r2']}  "
            f"[{rule['tree_rule']}]")

        # 4. novelty archive (descriptor centroid of the novel-scaffold test set)
        centroid = data["X"][te].mean(0)
        is_novel = archive.maybe_add(centroid, threshold=0.0)

        # strip numpy arrays before serialising the record / lineage row
        orc_clean = {k: v for k, v in orc.items() if k not in ("pred_signal", "y_test")}

        # 5. lineage: log the experiment to DuckDB
        eid = catalog.record_experiment(
            dataset=f"admet_{ep}", endpoint=ep, model_type=kind,
            config_hash=f"e8_{ep}_{SEED}",
            hyperparams={"meaningful_effect": MEANINGFUL_EFFECT,
                         "conformal_alpha": CONFORMAL_ALPHA, "top_frac": TOPFRAC},
            metrics={"oracle": orc_clean, "sprt": dec, "rule": rule,
                     "novel_scaffold_region": bool(is_novel)},
            surrogate_prediction=float(surr_pred),
            surrogate_uncertainty=float(surr_unc),
            duration_seconds=time.time() - t0,
        )
        records.append({"endpoint": ep, "kind": kind, "experiment_id": eid,
                        "oracle": orc_clean, "sprt": dec, "rule": rule})
        dt = time.time() - t0
        ep_times.append(dt)
        remain = (len(CAMPAIGN) - step) * float(np.mean(ep_times))
        log(f"    logged experiment {eid[:8]} to DuckDB  ({dt:.1f}s)  "
            f"[{step}/{len(CAMPAIGN)} done, ETA ~{remain / 60:.1f}m]")

    catalog.close()

    # campaign-level roll-up
    total_used = sum(r["sprt"]["n_measurements"] for r in records)
    total_fixed = sum(r["sprt"]["fixed_sample_n"] for r in records)
    go = sum(r["sprt"]["decision"] == "reject_h0" for r in records)
    nogo = sum(r["sprt"]["decision"] == "accept_h0" for r in records)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "n_endpoints": len(records),
        "total_measurements_used": int(total_used),
        "total_fixed_sample": round(total_fixed, 1),
        "total_measurements_saved": round(total_fixed - total_used, 1),
        "fraction_saved": round((total_fixed - total_used) / total_fixed, 3),
        "decisions_go": int(go),
        "decisions_nogo": int(nogo),
        "records": records,
        "duckdb_path": str(DB_PATH.relative_to(ROOT)),
        "_runtime_s": round(time.time() - t_start, 1),
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    log("")
    log("=" * 64)
    log("E8 CAMPAIGN SUMMARY")
    for r in records:
        s = r["sprt"]
        log(f"  {r['endpoint']:12s} {s['verdict']:26s} "
            f"used {s['n_measurements']:3d} vs {s['fixed_sample_n']:5.1f}  "
            f"rule={r['rule']['tree_feature']}")
    log(f"  totals: used {total_used} vs fixed {total_fixed:.0f} measurements "
        f"-> saved {summary['fraction_saved'] * 100:.0f}%")
    log(f"  decisions: {go} GO / {nogo} NO-GO across {len(records)} endpoints")
    log(f"  DuckDB lineage: {DB_PATH}")
    log(f"  wrote {OUT_JSON}  ({summary['_runtime_s']}s)")
    log("=" * 64)

    # figures
    fig_timeline(records)
    fig_lineage(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
