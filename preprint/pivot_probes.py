"""Cheap confirm-or-kill probes for five candidate preprint angles.

Each probe is CPU-only, uses real TDC ADMET data, and reuses the SNPTX
intelligence-layer modules so a positive here maps directly onto the wired system.
Every probe prints a CONFIRM / KILL / SOFTEN verdict with the numbers behind it.

Angles:
  P_SEQ  sequential decision efficiency  -> src.intelligence.experiment_design.SPRT
  P_BO   Bayesian optimization for molecular optimization
                                         -> src.intelligence.surrogate (GP + EI)
  P_CONF conformal validity under scaffold shift
                                         -> src.safety.uncertainty.UncertaintyQuantifier
  P_DRIFT autonomous drift monitoring    -> src.deployment.drift_monitor (Page-Hinkley)
  P_ATTR uncertainty vs scaffold novelty -> bootstrap ensemble + Tanimoto OOD distance

Design choices for speed: molecular features are RDKit physchem descriptors (for the
GP) or Morgan fingerprints (for RF / Tanimoto), so nothing needs a GPU. These probes
test whether each phenomenon EXISTS, not to reach SOTA.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import _tdc_download_patch  # noqa: E402,F401
from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from src.adapters.admet import ADMETAdapter  # noqa: E402
from src.deployment.drift_monitor import ChangePointDetector  # noqa: E402
from src.intelligence.experiment_design import SPRT  # noqa: E402
from src.intelligence.surrogate import SurrogateModel, expected_improvement  # noqa: E402
from src.safety.uncertainty import UncertaintyQuantifier  # noqa: E402

RDLogger.DisableLog("rdApp.*")
np.random.seed(0)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Shared featurization
# ---------------------------------------------------------------------------
DESC_FNS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Crippen.MolLogP),
    ("TPSA", rdMolDescriptors.CalcTPSA),
    ("HBA", rdMolDescriptors.CalcNumHBA),
    ("HBD", rdMolDescriptors.CalcNumHBD),
    ("RotB", rdMolDescriptors.CalcNumRotatableBonds),
    ("ArRings", rdMolDescriptors.CalcNumAromaticRings),
    ("FrCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("Rings", rdMolDescriptors.CalcNumRings),
    ("HeavyAtoms", lambda m: m.GetNumHeavyAtoms()),
]


def murcko(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)) or "(acyclic)"
    except Exception:
        return None


def load(endpoint: str):
    """Return dict with smiles, y, descriptors X, morgan fps, scaffold ids."""
    adapter = ADMETAdapter(raw_dir=str(ROOT / "data" / "raw" / "admet"))
    df = adapter.build(endpoint, split="all")
    smis, ys, X, fps, scafs = [], [], [], [], []
    scaf_index: dict[str, int] = {}
    for smi, y in zip(df["Drug"], df["property_value"]):
        mol = Chem.MolFromSmiles(str(smi))
        s = murcko(smi)
        if mol is None or s is None or not np.isfinite(float(y)):
            continue
        try:
            row = [float(fn(mol)) for _, fn in DESC_FNS]
        except Exception:
            continue
        if not all(np.isfinite(row)):
            continue
        smis.append(str(smi))
        ys.append(float(y))
        X.append(row)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
        scafs.append(scaf_index.setdefault(s, len(scaf_index)))
    return {"smiles": smis, "y": np.asarray(ys), "X": np.asarray(X),
            "fps": fps, "scaf": np.asarray(scafs), "n_scaf": len(scaf_index)}


def zscore(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd


def tanimoto_max_to_set(fp, ref_fps):
    return float(np.max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps)))


# ===========================================================================
# P_SEQ  sequential decision efficiency (SPRT vs fixed-n)
# ===========================================================================
def probe_sequential(data) -> dict:
    y = data["y"]
    yz = (y - y.mean()) / (y.std() + 1e-9)
    ar = data["X"][:, 6]  # NumAromaticRings, correlates with lipophilicity
    hi = yz[ar >= np.quantile(ar, 0.66)]           # a genuinely higher-mean subgroup
    effect = float(hi.mean())                       # empirical effect size (sigma units)
    if effect < 0.15:
        return {"verdict": "SOFTEN", "reason": f"subgroup effect small ({effect:.2f})"}
    alpha, beta = 0.05, 0.20
    n_fixed = ((norm.ppf(1 - alpha) + norm.ppf(1 - beta)) / effect) ** 2

    rng = np.random.default_rng(0)
    R = 400
    n_h1, correct_h1, n_h0, correct_h0 = [], 0, [], 0
    for _ in range(R):
        s = SPRT(theta_0=0.0, theta_1=round(effect, 3), alpha=alpha, beta=beta, sigma=1.0)
        for _ in range(2000):
            d = s.update(float(rng.choice(hi)))
            if d != "continue":
                break
        n_h1.append(s.n_observations)
        correct_h1 += int(s.decision == "reject_h0")
        s2 = SPRT(theta_0=0.0, theta_1=round(effect, 3), alpha=alpha, beta=beta, sigma=1.0)
        for _ in range(2000):
            d = s2.update(float(rng.choice(yz)))       # H0-true: full population, mean 0
            if d != "continue":
                break
        n_h0.append(s2.n_observations)
        correct_h0 += int(s2.decision == "accept_h0")
    mean_n = float(np.mean(n_h1))
    power = correct_h1 / R
    typeI = 1 - correct_h0 / R
    saving = (n_fixed - mean_n) / n_fixed
    verdict = "CONFIRM" if (mean_n < n_fixed and power >= 0.75 and typeI <= 0.12) else "SOFTEN"
    return {"verdict": verdict, "effect_sigma": round(effect, 3),
            "n_fixed": round(n_fixed, 1), "sprt_mean_n": round(mean_n, 1),
            "sample_saving": round(saving, 3), "power": round(power, 3),
            "typeI": round(typeI, 3)}


# ===========================================================================
# P_BO  Bayesian optimization for molecular optimization (GP + EI vs random)
# ===========================================================================
def probe_bo(data) -> dict:
    rng_master = np.random.default_rng(0)
    N = min(1200, len(data["y"]))
    idx_all = rng_master.choice(len(data["y"]), N, replace=False)
    Xz, _, _ = zscore(data["X"][idx_all])
    y = data["y"][idx_all]
    top_thresh = np.quantile(y, 0.99)              # top 1% = "hits"
    seeds = list(range(8))
    seed_init, n_iter = 12, 40

    def run(strategy, seed):
        rng = np.random.default_rng(100 + seed)
        labeled = list(rng.choice(N, seed_init, replace=False))
        found_at = np.nan
        best_curve = []
        for t in range(n_iter):
            best = y[labeled].max()
            best_curve.append(best)
            if np.isnan(found_at) and best >= top_thresh:
                found_at = seed_init + t
            remaining = [i for i in range(N) if i not in set(labeled)]
            if strategy == "random":
                pick = int(rng.choice(remaining))
            else:
                gp = SurrogateModel(n_restarts=2, random_state=seed)
                gp.fit(Xz[labeled], y[labeled])
                mu, sigma = gp.predict(Xz[remaining])
                ei = expected_improvement(mu, sigma, best_f=best)
                pick = remaining[int(np.argmax(ei))]
            labeled.append(pick)
        best = y[labeled].max()
        best_curve.append(best)
        if np.isnan(found_at) and best >= top_thresh:
            found_at = seed_init + n_iter
        return found_at, np.trapz(best_curve)

    bo_found, rnd_found, bo_auc, rnd_auc = [], [], [], []
    for sd in seeds:
        f, a = run("bo", sd); bo_found.append(f); bo_auc.append(a)
        f, a = run("random", sd); rnd_found.append(f); rnd_auc.append(a)
    bo_found = np.array(bo_found, float); rnd_found = np.array(rnd_found, float)
    # AUC of best-so-far: higher = reaches good values sooner
    auc_gap = np.array(bo_auc) - np.array(rnd_auc)
    # evals-to-hit: lower is better; treat nan (never) as worst = seed_init+n_iter+1
    worst = seed_init + n_iter + 1
    bo_h = np.nan_to_num(bo_found, nan=worst); rnd_h = np.nan_to_num(rnd_found, nan=worst)
    evals_saved = float(np.mean(rnd_h - bo_h))
    auc_frac_pos = float((auc_gap > 0).mean())
    verdict = "CONFIRM" if (np.mean(auc_gap) > 0 and auc_frac_pos >= 0.75
                            and evals_saved > 0) else "SOFTEN"
    return {"verdict": verdict, "auc_gap_mean": round(float(np.mean(auc_gap)), 3),
            "auc_frac_seeds_positive": round(auc_frac_pos, 3),
            "bo_mean_evals_to_top1pct": round(float(np.nanmean(bo_found)), 1),
            "random_mean_evals_to_top1pct": round(float(np.nanmean(rnd_found)), 1),
            "evals_saved_mean": round(evals_saved, 1)}


# ===========================================================================
# P_CONF  conformal validity under scaffold shift (marginal vs Mondrian)
# ===========================================================================
def _rf_probs(Xtr, ytr, Xq):
    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    rf.fit(Xtr, ytr)
    return rf.predict_proba(Xq)


def _fp_matrix(fps, idx):
    arr = np.zeros((len(idx), 1024), dtype=np.float32)
    for r, i in enumerate(idx):
        DataStructs.ConvertToNumpyArray(fps[i], arr[r])
    return arr


def _coverage(probs, labels, q):
    scores = 1.0 - probs
    covered = 0
    for i, lab in enumerate(labels):
        pset = np.where(scores[i] <= q)[0]
        covered += int(lab in pset)
    return covered / len(labels)


def probe_conformal(data) -> dict:
    alpha = 0.10
    y = data["y"].astype(int)
    if set(np.unique(y)) - {0, 1}:
        return {"verdict": "SOFTEN", "reason": "non-binary labels"}
    n = len(y)
    rng = np.random.default_rng(0)

    # ---- random (exchangeable) split: train / cal / test ----
    perm = rng.permutation(n)
    tr, cal, te = perm[: n // 2], perm[n // 2: int(0.75 * n)], perm[int(0.75 * n):]
    probs_cal = _rf_probs(_fp_matrix(data["fps"], tr), y[tr], _fp_matrix(data["fps"], cal))
    probs_te = _rf_probs(_fp_matrix(data["fps"], tr), y[tr], _fp_matrix(data["fps"], te))
    uq = UncertaintyQuantifier(alpha=alpha)
    q_rand = uq.calibrate_conformal(probs_cal, y[cal])
    cov_random = _coverage(probs_te, y[te], q_rand)

    # ---- scaffold (shifted) split: cal from train scaffolds, test = held-out scaffolds ----
    uniq = np.unique(data["scaf"])
    sperm = rng.permutation(uniq)
    test_scaf = set(sperm[: max(1, int(0.25 * len(uniq)))].tolist())
    is_test = np.array([s in test_scaf for s in data["scaf"]])
    tr_idx = np.where(~is_test)[0]
    te_idx = np.where(is_test)[0]
    rng.shuffle(tr_idx)
    cut = int(0.7 * len(tr_idx))
    tr2, cal2 = tr_idx[:cut], tr_idx[cut:]
    pc = _rf_probs(_fp_matrix(data["fps"], tr2), y[tr2], _fp_matrix(data["fps"], cal2))
    pt = _rf_probs(_fp_matrix(data["fps"], tr2), y[tr2], _fp_matrix(data["fps"], te_idx))
    uq2 = UncertaintyQuantifier(alpha=alpha)
    q_marg = uq2.calibrate_conformal(pc, y[cal2])
    cov_scaffold_marginal = _coverage(pt, y[te_idx], q_marg)

    # ---- Mondrian (class-conditional) recalibration under the shifted split ----
    q_by_class = {}
    for k in (0, 1):
        m = y[cal2] == k
        if m.sum() >= 10:
            s = 1.0 - pc[m][:, k]
            ql = np.ceil((m.sum() + 1) * (1 - alpha)) / m.sum()
            q_by_class[k] = float(np.quantile(s, min(ql, 1.0)))
        else:
            q_by_class[k] = q_marg
    covered = 0
    for i, lab in enumerate(y[te_idx]):
        pset = [k for k in (0, 1) if (1.0 - pt[i, k]) <= q_by_class[k]]
        covered += int(lab in pset)
    cov_scaffold_mondrian = covered / len(te_idx)

    undercov = cov_scaffold_marginal < (1 - alpha) - 0.03
    fixed = cov_scaffold_mondrian > cov_scaffold_marginal + 0.01
    verdict = "CONFIRM" if (undercov and fixed) else ("SOFTEN" if undercov else "KILL")
    return {"verdict": verdict, "target_coverage": 1 - alpha,
            "cov_random_split": round(cov_random, 3),
            "cov_scaffold_marginal": round(cov_scaffold_marginal, 3),
            "cov_scaffold_mondrian": round(cov_scaffold_mondrian, 3),
            "undercoverage_under_shift": bool(undercov),
            "mondrian_helps": bool(fixed)}


# ===========================================================================
# P_DRIFT  autonomous drift monitoring (Page-Hinkley on a real induced shift)
# ===========================================================================
def probe_drift(data) -> dict:
    X, y = data["X"], data["y"]
    mw = X[:, 0]
    low = np.where(mw <= np.quantile(mw, 0.5))[0]
    high = np.where(mw > np.quantile(mw, 0.5))[0]
    rng = np.random.default_rng(0)
    rng.shuffle(low); rng.shuffle(high)
    tr = low[:int(0.6 * len(low))]
    rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    rf.fit(X[tr], y[tr])

    def err_stream(idx):
        return np.abs(rf.predict(X[idx]) - y[idx])

    ind = low[int(0.6 * len(low)):]                 # in-distribution
    shift = high                                    # shifted (high MW)
    L = min(len(ind) // 2, len(shift), 120)
    # shifted stream: L in-dist then L shifted
    stream_shift = np.concatenate([err_stream(ind[:L]), err_stream(shift[:L])])
    # control stream: 2L in-dist
    stream_ctrl = err_stream(ind[:2 * L]) if len(ind) >= 2 * L else err_stream(
        np.concatenate([ind, ind])[:2 * L])

    det = ChangePointDetector(threshold=5.0, allowance=0.5, min_observations=20)
    r_shift = det.page_hinkley(stream_shift, delta=0.005)
    r_ctrl = det.page_hinkley(stream_ctrl, delta=0.005)
    detected_shift = bool(r_shift.detected)
    idx_shift = r_shift.change_index if r_shift.detected else None
    latency = (idx_shift - L) if (detected_shift and idx_shift is not None and idx_shift >= L) else None
    early_false = bool(detected_shift and idx_shift is not None and idx_shift < L)
    ctrl_quiet = (not r_ctrl.detected) or (r_ctrl.change_index is not None and r_ctrl.change_index > int(1.5 * L))
    verdict = "CONFIRM" if (detected_shift and not early_false and ctrl_quiet) else "SOFTEN"
    return {"verdict": verdict, "transition_index": L,
            "detected_shift_at": idx_shift, "detection_latency": latency,
            "false_alarm_before_shift": early_false,
            "control_quiet": bool(ctrl_quiet),
            "mean_err_indist": round(float(stream_shift[:L].mean()), 3),
            "mean_err_shifted": round(float(stream_shift[L:].mean()), 3)}


# ===========================================================================
# P_ATTR  predictive uncertainty tracks scaffold novelty (OOD distance)
# ===========================================================================
def probe_attribution(data) -> dict:
    rng = np.random.default_rng(0)
    uniq = np.unique(data["scaf"])
    sperm = rng.permutation(uniq)
    test_scaf = set(sperm[: max(1, int(0.3 * len(uniq)))].tolist())
    is_test = np.array([s in test_scaf for s in data["scaf"]])
    tr = np.where(~is_test)[0]
    te = np.where(is_test)[0]
    Xtr = _fp_matrix(data["fps"], tr); ytr = data["y"][tr]
    Xte = _fp_matrix(data["fps"], te)

    # bootstrap ensemble -> per-point predictive std = epistemic uncertainty
    preds = []
    for b in range(15):
        bi = rng.integers(0, len(tr), len(tr))
        rf = RandomForestRegressor(n_estimators=80, random_state=b, n_jobs=-1)
        rf.fit(Xtr[bi], ytr[bi])
        preds.append(rf.predict(Xte))
    unc = np.std(np.stack(preds, 0), axis=0)

    tr_fps = [data["fps"][i] for i in tr]
    ood = np.array([1.0 - tanimoto_max_to_set(data["fps"][i], tr_fps) for i in te])
    rho, p = spearmanr(unc, ood)
    verdict = "CONFIRM" if (rho > 0.1 and p < 0.05) else "SOFTEN"
    return {"verdict": verdict, "spearman_unc_vs_ood": round(float(rho), 3),
            "p_value": float(f"{p:.2e}"), "n_test": len(te)}


# ===========================================================================
def main() -> None:
    t0 = time.time()
    log("Loading lipophilicity (regression) + BBB (classification) ...")
    lipo = load("lipophilicity")
    bbb = load("bbb")
    log(f"  lipophilicity n={len(lipo['y'])} scaffolds={lipo['n_scaf']}")
    log(f"  bbb           n={len(bbb['y'])} scaffolds={bbb['n_scaf']} "
        f"pos_rate={bbb['y'].mean():.2f}")

    results = {}
    probes = [
        ("P_SEQ  (sequential SPRT)", lambda: probe_sequential(lipo)),
        ("P_BO   (Bayesian optimization)", lambda: probe_bo(lipo)),
        ("P_CONF (conformal under shift)", lambda: probe_conformal(bbb)),
        ("P_DRIFT(drift monitoring)", lambda: probe_drift(lipo)),
        ("P_ATTR (uncertainty vs novelty)", lambda: probe_attribution(lipo)),
    ]
    for name, fn in probes:
        ts = time.time()
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            r = {"verdict": "ERROR", "error": repr(e)}
        r["_seconds"] = round(time.time() - ts, 1)
        results[name] = r
        log(f"\n### {name}  -> {r['verdict']}  ({r['_seconds']}s)")
        for k, v in r.items():
            if k not in ("verdict", "_seconds"):
                log(f"      {k}: {v}")

    log("\n" + "=" * 64)
    log("VERDICT SUMMARY")
    for name, r in results.items():
        log(f"  {name:34s} {r['verdict']}")
    log(f"total {round(time.time() - t0, 1)}s")
    log("=" * 64)


if __name__ == "__main__":
    main()
