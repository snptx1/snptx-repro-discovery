"""Pillar-2 driver: deep-ensemble uncertainty to decision quality (AGENT_09).

Claim under test: a K=5 deep ensemble plus temperature scaling gives better-calibrated,
sharper uncertainty than RF, a single GNN, or MC-dropout, and it propagates to better
decisions (smaller conformal sets at fixed 90% coverage, dominating risk-coverage).

Design note (important): a deep ensemble requires ONE fixed train/val/test split with K
differently-initialized members. Reusing the Pillar-1 per-seed splits (which differ by
seed) would leak test molecules into other members' training sets. So this driver trains
K members with init seeds 0..K-1 on a single fixed scaffold split.

Scope: the classification endpoints (BBB, AMES, hERG, HIA). Conformal sets, ECE, and
selective accuracy are classification decision-quality metrics; regression uncertainty is
handled separately (ensemble variance) and is not part of these tables.

Reuses src/safety/uncertainty.py (conformal, ECE) and the Pillar-1 harness.

Outputs (non-destructive) under dl_forward/artifacts/:
    - ensemble_uncertainty.json : per endpoint x method calibration + decision metrics.
    - ensemble_predictions.npz  : per-endpoint test probabilities per method (learned
      oracle analogue of conformal_selective_real.npz).

Usage:
    PYTHONPATH=src:. python pilot_phd/preprint/dl_forward/make_ensemble_uncertainty.py --smoke
    PYTHONPATH=src:. python pilot_phd/preprint/dl_forward/make_ensemble_uncertainty.py   # GPU, HITL-gated
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import mtl_harness as H
from mtl_config import CLF_KEYS, SMOKE, TrainConfig
from src.safety.uncertainty import UncertaintyQuantifier

HERE = Path(__file__).resolve().parent
ART = Path(os.environ.get("DL_ART_DIR", str(HERE / "artifacts")))
SPLIT_SEED = 0          # the single fixed evaluation split
ALPHA = 0.10            # 90% conformal target
COVERAGE_TARGET = 0.70  # selective-accuracy operating point


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Member logits / probabilities.
# ---------------------------------------------------------------------------
@torch.no_grad()
def member_logits(model, data: H.EndpointData, idx: np.ndarray, key: str):
    """Return (logits (n,2), y (n,)) for a single-endpoint classification member."""
    model.eval()
    loader = DataLoader([data.graphs[i] for i in idx], batch_size=256, shuffle=False)
    logits, ys = [], []
    for batch in loader:
        batch = batch.to(H.DEVICE)
        logits.append(model(batch, key).cpu().numpy())
        ys.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(logits), np.concatenate(ys).astype(int)


@torch.no_grad()
def mc_dropout_probs(model, data: H.EndpointData, idx: np.ndarray, key: str, passes: int):
    """Mean softmax over `passes` stochastic forward passes (dropout active)."""
    model.train()  # keep dropout on
    loader = DataLoader([data.graphs[i] for i in idx], batch_size=256, shuffle=False)
    runs = []
    for _ in range(passes):
        ps = []
        for batch in loader:
            batch = batch.to(H.DEVICE)
            ps.append(F.softmax(model(batch, key), dim=-1).cpu().numpy())
        runs.append(np.concatenate(ps))
    return np.stack(runs).mean(axis=0)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    logits = torch.tensor(val_logits, dtype=torch.float32)
    labels = torch.tensor(val_labels, dtype=torch.long)
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_T.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp().item())


# ---------------------------------------------------------------------------
# Decision-quality metrics.
# ---------------------------------------------------------------------------
def nll(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((probs[:, 1] - y) ** 2))


def aurc(probs: np.ndarray, y: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    order = np.argsort(-conf)
    err = 1.0 - correct[order]
    risks = np.cumsum(err) / (np.arange(len(err)) + 1)
    return float(np.mean(risks))


def selective_accuracy(probs: np.ndarray, y: np.ndarray, coverage: float) -> float:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    order = np.argsort(-conf)
    k = max(1, int(round(coverage * len(y))))
    return float(np.mean(correct[order][:k]))


def conformal_efficiency(val_probs, val_y, test_probs, test_y):
    """Mean conformal set size + empirical coverage at the 90% target."""
    q = UncertaintyQuantifier(alpha=ALPHA)
    q.calibrate_conformal(val_probs, val_y)
    sizes, covered = [], []
    for i in range(len(test_y)):
        cset = q.conformal_set(test_probs[i])
        sizes.append(len(cset))
        covered.append(int(test_y[i]) in cset)
    return float(np.mean(sizes)), float(np.mean(covered))


def decision_metrics(val_probs, val_y, test_probs, test_y) -> dict:
    ece = UncertaintyQuantifier(alpha=ALPHA).compute_calibration(test_probs, test_y)["ece"]
    set_size, coverage = conformal_efficiency(val_probs, val_y, test_probs, test_y)
    return {
        "ece": float(ece),
        "nll": nll(test_probs, test_y),
        "brier": brier(test_probs, test_y),
        "conf_set_size_at_90": set_size,
        "conf_coverage_at_90": coverage,
        "aurc": aurc(test_probs, test_y),
        "sel_acc_at_70": selective_accuracy(test_probs, test_y, COVERAGE_TARGET),
        "test_auroc": H._clf_metric(test_y, test_probs[:, 1], "auroc"),
    }


# ---------------------------------------------------------------------------
# Per-endpoint ensemble build + baseline comparison.
# ---------------------------------------------------------------------------
def run_endpoint(key: str, cfg: TrainConfig, K: int, mc_passes: int) -> tuple[dict, dict]:
    data = H.load_endpoint(key, limit=(300 if cfg.epochs <= 3 else None))
    train_idx, val_idx, test_idx = H.scaffold_three_way(
        data.scaf, SPLIT_SEED, cfg.test_scaffold_frac, cfg.val_scaffold_frac)
    log(f"[{key}] fixed split train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}; "
        f"training K={K} members ...")

    val_p_members, test_p_members = [], []
    first_model = None
    for m in range(K):
        model, _ = H.train_single_task(data, train_idx, val_idx, cfg, seed=m)
        if first_model is None:
            first_model = model
        vlog, vy = member_logits(model, data, val_idx, key)
        tlog, ty = member_logits(model, data, test_idx, key)
        T = fit_temperature(vlog, vy)
        val_p_members.append(_softmax(vlog / T))
        test_p_members.append(_softmax(tlog / T))
    val_y = vy
    test_y = ty

    # Methods -------------------------------------------------------------
    results, preds = {}, {}

    # (1) deep ensemble: mean of temp-scaled member probabilities.
    ens_val = np.mean(val_p_members, axis=0)
    ens_test = np.mean(test_p_members, axis=0)
    results["ensemble"] = decision_metrics(ens_val, val_y, ens_test, test_y)
    preds["ensemble"] = ens_test[:, 1]

    # (2) single GNN: first member (already temp-scaled).
    results["gnn_single"] = decision_metrics(
        val_p_members[0], val_y, test_p_members[0], test_y)
    preds["gnn_single"] = test_p_members[0][:, 1]

    # (3) MC-dropout on the first member.
    mc_val = mc_dropout_probs(first_model, data, val_idx, key, mc_passes)
    mc_test = mc_dropout_probs(first_model, data, test_idx, key, mc_passes)
    results["mc_dropout"] = decision_metrics(mc_val, val_y, mc_test, test_y)
    preds["mc_dropout"] = mc_test[:, 1]

    # (4) RF descriptor baseline (2-class probs).
    _, rf_val_p1 = H.rf_baseline(data, train_idx, val_idx, SPLIT_SEED)
    _, rf_test_p1 = H.rf_baseline(data, train_idx, test_idx, SPLIT_SEED)
    rf_val = np.stack([1 - rf_val_p1, rf_val_p1], axis=1)
    rf_test = np.stack([1 - rf_test_p1, rf_test_p1], axis=1)
    results["rf"] = decision_metrics(rf_val, val_y, rf_test, test_y)
    preds["rf"] = rf_test_p1

    preds["_y_true"] = test_y.astype(np.int64)
    return results, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU validation of the metrics")
    ap.add_argument("--endpoints", nargs="*", default=None)
    args = ap.parse_args()

    cfg = SMOKE if args.smoke else TrainConfig()
    K = 3 if args.smoke else len(cfg.seeds)
    mc_passes = 5 if args.smoke else 30
    keys = tuple(args.endpoints) if args.endpoints else CLF_KEYS
    if args.smoke and args.endpoints is None:
        keys = ("bbb", "herg")

    ART.mkdir(exist_ok=True)
    log(f"device={H.DEVICE}  K={K}  endpoints={keys}  mc_passes={mc_passes}")

    all_results, npz = {}, {}
    for key in keys:
        res, preds = run_endpoint(key, cfg, K, mc_passes)
        all_results[key] = res
        for method, arr in preds.items():
            npz[f"{key}__{method}"] = arr
        log(f"[{key}] ensemble ECE={res['ensemble']['ece']:.4f} "
            f"set@90={res['ensemble']['conf_set_size_at_90']:.3f}  |  "
            f"rf ECE={res['rf']['ece']:.4f} set@90={res['rf']['conf_set_size_at_90']:.3f}")

    payload = {
        "study": "AGENT_09 Pillar 2 ensemble uncertainty-to-decisions",
        "generated_at": datetime.now(UTC).isoformat(),
        "device": str(H.DEVICE),
        "smoke": args.smoke,
        "K": K,
        "split_seed": SPLIT_SEED,
        "alpha": ALPHA,
        "coverage_target": COVERAGE_TARGET,
        "config": asdict(cfg),
        "results": all_results,
    }
    (ART / "ensemble_uncertainty.json").write_text(json.dumps(payload, indent=2))
    np.savez(ART / "ensemble_predictions.npz", **npz)
    log(f"wrote {ART/'ensemble_uncertainty.json'} and {ART/'ensemble_predictions.npz'}")


if __name__ == "__main__":
    main()
