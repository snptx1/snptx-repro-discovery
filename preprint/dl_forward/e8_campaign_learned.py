"""Learned-oracle decision campaign (AGENT_09 Pillar-2 -> decisions).

Non-destructive companion to `pilot_phd/preprint/e8_campaign.py`. That campaign drives the
SPRT go/no-go loop with a random-forest descriptor oracle. Here we swap in the *learned*
oracle -- the K=5 deep-ensemble graph model from `make_ensemble_uncertainty.py` -- and run
the identical decision logic, so the two are directly comparable.

The SPRT step is oracle-agnostic: it needs only a per-molecule ranking signal (here the
ensemble's P(positive)) and the realized test labels. We reuse the ensemble test
probabilities and labels already committed to `artifacts/ensemble_predictions.npz`
(`{key}__ensemble` and `{key}___y_true`), and the conformal / selective decision metrics
from `artifacts/ensemble_uncertainty.json`. The SPRT parameters and the top-fraction
subgroup match `e8_campaign.py` exactly (ALPHA=0.05, BETA=0.20, MEANINGFUL_EFFECT=0.30 in
sigma units, TOPFRAC=0.30) so the measurement-saving numbers are on the same footing.

Output (non-destructive) under dl_forward/artifacts/:
    - campaign_learned.json : per-endpoint go/no-go + measurements saved, plus the
      committed conformal coverage and selective accuracy for the learned oracle.
    - campaign_learned_lineage.duckdb : a DuckDB provenance store (via
      ``src.intelligence.catalog.ExperimentCatalog``) with one row per endpoint recording
      the learned oracle, its SPRT decision, and the conformal / selective metrics, so every
      go/no-go call is traceable from task to decision.

Usage:
    PYTHONPATH=src:. python pilot_phd/preprint/dl_forward/e8_campaign_learned.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
ROOT = HERE.parents[2]  # snptx-core
sys.path.insert(0, str(ROOT))

from src.intelligence.catalog import ExperimentCatalog  # noqa: E402
from src.intelligence.experiment_design import SPRT  # noqa: E402

# Decision parameters, identical to e8_campaign.py (single source of truth for the values).
ALPHA, BETA = 0.05, 0.20
MEANINGFUL_EFFECT = 0.30
TOPFRAC = 0.30
SEED = 20260905
CLF_KEYS = ("bbb", "ames", "herg", "hia")
DB_PATH = ART / "campaign_learned_lineage.duckdb"


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def sprt_decision(pred_signal: np.ndarray, y_test: np.ndarray, rng) -> dict:
    """Sequentially measure the oracle's top-predicted subgroup; SPRT decides whether its
    z-scored property is shifted from the pool baseline by >= MEANINGFUL_EFFECT.

    Mirrors ``e8_campaign.sprt_decision`` (same constants), so the learned-oracle and
    descriptor-oracle campaigns are directly comparable.
    """
    yz = (y_test - y_test.mean()) / (y_test.std() + 1e-9)
    order = np.argsort(-pred_signal)
    k = max(8, int(TOPFRAC * len(order)))
    subgroup = order[:k]
    measurements = yz[subgroup]
    sign = 1.0 if measurements.mean() >= 0 else -1.0
    stream = sign * measurements
    perm = rng.permutation(len(stream))

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


def main() -> None:
    preds = np.load(ART / "ensemble_predictions.npz")
    unc = json.loads((ART / "ensemble_uncertainty.json").read_text())["results"]
    rng = np.random.default_rng(SEED)

    # Fresh DuckDB lineage store: one provenance row per endpoint decision.
    if DB_PATH.exists():
        DB_PATH.unlink()
    catalog = ExperimentCatalog(DB_PATH)

    records, tot_used, tot_fixed, n_go = {}, 0, 0.0, 0
    for key in CLF_KEYS:
        t0 = time.time()
        signal = preds[f"{key}__ensemble"]
        y_test = preds[f"{key}___y_true"].astype(float)
        dec = sprt_decision(signal, y_test, rng)
        ens = unc[key]["ensemble"]
        record = {
            "sprt": dec,
            "conformal_coverage_target": 0.90,
            "conf_set_size_at_90": round(float(ens["conf_set_size_at_90"]), 3),
            "selective_acc_at_70": round(float(ens["sel_acc_at_70"]), 3),
            "ece": round(float(ens["ece"]), 4),
        }

        # Lineage: log this endpoint's learned-oracle decision to DuckDB. The oracle's
        # mean P(positive) and its dispersion stand in as the surrogate prediction and
        # uncertainty, so the store captures what the oracle believed and what was decided.
        eid = catalog.record_experiment(
            dataset=f"admet_{key}", endpoint=key,
            model_type="learned_ensemble_oracle",
            config_hash=f"e8learned_{key}_{SEED}",
            hyperparams={"alpha": ALPHA, "beta": BETA,
                         "meaningful_effect_sigma": MEANINGFUL_EFFECT, "top_fraction": TOPFRAC},
            metrics={"sprt": dec, "conformal": {
                "conf_set_size_at_90": record["conf_set_size_at_90"],
                "selective_acc_at_70": record["selective_acc_at_70"],
                "ece": record["ece"]}},
            surrogate_prediction=float(np.mean(signal)),
            surrogate_uncertainty=float(np.std(signal)),
            duration_seconds=time.time() - t0,
        )
        record["experiment_id"] = eid
        records[key] = record

        tot_used += dec["n_measurements"]
        tot_fixed += dec["fixed_sample_n"]
        n_go += int(dec["decision"] == "reject_h0")
        log(f"[{key}] {dec['verdict']}  n={dec['n_measurements']} vs fixed "
            f"{dec['fixed_sample_n']}  (saved {dec['measurements_saved']})  "
            f"lineage={eid[:8]}")

    catalog.close()

    saved = tot_fixed - tot_used
    pct = round(100.0 * saved / tot_fixed, 1) if tot_fixed else 0.0
    summary = {
        "n_endpoints": len(CLF_KEYS),
        "n_go_decisions": int(n_go),
        "total_measurements_used": int(tot_used),
        "total_fixed_sample": round(tot_fixed, 1),
        "measurements_saved": round(saved, 1),
        "pct_saved": pct,
    }
    log(f"aggregate: {n_go}/{len(CLF_KEYS)} GO  ·  {tot_used} vs "
        f"{round(tot_fixed,1)} measurements  ·  {pct}% saved")

    payload = {
        "study": "AGENT_09 learned-oracle decision campaign",
        "generated_at": datetime.now(UTC).isoformat(),
        "oracle": "K=5 deep ensemble (multi-task graph stack)",
        "sprt": {"alpha": ALPHA, "beta": BETA, "meaningful_effect_sigma": MEANINGFUL_EFFECT,
                 "top_fraction": TOPFRAC},
        "lineage_duckdb": str(DB_PATH.relative_to(ROOT)),
        "per_endpoint": records,
        "summary": summary,
    }
    (ART / "campaign_learned.json").write_text(json.dumps(payload, indent=2))
    log(f"wrote campaign_learned.json  ·  DuckDB lineage: {DB_PATH.name}")


if __name__ == "__main__":
    main()
