"""Pillar-1 driver: train single-task and multi-task GNNs vs the RF descriptor
baseline under Murcko scaffold cold-splits, across seeds and training-data
fractions, and emit the transfer evidence (data-efficiency curves) with CIs.

Outputs (non-destructive) under dl_forward/artifacts/:
    - multitask_metrics.json  : per endpoint x method x fraction x seed metrics + CIs.
    - learning_curves.npz     : arrays (n_fractions, n_seeds) per endpoint/method/metric.
    - checkpoints/{mtl,single}_seed{S}.pt : full-data checkpoints for Pillar 2.

Usage:
    # tiny CPU plumbing check (no GPU):
    PYTHONPATH=src:. python pilot_phd/preprint/dl_forward/train_multitask_gnn.py --smoke
    # full A10G run (HITL-gated):
    PYTHONPATH=src:. python pilot_phd/preprint/dl_forward/train_multitask_gnn.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
import os
from pathlib import Path

import numpy as np
import torch

import mtl_harness as H
from mtl_config import DRYRUN, ENDPOINTS, SMOKE, TrainConfig

HERE = Path(__file__).resolve().parent
ART = Path(os.environ.get("DL_ART_DIR", str(HERE / "artifacts")))
CKPT = ART / "checkpoints"


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def _primary(task: str) -> str:
    return "auroc" if task == "clf" else "mae"


def run(cfg: TrainConfig, endpoint_keys: tuple[str, ...], smoke: bool,
        out_dir: Path | None = None, resume: bool = False) -> dict:
    global ART, CKPT
    if out_dir is not None:
        ART = out_dir
        CKPT = ART / "checkpoints"
    ART.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    t0 = time.time()
    log(f"device={H.DEVICE}  backbone={cfg.backbone}  seeds={cfg.seeds}  "
        f"fractions={cfg.train_fractions}  endpoints={endpoint_keys}  resume={resume}")

    limit = 300 if smoke else None
    datas = {k: H.load_endpoint(k, limit=limit) for k in endpoint_keys}

    # Incremental persistence so a host failure mid-run is resumable: each completed
    # (seed, fraction) unit is flushed to records.jsonl and logged in progress.json.
    records_path = ART / "records.jsonl"
    progress_path = ART / "progress.json"
    records: list[dict] = []
    done_units: set[tuple[int, float]] = set()
    if resume and records_path.exists() and progress_path.exists():
        records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        done_units = {(u[0], u[1]) for u in json.loads(progress_path.read_text())["done_units"]}
        log(f"resume: loaded {len(records)} records, {len(done_units)} completed units")
    else:
        records_path.write_text("")  # fresh run

    for seed in cfg.seeds:
        log(f"=== seed {seed} ===")
        # Per-endpoint scaffold three-way split (leakage-controlled).
        splits = {k: H.scaffold_three_way(datas[k].scaf, seed,
                                           cfg.test_scaffold_frac, cfg.val_scaffold_frac)
                  for k in endpoint_keys}

        for frac in cfg.train_fractions:
            if (seed, frac) in done_units:
                log(f"  seed {seed} fraction {frac}: skip (already done)")
                continue
            unit_records: list[dict] = []
            # Subsample each endpoint's train pool by whole scaffolds.
            train_map = {k: H.subsample_scaffolds(datas[k].scaf, splits[k][0], frac, seed)
                         for k in endpoint_keys}

            # (c) multi-task: one shared trunk, all endpoints.
            mt_model, mt_reg = H.train_multitask(datas, train_map, cfg, seed)

            for k in endpoint_keys:
                data = datas[k]
                ep = data.endpoint
                _, val_idx, test_idx = splits[k]

                # (a) RF descriptor baseline.
                yte, prf = H.rf_baseline(data, train_map[k], test_idx, seed)
                unit_records.append({"endpoint": k, "task": ep.task, "method": "rf",
                                "fraction": frac, "seed": seed, "n_train": int(len(train_map[k])),
                                "n_test": int(len(test_idx)),
                                **H.evaluate(yte, prf, ep.task, cfg.bootstrap_resamples, seed)})

                # (b) single-task GNN.
                st_model, st_reg = H.train_single_task(data, train_map[k], val_idx, cfg, seed)
                yte, pst = H.predict_gnn(st_model, data, test_idx, cfg, st_reg)
                unit_records.append({"endpoint": k, "task": ep.task, "method": "gnn_single",
                                "fraction": frac, "seed": seed, "n_train": int(len(train_map[k])),
                                "n_test": int(len(test_idx)),
                                **H.evaluate(yte, pst, ep.task, cfg.bootstrap_resamples, seed)})

                # (c) multi-task GNN.
                yte, pmt = H.predict_gnn(mt_model, data, test_idx, cfg, mt_reg.get(k))
                unit_records.append({"endpoint": k, "task": ep.task, "method": "gnn_multi",
                                "fraction": frac, "seed": seed, "n_train": int(len(train_map[k])),
                                "n_test": int(len(test_idx)),
                                **H.evaluate(yte, pmt, ep.task, cfg.bootstrap_resamples, seed)})

                # Save single-task full-data checkpoint (for the deep ensemble).
                if frac >= 1.0:
                    torch.save({"state_dict": st_model.state_dict(), "reg_stat": st_reg,
                                "endpoint": k, "seed": seed, "cfg": asdict(cfg)},
                               CKPT / f"single_{k}_seed{seed}.pt")

            # Save multi-task full-data checkpoint (for the deep ensemble).
            if frac >= 1.0:
                torch.save({"state_dict": mt_model.state_dict(), "reg_stats": mt_reg,
                            "seed": seed, "endpoints": list(endpoint_keys), "cfg": asdict(cfg)},
                           CKPT / f"mtl_seed{seed}.pt")

            # Flush this unit's records + progress (crash-safe boundary).
            records.extend(unit_records)
            done_units.add((seed, frac))
            with records_path.open("a") as fh:
                for r in unit_records:
                    fh.write(json.dumps(r) + "\n")
            progress_path.write_text(json.dumps(
                {"done_units": sorted([list(u) for u in done_units])}, indent=2))
            log(f"  seed {seed} fraction {frac}: done ({time.time()-t0:.0f}s elapsed)")

    return _summarize(records, cfg, endpoint_keys, smoke, time.time() - t0)


def _summarize(records: list[dict], cfg: TrainConfig, endpoint_keys: tuple[str, ...],
               smoke: bool, runtime: float) -> dict:
    """Aggregate across seeds and build the learning-curve arrays."""
    methods = ("rf", "gnn_single", "gnn_multi")
    fractions = list(cfg.train_fractions)
    seeds = list(cfg.seeds)

    agg: dict = {}
    curves: dict[str, np.ndarray] = {}
    for k in endpoint_keys:
        task = next(e.task for e in ENDPOINTS if e.key == k)
        metrics = ("auroc", "auprc") if task == "clf" else ("mae", "r2")
        agg[k] = {"task": task}
        for method in methods:
            agg[k][method] = {}
            for m in metrics:
                arr = np.full((len(fractions), len(seeds)), np.nan)
                for fi, frac in enumerate(fractions):
                    for si, seed in enumerate(seeds):
                        rec = next((r for r in records if r["endpoint"] == k
                                    and r["method"] == method and r["fraction"] == frac
                                    and r["seed"] == seed), None)
                        if rec is not None:
                            arr[fi, si] = rec[m]
                curves[f"{k}__{method}__{m}"] = arr
                agg[k][method][m] = {
                    "by_fraction": [
                        {"fraction": frac,
                         "mean": float(np.nanmean(arr[fi])),
                         "std": float(np.nanstd(arr[fi]))}
                        for fi, frac in enumerate(fractions)
                    ]
                }
    curves["_fractions"] = np.asarray(fractions, dtype=np.float32)
    curves["_seeds"] = np.asarray(seeds, dtype=np.int64)

    payload = {
        "study": "AGENT_09 Pillar 1 multi-task transfer",
        "generated_at": datetime.now(UTC).isoformat(),
        "device": str(H.DEVICE),
        "smoke": smoke,
        "config": asdict(cfg),
        "protocol": {"split": "murcko_scaffold_cold", "seeds": seeds,
                     "fractions": fractions, "bootstrap": cfg.bootstrap_resamples},
        "runtime_seconds": round(runtime, 1),
        "records": records,
        "summary": agg,
    }
    (ART / "multitask_metrics.json").write_text(json.dumps(payload, indent=2))
    np.savez(ART / "learning_curves.npz", **curves)
    log(f"wrote {ART/'multitask_metrics.json'} and {ART/'learning_curves.npz'}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU plumbing check")
    ap.add_argument("--dryrun", action="store_true",
                    help="full-size real data, 1 seed/1 fraction/few epochs -> dryrun/ dir")
    ap.add_argument("--backbone", choices=["gin", "gine"], default=None)
    ap.add_argument("--endpoints", nargs="*", default=None,
                    help="subset of endpoint keys (default: all six)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from artifacts/records.jsonl, skipping completed units")
    args = ap.parse_args()

    cfg = SMOKE if args.smoke else (DRYRUN if args.dryrun else TrainConfig())
    if args.backbone:
        cfg = TrainConfig(**{**asdict(cfg), "backbone": args.backbone})
    keys = tuple(args.endpoints) if args.endpoints else tuple(e.key for e in ENDPOINTS)
    if args.smoke and args.endpoints is None:
        keys = ("bbb", "solubility")  # one clf + one reg is enough to test plumbing

    out_dir = (HERE / "dryrun") if args.dryrun else None
    payload = run(cfg, keys, smoke=args.smoke, out_dir=out_dir, resume=args.resume)
    log("=== summary (mean across seeds, full-data fraction) ===")
    frac_last = cfg.train_fractions[-1]
    for k in keys:
        task = payload["summary"][k]["task"]
        pm = _primary(task)
        row = []
        for method in ("rf", "gnn_single", "gnn_multi"):
            bf = payload["summary"][k][method][pm]["by_fraction"]
            val = next(x["mean"] for x in bf if x["fraction"] == frac_last)
            row.append(f"{method}={val:.4f}")
        log(f"  {k} [{pm}]: " + "  ".join(row))


if __name__ == "__main__":
    main()
