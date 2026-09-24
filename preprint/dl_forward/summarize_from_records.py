"""Rebuild multitask_metrics.json + learning_curves.npz from the flushed records.jsonl.

Two uses:
  1. Crash recovery: if the run dies before the final summarize, the per-unit records are
     still on disk; this reconstructs the summary artifacts from them.
  2. Mid-run preview: after some seeds finish, build a partial summary to preview figures
     without waiting for the whole run.

Usage:
    PYTHONPATH=src:. python pilot_phd/preprint/dl_forward/summarize_from_records.py
"""

from __future__ import annotations

import json
from pathlib import Path

from mtl_config import TrainConfig
from train_multitask_gnn import ART, _summarize

HERE = Path(__file__).resolve().parent


def main() -> None:
    records_path = ART / "records.jsonl"
    if not records_path.exists():
        raise SystemExit(f"no records at {records_path}")
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit("records.jsonl is empty")

    seeds = sorted({r["seed"] for r in records})
    fractions = sorted({r["fraction"] for r in records})
    endpoints = tuple(dict.fromkeys(r["endpoint"] for r in records))
    boot = max(r.get("_boot", 2000) for r in records) if records else 2000
    cfg = TrainConfig(seeds=tuple(seeds), train_fractions=tuple(fractions),
                      bootstrap_resamples=boot)
    print(f"rebuilding from {len(records)} records: seeds={seeds} fractions={fractions} "
          f"endpoints={endpoints}")
    _summarize(records, cfg, endpoints, smoke=False, runtime=0.0)
    print("wrote multitask_metrics.json + learning_curves.npz (partial if run incomplete)")


if __name__ == "__main__":
    main()
