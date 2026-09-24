"""AGENT_09 Pillar-1 ablation: self-supervised node-mask pretraining (Hu et al. 2020).

Question
--------
Does self-supervised attribute-mask pretraining of the shared GIN trunk improve
low-data transfer beyond training the same trunk from scratch? This is the deferred
second-pass ablation listed in the Pillar-1 spec (shared trunk vs independent heads,
PairNorm/DropEdge, number of tasks, and *optional self-supervised node/edge-mask
pretraining*).

Protocol (matched to the Pillar-1 main run so the deltas are exact)
------------------------------------------------------------------
- Same Murcko scaffold cold-splits, same seeds, same low-data training fractions.
- Four conditions per (seed, fraction), all evaluated on identical test scaffolds:
    single-task scratch      -> mtl_harness.train_single_task (canonical, unchanged)
    single-task pretrained   -> SSL trunk init, then the same single-task loop
    multi-task scratch       -> mtl_harness.train_multitask   (canonical, unchanged)
    multi-task pretrained    -> SSL trunk init, then the same multi-task loop
  The scratch conditions call the validated Pillar-1 code so the comparison isolates
  the pretraining effect alone.
- Pretraining corpus: the union of all six endpoints' *training-pool* graphs for that
  (seed, fraction). Labels are never used, and val/test scaffolds are excluded, so the
  self-supervised step sees exactly the molecules the supervised models can already see
  under the accepted multi-task protocol.

Attribute-mask objective (Hu et al. 2020, "Strategies for Pre-training GNNs")
----------------------------------------------------------------------------
Mask a fraction of atoms by zeroing their 9-dim feature row, embed the graph with a
node-level trunk, and predict each masked atom's element (feature[0] = atomic number)
with cross-entropy over the atomic-number vocabulary of the corpus.

Nothing here writes to protected data paths. Reuses mtl_harness for data, splits,
featurization, evaluation, and the canonical scratch trainers.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import mtl_harness as H
from mtl_config import ENDPOINT_KEYS, ENDPOINTS, TrainConfig
from mtl_harness import MultiTaskGNN, _class_weights, _head_loss, _reg_stats, set_determinism

HERE = Path(__file__).resolve().parent
ART = Path(os.environ.get("DL_ART_DIR", str(HERE / "artifacts")))
DEVICE = H.DEVICE

# Data-poor endpoints are where transfer lives; we still report all six.
LOW_FRACTIONS = (0.1, 0.25, 0.5)
MASK_RATE = 0.15
PRETRAIN_EPOCHS = 40


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Self-supervised attribute-mask pretraining.
# ---------------------------------------------------------------------------
class MaskedAtomPretrainer(nn.Module):
    """Node-level trunk + linear atom-type head for attribute-mask pretraining."""

    def __init__(self, cfg: TrainConfig, vocab_size: int):
        super().__init__()
        from src.models.gnn import build_gnn_model
        self.backbone = cfg.backbone
        self.trunk = build_gnn_model(
            cfg.backbone,
            in_channels=cfg.node_dim,
            out_channels=cfg.hidden,
            hidden_channels=cfg.hidden,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            task="node",  # return per-node embeddings (no graph pooling)
            edge_dropout=cfg.edge_dropout,
            pair_norm_scale=cfg.pair_norm_scale,
            edge_dim=cfg.edge_dim,
        )
        self.head = nn.Linear(cfg.hidden, vocab_size)

    def node_embed(self, batch) -> torch.Tensor:
        if self.backbone == "gine":
            return self.trunk.extract_embeddings(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
        return self.trunk.extract_embeddings(batch.x, batch.edge_index, batch.batch)


def _build_vocab(graphs: list) -> dict[int, int]:
    """Map each atomic number present in the corpus to a contiguous class index."""
    zs: set[int] = set()
    for g in graphs:
        zs.update(int(z) for z in g.x[:, 0].tolist())
    return {z: i for i, z in enumerate(sorted(zs))}


def pretrain_trunk(corpus: list, cfg: TrainConfig, seed: int):
    """Attribute-mask pretraining on the pooled (unlabeled) training corpus.

    Returns the trunk state_dict for transfer into the supervised models.
    """
    set_determinism(seed)
    vocab = _build_vocab(corpus)
    lut = torch.full((200,), -1, dtype=torch.long)
    for z, idx in vocab.items():
        lut[z] = idx
    lut = lut.to(DEVICE)

    model = MaskedAtomPretrainer(cfg, len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(corpus, batch_size=cfg.batch_size, shuffle=True)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    model.train()
    last = 0.0
    for _ in range(PRETRAIN_EPOCHS):
        for batch in loader:
            batch = batch.to(DEVICE)
            n = batch.x.shape[0]
            mask = torch.rand(n, generator=gen).to(DEVICE) < MASK_RATE
            if not bool(mask.any()):
                continue
            target = lut[batch.x[:, 0].long().clamp(0, 199)][mask]
            x_masked = batch.x.clone()
            x_masked[mask] = 0.0  # all-zero mask token
            batch.x = x_masked
            opt.zero_grad()
            node_emb = model.node_embed(batch)
            loss = F.cross_entropy(model.head(node_emb)[mask], target)
            loss.backward()
            opt.step()
            last = float(loss.detach().cpu())
    log(f"    [pretrain seed {seed}] corpus={len(corpus)} graphs vocab={len(vocab)} "
        f"final_masked_ce={last:.3f}")
    return {k: v.detach().cpu() for k, v in model.trunk.state_dict().items()}


def _init_trunk(model: MultiTaskGNN, trunk_state: dict) -> None:
    """Load pretrained trunk weights (conv/bn/pair_norm) into a supervised model."""
    state = {k: v.to(DEVICE) for k, v in trunk_state.items()}
    model.trunk.load_state_dict(state, strict=False)


# ---------------------------------------------------------------------------
# Supervised finetune loops (mirror mtl_harness exactly, plus a trunk-init hook).
# ---------------------------------------------------------------------------
def train_single_pretrained(data, train_idx, val_idx, cfg, seed, trunk_state):
    set_determinism(seed)
    ep = data.endpoint
    model = MultiTaskGNN(cfg, (ep,)).to(DEVICE)
    _init_trunk(model, trunk_state)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    weight = _class_weights(data.y, train_idx) if ep.task == "clf" else None
    reg_stat = _reg_stats(data.y, train_idx) if ep.task == "reg" else None
    loader = DataLoader([data.graphs[i] for i in train_idx],
                        batch_size=cfg.batch_size, shuffle=True)
    model.train()
    for _ in range(cfg.epochs):
        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            loss = _head_loss(model(batch, ep.key), batch.y, ep.task, weight, reg_stat)
            loss.backward()
            opt.step()
    return model, reg_stat


def train_multi_pretrained(datas, train_idx_map, cfg, seed, trunk_state):
    set_determinism(seed)
    endpoints = tuple(datas[k].endpoint for k in datas)
    model = MultiTaskGNN(cfg, endpoints).to(DEVICE)
    _init_trunk(model, trunk_state)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    weights, reg_stats, loaders = {}, {}, {}
    for k, data in datas.items():
        ep = data.endpoint
        idx = train_idx_map[k]
        weights[k] = _class_weights(data.y, idx) if ep.task == "clf" else None
        if ep.task == "reg":
            reg_stats[k] = _reg_stats(data.y, idx)
        loaders[k] = DataLoader([data.graphs[i] for i in idx],
                                batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        iters = {k: iter(dl) for k, dl in loaders.items()}
        n_steps = max(len(dl) for dl in loaders.values())
        for _step in range(n_steps):
            for k, data in datas.items():
                try:
                    batch = next(iters[k])
                except StopIteration:
                    iters[k] = iter(loaders[k])
                    batch = next(iters[k])
                batch = batch.to(DEVICE)
                ep = data.endpoint
                opt.zero_grad()
                w = cfg.task_weights.get(k, 1.0)
                loss = w * _head_loss(model(batch, k), batch.y, ep.task,
                                      weights[k], reg_stats.get(k))
                loss.backward()
                opt.step()
    return model, reg_stats


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def run(cfg: TrainConfig, endpoint_keys: tuple[str, ...], smoke: bool,
        out_dir: Path | None = None) -> dict:
    global ART
    if out_dir is not None:
        ART = out_dir
    ART.mkdir(exist_ok=True)
    t0 = time.time()
    log(f"device={DEVICE} backbone={cfg.backbone} seeds={cfg.seeds} "
        f"fractions={cfg.train_fractions} endpoints={endpoint_keys}")

    limit = 300 if smoke else None
    datas = {k: H.load_endpoint(k, limit=limit) for k in endpoint_keys}
    records: list[dict] = []

    for seed in cfg.seeds:
        log(f"=== seed {seed} ===")
        splits = {k: H.scaffold_three_way(datas[k].scaf, seed,
                                           cfg.test_scaffold_frac, cfg.val_scaffold_frac)
                  for k in endpoint_keys}

        for frac in cfg.train_fractions:
            train_map = {k: H.subsample_scaffolds(datas[k].scaf, splits[k][0], frac, seed)
                         for k in endpoint_keys}

            # Unlabeled pooled training corpus -> self-supervised trunk.
            corpus = [datas[k].graphs[i] for k in endpoint_keys for i in train_map[k]]
            trunk_state = pretrain_trunk(corpus, cfg, seed)

            # Multi-task (scratch = canonical Pillar-1 code; pretrained = SSL init).
            mt_scratch, mt_reg_s = H.train_multitask(datas, train_map, cfg, seed)
            mt_pre, mt_reg_p = train_multi_pretrained(datas, train_map, cfg, seed, trunk_state)

            for k in endpoint_keys:
                data = datas[k]
                ep = data.endpoint
                _, val_idx, test_idx = splits[k]
                common = {"endpoint": k, "task": ep.task, "fraction": frac, "seed": seed,
                          "n_train": int(len(train_map[k])), "n_test": int(len(test_idx))}

                # single-task scratch
                st_s, st_reg_s = H.train_single_task(data, train_map[k], val_idx, cfg, seed)
                yte, p = H.predict_gnn(st_s, data, test_idx, cfg, st_reg_s)
                records.append({**common, "method": "single_scratch",
                                **H.evaluate(yte, p, ep.task, cfg.bootstrap_resamples, seed)})

                # single-task pretrained
                st_p, st_reg_p = train_single_pretrained(
                    data, train_map[k], val_idx, cfg, seed, trunk_state)
                yte, p = H.predict_gnn(st_p, data, test_idx, cfg, st_reg_p)
                records.append({**common, "method": "single_pretrained",
                                **H.evaluate(yte, p, ep.task, cfg.bootstrap_resamples, seed)})

                # multi-task scratch
                yte, p = H.predict_gnn(mt_scratch, data, test_idx, cfg, mt_reg_s.get(k))
                records.append({**common, "method": "multi_scratch",
                                **H.evaluate(yte, p, ep.task, cfg.bootstrap_resamples, seed)})

                # multi-task pretrained
                yte, p = H.predict_gnn(mt_pre, data, test_idx, cfg, mt_reg_p.get(k))
                records.append({**common, "method": "multi_pretrained",
                                **H.evaluate(yte, p, ep.task, cfg.bootstrap_resamples, seed)})

            log(f"  seed {seed} fraction {frac}: done ({time.time()-t0:.0f}s elapsed)")

    return _summarize(records, cfg, endpoint_keys, smoke, time.time() - t0)


def _primary_metric(task: str) -> str:
    return "auroc" if task == "clf" else "mae"


def _summarize(records, cfg, endpoint_keys, smoke, runtime) -> dict:
    """Aggregate primary-metric deltas (pretrained - scratch) across seeds, with CIs."""
    fractions = list(cfg.train_fractions)
    seeds = list(cfg.seeds)
    summary: dict = {}

    for k in endpoint_keys:
        task = next(e.task for e in ENDPOINTS if e.key == k)
        m = _primary_metric(task)
        # For MAE lower is better, so a beneficial pretraining delta is negative.
        summary[k] = {"task": task, "metric": m, "single": [], "multi": []}
        for frac in fractions:
            def _vals(method):
                return np.array([
                    next(r[m] for r in records if r["endpoint"] == k
                         and r["method"] == method and r["fraction"] == frac
                         and r["seed"] == s)
                    for s in seeds], dtype=float)
            s_scr, s_pre = _vals("single_scratch"), _vals("single_pretrained")
            m_scr, m_pre = _vals("multi_scratch"), _vals("multi_pretrained")
            summary[k]["single"].append({
                "fraction": frac,
                "scratch_mean": float(s_scr.mean()), "pretrained_mean": float(s_pre.mean()),
                "delta_mean": float((s_pre - s_scr).mean()),
                "delta_std": float((s_pre - s_scr).std()),
            })
            summary[k]["multi"].append({
                "fraction": frac,
                "scratch_mean": float(m_scr.mean()), "pretrained_mean": float(m_pre.mean()),
                "delta_mean": float((m_pre - m_scr).mean()),
                "delta_std": float((m_pre - m_scr).std()),
            })

    payload = {
        "study": "AGENT_09 Pillar 1 ablation: self-supervised node-mask pretraining",
        "objective": "Hu et al. 2020 attribute masking (predict masked atomic number)",
        "generated_at": datetime.now(UTC).isoformat(),
        "device": str(DEVICE),
        "smoke": smoke,
        "config": asdict(cfg),
        "pretrain": {"epochs": PRETRAIN_EPOCHS, "mask_rate": MASK_RATE},
        "protocol": {"split": "murcko_scaffold_cold", "seeds": seeds,
                     "fractions": fractions, "bootstrap": cfg.bootstrap_resamples},
        "runtime_seconds": round(runtime, 1),
        "records": records,
        "summary": summary,
    }
    (ART / "pretrain_ablation.json").write_text(json.dumps(payload, indent=2))
    log(f"wrote {ART/'pretrain_ablation.json'}  ({runtime:.0f}s)")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU plumbing check")
    ap.add_argument("--full", action="store_true", help="full GPU ablation run")
    args = ap.parse_args()

    if args.smoke:
        cfg = replace(TrainConfig(), hidden=32, num_layers=2, epochs=3,
                      seeds=(0,), train_fractions=(0.25,), bootstrap_resamples=50)
        global PRETRAIN_EPOCHS
        PRETRAIN_EPOCHS = 2
        run(cfg, ENDPOINT_KEYS, smoke=True)
    else:
        cfg = replace(TrainConfig(), train_fractions=LOW_FRACTIONS)
        run(cfg, ENDPOINT_KEYS, smoke=False)


if __name__ == "__main__":
    main()
