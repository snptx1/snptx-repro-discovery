"""Historical GAT training and incoming-attention score diagnostic.

The original experiment tried to compare an atom score with an aromatic-or-nitrogen
rule. That attribution interpretation is withdrawn: the score sums softmax-normalized
incoming coefficients, including self-loops, and is identically one in evaluation mode.
Rule AUROC and top-three enrichment therefore rank floating-point residuals rather
than meaningful atom importance. Rerunning this driver does not repair the statistic.

This script retains the original training, eligibility filter, scoring, and archived
metric names for provenance. Predictive GAT AUROC is a separate result. Fresh output
explicitly marks attribution as invalid and records the numerical score spread.
Figure 6 and the walkthrough diagnose the committed scores without retraining.

Usage from the repository root (GPU recommended; training is unseeded):
    PYTHONPATH=src:. python preprint/dl_forward/attention_attribution.py --smoke
    PYTHONPATH=src:. python preprint/dl_forward/attention_attribution.py

Outputs overwrite attention_attribution.json and attention_nodes.npz in DL_ART_DIR
(or the local artifacts directory). Use a separate DL_ART_DIR to preserve archived runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

import mtl_harness as H
from mtl_config import TrainConfig
from src.models.gnn import build_gnn_model

HERE = Path(__file__).resolve().parent
ART = Path(os.environ.get("DL_ART_DIR", str(HERE / "artifacts")))
ART.mkdir(exist_ok=True)
SPLIT_SEED = 0
ENDPOINTS = ("herg", "bbb")
TOPK = 3


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _labels(data: H.EndpointData, idx: np.ndarray) -> torch.Tensor:
    return torch.tensor((data.y[idx] > 0.5).astype(np.int64))


def train_gat(data: H.EndpointData, train_idx: np.ndarray, val_idx: np.ndarray,
              cfg: TrainConfig, device: torch.device):
    """Train a GAT classifier; return the best-val-AUROC model state."""
    model = build_gnn_model(
        "gat", in_channels=9, hidden_channels=64, out_channels=2,
        num_layers=3, heads=4, dropout=0.3, task="graph",
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_loader = DataLoader([data.graphs[i] for i in train_idx],
                              batch_size=cfg.batch_size, shuffle=True)
    y_val = _labels(data, val_idx).numpy()

    best_auc, best_state = -1.0, None
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            y = (batch.y.view(-1) > 0.5).long()
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
        if (epoch + 1) % 10 == 0 or epoch == cfg.epochs - 1:
            probs = predict_probs(model, data, val_idx, cfg, device)
            try:
                auc = roc_auc_score(y_val, probs)
            except ValueError:
                auc = 0.5
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


@torch.no_grad()
def predict_probs(model, data: H.EndpointData, idx: np.ndarray,
                  cfg: TrainConfig, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader([data.graphs[i] for i in idx],
                        batch_size=cfg.batch_size, shuffle=False)
    out = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        out.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def node_attention(model, graph, device: torch.device) -> np.ndarray:
    """Legacy normalized incoming sum: constant one, invalid as atom attribution."""
    model.eval()
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    n = x.shape[0]
    # replicate GATModel.extract_embeddings, capturing final-layer attention.
    for i, (conv, bn) in enumerate(zip(model.convs, model.bns)):
        if i == len(model.convs) - 1:
            x, (att_ei, alpha) = conv(x, edge_index, return_attention_weights=True)
        else:
            x = conv(x, edge_index)
        x = bn(x)
        if i < len(model.convs) - 1:
            x = F.elu(x)
    alpha = alpha.mean(dim=1).cpu().numpy()          # (num_edges,) average over heads
    tgt = att_ei[1].cpu().numpy()
    score = np.zeros(n, dtype=np.float64)
    for e, t in enumerate(tgt):
        if t < n:                                    # ignore any self-loop padding beyond n
            score[t] += alpha[e]
    return score


def salient_labels(graph) -> np.ndarray:
    """Chemically motivated atom salience from node features: aromatic OR nitrogen."""
    x = graph.x.cpu().numpy()
    aromatic = x[:, 4] > 0.5
    nitrogen = np.abs(x[:, 0] - 7.0) < 0.5
    return (aromatic | nitrogen).astype(np.int64)


def evaluate_endpoint(key: str, cfg: TrainConfig, device: torch.device) -> dict:
    data = H.load_endpoint(key, limit=(200 if cfg.epochs <= 5 else None))
    tr, va, te = H.scaffold_three_way(data.scaf, SPLIT_SEED,
                                      cfg.test_scaffold_frac, cfg.val_scaffold_frac)
    log(f"[{key}] split train={len(tr)} val={len(va)} test={len(te)}; training GAT ...")
    model, val_auc = train_gat(data, tr, va, cfg, device)
    test_probs = predict_probs(model, data, te, cfg, device)
    try:
        test_auc = float(roc_auc_score(_labels(data, te).numpy(), test_probs))
    except ValueError:
        test_auc = float("nan")

    all_scores, all_sal, enrich = [], [], []
    for i in te:
        g = data.graphs[i]
        if g.x.shape[0] < TOPK + 1:
            continue
        att = node_attention(model, g, device)
        sal = salient_labels(g)
        if sal.sum() == 0 or sal.sum() == len(sal):
            continue
        all_scores.append(att)
        all_sal.append(sal)
        k = min(TOPK, len(att))
        top = np.argsort(att)[-k:]
        enrich.append(sal[top].mean() - sal.mean())

    flat_scores = np.concatenate(all_scores)
    flat_sal = np.concatenate(all_sal)
    try:
        node_auc = float(roc_auc_score(flat_sal, flat_scores))
    except ValueError:
        node_auc = float("nan")

    res = {
        "endpoint": key,
        "attribution_valid": False,
        "score_population_std": float(flat_scores.std(ddof=0)),
        "score_max_abs_deviation_from_one": float(np.max(np.abs(flat_scores - 1.0))),
        "val_auroc": round(val_auc, 4),
        "test_auroc": round(test_auc, 4),
        "n_test_molecules": int(len(all_scores)),
        "node_attention_auroc": round(node_auc, 4),
        "topk_enrichment": round(float(np.mean(enrich)), 4),
        "salient_base_rate": round(float(flat_sal.mean()), 4),
        "topk": TOPK,
    }
    log(f"[{key}] test_auroc={res['test_auroc']} node_att_auroc={res['node_attention_auroc']} "
        f"topk_enrichment={res['topk_enrichment']}")
    return res, flat_scores, flat_sal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU sanity run")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.smoke:
        cfg = replace(cfg, epochs=5, batch_size=32)

    device = _device()
    log("WARNING: incoming attention sums are constant; rule metrics are invalid attribution diagnostics.")
    log(f"device={device}  endpoints={ENDPOINTS}  epochs={cfg.epochs}")

    results, npz = {}, {}
    for key in ENDPOINTS:
        res, scores, sal = evaluate_endpoint(key, cfg, device)
        results[key] = res
        npz[f"{key}__scores"] = scores
        npz[f"{key}__salient"] = sal

    meta = {
        "study": "attention_attribution",
        "attribution_valid": False,
        "diagnostic": "Incoming softmax-normalized attention sums equal one in exact arithmetic.",
        "generated_at": datetime.now(UTC).isoformat(),
        "device": str(device),
        "smoke": args.smoke,
        "split_seed": SPLIT_SEED,
        "rule": "salient atom = aromatic OR nitrogen (from node features)",
        "results": results,
    }
    (ART / "attention_attribution.json").write_text(json.dumps(meta, indent=2))
    np.savez(ART / "attention_nodes.npz", **npz)
    log("wrote attention_attribution.json + attention_nodes.npz")


if __name__ == "__main__":
    main()
