"""Pillar-1 harness: multi-task representation transfer for ADMET.

Reuses the SNPTX engine rather than rebuilding it:
    - Molecular featurizer: DrugCombAdapter.smiles_to_graph (node 9-dim, edge 3-dim).
    - ADMET data: ADMETAdapter (TDC, dvc-pulled under data/raw/admet/).
    - GNN trunk: build_gnn_model(gin|gine) with PairNorm + DropEdge; we read the
      pooled graph embedding via extract_embeddings and attach our own per-endpoint
      heads for the multi-task and single-task models.

Baselines compared under identical Murcko scaffold cold-splits:
    (a) RF on RDKit descriptors + Morgan fingerprints (descriptor baseline / oracle),
    (b) single-task GNN (one trunk + one head per endpoint),
    (c) multi-task GNN (one shared trunk + per-endpoint heads).

Nothing here writes to protected data paths. Featurized graphs are cached under
dl_forward/cache/.
"""

from __future__ import annotations

import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # snptx-core
PREPRINT = HERE.parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

# WAF-403 shim must come before any TDC import.
sys.path.insert(0, str(PREPRINT))
sys.path.insert(0, str(ROOT))
import _tdc_download_patch  # noqa: E402,F401

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, Descriptors  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader  # noqa: E402

from src.adapters.admet import ADMETAdapter  # noqa: E402
from src.adapters.drugcomb import DrugCombAdapter  # noqa: E402
from src.models.gnn import build_gnn_model  # noqa: E402

from mtl_config import ENDPOINTS, Endpoint, TrainConfig  # noqa: E402

RDLogger.DisableLog("rdApp.*")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ENDPOINT_BY_KEY = {e.key: e for e in ENDPOINTS}
_TASK_INDEX = {e.key: i for i, e in enumerate(ENDPOINTS)}

# Morgan fingerprint width for the RF descriptor baseline.
_MORGAN_BITS = 1024


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Featurization (graphs + descriptors), cached per endpoint.
# ---------------------------------------------------------------------------
def _murcko(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)) or "(acyclic)"
    except Exception:
        return None


_DESCRIPTORS = (
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.TPSA,
    Descriptors.NumHDonors,
    Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds,
    Descriptors.NumAromaticRings,
    Descriptors.FractionCSP3,
    Descriptors.HeavyAtomCount,
    Descriptors.RingCount,
)


def _descriptor_vector(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    desc = np.array([float(fn(mol)) for fn in _DESCRIPTORS], dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=_MORGAN_BITS)
    bits = np.zeros((_MORGAN_BITS,), dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(fp, bits)
    return np.concatenate([desc, bits])


class EndpointData:
    """Featurized molecules for one endpoint, aligned across representations."""

    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint
        self.graphs: list = []          # PyG Data, .y = float raw label
        self.desc: np.ndarray | None = None  # (N, D) RF features
        self.y: np.ndarray | None = None     # (N,) raw labels
        self.scaf: np.ndarray | None = None  # (N,) integer scaffold ids
        self.task_index: int = _TASK_INDEX[endpoint.key]

    def __len__(self) -> int:
        return len(self.graphs)


def load_endpoint(key: str, *, limit: int | None = None) -> EndpointData:
    """Featurize one ADMET endpoint (cached). ``limit`` truncates for smoke runs."""
    ep = _ENDPOINT_BY_KEY[key]
    cache_file = CACHE / f"{key}.pt"
    if cache_file.exists() and limit is None:
        blob = torch.load(cache_file, weights_only=False)
        data = EndpointData(ep)
        data.graphs = blob["graphs"]
        data.desc = blob["desc"]
        data.y = blob["y"]
        data.scaf = blob["scaf"]
        log(f"  [{key}] loaded cache: {len(data.graphs)} molecules")
        return data

    log(f"  [{key}] featurizing {ep.pretty} ...")
    adapter = ADMETAdapter(raw_dir=str(ROOT / "data" / "raw" / "admet"))
    df = adapter.build(key, split="all")
    featurizer = DrugCombAdapter(raw_dir=str(ROOT / "data" / "raw" / "drugcomb"))

    graphs, descs, ys, scafs = [], [], [], []
    scaf_index: dict[str, int] = {}
    rows = list(zip(df["Drug"], df["property_value"]))
    if limit is not None:
        rows = rows[:limit]
    for smi, yval in rows:
        g = featurizer.smiles_to_graph(str(smi))
        dv = _descriptor_vector(str(smi))
        s = _murcko(smi)
        if g is None or g.x.shape[0] == 0 or dv is None or s is None:
            continue
        sid = scaf_index.setdefault(s, len(scaf_index))
        g.y = torch.tensor([float(yval)], dtype=torch.float32)
        g.task_index = _TASK_INDEX[ep.key]
        graphs.append(g)
        descs.append(dv)
        ys.append(float(yval))
        scafs.append(sid)

    data = EndpointData(ep)
    data.graphs = graphs
    data.desc = np.asarray(descs, dtype=np.float32)
    data.y = np.asarray(ys, dtype=np.float32)
    data.scaf = np.asarray(scafs, dtype=np.int64)
    log(f"  [{key}] {len(graphs)} molecules across {len(scaf_index)} scaffolds")

    if limit is None:
        torch.save(
            {"graphs": graphs, "desc": data.desc, "y": data.y, "scaf": data.scaf},
            cache_file,
        )
    return data


# ---------------------------------------------------------------------------
# Scaffold cold-splits (pre-declared protocol).
# ---------------------------------------------------------------------------
def scaffold_three_way(
    scaf: np.ndarray, seed: int, test_frac: float, val_frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whole-scaffold disjoint train/val/test indices (leakage-controlled)."""
    uniq = np.unique(scaf)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_test = max(1, int(round(test_frac * len(uniq))))
    n_val = max(1, int(round(val_frac * len(uniq))))
    test_s = set(perm[:n_test].tolist())
    val_s = set(perm[n_test : n_test + n_val].tolist())
    is_test = np.array([s in test_s for s in scaf])
    is_val = np.array([s in val_s for s in scaf])
    train_idx = np.where(~(is_test | is_val))[0]
    val_idx = np.where(is_val)[0]
    test_idx = np.where(is_test)[0]
    return train_idx, val_idx, test_idx


def subsample_scaffolds(
    scaf: np.ndarray, idx: np.ndarray, fraction: float, seed: int
) -> np.ndarray:
    """Subsample a training index set by whole scaffolds to fraction of scaffolds."""
    if fraction >= 1.0:
        return idx
    sub_scaf = scaf[idx]
    uniq = np.unique(sub_scaf)
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(uniq)
    keep_n = max(1, int(round(fraction * len(uniq))))
    keep = set(perm[:keep_n].tolist())
    mask = np.array([s in keep for s in sub_scaf])
    return idx[mask]


# ---------------------------------------------------------------------------
# Models: shared GIN/GINE trunk + per-endpoint heads.
# ---------------------------------------------------------------------------
def _make_trunk(cfg: TrainConfig) -> nn.Module:
    return build_gnn_model(
        cfg.backbone,
        in_channels=cfg.node_dim,
        out_channels=cfg.hidden,  # unused; we read extract_embeddings
        hidden_channels=cfg.hidden,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        task="graph",
        edge_dropout=cfg.edge_dropout,
        pair_norm_scale=cfg.pair_norm_scale,
        edge_dim=cfg.edge_dim,
    )


def _embed(trunk: nn.Module, backbone: str, batch) -> torch.Tensor:
    if backbone == "gine":
        return trunk.extract_embeddings(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
    return trunk.extract_embeddings(batch.x, batch.edge_index, batch.batch)


class MultiTaskGNN(nn.Module):
    """One shared trunk, per-endpoint heads (2 logits for clf, 1 value for reg)."""

    def __init__(self, cfg: TrainConfig, endpoints: tuple[Endpoint, ...]):
        super().__init__()
        self.cfg = cfg
        self.backbone = cfg.backbone
        self.trunk = _make_trunk(cfg)
        self.heads = nn.ModuleDict()
        self.head_task: dict[str, str] = {}
        for ep in endpoints:
            out = 2 if ep.task == "clf" else 1
            self.heads[ep.key] = nn.Linear(cfg.hidden, out)
            self.head_task[ep.key] = ep.task

    def forward(self, batch, key: str) -> torch.Tensor:
        emb = _embed(self.trunk, self.backbone, batch)
        return self.heads[key](emb)

    def embed(self, batch) -> torch.Tensor:
        return _embed(self.trunk, self.backbone, batch)


# ---------------------------------------------------------------------------
# Training utilities.
# ---------------------------------------------------------------------------
def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _reg_stats(y: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    vals = y[idx]
    mu = float(vals.mean())
    sd = float(vals.std()) or 1.0
    return mu, sd


def _class_weights(y: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    yi = y[idx].astype(int)
    n = len(yi)
    n0 = max((yi == 0).sum(), 1)
    n1 = max((yi == 1).sum(), 1)
    return torch.tensor([n / (2 * n0), n / (2 * n1)], dtype=torch.float32, device=DEVICE)


def _head_loss(logits: torch.Tensor, y: torch.Tensor, task: str,
               weight: torch.Tensor | None, reg_stat: tuple[float, float] | None) -> torch.Tensor:
    if task == "clf":
        target = y.view(-1).long()
        return F.cross_entropy(logits, target, weight=weight)
    mu, sd = reg_stat
    target = (y.view(-1) - mu) / sd
    return F.mse_loss(logits.view(-1), target)


def train_single_task(data: EndpointData, train_idx: np.ndarray, val_idx: np.ndarray,
                      cfg: TrainConfig, seed: int):
    """Train one trunk + one head on a single endpoint. Returns (model, reg_stat)."""
    set_determinism(seed)
    ep = data.endpoint
    model = MultiTaskGNN(cfg, (ep,)).to(DEVICE)
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
            logits = model(batch, ep.key)
            loss = _head_loss(logits, batch.y, ep.task, weight, reg_stat)
            loss.backward()
            opt.step()
    return model, reg_stat


def train_multitask(datas: dict[str, EndpointData], train_idx_map: dict[str, np.ndarray],
                    cfg: TrainConfig, seed: int):
    """Train one shared trunk + per-endpoint heads across all endpoints.

    Returns (model, reg_stats) where reg_stats maps reg-endpoint key -> (mu, sd).
    Balanced task updates: one gradient step per task-batch each round.
    """
    set_determinism(seed)
    endpoints = tuple(datas[k].endpoint for k in datas)
    model = MultiTaskGNN(cfg, endpoints).to(DEVICE)
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
        # Round-robin over tasks until the largest loader is exhausted.
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
                logits = model(batch, k)
                w = cfg.task_weights.get(k, 1.0)
                loss = w * _head_loss(logits, batch.y, ep.task, weights[k],
                                      reg_stats.get(k))
                loss.backward()
                opt.step()
    return model, reg_stats


@torch.no_grad()
def predict_gnn(model: MultiTaskGNN, data: EndpointData, idx: np.ndarray,
                cfg: TrainConfig, reg_stat: tuple[float, float] | None):
    """Return (y_true, y_pred) on idx. clf -> P(y=1); reg -> destandardized value."""
    model.eval()
    ep = data.endpoint
    loader = DataLoader([data.graphs[i] for i in idx],
                        batch_size=256, shuffle=False)
    preds, ys = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out = model(batch, ep.key)
        if ep.task == "clf":
            p = F.softmax(out, dim=-1)[:, 1]
            preds.append(p.cpu().numpy())
        else:
            mu, sd = reg_stat
            preds.append((out.view(-1).cpu().numpy() * sd + mu))
        ys.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds)


def rf_baseline(data: EndpointData, train_idx: np.ndarray, test_idx: np.ndarray,
                seed: int):
    """RF on RDKit descriptors + Morgan fingerprints. Returns (y_true, y_pred)."""
    ep = data.endpoint
    Xtr, ytr = data.desc[train_idx], data.y[train_idx]
    Xte, yte = data.desc[test_idx], data.y[test_idx]
    if ep.task == "clf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     n_jobs=-1, random_state=seed)
        clf.fit(Xtr, ytr.astype(int))
        # Robust to single-class training folds (small endpoints): map P(y=1)
        # onto the classifier's actual classes_.
        proba = clf.predict_proba(Xte)
        if 1 in clf.classes_:
            pred = proba[:, list(clf.classes_).index(1)]
        else:
            pred = np.zeros(len(Xte), dtype=np.float32)
    else:
        reg = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
        reg.fit(Xtr, ytr)
        pred = reg.predict(Xte)
    return yte, pred


# ---------------------------------------------------------------------------
# Metrics + bootstrap CIs.
# ---------------------------------------------------------------------------
def _clf_metric(y: np.ndarray, p: np.ndarray, which: str) -> float:
    yi = y.astype(int)
    if len(np.unique(yi)) < 2:
        return float("nan")
    if which == "auroc":
        return float(roc_auc_score(yi, p))
    return float(average_precision_score(yi, p))


def _reg_metric(y: np.ndarray, p: np.ndarray, which: str) -> float:
    if which == "mae":
        return float(mean_absolute_error(y, p))
    return float(r2_score(y, p))


def evaluate(y: np.ndarray, p: np.ndarray, task: str, resamples: int, seed: int) -> dict:
    """Point metrics + 2000-resample bootstrap 95% CIs."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if task == "clf":
        names, fn = ("auroc", "auprc"), _clf_metric
    else:
        names, fn = ("mae", "r2"), _reg_metric
    point = {m: fn(y, p, m) for m in names}
    boot = {m: [] for m in names}
    for _ in range(resamples):
        b = rng.integers(0, n, n)
        for m in names:
            v = fn(y[b], p[b], m)
            if not np.isnan(v):
                boot[m].append(v)
    out = {}
    for m in names:
        arr = np.asarray(boot[m]) if boot[m] else np.asarray([point[m]])
        out[m] = float(point[m])
        out[f"{m}_ci"] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
    return out
