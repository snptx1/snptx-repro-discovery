"""Configuration for the DL-forward multi-task ADMET study (AGENT_09, Pillar 1).

Endpoint typing (locked): four classification, two regression. All six are trained
jointly by the multi-task encoder and singly by the single-task baselines, then
compared against an RF descriptor baseline under identical Murcko scaffold cold-splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Endpoint:
    key: str            # ADMETAdapter endpoint key (see src/adapters/admet.py)
    task: str           # "clf" or "reg"
    pretty: str         # display name


# Order is stable and used for head indexing / reporting.
ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint("bbb", "clf", "BBB (Martins)"),
    Endpoint("ames", "clf", "AMES"),
    Endpoint("herg", "clf", "hERG"),
    Endpoint("hia", "clf", "HIA (Hou)"),
    Endpoint("solubility", "reg", "Solubility (AqSolDB)"),
    Endpoint("caco2", "reg", "Caco2 (Wang)"),
)

ENDPOINT_KEYS: tuple[str, ...] = tuple(e.key for e in ENDPOINTS)
CLF_KEYS: tuple[str, ...] = tuple(e.key for e in ENDPOINTS if e.task == "clf")
REG_KEYS: tuple[str, ...] = tuple(e.key for e in ENDPOINTS if e.task == "reg")


@dataclass(frozen=True)
class TrainConfig:
    # Architecture
    backbone: str = "gin"          # "gin" (edge-agnostic) or "gine" (edge-conditioned)
    hidden: int = 128
    num_layers: int = 4
    dropout: float = 0.3
    edge_dropout: float = 0.1      # DropEdge
    pair_norm_scale: float = 1.0   # PairNorm

    # Optimization
    epochs: int = 150
    lr: float = 5e-4
    weight_decay: float = 5e-4
    batch_size: int = 128

    # Featurizer dims (from DrugCombAdapter.smiles_to_graph)
    node_dim: int = 9
    edge_dim: int = 3

    # Evaluation protocol (pre-declared)
    test_scaffold_frac: float = 0.20
    val_scaffold_frac: float = 0.20
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    # Data-efficiency curve: fractions of the (scaffold) training pool.
    train_fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
    bootstrap_resamples: int = 2000

    # Multi-task loss weighting: regression losses are on standardized targets, so
    # per-task weights of 1.0 keep classification and regression on comparable scales.
    task_weights: dict[str, float] = field(default_factory=dict)


# Dry-run: FULL-size real data, all endpoints, but 1 seed / 1 fraction / few epochs.
# Exercises the exact GPU code path on CPU to de-risk before the multi-hour run.
DRYRUN = TrainConfig(
    epochs=5,
    seeds=(0,),
    train_fractions=(1.0,),
    bootstrap_resamples=200,
)

# Smoke config: tiny, CPU-only, validates plumbing without a GPU run.
SMOKE = TrainConfig(
    hidden=32,
    num_layers=2,
    epochs=3,
    seeds=(0,),
    train_fractions=(0.25, 1.0),
    bootstrap_resamples=50,
)
