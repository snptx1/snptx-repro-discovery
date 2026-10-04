"""Cross-endpoint structural overlap under the per-endpoint scaffold splits (CPU, minutes).

The transfer and pretraining studies split each endpoint into scaffold-disjoint train /
validation / test sets independently. A shared multi-task trunk (and the pooled pretraining
corpus) also trains on the *other* endpoints' training molecules, which can share scaffolds,
or be the same molecules, as the target endpoint's test set. The single-task baseline never
sees them. This script measures that exposure so it can be reported.

For each seed, training fraction, and target endpoint t, it reports the fraction of t's test
molecules whose Murcko scaffold (or canonical SMILES) appears among the training molecules of
any other endpoint. Splits are reproduced exactly with ``mtl_harness.scaffold_three_way`` and
``subsample_scaffolds`` on the same featurization order as ``mtl_harness.load_endpoint``.

Output: artifacts/scaffold_overlap.json

Usage (from the repository root):
    PYTHONPATH=.:src python preprint/dl_forward/scaffold_overlap.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from rdkit import Chem

import mtl_harness as H
from mtl_config import TrainConfig

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"


def endpoint_structures(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (scaffold ids, scaffold SMILES, canonical SMILES) in load_endpoint order."""
    adapter = H.ADMETAdapter(raw_dir=str(H.ROOT / "data" / "raw" / "admet"))
    (H.ROOT / "data" / "raw" / "drugcomb").mkdir(parents=True, exist_ok=True)
    featurizer = H.DrugCombAdapter(raw_dir=str(H.ROOT / "data" / "raw" / "drugcomb"))
    df = adapter.build(key, split="all")
    ids, scafs, canon = [], [], []
    index: dict[str, int] = {}
    for smi in df["Drug"]:
        # Same inclusion rule as load_endpoint: graph, descriptors and scaffold must all exist.
        g = featurizer.smiles_to_graph(str(smi))
        dv = H._descriptor_vector(str(smi))
        s = H._murcko(smi)
        if g is None or g.x.shape[0] == 0 or dv is None or s is None:
            continue
        ids.append(index.setdefault(s, len(index)))
        scafs.append(s)
        canon.append(Chem.MolToSmiles(Chem.MolFromSmiles(str(smi))))
    return np.asarray(ids), np.asarray(scafs), np.asarray(canon)


def main() -> None:
    cfg = TrainConfig()
    keys = [e.key for e in H.ENDPOINTS]
    data = {k: endpoint_structures(k) for k in keys}
    for k in keys:
        H.log(f"[{k}] {len(data[k][0])} molecules, {len(set(data[k][0]))} scaffolds")

    results: dict[str, dict[str, dict[str, float]]] = {k: {} for k in keys}
    for frac in cfg.train_fractions:
        per = {k: {"scaffold": [], "molecule": []} for k in keys}
        for seed in cfg.seeds:
            splits, train = {}, {}
            for k in keys:
                ids = data[k][0]
                splits[k] = H.scaffold_three_way(ids, seed, cfg.test_scaffold_frac,
                                                 cfg.val_scaffold_frac)
                train[k] = H.subsample_scaffolds(ids, splits[k][0], frac, seed)
            for t in keys:
                other_scaf, other_mol = set(), set()
                for u in keys:
                    if u != t:
                        other_scaf |= set(data[u][1][train[u]].tolist())
                        other_mol |= set(data[u][2][train[u]].tolist())
                test = splits[t][2]
                per[t]["scaffold"].append(float(np.mean([s in other_scaf for s in data[t][1][test]])))
                per[t]["molecule"].append(float(np.mean([m in other_mol for m in data[t][2][test]])))
        for t in keys:
            results[t][f"{frac:.2f}"] = {
                "test_scaffold_seen_mean": round(float(np.mean(per[t]["scaffold"])), 3),
                "test_molecule_seen_mean": round(float(np.mean(per[t]["molecule"])), 3),
            }
            H.log(f"[{t}] f={frac:.2f} scaffold-seen {results[t][f'{frac:.2f}']['test_scaffold_seen_mean']} "
                  f"molecule-seen {results[t][f'{frac:.2f}']['test_molecule_seen_mean']}")

    payload = {
        "study": "cross-endpoint structural overlap under per-endpoint scaffold splits",
        "generated_at": datetime.now(UTC).isoformat(),
        "definition": ("fraction of a target endpoint's test molecules whose Murcko scaffold "
                       "(or canonical SMILES) appears in any other endpoint's training set at "
                       "the same seed and training fraction; mean over seeds"),
        "seeds": list(cfg.seeds),
        "n_molecules": {k: int(len(data[k][0])) for k in keys},
        "results": results,
    }
    (ART / "scaffold_overlap.json").write_text(json.dumps(payload, indent=2))
    H.log("wrote artifacts/scaffold_overlap.json")


if __name__ == "__main__":
    main()
