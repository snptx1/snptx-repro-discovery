"""E7: does the abductive discovery cycle produce a real, checkable discovery?

Two outputs on real ADMET data, CPU-only:

  1. RULE REDISCOVERY. Feed RDKit descriptors + a property into the SNPTX abductive
     discovery cycle (`src.intelligence.scientific_discovery.run_discovery_cycle`)
     and check it recovers the established structure-property driver from data. Led
     by BBB (blood-brain-barrier penetration; the clean, non-circular demo where the
     driver is polar surface area / H-bond donors), plus solubility, HIA and Caco2.
     Lipophilicity is included only as a labelled semi-circular reference (its driver
     is calculated logP, which is itself a descriptor). Reports the symbolic tree
     rule, the distilled meta-model R2, and an abductive natural-language hypothesis
     for the top surprise.

  2. STRUCTURE-PROPERTY CLIFFS (the discovery surface). Full-dataset scan for pairs of
     structurally similar molecules (Morgan Tanimoto >= 0.70) with a large property
     gap. These interpretable "exceptions" pinpoint the single change that flips the
     property. Ranked by SALI = |dY| / (1 - similarity).

Verdict CONFIRM if the cycle recovers the known driver on a majority of endpoints
(with a positive meta-model R2) AND a non-trivial set of structure-property cliffs is
found. Writes e7_discovery_results.json.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from pivot_probes import DESC_FNS, load  # noqa: E402
from rdkit import DataStructs  # noqa: E402

from src.intelligence.scientific_discovery import (  # noqa: E402
    generate_abductive_hypotheses,
    run_discovery_cycle,
)

DESC_NAMES = [n for n, _ in DESC_FNS]
OUT_JSON = HERE / "e7_discovery_results.json"

# endpoint -> (known structure-property drivers, note)
REDISCOVERY = {
    "bbb": ({"TPSA", "HBD", "HBA", "MolWt"}, "lead: low TPSA/HBD penetrate the CNS"),
    "solubility": ({"MolLogP", "MolWt", "TPSA"}, "high logP -> low aqueous solubility"),
    "hia": ({"TPSA", "MolLogP", "HBD"}, "low TPSA -> good intestinal absorption"),
    "caco2": ({"TPSA", "MolWt", "HBD"}, "low TPSA -> high permeability"),
    "lipophilicity": ({"MolLogP"}, "semi-circular reference (driver is calc logP)"),
}
CLIFF_SCAN = {"lipophilicity": 2.0, "solubility": 2.0}  # endpoint -> |dY| threshold


def log(m):
    print(m, flush=True)


def zscore(X):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd


def rule_rediscovery(name, data, known_drivers, note):
    y = data["y"]
    Xz = zscore(data["X"])
    meta = pd.DataFrame(Xz, columns=DESC_NAMES)
    results = pd.DataFrame({"experiment_id": [f"mol_{i}" for i in range(len(y))],
                            "property_value": y})
    rep = run_discovery_cycle(results, meta, metric_col="property_value",
                              z_threshold=2.5, n_novel=5)

    corrs = {n: float(spearmanr(data["X"][:, i], y).correlation)
             for i, n in enumerate(DESC_NAMES)}
    top_corr = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    tree = rep.symbolic_rules[0]
    tree_feat = tree.expression.split()[1] if tree.expression.startswith("if ") else "(leaf)"

    hyp = ""
    if rep.surprises:
        s = rep.surprises[0]
        idx = int(s.experiment_id.split("_")[1])
        deltas = {DESC_NAMES[i]: float(Xz[idx, i]) for i in range(len(DESC_NAMES))}
        hyp = generate_abductive_hypotheses(s, deltas, top_k=3)[0]

    driver_hit = tree_feat in known_drivers or top_corr[0][0] in known_drivers
    log(f"\n### RULE REDISCOVERY: {name}  ({note})")
    log(f"    meta-model R2={rep.meta_model_r2:.3f}  tree rule: {tree.expression}")
    log(f"    tree split feature={tree_feat}  top corr={top_corr[0][0]}={top_corr[0][1]:+.2f}"
        f"  known={sorted(known_drivers)} -> {'HIT' if driver_hit else 'miss'}")
    if hyp:
        log(f"      abductive: {hyp}")
    return {"endpoint": name, "note": note, "meta_r2": round(rep.meta_model_r2, 3),
            "tree_feature": tree_feat, "tree_rule": tree.expression,
            "top_corr_feature": top_corr[0][0], "top_corr": round(top_corr[0][1], 3),
            "driver_hit": bool(driver_hit), "n_surprises": len(rep.surprises)}


def structure_property_cliffs(name, data, dy_thresh, sim_thresh=0.70):
    fps, y, smi = data["fps"], data["y"], data["smiles"]
    n = len(y)
    cliffs = []
    for i in range(n):
        rest = fps[i + 1:]
        if not rest:
            break
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], rest))
        # exclude sim>=0.999: identical ECFP for non-identical molecules (e.g.
        # different-size siloxane oligomers) are fingerprint collisions, not cliffs.
        for h in np.where((sims >= sim_thresh) & (sims < 0.999))[0]:
            j = i + 1 + int(h)
            dy = abs(float(y[i]) - float(y[j]))
            if dy >= dy_thresh:
                s = float(sims[h])
                cliffs.append((dy / (1.0 - s + 1e-6), s, dy, smi[i], smi[j],
                               float(y[i]), float(y[j])))
    cliffs.sort(reverse=True)
    log(f"\n### STRUCTURE-PROPERTY CLIFFS: {name} (sim>={sim_thresh}, |dY|>={dy_thresh}, n={n})")
    log(f"    cliffs found: {len(cliffs)}")
    for sali, s, dy, a, b, ya, yb in cliffs[:3]:
        log(f"    SALI={sali:5.1f} sim={s:.2f} dY={dy:.2f} | {a} ({ya:.2f}) vs {b} ({yb:.2f})")
    return {"endpoint": name, "n_molecules": n, "n_cliffs": len(cliffs),
            "top_sali": round(cliffs[0][0], 2) if cliffs else None,
            "examples": [{"sim": round(s, 3), "dY": round(dy, 3), "smiles_a": a,
                          "smiles_b": b, "y_a": round(ya, 3), "y_b": round(yb, 3)}
                         for _, s, dy, a, b, ya, yb in cliffs[:8]]}


def main():
    t0 = time.time()
    rediscovery, cliffs, cache = [], [], {}
    for ep, (drivers, note) in REDISCOVERY.items():
        log(f"loading {ep} ...")
        cache[ep] = load(ep)
        rediscovery.append(rule_rediscovery(ep, cache[ep], drivers, note))
    for ep, dy in CLIFF_SCAN.items():
        d = cache.get(ep) or load(ep)
        cliffs.append(structure_property_cliffs(ep, d, dy))

    n_hits = sum(r["driver_hit"] and r["meta_r2"] > 0 for r in rediscovery)
    n_ep = len(rediscovery)
    total_cliffs = sum(c["n_cliffs"] for c in cliffs)
    verdict = "CONFIRM" if (n_hits >= (n_ep + 1) // 2 and total_cliffs >= 20) else "SOFTEN"

    payload = {"probe": "E7_discovery", "generated_at": datetime.now(UTC).isoformat(),
               "rediscovery": rediscovery, "cliffs": cliffs,
               "driver_hits": f"{n_hits}/{n_ep}", "total_cliffs": total_cliffs,
               "verdict": verdict, "runtime_seconds": round(time.time() - t0, 1)}
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    log("\n" + "=" * 64)
    log("E7 DISCOVERY VERDICT")
    for r in rediscovery:
        log(f"  {r['endpoint']:14s} driver_hit={r['driver_hit']} "
            f"(feat={r['tree_feature']}, R2={r['meta_r2']}) {r['note']}")
    for c in cliffs:
        log(f"  cliffs[{c['endpoint']}] = {c['n_cliffs']} (top SALI {c['top_sali']})")
    log(f"  driver hits {n_hits}/{n_ep}; total cliffs {total_cliffs}")
    log(f"  VERDICT = {verdict}   ({payload['runtime_seconds']}s)  wrote {OUT_JSON.name}")
    log("=" * 64)


if __name__ == "__main__":
    main()
