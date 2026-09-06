# snptx-repro-discovery

Public reproducibility artifact for the SNPTX decision engine
for molecular property discovery. Everything here runs **end to end on CPU**,
deterministic at `seed=20260905`, from a tagged commit.

The engine turns ADMET property discovery into three coupled decisions and wires them
into one autonomous loop:

1. **Measure less.** Wald's Sequential Probability Ratio Test (SPRT) reaches a
   confident go/no-go call with materially fewer measurements than a fixed-sample
   test — 65 vs 99 at the empirical effect size (34% fewer), 23% fewer on average, at
   power 0.85 and Type I 0.043.
2. **Know what you don't know.** Split-conformal prediction holds its 90% coverage
   target under leakage-controlled (Murcko scaffold) shift (BBB 0.884, hERG 0.869,
   AMES 0.899; ECE 0.044).
3. **Abstain wisely.** Selective prediction lifts retained accuracy from 0.864 to 0.94
   at 70% coverage, under both random and scaffold splits.

Wired together with a pluggable oracle, a novelty archive, an abductive discovery
cycle, and DuckDB provenance, the engine runs an **end-to-end autonomous campaign**
that reaches 6 go/no-go decisions using **140 vs 357** fixed-sample measurements
(**61% fewer**), recovers the established structure-property driver on **5 of 5** ADMET
endpoints, and surfaces **1398** interpretable structure-property cliffs. Every
decision is traced from task to discovered rule.

## To be fair

This is a retrospective, in-silico reproduction over public libraries. It closes no
wet-lab loop and makes no clinical-performance claim. Uncertainty-driven label
acquisition is reported as a regime-dependent effect (it helps only when the passive
baseline is unstable), not a task-general win.

## Why CPU-only when the project deployed GPU

Every result in this repository regenerates on **CPU**, yet the wider SNPTX project
ran on an NVIDIA A10G during development. Both statements are true, and the split is
deliberate:

- **The confirmed results here use CPU oracles.** The calibration/SPRT/selective-
  prediction spine (E1-E5), the E7 discovery cycle, and the E8 campaign use a
  bootstrap / random-forest oracle over RDKit physicochemical descriptors and Morgan
  fingerprints. These are fast on CPU, so the engine's decision
  logic (SPRT stopping, conformal coverage, selective prediction, abductive rules,
  DuckDB lineage) is fully reproducible without a GPU.
- **GPU was used for a line that did *not* survive scrutiny.** During
  development we trained deep-ensemble GIN graph-neural-network oracles on the GPU
  (~1 hr per run on one A10G) to test an uncertainty-driven *label-acquisition*
  headline. Under hardening (scaffold cold splits, 3 seeds, 2000x bootstrap CIs) that
  headline was **refuted** and demoted to a characterized regime law (E5, fig5). Only
  the pre-computed curves from that GPU work are committed here
  (`preprint/g1_harden_curves.npz`); fig5 just *plots* them, so no GPU is needed to
  reproduce the figure.
- **The GNN oracle remains a pluggable option, not a requirement.** The methods
  describe a GIN/GAT deep-learning oracle (`src/models/gnn.py`); `make install-gpu`
  adds the CUDA wheels for anyone who wants to retrain it. The headline numbers do
  not depend on it.

In short: GPU was for deep-learning acquisition experiments; the
*confirmed* engine and every number in the manuscript run on CPU.

## Quickstart

```bash
git clone https://github.com/snptx1/snptx-repro-discovery.git
cd snptx-repro-discovery
make install            # .venv + pinned CPU deps (~5 min, ~1.5 GB)
make test               # fast, no-download determinism regression (~10 s)
make repro-spine        # E1-E5 spine + E7 discovery + rule cards (CPU, ~5 min; first run downloads TDC)
make repro-campaign     # E8 end-to-end autonomous campaign + figures (CPU, ~5 min)
```

Figures land in `preprint/figures/`; the
campaign roll-up is `preprint/e8_campaign_results.json` and its provenance store is
`preprint/e8_campaign_lineage.duckdb` (regenerable, gitignored).

## What runs

| Target | Script | Produces |
|---|---|---|
| `repro-spine` | `preprint/feasibility_and_figures.py` | E1-E5: `fig1_sprt_efficiency` … `fig6_molecules`, `feasibility_summary.json` |
|  | `preprint/e7_discovery_probe.py` | 5/5 driver rediscovery + 1398 cliffs, `e7_discovery_results.json` |
|  | `preprint/e7_render.py` | `fig7_rule_cards`, `fig8_cliff_panel` |
| `repro-campaign` | `preprint/e8_campaign.py` | E8: `fig9_campaign_timeline`, `fig10_lineage_graph`, `e8_campaign_results.json`, DuckDB lineage |

## The engine (vendored under `src/`)

- `intelligence/experiment_design.py` — Wald SPRT sequential stopping
- `safety/uncertainty.py` — split-conformal, temperature scaling / ECE, MC-dropout
- `intelligence/scientific_discovery.py` — novelty archive + abductive discovery cycle
- `intelligence/catalog.py` — DuckDB experiment lineage
- `intelligence/surrogate.py` — GP surrogate + EI/UCB/KG/Thompson acquisition
- `models/gnn.py` — GIN/GAT graph encoders
- `adapters/admet.py`, `adapters/drugcomb.py` — TDC ADMET + molecular-graph featurizer
- `snptx/viz/theme.py` — the visualization theme

## Reproducibility invariants

- **SEED** deterministic at `seed=20260905` across Python, NumPy, and scikit-learn.
- **ENV** pinned via `requirements.txt`; Python `3.11.2` (see `PYTHON_VERSION.txt`).
- **DATA** public TDC ADMET benchmarks, fetched on first run.
- **LINEAGE** every campaign decision is logged to a DuckDB catalog with its oracle
  metrics, SPRT verdict, and discovered rule.

## License

MIT — see [LICENSE](LICENSE).
