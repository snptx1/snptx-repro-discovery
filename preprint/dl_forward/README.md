# dl_forward - learned representations and calibrated ensemble uncertainty

Companion study to the top-level engine (`../MANUSCRIPT_calibrated_sequential_discovery.md`).
Where the top-level work uses a descriptor/random-forest oracle, this folder swaps in a
learned multi-task graph encoder and asks two questions:

1. **Representation transfer.** Does a single graph encoder trained jointly across six
   ADMET endpoints help the small ones? (It helps Caco2 clearly and HIA partly, and hurts
   the largest endpoints.)
2. **Uncertainty-to-decisions.** Does a deep ensemble of temperature-scaled single-task
   graph networks turn graph-model uncertainty into better go/no-go decisions? (It is the
   strongest graph-model method on proper scoring rules and conformal set size, though not
   on ECE; the random forest remains stronger overall.)

See `MANUSCRIPT_learned_representations_admet.md` for the full write-up and
`walkthrough_learned_representations_admet.ipynb` for a rendered, code-to-figure
walkthrough.

## Reproducibility tiers

- **Tier 1 (CPU, fast).** The notebook recomputes every table from the committed
  artifacts in `artifacts/` and displays the committed figures. `figures.py`,
  `architecture_figure_v2.py`, and `e8_campaign_learned.py` regenerate the figures and the
  campaign from the same artifacts on CPU.
- **Tier 2 (GPU, full retrain).** `train_multitask_gnn.py`,
  `make_ensemble_uncertainty.py`, `pretrain_ablation.py`, and `attention_attribution.py`
  retrain from scratch on the six TDC ADMET endpoints. Seeds are pinned and
  `cudnn.deterministic` is set; report confidence intervals over seeds rather than
  expecting bitwise reproduction.

## What each script produces

| Script | Produces |
|---|---|
| `train_multitask_gnn.py` | `artifacts/multitask_metrics.json`, per-seed checkpoints |
| `pretrain_ablation.py` | `artifacts/pretrain_ablation.json`, `fig_pretrain_ablation.png` |
| `make_ensemble_uncertainty.py` | `artifacts/ensemble_uncertainty.json`, `artifacts/ensemble_predictions.npz` |
| `attention_attribution.py` | `artifacts/attention_attribution.json`, `artifacts/attention_nodes.npz` |
| `e8_campaign_learned.py` | `artifacts/campaign_learned.json`, DuckDB lineage store (not committed) |
| `architecture_figure_v2.py` | `fig0_architecture_v2.png` |
| `scaffold_overlap.py` | `artifacts/scaffold_overlap.json` (cross-endpoint test-set exposure, Table A1) |
| `figures.py` | `fig_transfer_curves.png`, `fig_pretrain_ablation.png`, `fig_ensemble_calibration.png`, `fig_conformal_efficiency.png`, `fig_campaign_efficiency.png`, `fig_attention_probe.png` from committed artifacts |

## Engine modules reused (not duplicated)

- Molecular featurizer: `../../src/adapters/drugcomb.py::DrugCombAdapter.smiles_to_graph`
- ADMET data: `../../src/adapters/admet.py::ADMETAdapter` (TDC, fetched on first run)
- GNN encoders: `../../src/models/gnn.py::build_gnn_model` (GIN/GAT trunk)
- Uncertainty toolkit: `../../src/safety/uncertainty.py` (conformal, temperature
  scaling, ECE, MC-dropout)

## Integrity guardrail

Pre-specified evaluation: Murcko scaffold cold-splits within each endpoint, five seeds for
the transfer and pretraining studies (reported as paired mean ± sample s.d. with seed
counts), every endpoint reported. Because splits are drawn per endpoint, the multi-task trunk
sees other endpoints' molecules that overlap a target's test set (`scaffold_overlap.py`), so
the transfer gains may be optimistic. This work does not claim graph-model accuracy supremacy over the random-forest
descriptor baseline, which leads on every endpoint at full data. The supported claims are
endpoint-specific transfer to small endpoints (subject to that caveat) and graph-model decision quality (proper
scoring rules, conformal set size at reported coverage, risk-coverage).
