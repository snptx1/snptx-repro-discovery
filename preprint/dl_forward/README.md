# dl_forward — learned representations and calibrated ensemble uncertainty

Companion study to the top-level engine (`../MANUSCRIPT_calibrated_sequential_discovery.md`).
Where the top-level work uses a descriptor/random-forest oracle, this folder swaps in a
learned multi-task graph encoder and asks two questions:

1. **Representation transfer.** Does a single graph encoder trained jointly across six
   ADMET endpoints help the data-poor ones?
2. **Uncertainty-to-decisions.** Is a calibrated deep ensemble the best way to turn a
   graph model's uncertainty into a go/no-go decision?

See `MANUSCRIPT_learned_representations_admet.md` for the full write-up and
`walkthrough_learned_representations_admet.ipynb` for a rendered, code-to-figure
walkthrough.

## Reproducibility tiers

- **Tier 1 (CPU, fast).** The notebook renders the committed artifacts in
  `artifacts/` and regenerates every figure in `figures/` from them.
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
| `make_ensemble_uncertainty.py` | `artifacts/ensemble_uncertainty.json`, `fig_ensemble_calibration.png`, `fig_conformal_efficiency.png` |
| `attention_attribution.py` | `artifacts/attention_attribution.json`, `fig_attention_probe.png` |
| `e8_campaign_learned.py` | `artifacts/campaign_learned.json`, `fig_campaign_efficiency.png` |
| `architecture_figure_v2.py` | `fig0_architecture_v2.png` |
| `figures.py` | renders the figures above from committed artifacts |

## Engine modules reused (not duplicated)

- Molecular featurizer: `../../src/adapters/drugcomb.py::DrugCombAdapter.smiles_to_graph`
- ADMET data: `../../src/adapters/admet.py::ADMETAdapter` (TDC, fetched on first run)
- GNN encoders: `../../src/models/gnn.py::build_gnn_model` (GIN/GAT trunk)
- Uncertainty toolkit: `../../src/safety/uncertainty.py` (conformal, temperature
  scaling, ECE, MC-dropout)

## Integrity guardrail

Pre-registered evaluation: Murcko scaffold cold-splits, 5 seeds, 2000-resample
bootstrap confidence intervals, every endpoint reported, no cherry-picking. This work
does not claim graph-model accuracy supremacy over the random-forest descriptor
baseline; the defensible wins are low-data-regime transfer and decision quality
(calibration, conformal efficiency, selective prediction).
