# dl_forward - learned representations and ADMET decisions

Companion study to the top-level engine (`../MANUSCRIPT_calibrated_sequential_discovery.md`).
Where the top-level work uses a descriptor/random-forest oracle, this folder evaluates
single- and multi-task graph encoders and a single-task deep ensemble:

1. **Representation sharing.** Joint training has favorable observed differences on Caco2
   and some HIA settings, with mixed or negative differences elsewhere. Cross-endpoint test
   exposure and unequal training updates prevent attributing these differences to transfer
   alone.
2. **Uncertainty and decisions.** On one scaffold split, a five-member ensemble improves
   mean proper scoring rules and risk-coverage among the graph models. It has the smallest
   mean conformal sets at reported coverage, but not the lowest ECE. The random forest
   remains stronger overall. The campaign illustrates retrospective boundary crossings;
   it does not establish an improvement caused by uncertainty estimates.

See [the manuscript](MANUSCRIPT_learned_representations_admet.md) and
[the executed walkthrough](walkthrough_learned_representations_admet.ipynb). The notebook
recomputes tables from saved experimental results; it does not retrain the models by
default. Its source is `build_notebook.py`.

## Reproducibility tiers

- **Tier 1 (CPU, committed artifacts).** With the repository's Python environment installed,
  execute the notebook with `REGENERATE_CPU = False` and `RETRAIN_MODELS = False` to reconstruct the reported tables and display
  committed figures. No model fitting or raw-data download is required. `figures.py` and
  `architecture_figure_v2.py` redraw figures from the saved results; `e8_campaign_learned.py`
  replays the campaign from saved probabilities and labels and recreates its DuckDB store.
- **CPU data check.** `scaffold_overlap.py` reproduces cross-endpoint molecule and scaffold
  exposure from the endpoint data. It requires cached or downloaded TDC data and RDKit, but
  does not train a model. This is separate from recomputing tables from committed artifacts.
- **Tier 2 (model retraining, GPU recommended).** `train_multitask_gnn.py`,
  `pretrain_ablation.py`, and `make_ensemble_uncertainty.py` refit the models using TDC data.
  They use CUDA when available and otherwise fall back to CPU. Full runs are substantially
  more expensive than Tier 1. NumPy and PyTorch seeds are set for these studies; Python's
  `random` is not seeded, and bitwise reproduction across environments is not established.
  The original `attention_attribution.py` driver is retained for provenance, but its
  normalized incoming-attention sum is constant in exact arithmetic, so rerunning it cannot
  produce a valid atom-importance assessment.

Run these notebook build commands from the repository root:

```bash
python preprint/dl_forward/build_notebook.py
python -m jupyter nbconvert --to notebook --execute --inplace \
  preprint/dl_forward/walkthrough_learned_representations_admet.ipynb
```

These commands rebuild the walkthrough from its source and saved artifacts. Full training
uses the Tier 2 drivers separately. Consult the repository environment requirements and
the manuscript's reproducibility section before retraining.

## What each script produces

| Script | Computation | Produces |
|---|---|---|
| `build_notebook.py` | CPU, builds notebook source | `walkthrough_learned_representations_admet.ipynb` (execute separately) |
| `train_multitask_gnn.py` | Model training | `artifacts/multitask_metrics.json`, `artifacts/learning_curves.npz`, per-seed checkpoints |
| `pretrain_ablation.py` | Model training | `artifacts/pretrain_ablation.json` |
| `make_ensemble_uncertainty.py` | Model training and calibration | `artifacts/ensemble_uncertainty.json`, `artifacts/ensemble_predictions.npz` |
| `attention_attribution.py` | Historical model training and invalid attribution statistic | `artifacts/attention_attribution.json`, `artifacts/attention_nodes.npz` |
| `e8_campaign_learned.py` | CPU, saved predictions and labels | `artifacts/campaign_learned.json`, DuckDB lineage store (not committed) |
| `architecture_figure_v2.py` | CPU | `figures/fig0_architecture_v2.png` |
| `scaffold_overlap.py` | CPU, endpoint data | `artifacts/scaffold_overlap.json` (Table A1) |
| `figures.py` | CPU, saved artifacts | Figures 1-6 in `figures/`; Figure 6 diagnoses the invalid attention statistic |

## Engine modules reused (not duplicated)

- Molecular featurizer: `../../src/adapters/drugcomb.py::DrugCombAdapter.smiles_to_graph`
- ADMET data: `../../src/adapters/admet.py::ADMETAdapter` (TDC, fetched on first run)
- GNN encoders: `../../src/models/gnn.py::build_gnn_model` (GIN/GAT trunk)
- Uncertainty toolkit: `../../src/safety/uncertainty.py` (conformal, ECE, MC-dropout);
  per-member temperature fitting is in `make_ensemble_uncertainty.py`.

## Integrity guardrail

Pre-specified evaluation: Murcko scaffold cold-splits within each endpoint, five seeds for
the transfer and pretraining studies (reported as paired mean ± sample s.d. with seed
counts), every endpoint reported. Because splits are drawn per endpoint, the multi-task trunk
and pooled pretraining corpus can see other endpoints' training molecules that overlap a
target's test set (`scaffold_overlap.py`). Smaller tasks also receive repeated updates in
multi-task training. The observed differences are exploratory comparisons under this protocol.
The random-forest descriptor baseline leads on every endpoint at full data.

The ensemble reuses validation labels for fitting its member temperatures and calibrating
its prediction sets, without an established fixed-score rank argument. A single binary
network has an exception: scalar temperature scaling preserves the exact order-statistic
sets, even with a data-fitted temperature. The scaffold-group split does not establish
individual-score exchangeability for any method. Coverage is therefore reported empirically;
set-size comparisons must include it. The campaign ranks by positive-class probability and
uses future labels for standardization and direction selection, with nominal Gaussian error
parameters applied to finite binary data. A GO records an assay-enrichment boundary crossing;
for hERG and AMES, positives mean blockade and mutagenicity, respectively. The attention
artifacts document a constant-score implementation defect and support no interpretation of
learned atom importance. Corrected training, calibration, and attribution experiments remain
future work.
