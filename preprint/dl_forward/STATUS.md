# STATUS / handoff — dl_forward track (AGENT_09)

Last updated by the Block B agent line. This captures exactly where the DL-forward
re-scope stands so the applicant can launch the GPU work with one command and review
autonomous progress.

## What is done and validated (no GPU used yet)

Non-destructive: nothing under `pilot_phd/preprint/` outside this folder was modified.
The prior feasibility manuscript, probes, figures, and artifacts are untouched.

- `README.md` — locked decisions + plan.
- `mtl_config.py` — endpoint typing (BBB/AMES/hERG/HIA = clf; Solubility/Caco2 = reg),
  pre-declared protocol (5 seeds, fractions {0.1,0.25,0.5,1.0}, 2000-bootstrap CIs).
- `mtl_harness.py` — reuses `DrugCombAdapter.smiles_to_graph`, `build_gnn_model`, and
  `src/safety/uncertainty.py`. Provides: featurization+cache (all six endpoints cached
  under `cache/`), Murcko scaffold three-way cold-splits, `MultiTaskGNN` (shared GIN
  trunk + per-endpoint mixed clf/reg heads), single-task GNN, RF descriptor baseline,
  metrics + bootstrap CIs.
- `train_multitask_gnn.py` — Pillar-1 driver (single-task + multi-task + RF across seeds
  and data fractions). Emits `artifacts/multitask_metrics.json`,
  `artifacts/learning_curves.npz`, and per-seed checkpoints.
- `make_ensemble_uncertainty.py` — Pillar-2 driver (K=5 deep ensemble + temp scaling vs
  RF / single GNN / MC-dropout; ECE/NLL/Brier, conformal set-size@90, AURC, SelAcc@70).
  Fixed-split / K-init design to avoid ensemble leakage.
- `Makefile` (repo root) — `make train-gnn` (Tier 2 GPU) and `make train-gnn-smoke`.

Validation performed (CPU only, no GPU gate crossed):
- Pillar-1 plumbing smoke: PASS.
- Pillar-1 full-size dry-run (all six endpoints, real data, 1 seed, 5 epochs):
  PASS — see `dryrun/`. Even under-trained it shows the predicted transfer signal
  (multi-task > single-task on the data-poor hERG/HIA/Caco2; RF still leads on raw
  accuracy, as intended).
- Pillar-2 metric smoke (K=3, tiny data): PASS — all four methods and every decision
  metric compute; RF hardened against single-class folds.

## The one gate: GPU training (HITL, per applicant request)

STATUS UPDATE (2026-09-09 ~04:50 UTC): BOTH GPU runs COMPLETE and all downstream
artifacts, figures, manuscript, and notebook are built. Summary of results:

- Pillar-1 (multi-task transfer, 3.06h, 20/20 units): data-poor endpoints gain in the
  low-data regime (Caco2 MAE -0.11..-0.17 all fractions; hERG AUROC +0.008..+0.027; HIA
  +0.058..+0.071 at 25-50%). Data-rich endpoints show mild negative transfer. RF stays
  accuracy-competitive throughout (no supremacy claim).
- Pillar-2 (deep ensemble K=5): ensemble is the best GRAPH uncertainty method (lowest
  NLL/Brier/AURC, sharpest 90% conformal sets, beats RF set-size on AMES/hERG/HIA). RF
  still leads raw ECE overall - reported in full, not overclaimed.
- Interpretability probe: NEGATIVE result - GAT attention does not recover the
  descriptor rule (node AUROC ~0.5), reported as a limitation (Jain & Wallace 2019).
- Learned-oracle campaign: 3/4 GO, 29.7% fewer measurements; BBB weak-shift and HIA
  undecided reported in full.

```bash
cd /home/snptx/snptx-core && source .venv/bin/activate
make train-gnn                 # Pillar 1: ~3-5h on the A10G, detached logging recommended
# then, consuming the checkpoints:
PYTHONPATH=src:. python pilot_phd/preprint/dl_forward/make_ensemble_uncertainty.py
```

Recommended detached invocation with a watchable log:
```bash
PYTHONPATH=src:. nohup make train-gnn > pilot_phd/preprint/dl_forward/artifacts/train.log 2>&1 & disown
```

## Preliminary dry-run numbers (5 epochs, 1 seed — NOT final, directional only)

| Endpoint | metric | RF | GNN-single | GNN-multi | multi vs single |
|---|---|---|---|---|---|
| hERG | AUROC | 0.871 | 0.645 | 0.776 | +0.131 |
| HIA  | AUROC | 0.911 | 0.777 | 0.887 | +0.110 |
| AMES | AUROC | 0.864 | 0.702 | 0.725 | +0.023 |
| BBB  | AUROC | 0.881 | 0.743 | 0.725 | -0.019 |
| Caco2| MAE   | 0.381 | 1.736 | 0.569 | better |
| Sol. | MAE   | 0.846 | 1.989 | 2.572 | worse |

Reading: multi-task transfer helps most on the data-poor endpoints (hERG 655, HIA 578,
Caco2 910), the pre-declared low-data-regime claim. RF remains the accuracy reference
(we do not claim GNN accuracy supremacy). Solubility (10k) needs the full 150-epoch
schedule; 5-epoch regression is undertrained. The real run produces the curves + CIs.

## Remaining sequence (after the GPU numbers land)

Build shells DONE in parallel with the run (2026-09-09), pending real numbers:
- `MANUSCRIPT_dl_forward_v2.md` (NEW) - DL-forward, two-pillar lead, `[[TOKEN]]`
  number placeholders filled only from the committed JSON.
- `figures.py` (NEW) - dark-theme transfer curves, ensemble calibration, conformal
  efficiency, from the artifacts.
- `build_notebook.py` -> `walkthrough_v2.ipynb` (NEW) - Tier-1 render + gated Tier-2
  RECOMPUTE cell; upgraded per-figure `show_fig(*names, width=...)`.
- `summarize_from_records.py` (NEW) - crash recovery / mid-run preview.

Still to do once artifacts land:
1. [DONE] Fill manuscript `[[TOKEN]]`s from all four artifact JSONs.
2. [DONE] Pillar-1 pretraining ablation (Hu et al. attribute-mask) - second pass.
   Script `pretrain_ablation.py`; ran 15 units (5 seeds x 3 low-data fractions) in ~2.3h.
   Result: classification deltas within seed noise (single +0.002 / multi +0.005 AUROC at
   f=0.1); only single-task regression exceeds noise (Solubility MAE -0.126 +/- 0.053).
   Reported as a supporting ablation (manuscript S4.8, fig_pretrain_ablation.png, notebook
   S4b). Headline stays supervised multi-task transfer.
3. [DONE] Pillar-2 full K=5 ensemble run (`make_ensemble_uncertainty.py`).
4. [DONE] `attention_attribution.py` (negative result, fully reported).
5. [DONE] Learned oracle in the campaign: `e8_campaign_learned.py` (non-destructive,
   reuses SPRT; 3/4 GO, 29.7% saved).
6. [DONE] `figures.py` (4 figs incl attention probe); [DONE] `walkthrough_v2.ipynb`
   executed (20 cells, 0 errors, 4 figs); [DONE] definitive Table 1 in the manuscript.
7. [DONE] Revised fig0 `architecture_figure_v2.py` (learned oracle primary, RF demoted,
   training sub-workflow, attention->interpretability branch) in the public repo.
8. [DONE] Mirrored scripts + committed artifacts + figures + notebook + manuscript to
   `/home/snptx/snptx-repro-discovery/preprint/dl_forward/`.

REMAINING / not done: optional wiring of the original
`e8_campaign.py` regression endpoints; nothing is committed yet in either repo - awaiting
applicant review.
