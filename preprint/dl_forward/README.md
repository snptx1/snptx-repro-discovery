# dl_forward — DL-forward elevation of Block B (AGENT_09)

New, non-destructive work area for the re-scoped study
(`pilot_phd/agent_prompts/AGENT_09_dl_forward_multitask_uncertainty_study.md`).
It elevates the already-built SNPTX ML/DL stack into the star of the study, with two
positive headline pillars:

1. Multi-task representation transfer (lead result).
2. Uncertainty-to-decisions (novelty).

Nothing here overwrites the prior feasibility work. The existing CPU probes,
`MANUSCRIPT_calibrated_sequential_discovery.md`, figures fig1..fig10, and committed
artifacts stay intact as the Tier-1 fast-read track. New GPU artifacts, a new
manuscript file, and a new elevated notebook are added alongside them.

## Locked decisions (confirmed by applicant, 2026-09-08)

1. Endpoint typing (agent judgement): classification = BBB, AMES, hERG, HIA
   (AUROC/AUPRC); regression = Solubility (AqSolDB), Caco2 (Wang) (MAE/R2). Mixed
   multi-task head with per-endpoint loss weighting.
2. Ensemble + seeds: K=5 deep ensemble, 5 seeds (as spec).
3. Self-supervised pretraining (Hu et al. node/edge-mask): first pass without it,
   then add it as a Pillar-1 ablation.
4. GPU target: train on the local A10G (main remote session), wire a `make train-gnn`
   entry point (Tier 2). Confirmed device: NVIDIA A10G 23 GB, torch 2.5.1+cu121.
5. Old active-learning section: moved fully to future work (no longer a headline).

## Reproducibility tiers

- Tier 1 (CPU, fast): the notebook renders committed GPU artifacts for a fast reviewer read.
- Tier 2 (GPU): `make train-gnn` regenerates checkpoints/curves; seeds pinned,
  `cudnn.deterministic` set, residual-nondeterminism caveat stated (report CIs over
  seeds, not bitwise reproduction).

## Engine modules reused (not rebuilt)

- Molecular featurizer: `src/adapters/drugcomb.py::DrugCombAdapter.smiles_to_graph`
  (node feats 9-dim, edge feats 3-dim).
- ADMET data: `src/adapters/admet.py::ADMETAdapter` (TDC, already dvc-pulled under
  `data/raw/admet/`).
- GNN zoo + factory: `src/models/gnn.py::build_gnn_model` (GIN/GINE trunk,
  PairNorm + DropEdge, `extract_embeddings` gives the shared graph embedding).
- Uncertainty toolkit: `src/safety/uncertainty.py` (conformal, temp scaling, ECE,
  MC-dropout) — extended in Pillar 2.

## Build sequence (this track)

1. Pillar 1 harness + train script (this folder), single-task + multi-task, 5 seeds,
   learning-curve fractions. <- current
2. HITL checkpoint, then the full A10G run.
3. Pillar 2 ensemble uncertainty (extends g2 + uncertainty.py), consumes P1 checkpoints.
4. Attention attribution.
5. Learned-oracle campaign (patch e8), figures, definitive table.
6. New manuscript + elevated notebook, kept in lockstep and in sync with the public mirror.

## Integrity guardrail

Pre-declared evaluation: Murcko scaffold cold-splits, >= 5 seeds, 2000-resample
bootstrap CIs, every endpoint reported, no cherry-picking. We do NOT claim GNN accuracy
supremacy over the RF descriptor baseline. The defensible wins are low-data-regime
transfer and decision quality (calibration + measurements-to-decision).
