# Submission Checklist (Preprint 2: dl_forward)

Fill the citable-repository slot after tagging, then act on the upload steps
in `SUBMISSION_README.md`. The applicant performs the submission; this
checklist only stages it.

## Venue

- **Chosen: bioRxiv**, category **Bioinformatics** (q-bio). No endorsement
  required. Same venue as Preprint 1 for consistency.
- Post to the primary venue first; the DOI/ID from that venue is what the CV
  and SOP references cite.

## Title

Learned molecular representations with calibrated deep-ensemble uncertainty
for label-efficient ADMET decisions

## Abstract

Early molecular discovery is bottlenecked by the cost of measurement and by a
trust gap: models rarely say how confident they are, or when enough has been
measured to make a call. We present a study of learned graph representations
coupled to calibrated deep-ensemble uncertainty, driving a sequential
decision engine that reaches defensible ADMET go/no-go calls with fewer
measurements. Two results lead. First, representation transfer: a single
multi-task graph encoder trained jointly across six ADMET endpoints improves
data-poor endpoints over single-task graph baselines, most in the low-label
regime, with the improvement traced by data-efficiency curves rather than a
single operating point (on the three data-poor endpoints multi-task lifts
hERG AUROC by up to +0.027 and HIA by +0.058 to +0.071 at 25-50% data, and
cuts Caco2 MAE by 0.11 to 0.17 at every fraction). Second,
uncertainty-to-decisions: a five-member deep ensemble with temperature
scaling is the best-calibrated graph model: within the graph-model family it
gives the lowest NLL, Brier score and risk-coverage AURC (the random-forest
descriptor baseline stays the strongest single model on raw ECE, which we do
not contest), and that calibration propagates to better decisions, the
sharpest 90% conformal sets of any graph model (matching the descriptor
baseline on average and beating it on AMES, hERG and HIA), and the highest
graph-model selective accuracy at 70% coverage. We do not claim graph-model
accuracy supremacy over the descriptor baseline; under a pre-registered,
leakage-controlled protocol (Murcko scaffold cold-splits, five seeds,
2000-resample bootstrap confidence intervals, every endpoint reported) the
defensible wins are transfer in the low-data regime and decision quality.
These learned components then power the engine already validated in prior
work: sequential probability ratio testing for when to stop, split-conformal
coverage under scaffold shift, selective prediction, abductive rule
discovery, and DuckDB provenance. Results regenerate from a tagged commit; a
fast tier renders committed artifacts on CPU, a GPU tier retrains from
scratch.

## Author and affiliation

- Author(s): Russell, D. R.
- Affiliation(s): Independent Researcher
- Corresponding email: drr508@g.harvard.edu

## Citable repository

- Repository: https://github.com/snptx1/snptx-repro-discovery (dl_forward
  subdirectory)
- Tag to cite: `dl-forward-preprint-v1`
- Commit to cite: `__________`  (record the exact SHA of the
  `dl-forward-preprint-v1` tag once cut)
- Every headline number regenerates from this tagged commit on CPU (fast
  tier renders committed artifacts; GPU tier retrains from scratch).

## Figure list (upload in this order)

0. `fig0_architecture_v2.png` — system schematic.
1. `fig_transfer_curves.png` — multi-task data-efficiency transfer curves.
2. `fig_pretrain_ablation.png` — self-supervised pretraining ablation.
3. `fig_ensemble_calibration.png` — deep-ensemble calibration.
4. `fig_conformal_efficiency.png` — conformal efficiency and selective
   prediction under scaffold shift.
5. `fig_attention_probe.png` — attention-attribution negative result.
6. `fig_campaign_efficiency.png` — end-to-end autonomous campaign with the
   learned oracle.

## Not a clinical claim

This work is a retrospective, computational methods study over public ADMET
libraries. No wet-lab loop is closed and no clinical or diagnostic claim is
made. Reported quantities are decision-engine operating characteristics
under leakage-controlled splits, not assertions about any specific
compound's safety or efficacy in humans.

## Generative AI disclosure

The manuscript includes a Generative AI Disclosure section. If the venue
asks for an AI-use statement separately, paste that section into the venue
field.

## Final step (applicant)

The actual upload to bioRxiv is performed by the applicant. This package
does not submit anything automatically.
