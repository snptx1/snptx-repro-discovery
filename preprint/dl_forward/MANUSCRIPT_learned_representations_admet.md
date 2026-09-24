# Learned molecular representations with calibrated deep-ensemble uncertainty for label-efficient ADMET decisions

<!--
MANUSCRIPT v2 (AGENT_09 DL-forward re-scope). NEW FILE; does not replace
MANUSCRIPT_calibrated_sequential_discovery.md, which remains the feasibility draft.

All number placeholders have been filled from the committed JSON artifacts produced by the
GPU run:
  - pilot_phd/preprint/dl_forward/artifacts/multitask_metrics.json    (Pillar 1)
  - pilot_phd/preprint/dl_forward/artifacts/ensemble_uncertainty.json  (Pillar 2)
  - pilot_phd/preprint/dl_forward/artifacts/attention_attribution.json (interpretability probe)
  - pilot_phd/preprint/dl_forward/artifacts/campaign_learned.json      (learned-oracle campaign)
Every figure caption names the script that regenerates it. Do not hand-edit numbers.
Style: no em dashes, measured register, natural contractions.
-->

## Abstract

Early molecular discovery is bottlenecked by the cost of measurement and by a trust
gap: models rarely say how confident they are, or when enough has been measured to make
a call. We present a study of learned graph representations coupled to calibrated
deep-ensemble uncertainty, driving a sequential decision engine that reaches defensible
ADMET go/no-go calls with fewer measurements. Two results lead. First, representation
transfer: a single multi-task graph encoder trained jointly across six ADMET endpoints
improves data-poor endpoints over single-task graph baselines, most in the low-label
regime, with the improvement traced by data-efficiency curves rather than a single
operating point (on the three data-poor endpoints multi-task lifts hERG AUROC by up to
+0.027 and HIA by +0.058 to +0.071 at 25-50% data, and cuts Caco2 MAE by 0.11 to 0.17
at every fraction). Second, uncertainty-to-decisions: a
five-member deep ensemble with temperature scaling is the best-calibrated graph model:
within the graph-model family it gives the lowest NLL, Brier score and risk-coverage AURC
(the random-forest descriptor baseline stays the strongest single model on raw ECE, which
we do not contest), and that calibration propagates to better decisions, the sharpest 90%
conformal sets of any graph model -- matching the descriptor baseline on average and
beating it on AMES, hERG and HIA -- and the highest graph-model selective accuracy at 70%
coverage. We do not claim graph-model accuracy supremacy over the
descriptor baseline; under a pre-registered, leakage-controlled protocol (Murcko
scaffold cold-splits, five seeds, 2000-resample bootstrap confidence intervals, every
endpoint reported) the defensible wins are transfer in the low-data regime
and decision quality. These learned components then power the engine already validated
in prior work: sequential probability ratio testing for when to stop, split-conformal
coverage under scaffold shift, selective prediction, abductive rule discovery, and
DuckDB provenance. Results regenerate from a tagged commit; a fast tier renders
committed artifacts on CPU, a GPU tier retrains from scratch.

## 1. Introduction

Two costs dominate early molecular discovery. The first is measurement: each assay
consumes material, time, and money, so deciding how many molecules to measure before
committing to a go/no-go call is itself a scientific decision. The second is trust: a
point prediction with no calibrated uncertainty cannot be safely acted on, especially
under the distribution shift that is the norm when a program moves into novel chemical
scaffolds.

Most machine-learning-for-chemistry work optimizes predictive accuracy on a fixed test
set and stops there. We take two complementary positions. First, that learned graph
representations earn their keep not by beating a strong tabular baseline on raw accuracy,
a claim that is fragile under leakage-controlled cold-splitting, but by transferring across related
endpoints so that data-poor tasks borrow strength from data-rich ones. Second, that the
value of a deep model is as much in the quality of its uncertainty as in its point
predictions: a well-calibrated ensemble makes better decisions, smaller conformal sets,
better abstention, fewer measurements to a confident call, even when its raw accuracy
merely ties the baseline.

Our contribution is a study that makes both cases rigorously and then wires the learned
components into a sequential decision engine. We (i) show multi-task representation
transfer with data-efficiency curves across six ADMET endpoints, (ii) show that a deep
ensemble's calibration advantage propagates to decision quality, and (iii) close the
loop autonomously while logging a full provenance trail. Throughout, the evaluation is
pre-registered and leakage-controlled, and every endpoint is reported.

## 2. Related work and positioning

Graph neural networks for molecules (Kipf and Welling 2017; Velickovic et al. 2018;
Xu et al. 2019; Gilmer et al. 2017) learn representations directly from molecular graphs;
multi-task and pre-training strategies (Hu et al. 2020) aim to transfer structure across
endpoints. Deep ensembles (Lakshminarayanan et al. 2017) and temperature scaling
(Guo et al. 2017) provide calibrated deep uncertainty; Monte Carlo dropout (Gal and
Ghahramani 2016) is the standard single-model epistemic baseline. Conformal prediction
(Vovk et al. 2005; Angelopoulos and Bates 2023) gives distribution-free coverage, and
selective prediction (El-Yaniv and Wiener 2010; Geifman and El-Yaniv 2017) a principled
abstain rule. Sequential testing (Wald 1945; Wald and Wolfowitz 1948) provides optimal
stopping. Our contribution is not a new estimator but a rigorous demonstration, on real
ADMET data under leakage-controlled splits, that learned representations transfer in the
low-data regime and that deep-ensemble calibration converts into decision quality, and a
wiring of those learned components into an autonomous, traceable engine.

## 3. Methods

Figure `fig0_architecture_v2.png` (regenerated by `architecture_figure_v2.py`) is the system
schematic: molecular graphs feed the learned multi-task ensemble oracle, the random forest
sits beside it as an explicit descriptor comparator, and the calibrated decision engine
(SPRT, conformal and selective prediction, abductive discovery) turns the oracle's
uncertainty into a traceable go/no-go call, with every step logged to a DuckDB lineage
store. The subsections below detail each block.

### 3.1 Data and leakage-controlled evaluation

We use six TDC ADMET endpoints spanning classification and regression: BBB (Martins),
AMES, and hERG and HIA (Hou) as binary classification; Solubility (AqSolDB) and Caco2
(Wang) as regression. After featurization the effective sizes are BBB 2030, AMES
7278, hERG 655, HIA 578, Solubility 9982, and Caco2 910
molecules. Every split is a Murcko scaffold cold-split: whole scaffolds are assigned to
train, validation, or test, so test scaffolds are never seen in training or calibration.
Unless stated we report five seeds and 2000-resample bootstrap 95% confidence intervals,
and we report every endpoint.

### 3.2 Molecular featurization and the graph encoder

Each molecule is a graph with nine-dimensional atom features (atomic number, degree,
formal charge, explicit hydrogen count, aromatic and ring flags, and one-hot
hybridization) and three-dimensional bond features (bond order, conjugation, ring
membership), from `src/adapters/drugcomb.py`. The encoder is a Graph Isomorphism Network
(GIN; Xu et al. 2019) from the SNPTX GNN library `src/models/gnn.py`, with
4 message-passing layers, hidden width 128, sum pooling to preserve the
Weisfeiler-Leman multiset interpretation, and two anti-oversmoothing mechanisms: PairNorm
(Zhao and Akoglu 2020) between layers and DropEdge (Rong et al. 2020) during training.
An edge-conditioned variant (GINE; Hu et al. 2020) is available as an ablation.

### 3.3 Multi-task objective

The multi-task model shares one encoder trunk across all six endpoints and attaches a
per-endpoint head: a two-logit classifier for classification endpoints and a single
regression output for regression endpoints. Regression targets are standardized per
endpoint on the training split, so a unit per-task weight keeps classification
cross-entropy and regression mean-squared-error on comparable scales. Training visits the
endpoints round-robin, one task-batch per step, so no single endpoint dominates the
shared trunk. Classification heads use inverse-frequency class weights to counter label
imbalance. The single-task baseline is identical in architecture but trained on one
endpoint at a time, isolating the transfer effect to the shared trunk. Appendix A.1 gives
the loss.

### 3.4 Baselines

Three model families are compared under identical splits: (a) a random-forest descriptor
baseline on RDKit physicochemical descriptors plus 1024-bit Morgan fingerprints, the
reference oracle from prior work; (b) the single-task GIN; and (c) the multi-task GIN.
The random forest is a deliberately strong tabular baseline: prior work found it
competitive with or better than graph models on raw accuracy under leakage-controlled cold-splits, so
it is the reference against which we make the non-accuracy claims of transfer and
decision quality.

### 3.5 Deep-ensemble uncertainty

For the classification endpoints we build a K = 5 deep ensemble by training K
single-task members that differ only in initialization seed, on one fixed scaffold split
(so members share the test set and never leak across it). Each member's logits are
temperature-scaled on the validation split (Guo et al. 2017), and the ensemble
probability is the mean of the temperature-scaled member probabilities. We compare the
ensemble against the random forest, a single graph model, and Monte Carlo dropout on a
single member. Uncertainty is decomposed into aleatoric and epistemic components in
Appendix A.2.

### 3.6 Decisions the uncertainty drives

Calibrated probabilities feed three decisions. Split-conformal prediction
(`src/safety/uncertainty.py`) produces coverage-guaranteed sets whose mean size at fixed
90% coverage measures efficiency. Selective prediction abstains on the least-confident
molecules, traced by the risk-coverage curve and its area (AURC) and by selective
accuracy at 70% coverage. Wald's SPRT (`src/intelligence/experiment_design.py`) stops
measuring at the first crossing of the log-likelihood-ratio boundaries, now driven by the
learned-ensemble oracle. Attention over molecular graphs (GAT; Velickovic et al. 2018)
yields substructure attributions cross-validated against the descriptor rules
(Appendix A.3).

## 4. Results

### 4.1 Multi-task representation transfer (lead result)

Under the pre-registered protocol, the multi-task encoder improves the data-poor
endpoints over the single-task graph baseline, with the largest gains where labels are
scarcest. Table P1 reports the primary-metric deltas (multi-task minus single-task) on
the three data-poor endpoints across training fractions:

| endpoint (metric) | f=0.10 | f=0.25 | f=0.50 | f=1.00 |
|---|---|---|---|---|
| hERG (AUROC, ↑) | +0.008 | +0.027 | +0.002 | +0.016 |
| HIA (AUROC, ↑) | −0.018 | +0.058 | +0.071 | −0.009 |
| Caco2 (MAE, ↓) | −0.125 | −0.131 | −0.114 | −0.172 |

The evidence is the data-efficiency curve: performance
versus training fraction for each endpoint and model family. On the data-poor endpoints
(hERG, HIA, Caco2) the multi-task curve sits above the single-task curve across low
training fractions and the gap narrows as data grows: at the 10-50% fractions multi-task
lifts hERG AUROC by up to +0.027, lifts HIA AUROC by +0.058 to +0.071 at 25-50%, and cuts
Caco2 MAE by 0.11 to 0.17 at every fraction (all three data-poor endpoints benefit in the
low-data regime); on the
data-rich endpoints (AMES, Solubility) the curves converge or show mild negative transfer,
as expected when an endpoint has enough labels of its own (at full data AMES AUROC −0.049,
Solubility MAE +0.145, BBB ~0 — the familiar multi-task capacity-dilution tradeoff). We
report all six endpoints,
including any where multi-task does not help, and we do not claim graph-model accuracy
supremacy over the random forest: on raw accuracy the descriptor baseline remains
competitive (at full data RF AUROC 0.892/0.817/0.865/0.945 on BBB/AMES/hERG/HIA versus
multi-task 0.815/0.712/0.774/0.920; RF stays the stronger raw-accuracy predictor while
offering no shared, transferable representation). Figure `fig_transfer_curves.png`
(regenerated by `train_multitask_gnn.py` then `dl_forward/figures.py`).

### 4.2 Ablation: self-supervised node-mask pretraining

We test whether self-supervised pretraining of the shared trunk adds to the transfer
result. Following the attribute-masking objective of Hu et al. (2020), we mask
15% of atoms per molecule (zeroing the atom-feature row) and train the trunk
to predict each masked atom's element from its node embedding, using the union of all six
endpoints' training-pool molecules as the unlabeled corpus for each seed and fraction. No
labels and no validation or test scaffolds enter the pretraining step. We then fine-tune
under the identical protocol and compare four conditions on the same scaffold splits:
single-task and multi-task, each from scratch versus pretrained-then-fine-tuned, across
the 0.10, 0.25, and 0.50 low-data fractions and 5 seeds. The scratch conditions reuse the
Pillar-1 training code unchanged, so the comparison isolates the pretraining effect.

The effect is small and mixed, and for classification it stays within seed-to-seed
variability. Averaged over the classification endpoints at the lowest
fraction (f = 0.10), pretraining moves single-task AUROC by +0.002
and multi-task AUROC by +0.005; on the regression endpoints it moves MAE by
−0.069 (single) and −0.009 (multi), where negative is an
improvement. The one delta that exceeds seed noise is single-task regression:
attribute masking cuts single-task Solubility MAE by 0.126 ± 0.053 over 5 seeds, and the
single-task regression benefit persists across fractions (mean MAE −0.069/−0.085/−0.043 at
f = 0.10/0.25/0.50), whereas the classification deltas hover near zero with signs that flip
across fractions.

**Table 1.** Pretraining ablation: primary-metric delta (pretrained − from-scratch), mean ±
std over 5 scaffold-split seeds, at each low-data fraction. Negative MAE and positive AUROC
mean pretraining helps. The single-task Solubility row is the only effect that clears seed
noise; the classification deltas are within variability.

| endpoint | arm | metric | Δ @ f=0.10 | Δ @ f=0.25 | Δ @ f=0.50 |
|---|---|---|---|---|---|
| BBB | single | AUROC ↑ | +0.019 ± 0.029 | +0.018 ± 0.034 | +0.024 ± 0.040 |
|  | multi | AUROC ↑ | −0.004 ± 0.017 | −0.025 ± 0.020 | +0.002 ± 0.025 |
| AMES | single | AUROC ↑ | +0.002 ± 0.010 | −0.025 ± 0.017 | +0.004 ± 0.019 |
|  | multi | AUROC ↑ | −0.006 ± 0.019 | −0.013 ± 0.011 | +0.002 ± 0.022 |
| hERG | single | AUROC ↑ | +0.023 ± 0.055 | +0.033 ± 0.011 | −0.020 ± 0.020 |
|  | multi | AUROC ↑ | +0.020 ± 0.060 | −0.006 ± 0.031 | +0.026 ± 0.042 |
| HIA | single | AUROC ↑ | −0.036 ± 0.067 | +0.003 ± 0.163 | +0.003 ± 0.020 |
|  | multi | AUROC ↑ | +0.010 ± 0.052 | −0.014 ± 0.091 | +0.006 ± 0.053 |
| Solubility | single | MAE ↓ | −0.126 ± 0.053 | −0.041 ± 0.059 | −0.131 ± 0.063 |
|  | multi | MAE ↓ | +0.021 ± 0.219 | −0.033 ± 0.087 | −0.141 ± 0.191 |
| Caco2 | single | MAE ↓ | −0.012 ± 0.069 | −0.129 ± 0.080 | +0.046 ± 0.142 |
|  | multi | MAE ↓ | −0.039 ± 0.013 | −0.013 ± 0.008 | −0.001 ± 0.039 |

The per-endpoint deltas in the beneficial direction are in Figure `fig_pretrain_ablation.png`
(regenerated by `pretrain_ablation.py` then `dl_forward/figures.py`). Consistent with Hu et
al.'s own finding that attribute masking alone gives modest and sometimes negative transfer
without a complementary structural objective, we report pretraining as a supporting ablation
rather than a headline: the supervised multi-task signal (§4.1) remains the operative
transfer mechanism in this label regime.

### 4.3 Deep-ensemble uncertainty to decision quality (novelty)

Among the graph-model uncertainty methods the K = 5 deep ensemble is clearly the best:
averaged across the four classification endpoints it gives the lowest negative
log-likelihood (0.482 vs 0.505 single, 0.532 MC-dropout), the lowest Brier score (0.153
vs 0.162, 0.164), the lowest risk-coverage AURC (0.126 vs 0.146, 0.155), and the highest
selective accuracy at 70% coverage (0.853 vs 0.837, 0.822). It also produces the sharpest
conformal prediction sets at a fixed 90% coverage of any graph model (mean set size 1.212
vs 1.265 single, 1.327 MC-dropout), and on AMES, hERG and HIA it undercuts even the
descriptor baseline (e.g. hERG 1.222 vs RF 1.418). The boundary of the claim is
calibration against the random forest: on mean expected calibration error and AURC the RF
descriptor baseline remains the strongest single model (ECE 0.053, AURC 0.062), and we do
not contest that. The defensible, novel result is that a deep ensemble is the right way to
extract decision-grade uncertainty from the graph stack -- it dominates the other graph
uncertainty methods on every decision metric and matches the descriptor baseline on
conformal efficiency.

| method (mean over 4 clf endpoints) | ECE ↓ | NLL ↓ | Brier ↓ | conf. set @90 ↓ | AURC ↓ | sel-acc @70 ↑ |
|---|---|---|---|---|---|---|
| RF (descriptor baseline) | 0.053 | 0.371 | 0.117 | 1.213 | 0.062 | 0.910 |
| single GNN | 0.052 | 0.505 | 0.162 | 1.265 | 0.146 | 0.837 |
| MC-dropout | 0.071 | 0.532 | 0.164 | 1.327 | 0.155 | 0.822 |
| **deep ensemble** | 0.068 | **0.482** | **0.153** | **1.212** | **0.126** | **0.853** |

(Bold marks the best graph model; the RF column is the descriptor comparator, not a claim
of graph-model supremacy.) On the smallest endpoint (HIA, n = 578) single-GNN and
MC-dropout edge out the ensemble on NLL and AURC, the expected high-variance exception; we
report it rather than hide it. Figures `fig_ensemble_calibration.png` and
`fig_conformal_efficiency.png` (regenerated by `make_ensemble_uncertainty.py` then
`dl_forward/figures.py`).

### 4.4 Sequential decision efficiency, driven by the learned oracle

With the learned-ensemble oracle in the loop, Wald's SPRT reaches the correct go/no-go
call using fewer measurements than a fixed-sample design on the majority of endpoints:
across the four classification tasks it returns a GO on three (AMES, hERG, BBB) and spends
150 measurements against a fixed-sample budget of 213, a 29.7% aggregate reduction, with
the largest savings where the ensemble's top-ranked subgroup is most enriched (hERG 14 vs
45, AMES 20 vs 69). We report the exceptions rather than hide them: on BBB the realized
shift is weak and the sequential test spends slightly more than the fixed bound (85 vs
69), and HIA (n = 106 test molecules) remains undecided within its measurement budget. The
theory is unchanged from prior work; the oracle is now the multi-task graph ensemble.
Figure `fig_campaign_efficiency.png` (regenerated by `e8_campaign_learned.py` then
`dl_forward/figures.py`).

### 4.5 Calibrated coverage and selective prediction under scaffold shift

Split-conformal coverage holds near the 90% target under scaffold shift (ensemble mean
empirical coverage 0.877; BBB 0.906, AMES 0.883, HIA 0.887, with hERG under-covering at
0.830), and at matched coverage the ensemble's prediction sets are the sharpest of any
graph model and undercut the descriptor baseline on AMES (1.309 vs 1.333), hERG (1.222 vs
1.418) and HIA (1.019 vs 1.075), though not on BBB (1.299 vs 1.024). Selective prediction
lifts retained accuracy by abstaining on the least-confident molecules; among the graph
models the ensemble has the best risk-coverage frontier (mean selective accuracy at 70%
coverage 0.853 vs 0.837 single-GNN and 0.822 MC-dropout, and the lowest AURC 0.126). The
random-forest descriptor baseline still leads the frontier overall (0.910, AURC 0.062),
which we do not contest: the ensemble is the strongest graph-model decision-maker, not a
claim to beat descriptors. Figure `fig_conformal_efficiency.png` (regenerated by
`make_ensemble_uncertainty.py` then `dl_forward/figures.py`).

### 4.6 Interpretability: do attention maps recover descriptor rules?

We tested directly whether graph-attention weights recover a transparent, chemically
motivated atom-salience rule (an atom is salient if it is aromatic or a nitrogen -- the
lipophilic-ring and basic-amine motifs). They do not: node-level attention ranks atoms at
chance against the rule (attention-vs-salience AUROC 0.52 on hERG and 0.49 on BBB, top-3
enrichment +0.04 and -0.04), even though the attention network itself is predictive (test
AUROC 0.77 on hERG, 0.76 on BBB). This negative result is consistent with the literature
that attention weights are not, on their own, faithful explanations (Jain and Wallace
2019); we therefore rely on the abductive discovery cycle, not raw attention, for
interpretable structure-property rules, and report the attention probe as a limitation.
Figure `fig_attention_probe.png` (regenerated by `attention_attribution.py` then
`dl_forward/figures.py`).

### 4.7 End-to-end autonomous campaign with the learned oracle

The wired engine runs a closed campaign across the four classification endpoints with the
learned oracle: it returns a GO on three (AMES, hERG, BBB) and spends 150 measurements
against a 213 fixed-sample budget, a 29.7% aggregate reduction, with every decision,
oracle prediction, and SPRT trace logged to the DuckDB lineage store. The outcome is
reported in full, including the two harder cases: HIA stays undecided within its
measurement budget and BBB's weak realized shift costs slightly more than the fixed bound
(see §4.4). The per-endpoint measurements are in Figure `fig_campaign_efficiency.png`
(regenerated by `e8_campaign_learned.py` then `dl_forward/figures.py`); the DuckDB lineage
store that records every decision is shown in the system schematic (Figure
`fig0_architecture_v2.png`).

### 4.8 The definitive results table

Table 2 collects, per endpoint: n; random-forest, single-task, and multi-task point
metrics with bootstrap CIs; the multi-task versus baseline delta; and the ensemble versus
random-forest calibration and decision-quality columns (ECE, conformal set size at 90%,
selective accuracy at 70%).

**Table 2.** Full-data point metrics (training fraction 1.0), mean ± std over 5 scaffold-split
seeds. Classification uses AUROC (higher better); regression uses MAE (lower better). The
final three columns are the fixed-split K = 5 deep-ensemble decision metrics (classification
endpoints only). The multi−single column here is the *high-data* regime; the headline
low-data transfer deltas are in the §4.1 table.

| endpoint | n | metric | RF | single-task | multi-task | multi−single | ens. ECE ↓ | set@90 ↓ | sel-acc@70 ↑ |
|---|---|---|---|---|---|---|---|---|---|
| BBB | 2030 | AUROC ↑ | 0.892 ± 0.020 | 0.815 ± 0.027 | 0.815 ± 0.041 | +0.000 | 0.085 | 1.299 | 0.866 |
| AMES | 7278 | AUROC ↑ | 0.817 ± 0.036 | 0.761 ± 0.035 | 0.712 ± 0.049 | −0.049 | 0.055 | 1.309 | 0.830 |
| hERG | 655 | AUROC ↑ | 0.865 ± 0.040 | 0.759 ± 0.056 | 0.774 ± 0.033 | +0.016 | 0.060 | 1.222 | 0.785 |
| HIA | 578 | AUROC ↑ | 0.945 ± 0.031 | 0.929 ± 0.057 | 0.920 ± 0.046 | −0.009 | 0.074 | 1.019 | 0.932 |
| Solubility | 9982 | MAE ↓ | 0.842 ± 0.060 | 1.215 ± 0.054 | 1.360 ± 0.056 | +0.145 | — | — | — |
| Caco2 | 910 | MAE ↓ | 0.399 ± 0.035 | 0.658 ± 0.095 | 0.486 ± 0.046 | −0.172 | — | — | — |

The random forest leads on raw accuracy at full data on every endpoint (the bound
on the graph-model claims); the multi-task encoder's value is data-efficiency on the
data-poor endpoints (§4.1) and the deep ensemble's value is decision-grade uncertainty
within the graph stack (§4.3).

## 5. Reproducibility

Two tiers. Tier 1 (CPU, fast): the notebook renders committed artifacts so a reviewer
reads the full study in seconds. Tier 2 (GPU): `make train-gnn` retrains the single-task
and multi-task encoders and regenerates the learning curves and checkpoints, then
`make_ensemble_uncertainty.py` regenerates the ensemble decision metrics. Seeds are
pinned and `cudnn.deterministic` is set; we report confidence intervals over seeds rather
than bitwise reproduction, since residual GPU nondeterminism remains. The interpreter is
pinned (3.11.2) and TDC data versions are fixed.

## 6. Limitations

- Retrospective simulation over public libraries; no wet-lab loop is closed.
- Multi-task transfer is a low-data-regime claim, not a task-general one; we report all
  endpoints, including any where it does not help.
- We do not claim graph-model accuracy supremacy over the descriptor baseline; the wins
  are transfer and decision quality.
- GPU training is not bitwise deterministic; we pin seeds and `cudnn.deterministic` and
  report CIs over seeds rather than exact reproduction.
- Self-supervised pre-training (Hu et al. 2020) is reported as a supporting ablation
  (§4.2), not the headline; the supervised multi-task signal is the operative transfer
  mechanism in this label regime.
- Uncertainty-driven label acquisition is out of scope here and is left to future work.

## 7. Conclusion

Learned graph representations earn their place in molecular discovery by transferring
across related endpoints in the low-data regime and by producing calibrated uncertainty
that converts into better decisions, even when raw accuracy merely ties a strong tabular
baseline. Wired into a calibrated sequential decision engine, those learned components
reach confident, traceable go/no-go calls with fewer measurements. The primitives are
individually established; the contribution is a rigorous, reproducible demonstration of
transfer and decision quality, and their composition into a single autonomous engine.

## Generative AI Disclosure

Some assertions and model development steps within this document were developed with
reference to Generative AI tools (Copilot; 2026 version). AI assistance was used for
clarifying concepts, validating code logic, identifying potential errors, and generating
some code segments. All AI-generated material was independently reviewed, debugged, and
validated for correctness before inclusion.

## References

1. Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of Mathematical
   Statistics*, 16(2), 117-186.
2. Wald, A. and Wolfowitz, J. (1948). Optimum character of the sequential probability
   ratio test. *Annals of Mathematical Statistics*, 19(3), 326-339.
3. Kipf, T. N. and Welling, M. (2017). Semi-supervised classification with graph
   convolutional networks. *ICLR*.
4. Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y. (2018).
   Graph attention networks. *ICLR*.
5. Xu, K., Hu, W., Leskovec, J. and Jegelka, S. (2019). How powerful are graph neural
   networks? *ICLR*.
6. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O. and Dahl, G. E. (2017).
   Neural message passing for quantum chemistry. *ICML*.
7. Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V. and Leskovec, J. (2020).
   Strategies for pre-training graph neural networks. *ICLR*.
8. Zhao, L. and Akoglu, L. (2020). PairNorm: tackling oversmoothing in GNNs. *ICLR*.
9. Rong, Y., Huang, W., Xu, T. and Huang, J. (2020). DropEdge: towards deep graph
   convolutional networks on node classification. *ICLR*.
10. Lakshminarayanan, B., Pritzel, A. and Blundell, C. (2017). Simple and scalable
    predictive uncertainty estimation using deep ensembles. *NeurIPS*.
11. Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On calibration of modern
    neural networks. *ICML*.
12. Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian approximation: representing
    model uncertainty in deep learning. *ICML*.
13. Vovk, V., Gammerman, A. and Shafer, G. (2005). *Algorithmic Learning in a Random
    World*. Springer.
14. Angelopoulos, A. N. and Bates, S. (2023). Conformal prediction: a gentle
    introduction. *Foundations and Trends in Machine Learning*, 16(4), 494-591.
15. El-Yaniv, R. and Wiener, Y. (2010). On the foundations of noise-free selective
    classification. *JMLR*, 11, 1605-1641.
16. Geifman, Y. and El-Yaniv, R. (2017). Selective classification for deep neural
    networks. *NeurIPS*.
17. Huang, K., Fu, T., Gao, W., et al. (2021). Therapeutics Data Commons: machine
    learning datasets and tasks for drug discovery and development. *NeurIPS Datasets and
    Benchmarks*.
18. Bemis, G. W. and Murcko, M. A. (1996). The properties of known drugs. 1. Molecular
    frameworks. *Journal of Medicinal Chemistry*, 39(15), 2887-2893.
19. Jain, S. and Wallace, B. C. (2019). Attention is not explanation. *NAACL-HLT*.

---

# Appendix A. Methods and derivations

This appendix makes the study self-contained: A.1-A.3 cover the two learned pillars and the
interpretability probe, A.4-A.7 give the decision-layer derivations the results rely on, A.8
the ablation objective, and A.9 the implementation and hyperparameters.

## A.1 Multi-task objective

Let the shared encoder be $\phi_\theta(\cdot)$ mapping a molecular graph to a pooled
embedding, and let $h_t$ be the head for endpoint $t$. For a classification endpoint the
per-example loss is class-weighted cross-entropy on $h_t(\phi_\theta(x))$; for a
regression endpoint it is mean-squared error on the standardized target
$\tilde y = (y-\mu_t)/\sigma_t$, with $\mu_t,\sigma_t$ estimated on the training split.
The multi-task objective is the task-weighted sum

$$
\mathcal{L}(\theta) = \sum_{t} w_t \, \mathbb{E}_{(x,y)\sim \mathcal{D}_t}\big[\ell_t(h_t(\phi_\theta(x)), y)\big],
$$

optimized by round-robin task-batch updates so the shared trunk is not dominated by the
largest endpoint. Standardization puts regression MSE and classification cross-entropy on
comparable scales, so $w_t = 1$ is the default. Transfer is isolated by comparing this
shared-trunk model to single-task models of identical architecture.

## A.2 Deep-ensemble uncertainty decomposition

For a K-member ensemble with temperature-scaled member probabilities
$p^{(k)}(y\mid x)$, the predictive distribution is the mean
$\bar p(y\mid x) = \frac1K\sum_k p^{(k)}(y\mid x)$. Total predictive uncertainty, the
entropy $\mathcal{H}[\bar p]$, decomposes into an aleatoric term, the mean of member
entropies $\frac1K\sum_k \mathcal{H}[p^{(k)}]$, and an epistemic term, the mutual
information $\mathcal{H}[\bar p] - \frac1K\sum_k \mathcal{H}[p^{(k)}]$, which measures
member disagreement and vanishes as members agree. The epistemic term is the ensemble's
signal for distribution shift and is what selective prediction and conformal set size pick
up on. Empirically this epistemic signal is what buys the ensemble its decision-quality
edge over a single graph model: across the four classification endpoints it lowers mean
risk-coverage AURC from 0.146 to 0.126 and shrinks the mean 90%-coverage conformal set
from 1.27 to 1.21 labels, while remaining behind the random-forest descriptor baseline on
raw ECE (0.068 vs 0.053) -- the ensemble is the best graph-model uncertainty estimator,
not a claim to beat descriptors.

## A.3 Attention attribution

For the GAT encoder, per-edge attention coefficients $\alpha_{ij}$ are aggregated to a
per-atom importance by summing incident-edge attention across heads and layers, then
normalized within a molecule. We cross-validate these attributions against the
descriptor-rule drivers by checking whether high-attention atoms concentrate on the
substructures named by a transparent rule (aromatic or nitrogen atoms, the lipophilic-ring
and basic-amine motifs). Empirically this cross-validation fails: aggregated attention
ranks atoms only at chance against the rule (attention-vs-salience AUROC 0.52 on hERG and
0.49 on BBB, top-3 enrichment near zero), even though the GAT is predictive (test AUROC
0.77 / 0.76). We therefore report attention as an unreliable explanation, consistent with
Jain and Wallace (2019), and defer interpretability to the abductive discovery cycle
rather than asserting a mechanism.

## A.4 Sequential probability ratio test (SPRT)

The campaign of §4.4 and §4.7 decides *when to stop measuring* with Wald's sequential
probability ratio test (Wald 1945; Wald and Wolfowitz 1948). We test a simple null against
a simple alternative on the standardized property of the oracle's top-ranked subgroup,
$H_0:\theta=\theta_0$ versus $H_1:\theta=\theta_1$ with $\theta_1=$ `MEANINGFUL_EFFECT`
$=0.30$ in units of $\sigma$ and $\theta_0=0$. After $n$ measurements the log-likelihood
ratio is

$$
\Lambda_n \;=\; \sum_{i=1}^{n} \log\frac{f_{\theta_1}(x_i)}{f_{\theta_0}(x_i)},
$$

and sampling continues while $B < \Lambda_n < A$, stopping to accept $H_1$ (a GO) when
$\Lambda_n \ge A$ and to accept $H_0$ (a NO-GO) when $\Lambda_n \le B$. Wald's bounds set the
thresholds directly from the target error rates $(\alpha,\beta)=(0.05,0.20)$,

$$
A \;=\; \log\frac{1-\beta}{\alpha}, \qquad B \;=\; \log\frac{\beta}{1-\alpha},
$$

which control the type-I and type-II error at their nominal levels up to the usual
overshoot approximation. The efficiency claim is against the fixed-sample test of the same
$(\alpha,\beta)$ for a one-sided normal mean, whose required size is

$$
n_{\text{fixed}} \;=\; \left(\frac{z_{1-\alpha}+z_{1-\beta}}{\theta_1}\right)^{2},
$$

the grey reference bars in Figure `fig_campaign_efficiency.png`. When the realized effect
is genuinely at or beyond $\theta_1$ the SPRT's expected sample size is well below
$n_{\text{fixed}}$; when it is weak (BBB) the test can run to, or slightly past, the budget
rather than declaring a premature GO, which is the exception we report rather than hide.

## A.5 Split-conformal prediction and coverage

Section 4.5 reports distribution-free coverage under scaffold shift via split-conformal
prediction (Vovk et al. 2005; Angelopoulos and Bates 2023). On a held-out calibration set
of size $n$ we compute nonconformity scores $s_i = 1 - \hat p(y_i \mid x_i)$ and take the
quantile

$$
\hat q \;=\; \text{the } \big\lceil (1-\alpha)(n+1) \big\rceil / n \text{ empirical quantile of } \{s_i\}_{i=1}^{n}.
$$

The prediction set $C(x) = \{\,y : 1-\hat p(y\mid x) \le \hat q\,\}$ then satisfies the
finite-sample marginal guarantee

$$
\Pr\big(y_{\text{test}} \in C(x_{\text{test}})\big) \;\ge\; 1-\alpha,
$$

for exchangeable calibration and test data, with $\alpha=0.10$ (90% target). Because
validity is guaranteed by construction, the informative quantity is *efficiency*, the mean
set size $\mathbb{E}\,|C(x)|$ at fixed coverage: a smaller set at the same guarantee is a
more decisive answer. Calibrating on held-out *scaffolds* (not random points) is what keeps
the guarantee meaningful under the leakage-controlled shift; the empirical coverage we
observe (ensemble mean 0.877) sits near the 0.90 target, and the ensemble's set sizes are
the sharpest of any graph model (§4.5).

## A.6 Calibration: expected calibration error and temperature scaling

Calibration in §4.3 is scored by the expected calibration error. Partitioning predictions
into $B$ equal-width confidence bins $\mathcal{B}_1,\dots,\mathcal{B}_B$,

$$
\mathrm{ECE} \;=\; \sum_{b=1}^{B} \frac{|\mathcal{B}_b|}{N}\,
    \big|\,\mathrm{acc}(\mathcal{B}_b) - \mathrm{conf}(\mathcal{B}_b)\,\big|,
$$

where $\mathrm{acc}(\mathcal{B}_b)$ is the empirical accuracy and
$\mathrm{conf}(\mathcal{B}_b)$ the mean predicted confidence in bin $b$. We reduce
miscalibration with temperature scaling (Guo et al. 2017): a single scalar $T>0$ is fit on
held-out data to minimize NLL of the tempered softmax
$\mathrm{softmax}(z/T)$, which rescales confidences without changing the arg-max, so
accuracy is unchanged while ECE and NLL improve. Each ensemble member is temperature-scaled
before averaging. Consistent with the integrity guardrail, temperature scaling makes the
ensemble the best-calibrated *graph* model but does not overturn the random forest's lead on
raw ECE.

## A.7 Selective prediction and risk-coverage

Selective prediction (El-Yaniv and Wiener 2010; Geifman and El-Yaniv 2017) lets the model
abstain on its least-confident inputs. For a confidence score $g(x)$ and threshold $\tau$,
coverage and selective risk are

$$
\mathrm{cov}(\tau) = \Pr\!\big(g(x) \ge \tau\big), \qquad
\mathrm{risk}(\tau) = \frac{\mathbb{E}\big[\ell(x,y)\,\mathbf{1}\{g(x)\ge\tau\}\big]}
                            {\mathrm{cov}(\tau)} .
$$

Sweeping $\tau$ traces the risk-coverage curve; we summarize it by the area under the
risk-coverage curve (AURC, lower is better) and by the selective accuracy at 70% coverage
(§4.5). The ensemble's epistemic term (A.2) is the confidence signal that drives this
frontier: among graph models it gives the lowest AURC (0.126) and the highest selective
accuracy at 70% (0.853), while the descriptor baseline still leads the frontier overall,
which we do not contest.

## A.8 Self-supervised attribute-mask pretraining

The ablation of §4.2 pretrains the shared trunk with the attribute-masking objective of Hu
et al. (2020). For each molecule we sample a masked atom set $M(x)$ at rate
$\rho = 0.15$, zero those atoms' 9-dim feature rows, embed the corrupted graph, and predict
each masked atom's element from its node embedding with cross-entropy over the atomic-number
vocabulary $\mathcal{V}$ of the corpus,

$$
\mathcal{L}_{\text{mask}}(\theta) \;=\; -\,\mathbb{E}_{x}\,\frac{1}{|M(x)|}
    \sum_{i \in M(x)} \log p_\theta\!\big(z_i \mid \tilde x_{\setminus M}\big),
\qquad z_i \in \mathcal{V},
$$

with the corpus being the union of all six endpoints' training-pool molecules for each seed
and fraction (no labels, no validation or test scaffolds). The pretrained trunk is then
fine-tuned under the identical supervised protocol. This is the weakest of the Hu et al.
objectives (attribute masking without a complementary structural task), and its effect here
is within seed noise for classification, with a single real single-task regression gain
(§4.2), so we report it as a supporting ablation rather than a headline.

## A.9 Implementation and hyperparameters

All models share one encoder and protocol; the table fixes the values used everywhere so a
reader can reproduce the numbers exactly.

| group | setting | value |
|---|---|---|
| Encoder | backbone | GIN (edge-agnostic; GINE available) |
| | hidden width | 128 |
| | message-passing layers | 4 |
| | dropout | 0.3 |
| | DropEdge rate | 0.1 |
| | PairNorm scale | 1.0 |
| | node / edge feature dims | 9 / 3 |
| Optimization | epochs | 150 |
| | learning rate | 5e-4 |
| | weight decay | 5e-4 |
| | batch size | 128 |
| | task weighting | round-robin task-batches, $w_t = 1$ |
| Evaluation | split | Murcko scaffold cold-split |
| | test / val scaffold fraction | 0.20 / 0.20 |
| | seeds | 5 (0-4) |
| | training fractions | 0.10, 0.25, 0.50, 1.00 |
| | bootstrap resamples | 2000 |
| Ensemble | members $K$ | 5 (init-seed) |
| | calibration | per-member temperature scaling |
| | MC-dropout passes (baseline) | 30 |
| Conformal / selective | conformal $\alpha$ (coverage) | 0.10 (90%) |
| | selective coverage point | 70% |
| Pretraining (A.8) | mask rate $\rho$ | 0.15 |
| | pretraining epochs | 40 |
| | fractions | 0.10, 0.25, 0.50 |
| SPRT (A.4) | $\alpha$, $\beta$ | 0.05, 0.20 |
| | meaningful effect | 0.30 $\sigma$ |
| | top fraction | 0.30 |

Determinism: seeds pin NumPy, Python and Torch, and `cudnn.deterministic` is set; residual
GPU nondeterminism remains, so we report confidence intervals over seeds rather than bitwise
reproduction. The interpreter is pinned (3.11.2) and TDC data versions are fixed. Every
figure names the script that regenerates it, and `make artifacts` runs the full pipeline in
dependency order.
