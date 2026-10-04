# Learned molecular representations with calibrated deep-ensemble uncertainty for label-efficient ADMET decisions

Daniel R. Russell<br>
Autonomous Discovery Systems · Biomedical ML, SNPTX<br>
Correspondence: dan@snptx.ai

## Abstract

Early molecular discovery is limited by the cost of measurement and by a trust gap: models
rarely say how confident they are, or when enough has been measured to make a call. A
companion study built a calibrated sequential decision engine around a random-forest
descriptor oracle. Here we replace that oracle with learned graph models and ask two
questions under a pre-specified protocol (Murcko scaffold cold-splits within each endpoint,
every endpoint reported). **First, does a shared graph encoder transfer across ADMET
endpoints?** Across six TDC endpoints and five seeds, a multi-task Graph Isomorphism Network
helps the smallest endpoints and hurts the largest. Caco2 permeability MAE falls by 0.11 to
0.17 at every training fraction (18 of 20 seed-fraction pairs improve), HIA AUROC rises by
0.071 at half the training data (5 of 5 seeds), and hERG gains are directional but within
seed variability. On the large AMES and Solubility endpoints the shared trunk costs accuracy
at full data (AUROC −0.049, MAE +0.145; 0 of 5 seeds improve), and BBB shows negative
transfer at low data. Because splits are drawn per endpoint, the shared trunk also trains on
other endpoints' molecules, and at full data 10-23% of each endpoint's test molecules appear
verbatim in another endpoint's training set, so these gains may be optimistic and can't be
attributed to transfer alone.
**Second, does a deep ensemble turn graph-model uncertainty into
better decisions?** On a fixed scaffold split, a five-member ensemble of temperature-scaled
single-task networks gives the best proper scores of the graph models (mean NLL 0.482 versus
0.505 for a single network and 0.532 for Monte Carlo dropout), the lowest risk-coverage AURC,
and the smallest 90% conformal sets, though not the lowest expected calibration error. The
descriptor random forest remains the stronger model on accuracy, NLL, Brier score, AURC and
selective accuracy. In a retrospective run of the engine's sequential probability ratio
test, the ensemble returns GO on three of four classification endpoints; on the two endpoints
where a fixed-sample design at the same nominal error rates is feasible, the sequential test
uses 105
measurements against 137.4. In one unseeded run, graph attention shows near-chance alignment with a
simple atom-salience rule. Every number regenerates on CPU from
committed artifacts; a GPU tier retrains from scratch.

## 1. Introduction

Two costs dominate early molecular discovery. The first is measurement: each assay consumes
material, time, and money, so deciding how many molecules to measure before committing to a
go/no-go call is itself a scientific decision. The second is trust: a point prediction with
no calibrated uncertainty can't be safely acted on, especially under the distribution shift
that is the norm when a program moves into new chemical scaffolds.

A companion study (Russell 2026) addressed both costs with a calibrated sequential decision
engine: Wald's sequential probability ratio test decides when to stop measuring,
split-conformal and selective prediction quantify what the oracle doesn't know, and every
decision is logged to a provenance store. That engine used a random-forest (RF) oracle on
molecular descriptors, and its calibration-and-stopping layer was designed to accept any
oracle that emits class probabilities. This paper swaps in learned graph models and asks
what they add.

The question has a known complication. Under scaffold splits, descriptor and
fingerprint models are often competitive with, or better than, graph neural networks on raw
accuracy (Yang et al. 2019; Jiang et al. 2021). If learned representations earn their place,
it is more likely through properties a tabular model lacks: a representation that can be
shared across related endpoints, and uncertainty estimates that can be improved by
ensembling. We test both.

**Contributions.**

1. A multi-task transfer study across six ADMET endpoints with data-efficiency curves over
   four training fractions and five scaffold-split seeds, reported as paired multi-task minus
   single-task differences with seed-level dispersion. Transfer benefits concentrate on the
   smallest endpoints and reverse on the largest (§4.1).
2. A pretraining ablation testing whether self-supervised attribute masking (Hu et al. 2020)
   adds to supervised transfer. It doesn't, detectably (§4.2).
3. A comparison of graph-model uncertainty methods (single network, Monte Carlo dropout, deep
   ensemble) against the RF baseline on calibration, conformal efficiency, and selective
   prediction under scaffold shift (§4.3-§4.4).
4. The ensemble wired into the companion engine's sequential go/no-go campaign, with every
   decision logged (§4.5), and a direct test of whether graph attention recovers a simple
   descriptor rule (§4.6).
5. A measurement of how much of each endpoint's test set the multi-task trunk sees through
   the other endpoints' training data (Appendix A.10), which bounds how the transfer results
   should be read.

**Scope of the claims.** We don't claim that graph models are more accurate than the
descriptor baseline: the RF leads on every endpoint at full data. We don't claim that the
ensemble is the best-calibrated model by expected calibration error: it isn't. Scaffold
splits are made within each endpoint, so the multi-task gains are measured against a baseline
that sees less related chemistry. The uncertainty results come from one fixed split, the
campaign is a retrospective simulation that uses test-pool statistics, and the attention
probe is a negative result. Each of these boundaries is
stated where the evidence is presented and collected in §6.

## 2. Related work and positioning

Graph neural networks learn representations directly from molecular graphs (Gilmer et al.
2017; Kipf and Welling 2017; Velickovic et al. 2018; Xu et al. 2019). Benchmarks such as
MoleculeNet (Wu et al. 2018) and systematic comparisons (Yang et al. 2019; Jiang et al. 2021)
show that their accuracy advantage over fingerprint and descriptor models is inconsistent,
particularly on small datasets and scaffold splits, which is why a strong RF baseline is
retained here. Multi-task learning shares statistical strength across related tasks
(Caruana 1997); in drug discovery, massively multi-task networks improve with more tasks and
data, with gains that vary by task (Ramsundar et al. 2015). Self-supervised pretraining of
graph networks can help or hurt depending on the objective, and node-level objectives alone
can transfer negatively (Hu et al. 2020).

Deep ensembles (Lakshminarayanan et al. 2017) and temperature scaling (Guo et al. 2017) are
standard routes to calibrated deep uncertainty, and Monte Carlo dropout (Gal and Ghahramani
2016) is the usual single-model baseline. Whether ensemble members should be calibrated
before or after averaging matters: averaging individually calibrated members tends to make
the ensemble underconfident (Rahaman and Thiery 2021; Wu and Gales 2021). Split-conformal
prediction (Vovk et al. 2005; Angelopoulos and Bates 2023) gives distribution-free marginal
coverage under exchangeability, which covariate shift breaks (Tibshirani et al. 2019).
Selective prediction (El-Yaniv and Wiener 2010; Geifman and El-Yaniv 2017) gives a principled
abstention rule. The sequential probability ratio test (Wald 1945) minimizes expected sample
size among tests with the same error rates (Wald and Wolfowitz 1948). Whether attention
weights explain predictions is contested (Jain and Wallace 2019; Wiegreffe and Pinter 2019).

Our contribution isn't a new estimator. It is a scaffold-split measurement, on real
ADMET data, of where multi-task transfer helps and hurts, of how much a deep ensemble
improves graph-model uncertainty and its downstream decisions relative to both graph and
descriptor alternatives, and of what changes when that ensemble drives an existing
sequential decision engine.

## 3. Methods

Figure 0 places the components in one pipeline. The learned oracle replaces the companion
study's RF oracle; the RF is retained as a descriptor comparator; and the decision engine is
reused unchanged. The abductive rule-discovery component of the engine belongs to the
companion study and isn't re-run here.

<p align="center"><img src="figures/fig0_architecture_v2.png" alt="Figure 0" width="800"></p>

<sub><strong>Figure 0.</strong> System architecture. Molecular graphs from six TDC ADMET endpoints feed the learned oracle, a K = 5 deep ensemble of temperature-scaled single-task GIN networks; the RF descriptor model sits beside it as a comparator. The decision engine turns the oracle's probabilities into a go/no-go call through sequential stopping (SPRT) and conformal and selective prediction, and every decision is logged to a DuckDB lineage store. The bottom strip shows how the learned models are built: the GIN encoder is trained single-task and multi-task for the transfer study (§4.1), and single-task members form the ensemble (§4.3). The GAT attention probe (§4.6) is a separate network trained on the same graphs. Blue marks inputs and feedback, yellow learned models and calibration, green decisions and outputs, purple interpretability, and grey the descriptor baseline and lineage. The dashed active-learning loop is future work.</sub>

### 3.1 Data and scaffold splits

We use six Therapeutics Data Commons ADMET endpoints (Huang et al. 2021): BBB (Martins),
AMES, hERG, and HIA (Hou) as binary classification, and Solubility (AqSolDB) and Caco2 (Wang)
as regression, with 2030, 7278, 655, 578, 9982, and 910 molecules respectively. Every split is
a Murcko scaffold cold-split (Bemis and Murcko 1996): unique scaffolds are randomly permuted
and assigned whole to test (20% of scaffolds), validation (20%), and training, so within an
endpoint no test or validation scaffold is seen in training. Splits are drawn independently
for each endpoint. A model that trains on several endpoints, such as the multi-task trunk or
the pretraining corpus, therefore also sees other endpoints' training molecules, some of which
share a scaffold with, or are identical to, the target endpoint's test molecules; Appendix
A.10 measures this. Training fractions of 0.10, 0.25, 0.50, and 1.00 are
formed by subsampling whole training scaffolds. The transfer and pretraining studies repeat
this over five seeds (0-4), each seed drawing a new split and initialization. The
uncertainty study and the campaign use one fixed split (seed 0). Apart from checkpoint
selection in the attention probe (§3.8), the validation set is used only for temperature
scaling and conformal calibration; models train for a fixed 150 epochs with no early stopping.

### 3.2 Featurization and graph encoder

Each molecule is a graph with nine atom features (atomic number as a scalar, degree, formal
charge, explicit hydrogen count, aromatic and ring flags, and a one-hot sp/sp2/sp3
hybridization), from `src/adapters/drugcomb.py`. The featurizer also computes three bond
attributes (bond order, conjugation, ring membership), but the GIN used here aggregates over
bond connectivity only and ignores them; only the edge-conditioned GINE variant, not used in
this study, would consume them. The encoder is a Graph Isomorphism Network (GIN; Xu et al. 2019)
from `src/models/gnn.py`, with four message-passing layers, hidden width 128, two-layer MLPs
with batch normalization inside each GIN update, sum pooling, PairNorm (Zhao and Akoglu 2020)
between layers, and DropEdge (Rong et al. 2020) at rate 0.1 during training.

### 3.3 Multi-task objective and training

The multi-task model shares one encoder trunk across all six endpoints and attaches a
per-endpoint head: two logits for classification and one output for regression. Regression
targets are standardized per endpoint on the training split so that cross-entropy and
mean-squared error sit on comparable scales with unit task weights. Training visits the
endpoints round-robin, one task batch per step, so no single endpoint dominates the trunk.
Classification losses use inverse-frequency class weights. The single-task baseline has the
same architecture trained on one endpoint, which isolates the effect of sharing the trunk.
All graph models use Adam (learning rate 5e-4, weight decay 5e-4, batch size 128). Appendix
A.1 gives the objective.

### 3.4 Descriptor baseline

The RF baseline uses ten RDKit physicochemical descriptors (molecular weight, cLogP, TPSA,
hydrogen-bond donors and acceptors, rotatable bonds, aromatic rings, fraction sp3, heavy
atoms, ring count) concatenated with a 1024-bit Morgan fingerprint of radius 2 (Rogers and
Hahn 2010). It has 300 trees and balanced class weights for classification. It is the oracle
of the companion study and a deliberately strong comparator.

### 3.5 Deep ensemble and uncertainty baselines

For the four classification endpoints we train K = 5 single-task GIN members on the fixed
split with different training seeds, which change the initialization, minibatch order, and
dropout and DropEdge masks; all members share one test set. Each
member's logits are temperature-scaled on the validation set by minimizing NLL (Guo et al.
2017), and the ensemble probability is the mean of the scaled member probabilities. The
single-network baseline is member 0 with its own temperature. The Monte Carlo dropout
baseline averages 30 stochastic passes of member 0 in training mode; in this implementation
training mode also activates DropEdge and batch-statistics normalization, and the dropout
probabilities aren't temperature-scaled (§6). The RF probabilities are used as produced.
Appendix A.2 gives the ensemble's uncertainty decomposition.

### 3.6 Decision metrics

Calibration is scored by top-label expected calibration error (ECE) over ten equal-width
bins, negative log-likelihood (NLL), and the Brier score. Split-conformal prediction
(`src/safety/uncertainty.py`) uses the score $1-\hat p(y\mid x)$, calibrated on the validation
set at a nominal 90% target; we report mean set size and realized test coverage. Selective
prediction ranks test molecules by maximum class probability; we report the area under the
risk-coverage curve (AURC) and accuracy on the most confident 70%. Appendices A.5-A.7 give
the definitions.

### 3.7 Sequential go/no-go campaign

The campaign reuses the companion engine's decision logic with the ensemble as oracle
(`e8_campaign_learned.py`). For each classification endpoint, test molecules are ranked by
ensemble probability of the positive class, and the top 30% form the candidate subgroup.
Labels are standardized with the mean and standard deviation of the full test pool, and the
subgroup's standardized labels are revealed one at a time in a random order. Wald's SPRT tests
$H_0:\theta=0$ against $H_1:\theta=0.30$ (in pool standard deviations) at
$(\alpha,\beta)=(0.05,0.20)$, with the test direction set by the sign of the whole subgroup's
mean. Both the standardization and the direction use labels that a real campaign would not
yet have measured, so the procedure is a retrospective simulation rather than an
implementable sequential design (§6). The comparator is the fixed-sample one-sided test at
the same nominal error rates under the Gaussian working model, which needs 68.7 measurements.
When the subgroup is smaller than that,
we cap the comparator at the subgroup size; a capped design has less than the nominal power,
so it isn't a same-error-rate comparator. Every decision, its parameters, and the
oracle's summary prediction are written to a DuckDB lineage store. Appendix A.4 gives the
derivation.

### 3.8 Attention probe

To test whether attention offers interpretability, we train a separate three-layer graph
attention network (GAT; Velickovic et al. 2018; hidden width 64, four heads in the first two
layers and one in the last) on hERG and BBB, selecting the checkpoint by validation AUROC.
Each atom's score is the sum of its incoming attention coefficients in the single-head final
layer. We compare these scores with a transparent rule (an atom is salient if it is aromatic
or a nitrogen) by the AUROC of attention against the rule, pooled over atoms, and by the
enrichment of salient atoms among each molecule's three most-attended atoms. Molecules with
fewer than four atoms, or whose atoms are all salient or all non-salient, are excluded, which
leaves 148 of 153 hERG and 370 of 458 BBB test molecules. The probe's training isn't seeded.
Appendix A.3 gives details.

### 3.9 Statistical reporting

For the transfer and pretraining studies we report paired differences between conditions
that share a split, as mean ± sample standard deviation over five seeds (divisor n − 1, as
for every ± in this paper), together with the number of seeds in which the difference favors
one condition. Per-run 95% bootstrap intervals over test molecules (2000 resamples) are
stored in the artifacts; the tables report variation across seeds, which also captures
variation in the split. With five seeds, single cells are
underpowered, so we read consistency of direction across seeds and fractions alongside the
means. For the 36-cell pretraining table we also apply a Holm correction (Holm 1979) to
paired t-tests. The uncertainty study and the campaign use one split, so their differences
carry no seed-level error bars; we treat small gaps there as unresolved.

## 4. Results

### 4.1 Multi-task transfer concentrates on the smallest endpoints

Table 1 gives the paired multi-task minus single-task difference in each endpoint's primary
metric, ordered by dataset size. The pattern follows endpoint size more than training
fraction. Caco2, the third-smallest endpoint, benefits at every fraction (18 of 20
seed-fraction pairs improve), with the largest gain at full data. HIA benefits consistently
only at half the data (+0.071 ± 0.032, 5 of 5 seeds) and is slightly worse at 10% and at full
data. hERG differences are positive on average at every fraction but small relative to their
seed spread (13 of 20 pairs improve). On the two largest endpoints the shared trunk costs
accuracy at full data (AMES −0.049 ± 0.017, Solubility MAE +0.145 ± 0.112, 0 of 5 seeds
improve in either), consistent with negative transfer when an endpoint with ample labels
shares capacity with smaller, possibly conflicting tasks (Caruana 1997). BBB shows negative
transfer at low data (−0.030 ± 0.024 at 10%, 0 of 5 seeds) that fades by full data.

**Table 1.** Multi-task minus single-task GIN, paired by seed, at each training fraction:
mean ± sample s.d. over five scaffold-split seeds, with the number of seeds in which multi-task is
better in parentheses. Endpoints are ordered by dataset size. Positive AUROC and negative MAE
differences favor multi-task.

| endpoint (metric) | n | f = 0.10 | f = 0.25 | f = 0.50 | f = 1.00 |
|---|---|---|---|---|---|
| HIA (AUROC ↑) | 578 | −0.018 ± 0.080 (2/5) | +0.058 ± 0.119 (3/5) | +0.071 ± 0.032 (5/5) | −0.009 ± 0.019 (1/5) |
| hERG (AUROC ↑) | 655 | +0.008 ± 0.035 (3/5) | +0.027 ± 0.069 (4/5) | +0.002 ± 0.020 (3/5) | +0.016 ± 0.029 (3/5) |
| Caco2 (MAE ↓) | 910 | −0.125 ± 0.171 (4/5) | −0.131 ± 0.116 (4/5) | −0.114 ± 0.070 (5/5) | −0.172 ± 0.095 (5/5) |
| BBB (AUROC ↑) | 2030 | −0.030 ± 0.024 (0/5) | −0.026 ± 0.037 (1/5) | −0.010 ± 0.011 (0/5) | 0.000 ± 0.031 (2/5) |
| AMES (AUROC ↑) | 7278 | +0.003 ± 0.025 (3/5) | −0.023 ± 0.022 (1/5) | −0.012 ± 0.037 (1/5) | −0.049 ± 0.017 (0/5) |
| Solubility (MAE ↓) | 9982 | 0.000 ± 0.357 (4/5) | +0.110 ± 0.173 (1/5) | +0.103 ± 0.254 (3/5) | +0.145 ± 0.112 (0/5) |

Figure 1 shows the underlying data-efficiency curves. Neither graph model approaches the
RF on raw accuracy: at full data the RF reaches AUROC 0.892, 0.817, 0.865, and 0.945 on BBB,
AMES, hERG, and HIA (multi-task 0.815, 0.712, 0.774, 0.920) and MAE 0.842 and 0.399 on
Solubility and Caco2 (multi-task 1.360, 0.486). The RF leads at every fraction on every
endpoint except HIA at f = 0.50, where multi-task (0.904) and RF (0.900) are level within
seed variability. What the shared trunk offers is a transferable representation that helps
some small endpoints, not a more accurate predictor.

These gains need one more qualification. Because splits are drawn per endpoint, the
multi-task trunk trains on other endpoints' molecules that the single-task baseline never
sees, and some of them are the target endpoint's test molecules or share their scaffolds
(Appendix A.10). The exposure grows with the training fraction: at f = 0.10, 1-4% of each
endpoint's test molecules appear verbatim in another endpoint's training set (4-14% share a
scaffold), rising to 10-23% (38-65%) at full data. The trunk never sees the target endpoint's
labels for those molecules, but it does learn their structures, so the multi-task gains in
Table 1 can't be attributed to transfer alone and may be optimistic. A globally
scaffold-disjoint protocol could show smaller gains, or, if the overlapping molecules carry
conflicting auxiliary signal, larger ones; only a rerun can tell. The
exposure can't explain everything: Caco2 already improves at f = 0.10 (4 of 5 seeds), where
only 2% of its test molecules are exposed. But its largest gain, at full data, coincides with
its largest exposure (19%), and the two can't be separated with the present design.

<p align="center"><img src="figures/fig_transfer_curves.png" alt="Figure 1" width="667"></p>

<sub><strong>Figure 1.</strong> Data-efficiency curves for the RF descriptor baseline (grey), single-task GIN (yellow), and multi-task GIN (cyan) on six ADMET endpoints; AUROC for classification (higher is better), MAE for regression (lower is better). Points are means over five scaffold-split seeds and bands are ±1 sample s.d. Multi-task improves on single-task for Caco2 at every fraction and for HIA at f = 0.50; it is worse than single-task on AMES and Solubility at full data and on BBB at low data. Table 1 gives the paired differences.</sub>

### 4.2 Ablation: self-supervised attribute-mask pretraining

We tested whether self-supervised pretraining of the shared trunk adds to supervised
transfer. Following the attribute-masking objective of Hu et al. (2020), we mask 15% of atoms
per molecule by zeroing their feature rows and train the trunk for 40 epochs to predict each
masked atom's element, using the union of all six endpoints' training molecules for each
seed and fraction as the unlabeled corpus. The corpus contains no labels and none of an
endpoint's own validation or test scaffolds, although, as in §4.1, other endpoints' training
molecules can overlap a target's test set (Appendix A.10). We then fine-tune under the identical protocol and compare pretrained against from-scratch
training for both single-task and multi-task models at the three low-data fractions, five
seeds each (Table 2).

Pretraining has no detectable effect. Of the 36 paired comparisons, seven reach p < 0.05 on
an uncorrected paired t-test, with mixed signs, and none survives Holm correction. The most
consistent direction is single-task Solubility, where pretraining lowers MAE at all three
fractions (−0.126 ± 0.060 at f = 0.10 and −0.131 ± 0.071 at f = 0.50). Averaged over the
classification endpoints at f = 0.10, pretraining changes AUROC by +0.002 (single-task) and
+0.005 (multi-task). This matches Hu et al.'s finding that pretraining at the level of
individual nodes alone gives limited improvement and can transfer negatively, so we treat supervised multi-task training (§4.1) as
the operative sharing mechanism in this label regime.

**Table 2.** Pretraining ablation: primary-metric difference (pretrained minus from-scratch),
mean ± s.d. over five scaffold-split seeds at each low-data fraction. Positive AUROC and
negative MAE differences favor pretraining. No cell survives Holm correction across the 36
comparisons.

| endpoint | arm | metric | Δ @ f = 0.10 | Δ @ f = 0.25 | Δ @ f = 0.50 |
|---|---|---|---|---|---|
| BBB | single | AUROC ↑ | +0.019 ± 0.033 | +0.018 ± 0.038 | +0.024 ± 0.044 |
|  | multi | AUROC ↑ | −0.004 ± 0.019 | −0.025 ± 0.022 | +0.002 ± 0.028 |
| AMES | single | AUROC ↑ | +0.002 ± 0.011 | −0.025 ± 0.019 | +0.004 ± 0.021 |
|  | multi | AUROC ↑ | −0.006 ± 0.021 | −0.013 ± 0.012 | +0.002 ± 0.025 |
| hERG | single | AUROC ↑ | +0.023 ± 0.062 | +0.033 ± 0.012 | −0.020 ± 0.022 |
|  | multi | AUROC ↑ | +0.020 ± 0.067 | −0.006 ± 0.034 | +0.026 ± 0.047 |
| HIA | single | AUROC ↑ | −0.036 ± 0.075 | +0.003 ± 0.182 | +0.003 ± 0.023 |
|  | multi | AUROC ↑ | +0.010 ± 0.058 | −0.014 ± 0.102 | +0.006 ± 0.059 |
| Solubility | single | MAE ↓ | −0.126 ± 0.060 | −0.041 ± 0.066 | −0.131 ± 0.071 |
|  | multi | MAE ↓ | +0.021 ± 0.245 | −0.033 ± 0.097 | −0.141 ± 0.214 |
| Caco2 | single | MAE ↓ | −0.012 ± 0.077 | −0.129 ± 0.090 | +0.046 ± 0.159 |
|  | multi | MAE ↓ | −0.039 ± 0.015 | −0.013 ± 0.009 | −0.001 ± 0.043 |

<p align="center"><img src="figures/fig_pretrain_ablation.png" alt="Figure 2" width="700"></p>

<sub><strong>Figure 2.</strong> Attribute-mask pretraining minus from-scratch training at the lowest training fraction (f = 0.10), in the beneficial direction (AUROC difference, or the negative of the MAE difference), for single-task (yellow) and multi-task (cyan) models. Bars are means and error bars ±1 s.d. over five seeds. Single-task Solubility is the largest and most consistent effect; no difference survives multiple-comparison correction.</sub>

### 4.3 Deep-ensemble uncertainty: scoring rules and calibration

Table 3 averages each uncertainty method over the four classification endpoints. Among the
graph models, the ensemble has the best mean NLL (0.482 versus 0.505 single and 0.532 MC
dropout), Brier score (0.153 versus 0.162 and 0.164), AURC (0.126 versus 0.146 and 0.155),
selective accuracy at 70% coverage (0.853 versus 0.837 and 0.822), and conformal set size
(1.212 versus 1.265 and 1.327). Per endpoint, it has the best NLL, Brier score, and AURC of
the graph models on BBB, AMES, and hERG; on HIA, the smallest test set (106 molecules, 13
negatives), the single network and MC dropout are better on all three.

The ensemble isn't the best-calibrated graph model by ECE. Its mean ECE (0.068) is above the
single temperature-scaled network (0.052) and the RF (0.053). This is the expected
consequence of averaging members that were each calibrated first: averaging pulls
probabilities toward the middle, so an ensemble of calibrated members tends to be
underconfident (Rahaman and Thiery 2021; Wu and Gales 2021). Inverse-frequency class weights
may add to this, since they shift predicted probabilities toward a balanced prior that a
single temperature can't undo. Calibrating the averaged ensemble, rather than each member,
is the natural fix and is left to future work.

Against the RF, the comparison is not close on most metrics. On this split the RF has the
higher test AUROC on every classification endpoint (0.881, 0.864, 0.871, 0.911 versus the
ensemble's 0.768, 0.833, 0.802, 0.844 on BBB, AMES, hERG, HIA) and better mean NLL, Brier
score, AURC, and selective accuracy. All values come from one split with no seed-level
error bars, so gaps of a few thousandths, such as the ensemble's and RF's mean set sizes,
are not resolved.

**Table 3.** Uncertainty methods on the four classification endpoints (fixed split, mean over
endpoints). Coverage is the realized test coverage of the nominal 90% conformal sets. Bold
marks the best graph model in each column; the RF row is the descriptor comparator.

| method | ECE ↓ | NLL ↓ | Brier ↓ | set size @ 90% ↓ | coverage | AURC ↓ | sel. acc. @ 70% ↑ |
|---|---|---|---|---|---|---|---|
| RF (descriptor baseline) | 0.053 | 0.371 | 0.117 | 1.213 | 0.918 | 0.062 | 0.910 |
| single GIN | **0.052** | 0.505 | 0.162 | 1.265 | 0.878 | 0.146 | 0.837 |
| MC dropout | 0.071 | 0.532 | 0.164 | 1.327 | 0.891 | 0.155 | 0.822 |
| deep ensemble (K = 5) | 0.068 | **0.482** | **0.153** | **1.212** | 0.877 | **0.126** | **0.853** |

<p align="center"><img src="figures/fig_ensemble_calibration.png" alt="Figure 3" width="700"></p>

<sub><strong>Figure 3.</strong> Per-endpoint ECE, NLL, and Brier score (lower is better) for the RF (grey), single GIN (yellow), MC dropout (green), and K = 5 deep ensemble (cyan) on the fixed scaffold split. The ensemble has the lowest NLL and Brier score among graph models on BBB, AMES, and hERG but not HIA, and it doesn't have the lowest ECE; the RF is lowest on NLL and Brier throughout.</sub>

### 4.4 Conformal coverage and selective prediction under scaffold shift

Split-conformal coverage is guaranteed only for exchangeable calibration and test data, and
a scaffold split is designed to break that (Appendix A.5). Realized coverage of the
ensemble's nominal 90% sets is 0.906, 0.883, 0.830, and 0.887 on BBB, AMES, hERG, and HIA
(Table 4; mean 0.877). On hERG, coverage of 0.830 over 153 test molecules is about 2.9
binomial standard errors below target. Molecules within a scaffold are correlated, so that
standard error understates the true uncertainty, but the shortfall is large enough that
nominal coverage shouldn't be assumed under scaffold shift. The RF covers at or above target on all four endpoints
(mean 0.918).

Set size is a fair efficiency measure only at matched coverage. The single network and the
ensemble agree in realized coverage within 0.01 on every endpoint, and the ensemble's sets
are smaller on all four. MC dropout's sets are larger still, though on HIA it also covers
more (0.925 versus 0.887). Against the RF the comparison is confounded. The ensemble's sets
are smaller on AMES (1.309 versus 1.333), hERG (1.222 versus 1.418), and HIA (1.019 versus
1.075), but in each case at lower realized coverage (0.883 versus 0.922, 0.830 versus 0.922,
and 0.887 versus 0.925), so part of the reduction is bought with coverage. On BBB, where
coverage matches, the RF's sets are much smaller (1.024 versus 1.299).

Selective prediction gives a mixed picture per endpoint. The ensemble has the highest
graph-model accuracy on its most confident 70% for BBB (0.866) and AMES (0.830), but the
lowest on hERG (0.785 versus 0.804 single and 0.794 MC dropout), and on HIA it ties the
single network (0.932) below MC dropout (0.946). Its higher mean comes from the first two
endpoints. Two of the endpoints are strongly imbalanced (83% positive test molecules for BBB
and 88% for HIA), so selective accuracy there sits close to ceiling. The RF has the highest
selective accuracy on all four endpoints.

**Table 4.** Conformal and selective-prediction results per endpoint on the fixed split.
Coverage is realized test coverage of the nominal 90% sets. Bold marks the best graph model.

| endpoint | test n | coverage RF / single / ens. | set size RF / single / MC / ens. | sel. acc. @ 70% RF / single / MC / ens. |
|---|---|---|---|---|
| BBB | 458 | 0.904 / 0.908 / 0.906 | 1.024 / 1.352 / 1.448 / **1.299** | 0.950 / 0.832 / 0.804 / **0.866** |
| AMES | 864 | 0.922 / 0.880 / 0.883 | 1.333 / 1.375 / 1.510 / **1.309** | 0.871 / 0.782 / 0.742 / **0.830** |
| hERG | 153 | 0.922 / 0.837 / 0.830 | 1.418 / 1.268 / 1.255 / **1.222** | 0.832 / **0.804** / 0.794 / 0.785 |
| HIA | 106 | 0.925 / 0.887 / 0.887 | 1.075 / 1.066 / 1.094 / **1.019** | 0.986 / 0.932 / **0.946** / 0.932 |

<p align="center"><img src="figures/fig_conformal_efficiency.png" alt="Figure 4" width="700"></p>

<sub><strong>Figure 4.</strong> Left: mean split-conformal set size at a nominal 90% target (lower is sharper; the dashed line marks a single-label set). Right: accuracy on the 70% most confident test molecules (higher is better). RF (grey), single GIN (yellow), MC dropout (green), deep ensemble (cyan). Set sizes are comparable only at matched realized coverage, which holds approximately between the single GIN and the ensemble, but not for MC dropout on HIA or for the RF (Table 4).</sub>

### 4.5 Sequential go/no-go campaign with the learned oracle

In this retrospective run the SPRT returns GO on three of the four classification endpoints
(BBB, AMES, hERG) and uses 150 measurements in total (Table 5, Figure 5). The comparison with
a fixed design needs care. A one-sided fixed-sample test at the same nominal error rates needs 68.7
measurements, which only the BBB and AMES subgroups can supply. On those two endpoints the
SPRT used 105 measurements against 137.4 (24% fewer), and the saving comes entirely from AMES
(20 versus 68.7; observed shift 0.64 standard deviations). BBB, whose shift (0.25) lies below
the design effect of 0.30, needed 85 measurements before crossing the GO boundary, 16 more
than the fixed design. hERG reached GO after 14 of its 45 subgroup molecules (shift 0.59).
HIA's subgroup has only 31 molecules, and its shift (0.28) never crossed either boundary, so
the test ended undecided when the subgroup ran out. Capping the fixed design at the subgroup
size gives a total of 213.4 and a 29.7% saving, but the capped hERG and HIA designs have less
than the nominal power, so that figure isn't a same-error-rate comparison.

A GO means that, in this measurement order, the working-model likelihood ratio crossed the
boundary favoring a 0.30 s.d. shift over no shift. It is evidence that the ensemble's
top-ranked 30% is enriched for positives relative to the test pool, consistent with the
ranking ability in §4.3, and it validates no particular chemistry; BBB reached GO with an
observed shift below the design effect. The procedure also looks ahead: labels are
standardized with the test pool's mean and standard deviation, and the direction is chosen
from the whole subgroup's mean, both of which use labels a real campaign wouldn't yet have.
The chosen direction was positive on all four endpoints, the direction one would pre-specify
for a subgroup ranked by probability of the positive class, so the decisions coincide with
those of a one-sided test, but the pool statistics remain a look-ahead. The Gaussian working
model is approximate for standardized binary labels drawn without replacement, so Wald's
error bounds (Appendix A.4) are nominal, and each count comes from one random measurement
order (§6). Each endpoint's decision is logged to the DuckDB lineage store with its
experiment ID, parameters, and the oracle's summary prediction.

**Table 5.** Campaign outcome per endpoint. Shift is the subgroup's standardized label mean
in pool standard deviations. The fixed-sample test at the same nominal error rates needs 68.7
measurements; for hERG and HIA it is capped at the subgroup size (marked *), which lowers its
power below nominal, and the total uses the capped values.

| endpoint | test n | subgroup | shift | verdict | SPRT n | fixed n | saved |
|---|---|---|---|---|---|---|---|
| BBB | 458 | 137 | 0.254 | GO | 85 | 68.7 | −16.3 |
| AMES | 864 | 259 | 0.636 | GO | 20 | 68.7 | 48.7 |
| hERG | 153 | 45 | 0.587 | GO | 14 | 45.0* | 31.0 |
| HIA | 106 | 31 | 0.276 | undecided | 31 | 31.0* | 0.0 |
| total |  |  |  | 3 GO | 150 | 213.4 | 63.4 (29.7%) |

<p align="center"><img src="figures/fig_campaign_efficiency.png" alt="Figure 5" width="700"></p>

<sub><strong>Figure 5.</strong> Measurements to a decision per endpoint: SPRT driven by the deep-ensemble oracle (cyan circles) against the fixed-sample test at the same nominal error rates (grey squares; 68.7 measurements, capped at the subgroup size for hERG and HIA, where the capped design has less than nominal power). AMES and hERG reach GO well below the fixed design, BBB reaches GO above it, and HIA exhausts its 31-molecule subgroup undecided. Retrospective run with test-pool standardization (§4.5).</sub>

### 4.6 Interpretability probe: attention against a descriptor rule

We tested whether graph-attention weights pick out atoms named by a transparent rule (aromatic
or nitrogen atoms, a coarse proxy for the lipophilic-ring and basic-amine motifs associated
with hERG binding and CNS penetration). In this run they don't. Node attention ranks atoms
close to chance against the rule (AUROC 0.52 on hERG and 0.49 on BBB), and the top three attended atoms per
molecule are barely richer or poorer in salient atoms than the molecule overall (enrichment +0.04 and
−0.04), even though the attention network is predictive (test AUROC 0.77 and 0.76). The
result has two readings that this probe can't separate: attention weights may not be
faithful explanations (Jain and Wallace 2019), or the rule may be too coarse a target, since
it marks 47% of hERG atoms and 44% of BBB atoms as salient. Whether attention can be an
explanation depends on how it is tested (Wiegreffe and Pinter 2019); here we simply don't
use it as one. The probe is a single unseeded run on one split, excludes molecules that are
too small or uniformly labeled (§3.8), and has no confidence intervals, so "near chance"
describes this run rather than a tested null.

<p align="center"><img src="figures/fig_attention_probe.png" alt="Figure 6" width="700"></p>

<sub><strong>Figure 6.</strong> Attention probe on hERG and BBB. Left: GAT test AUROC (cyan) against the AUROC of final-layer node attention for the aromatic-or-nitrogen salience rule (yellow); the dashed line marks chance. Right: salient-atom enrichment among each molecule's three most-attended atoms relative to the molecule's base rate (zero means no enrichment).</sub>

### 4.7 Summary of results

Table 6 collects the per-endpoint full-data accuracy of the three model families and the
ensemble's decision metrics on the fixed split.

**Table 6.** Full-data accuracy (training fraction 1.0; mean ± sample s.d. over five
scaffold-split seeds) and K = 5 ensemble decision metrics (fixed split, classification only). The
multi−single column is the full-data difference; the fraction-resolved differences are in
Table 1.

| endpoint | n | metric | RF | single-task | multi-task | multi−single | ens. ECE ↓ | coverage | set @ 90% ↓ | sel. acc. @ 70% ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| BBB | 2030 | AUROC ↑ | 0.892 ± 0.022 | 0.815 ± 0.030 | 0.815 ± 0.046 | 0.000 | 0.085 | 0.906 | 1.299 | 0.866 |
| AMES | 7278 | AUROC ↑ | 0.817 ± 0.040 | 0.761 ± 0.039 | 0.712 ± 0.055 | −0.049 | 0.055 | 0.883 | 1.309 | 0.830 |
| hERG | 655 | AUROC ↑ | 0.865 ± 0.044 | 0.759 ± 0.062 | 0.774 ± 0.037 | +0.016 | 0.060 | 0.830 | 1.222 | 0.785 |
| HIA | 578 | AUROC ↑ | 0.945 ± 0.035 | 0.929 ± 0.064 | 0.920 ± 0.051 | −0.009 | 0.074 | 0.887 | 1.019 | 0.932 |
| Solubility | 9982 | MAE ↓ | 0.842 ± 0.067 | 1.215 ± 0.060 | 1.360 ± 0.063 | +0.145 | - | - | - | - |
| Caco2 | 910 | MAE ↓ | 0.399 ± 0.039 | 0.658 ± 0.106 | 0.486 ± 0.052 | −0.172 | - | - | - | - |

Taken together: the RF is the most accurate model on every endpoint; sharing a graph encoder
helps some small endpoints (clearly Caco2, partly HIA) and hurts the large ones; and a deep
ensemble is the strongest graph-model uncertainty method on proper scoring rules and
decision metrics on average, without being the best calibrated by ECE or uniformly best per
endpoint.

## 5. Reproducibility

The study has two tiers. Tier 1 runs on CPU in minutes. The walkthrough notebook
recomputes every table in this paper from the committed artifacts in `artifacts/`, and three
scripts regenerate the figures and the campaign from the same artifacts:

```bash
pip install -r requirements.txt                               # pinned CPU environment
jupyter nbconvert --to notebook --execute --inplace \
    preprint/dl_forward/walkthrough_learned_representations_admet.ipynb
python preprint/dl_forward/figures.py                         # Figures 1-6
python preprint/dl_forward/architecture_figure_v2.py          # Figure 0
python preprint/dl_forward/e8_campaign_learned.py             # Table 5 + DuckDB lineage
PYTHONPATH=.:src python preprint/dl_forward/scaffold_overlap.py   # Appendix A.10 (fetches TDC data)
```

Tier 2 retrains from scratch on a GPU (`make install-gpu`), fetching the TDC data on first
run, with `PYTHONPATH=.:src`:

```bash
python preprint/dl_forward/train_multitask_gnn.py           # data for Table 1, Figure 1
python preprint/dl_forward/pretrain_ablation.py --full      # data for Table 2, Figure 2
python preprint/dl_forward/make_ensemble_uncertainty.py     # data for Tables 3-4, Figures 3-4
python preprint/dl_forward/attention_attribution.py         # data for Figure 6
```

For the transfer, pretraining, and ensemble scripts, seeds pin NumPy and PyTorch,
and `cudnn.deterministic` is set, but residual GPU nondeterminism remains, so we report
dispersion over seeds rather than expecting bitwise reproduction. The attention probe's
training isn't seeded. The interpreter is pinned to Python 3.11.2 and the data loader to PyTDC 1.1.15.
The DuckDB lineage store is regenerated by the campaign script rather than committed;
`artifacts/campaign_learned.json` records its path and the per-endpoint experiment IDs.

## 6. Limitations

- **Retrospective, public data.** Every result is a simulation over public TDC benchmarks; no
  wet-lab loop is closed and no prospective claim is made.
- **Transfer is endpoint-specific and possibly optimistic.** Multi-task benefits are clear for
  Caco2, partial for HIA, and within seed variability for hERG, and the shared trunk hurts
  AMES, Solubility, and BBB at some fractions. Splits are drawn per endpoint, so the
  multi-task trunk and the pretraining corpus see other endpoints' molecules that overlap the
  target's test set (1-4% of test molecules verbatim at f = 0.10, 10-23% at full data;
  Appendix A.10); a globally scaffold-disjoint split would be needed to remove this. Five
  seeds give low power for single cells, and we don't correct the 24 transfer comparisons for
  multiplicity.
- **One split for the uncertainty study and campaign.** Tables 3-5 come from a single
  scaffold split with no seed-level dispersion, and the hERG and HIA test sets have only 153
  and 106 molecules.
- **Ensemble calibration.** Members are calibrated before averaging, which is likely why the
  ensemble's ECE is worse than a single calibrated network's. The ensemble is built from
  single-task members; a multi-task ensemble wasn't evaluated.
- **Baselines and featurization.** The MC dropout baseline isn't temperature-scaled, and its
  training-mode passes also activate DropEdge and batch-statistics normalization, which may
  understate what a carefully tuned dropout baseline achieves. Atomic number enters the graph
  encoder as a scalar rather than a one-hot vector, and the GIN ignores bond attributes, a
  simpler featurization than common practice.
- **Conformal coverage is measured, not guaranteed.** Scaffold shift breaks the
  exchangeability that split-conformal coverage requires, and coverage falls to 0.830 on
  hERG. Shift-aware conformal methods (Tibshirani et al. 2019)
  weren't applied.
- **The campaign is retrospective.** Labels are standardized with the test pool's mean and
  standard deviation and the direction is chosen from the whole subgroup's mean, so the
  stopping counts aren't those of an implementable sequential design. The SPRT treats
  standardized binary labels as Gaussian, samples without replacement from a finite
  subgroup, and reports one random measurement order, so its error rates are nominal. The
  fixed-sample comparator at the same nominal error rates is feasible only for BBB and AMES. A GO is
  evidence of the oracle's ranking, not a discovery.
- **The attention probe** is one unseeded run with a coarse rule, two endpoints, one split,
  and no confidence intervals, and it can't distinguish unfaithful attention from an
  uninformative rule. The companion engine's
  abductive rule discovery wasn't re-run with the learned oracle.
- Uncertainty-driven label acquisition is out of scope and left to future work.

## 7. Conclusion

Replacing a descriptor oracle with learned graph models changes less than one might hope
about accuracy and more than one might expect about where the gains sit. A strong random
forest on descriptors and fingerprints remains the most accurate model on every endpoint
studied. A shared graph encoder helps the smallest endpoints, most clearly Caco2 and HIA, and
costs accuracy on the largest; its value depends on the endpoint, not on a general low-data
advantage, and because per-endpoint splits let the shared trunk see some of each target's
test structures, the measured gains may be optimistic. A five-member deep ensemble is the strongest graph-model uncertainty method on
proper scoring rules, risk-coverage, and conformal set size, but it isn't the best
calibrated by ECE, a gap that calibrating after averaging should close. Driving an existing
sequential decision engine in a retrospective run, the ensemble reaches traceable go/no-go
calls, and where a fixed design at the same nominal error rates is feasible it needs fewer
measurements, with
the saving concentrated where its top-ranked subgroup is strongly enriched. The practical message is to keep the descriptor baseline in the loop, use
learned representations where transfer demonstrably helps, and judge uncertainty methods by
the decisions they support, with coverage reported alongside every set size.

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
20. Russell, D. R. (2026). Knowing what to measure and when to stop: an autonomous decision
    engine for molecular property discovery. Preprint, SNPTX.
    https://github.com/snptx1/snptx-repro-discovery
21. Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., Guzman-Perez, A.,
    Hopper, T., Kelley, B., Mathea, M., Palmer, A., Settels, V., Jaakkola, T., Jensen, K.
    and Barzilay, R. (2019). Analyzing learned molecular representations for property
    prediction. *Journal of Chemical Information and Modeling*, 59(8), 3370-3388.
22. Jiang, D., Wu, Z., Hsieh, C.-Y., Chen, G., Liao, B., Wang, Z., Shen, C., Cao, D.,
    Wu, J. and Hou, T. (2021). Could graph neural networks learn better molecular
    representation for drug discovery? A comparison study of descriptor-based and
    graph-based models. *Journal of Cheminformatics*, 13, 12.
23. Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S.,
    Leswing, K. and Pande, V. (2018). MoleculeNet: a benchmark for molecular machine
    learning. *Chemical Science*, 9(2), 513-530.
24. Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.
25. Ramsundar, B., Kearnes, S., Riley, P., Webster, D., Konerding, D. and Pande, V.
    (2015). Massively multitask networks for drug discovery. *arXiv:1502.02072*.
26. Wu, X. and Gales, M. (2021). Should ensemble members be calibrated?
    *arXiv:2101.05397*.
27. Rahaman, R. and Thiery, A. H. (2021). Uncertainty quantification and deep ensembles.
    *NeurIPS*.
28. Tibshirani, R. J., Foygel Barber, R., Candès, E. and Ramdas, A. (2019). Conformal
    prediction under covariate shift. *NeurIPS*.
29. Wiegreffe, S. and Pinter, Y. (2019). Attention is not not explanation. *EMNLP-IJCNLP*.
30. Rogers, D. and Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of
    Chemical Information and Modeling*, 50(5), 742-754.
31. Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    *Scandinavian Journal of Statistics*, 6(2), 65-70.

---

# Appendix A. Methods and derivations

This appendix makes the study self-contained. A.1-A.3 cover the multi-task objective, the
ensemble's uncertainty decomposition, and the attention probe; A.4-A.7 give the
decision-layer definitions the results rely on; A.8 the pretraining objective; A.9 the
implementation settings; and A.10 the cross-endpoint exposure of the multi-task trunk.

## A.1 Multi-task objective

Let the shared encoder $\phi_\theta(\cdot)$ map a molecular graph to a pooled embedding, and
let $h_t$ be the head for endpoint $t$. For a classification endpoint the per-example loss is
class-weighted cross-entropy on $h_t(\phi_\theta(x))$; for a regression endpoint it is
mean-squared error on the standardized target $\tilde y = (y-\mu_t)/\sigma_t$, with
$\mu_t,\sigma_t$ estimated on the training split. The multi-task objective is the
task-weighted sum

$$
\mathcal{L}(\theta) = \sum_{t} w_t \, \mathbb{E}_{(x,y)\sim \mathcal{D}_t}\big[\ell_t(h_t(\phi_\theta(x)), y)\big],
$$

with $w_t = 1$, optimized by round-robin task-batch updates so that the trunk isn't dominated
by the largest endpoint. When $|\mathcal{D}_t|$ is small, a single-task estimate of the trunk
is high-variance, and gradients from the other endpoints act as a regularizer; when tasks
conflict or an endpoint has ample data, sharing capacity can cost accuracy. Both effects
appear in Table 1.

## A.2 Deep-ensemble uncertainty decomposition

For a K-member ensemble with temperature-scaled member probabilities $p^{(k)}(y\mid x)$, the
predictive distribution is $\bar p(y\mid x) = \frac1K\sum_k p^{(k)}(y\mid x)$. Its entropy
decomposes as

$$
\underbrace{\mathcal{H}[\bar p]}_{\text{total}}
  \;=\;
\underbrace{\tfrac1K\textstyle\sum_k \mathcal{H}[p^{(k)}]}_{\text{aleatoric}}
  \;+\;
\underbrace{\mathcal{H}[\bar p]-\tfrac1K\textstyle\sum_k \mathcal{H}[p^{(k)}]}_{\text{epistemic (mutual information)}},
$$

where the epistemic term is the mutual information between the label and the member index.
It is non-negative by concavity of entropy and vanishes when all members agree. The
decision metrics in this paper don't use the decomposition directly: conformal scores use
$1-\bar p(y\mid x)$ and selective prediction ranks by $\max_y \bar p(y\mid x)$. Member
disagreement enters those scores implicitly, because averaging disagreeing members flattens
$\bar p$. Testing the mutual information directly as an abstention score is left to future
work.

## A.3 Attention attribution

For the GAT, let $\alpha^{(L)}_{ji}$ be the attention coefficient from atom $j$ to atom $i$
in the final layer, which has a single head (the first two layers have four). The score of
atom $i$ is the incoming attention $a_i = \sum_{j \in \mathcal{N}(i)} \alpha^{(L)}_{ji}$. The
rule labels atom $i$ salient if it is aromatic or a nitrogen. Test molecules with fewer than
four atoms, or with all atoms salient or all non-salient, are excluded (5 of 153 on hERG, 88
of 458 on BBB). Over the remaining molecules we pool atoms and report the AUROC of $a_i$ for
predicting salience, and for each molecule the fraction of salient atoms among its three
highest-scoring atoms minus the molecule's salient fraction, averaged over molecules. An AUROC of 0.5 and an enrichment of 0 mean
attention carries no information about the rule.

## A.4 Sequential probability ratio test (SPRT)

The campaign of §4.5 decides when to stop measuring with Wald's SPRT (Wald 1945; Wald and
Wolfowitz 1948). Each measurement $x_i$ is a standardized label from the oracle's top-ranked
subgroup, modeled as $x_i \sim \mathcal{N}(\theta, 1)$. We test $H_0:\theta=\theta_0=0$
against $H_1:\theta=\theta_1=0.30$. The per-measurement log-likelihood ratio is

$$
z_i = \log\frac{f_{\theta_1}(x_i)}{f_{\theta_0}(x_i)}
    = (\theta_1-\theta_0)\Big(x_i - \frac{\theta_0+\theta_1}{2}\Big),
\qquad \Lambda_n = \sum_{i=1}^{n} z_i .
$$

Sampling continues while $B < \Lambda_n < A$, accepting $H_1$ (GO) when $\Lambda_n \ge A$ and
$H_0$ (NO-GO) when $\Lambda_n \le B$, with Wald's boundaries

$$
A = \log\frac{1-\beta}{\alpha}, \qquad B = \log\frac{\beta}{1-\alpha},
$$

and $(\alpha,\beta) = (0.05, 0.20)$. Wald's inequalities bound the realized error rates by
$\alpha' \le \alpha/(1-\beta) = 0.0625$ and $\beta' \le \beta/(1-\alpha) \approx 0.21$. The
comparator is the one-sided fixed-sample $z$-test with the same nominal error rates, which needs

$$
n_{\text{fixed}} = \left(\frac{z_{1-\alpha}+z_{1-\beta}}{\theta_1-\theta_0}\right)^{2}
= \left(\frac{1.645 + 0.842}{0.30}\right)^2 \approx 68.7
$$

measurements. Only the BBB and AMES subgroups are large enough for this design; for hERG and
HIA we cap it at the subgroup size (45 and 31), which gives those capped designs less than
the nominal power. When the true shift is at or above $\theta_1$, the SPRT's expected sample size is well below
$n_{\text{fixed}}$; when it lies between $\theta_0$ and $\theta_1$, the test can run longer than
the fixed design, as on BBB.

Four departures from the textbook setting matter. First, the measurements are standardized
with the mean and standard deviation of the whole test pool, which a real campaign wouldn't
know in advance. Second, the implementation chooses the test direction from the sign of the
whole subgroup's mean, which also looks ahead; under the working model a data-chosen
direction would loosen the per-direction bound of 0.0625 to at most 0.125 by a union bound.
Here the chosen direction was positive on all four endpoints, the direction one would
pre-specify for a subgroup ranked by probability of the positive class, so the decisions
equal those of a one-sided test. Third, the measurements are standardized binary labels,
which take two values, so the Gaussian likelihood is a working model rather than the data's
distribution. Fourth, measurements are drawn without replacement from a finite subgroup in
one random order, so the reported sample sizes are single realizations rather than expected
values. Together these make Wald's bounds nominal; we report the realized decisions and
counts of one retrospective run, not validated error rates.

## A.5 Split-conformal prediction and coverage

On a calibration set of size $n$ we compute scores $s_i = 1 - \hat p(y_i \mid x_i)$ and take

$$
\hat q = \text{the } \big\lceil (1-\alpha)(n+1) \big\rceil / n \text{ empirical quantile of } \{s_i\}_{i=1}^{n},
$$

with prediction set $C(x) = \{\,y : 1-\hat p(y\mid x) \le \hat q\,\}$. If calibration and test
points are exchangeable, then

$$
\Pr\big(y_{\text{test}} \in C(x_{\text{test}})\big) \;\ge\; 1-\alpha,
$$

marginally over the draw of calibration and test data (Vovk et al. 2005; Angelopoulos and
Bates 2023); here $\alpha = 0.10$. Because validity holds by construction under
exchangeability, efficiency, the mean set size $\mathbb{E}\,|C(x)|$, is the informative
comparison, but only between methods at the same realized coverage. In this study scaffolds
are assigned to validation and test at random, so the two sets are exchangeable at the level
of scaffold groups. The guarantee, however, is stated for exchangeable individual molecules,
and molecules within a scaffold are correlated, so realized coverage can depart from nominal,
as it does on hERG (0.830). The implementation computes $\hat q$ with NumPy's linear
interpolation at level $\lceil (1-\alpha)(n+1) \rceil / n$, which returns a value at or above
the exact order statistic, so its sets contain the exact construction's sets and the
guarantee still holds, slightly conservatively, under exchangeability whenever
$k = \lceil (1-\alpha)(n+1) \rceil \le n$, which is true of every calibration set here. (For
$k > n$ the exact construction returns every label, whereas the code caps the level at 1.)

## A.6 Calibration: expected calibration error and temperature scaling

Partitioning test predictions by top-label confidence into $B = 10$ equal-width bins
$\mathcal{B}_1,\dots,\mathcal{B}_B$,

$$
\mathrm{ECE} \;=\; \sum_{b=1}^{B} \frac{|\mathcal{B}_b|}{N}\,
    \big|\,\mathrm{acc}(\mathcal{B}_b) - \mathrm{conf}(\mathcal{B}_b)\,\big|,
$$

where $\mathrm{acc}(\mathcal{B}_b)$ is the empirical accuracy and $\mathrm{conf}(\mathcal{B}_b)$
the mean confidence in bin $b$. NLL is $-\frac1N\sum_i \log \hat p(y_i\mid x_i)$, and the Brier
score is $\frac1N\sum_i (\hat p(y{=}1\mid x_i) - y_i)^2$. Temperature scaling (Guo et al. 2017)
fits one scalar $T>0$ on the validation set to minimize the NLL of
$\mathrm{softmax}(z/T)$; it rescales confidence without changing the predicted class. Each
ensemble member is scaled before averaging. Averaging calibrated members flattens the
ensemble's probabilities and tends to make it underconfident (Rahaman and Thiery 2021; Wu
and Gales 2021), which is consistent with the ensemble's higher ECE in Table 3. ECE with ten
bins is also noisy on test sets of 106 to 864 molecules.

## A.7 Selective prediction and risk-coverage

For a confidence score $g(x)$ and threshold $\tau$, coverage and selective risk are

$$
\mathrm{cov}(\tau) = \Pr\!\big(g(x) \ge \tau\big), \qquad
\mathrm{risk}(\tau) = \frac{\mathbb{E}\big[\ell(x,y)\,\mathbf{1}\{g(x)\ge\tau\}\big]}
                            {\mathrm{cov}(\tau)} ,
$$

with 0-1 loss $\ell$ (El-Yaniv and Wiener 2010; Geifman and El-Yaniv 2017). We use
$g(x) = \max_y \hat p(y\mid x)$ for every method. Sweeping $\tau$ traces the risk-coverage
curve; AURC is the mean selective risk over all coverage levels $k/N$, $k = 1,\dots,N$, and
selective accuracy at 70% is the accuracy on the $\lfloor 0.7N \rceil$ most confident test
molecules. On imbalanced endpoints selective accuracy should be read against the majority
rate (BBB 0.83 and HIA 0.88 positive in the test set).

## A.8 Self-supervised attribute-mask pretraining

The ablation of §4.2 pretrains the shared trunk with the attribute-masking objective of Hu et
al. (2020). For each molecule we sample a masked atom set $M(x)$ at rate $\rho = 0.15$, zero
those atoms' feature rows to give the corrupted graph $\tilde x$, and predict each masked
atom's element $z_i$ from its node embedding by cross-entropy over the corpus's
atomic-number vocabulary $\mathcal{V}$:

$$
\mathcal{L}_{\text{mask}}(\theta) \;=\; -\,\mathbb{E}_{x}\,\frac{1}{|M(x)|}
    \sum_{i \in M(x)} \log p_\theta\!\big(z_i \mid \tilde x\big),
\qquad z_i \in \mathcal{V}.
$$

The corpus is the union of all six endpoints' training molecules for each seed and fraction,
without labels and without any endpoint's own validation or test scaffolds; as Appendix A.10
shows, other endpoints' training molecules can still overlap a target's test set. The pretrained trunk is then fine-tuned under
the identical supervised protocol. Hu et al. found that node-level or graph-level
pretraining alone gives limited improvement and can transfer negatively, and that combining
the two works best; this ablation tests the node-level objective alone.

## A.9 Implementation and hyperparameters

All graph models share the encoder and protocol below.

| group | setting | value |
|---|---|---|
| Encoder | backbone | GIN, sum pooling |
| | hidden width / layers | 128 / 4 |
| | normalization | BatchNorm in GIN MLPs; PairNorm (scale 1.0) between layers |
| | dropout / DropEdge rate | 0.3 / 0.1 |
| | atom features / bond attributes | 9 / computed but unused by GIN |
| Optimization | optimizer | Adam, learning rate 5e-4, weight decay 5e-4 |
| | epochs / batch size | 150 (fixed, no early stopping) / 128 |
| | task weighting | round-robin task batches, $w_t = 1$ |
| | class imbalance | inverse-frequency class weights |
| Descriptor baseline | features | 10 RDKit descriptors + 1024-bit Morgan (radius 2) |
| | model | random forest, 300 trees, balanced class weights |
| Evaluation | split | Murcko scaffold cold-split, scaffolds randomly assigned |
| | test / validation scaffold fraction | 0.20 / 0.20 |
| | seeds | 5 (0-4) for transfer and pretraining; split seed 0 for Tables 3-5 |
| | training fractions | 0.10, 0.25, 0.50, 1.00 (whole-scaffold subsampling) |
| Ensemble | members $K$ | 5 single-task GINs, training seeds 0-4 |
| | calibration | per-member temperature scaling on validation NLL |
| | MC dropout baseline | 30 passes of member 0, not temperature-scaled |
| Decision metrics | ECE bins | 10, equal width, top label |
| | conformal score / target | $1-\hat p(y\mid x)$ / 90% ($\alpha = 0.10$) |
| | selective coverage point | 70% |
| Pretraining (A.8) | mask rate $\rho$ / epochs | 0.15 / 40 |
| | fractions | 0.10, 0.25, 0.50 |
| SPRT (A.4) | $\alpha$, $\beta$ | 0.05, 0.20 |
| | design effect $\theta_1$ | 0.30 pool s.d. |
| | subgroup | top 30% by ensemble probability (minimum 8) |
| | measurement order seed | 20260905 |
| Attention probe | GAT | 3 layers, hidden 64, heads 4 / 4 / 1, checkpoint by validation AUROC, unseeded |

For the transfer, pretraining, and ensemble scripts, seeds pin NumPy and PyTorch,
and `cudnn.deterministic` is set; residual GPU nondeterminism remains, so we report
dispersion over seeds rather than bitwise reproduction. The interpreter is pinned to Python
3.11.2 and PyTDC to 1.1.15.

## A.10 Cross-endpoint exposure of the multi-task trunk

Scaffold splits are drawn independently for each endpoint, so a model trained on several
endpoints sees other endpoints' training molecules, and some of those are a target endpoint's
test molecules or share their Murcko scaffold. The single-task baseline never sees them. The
table gives, for each target endpoint and training fraction, the fraction of its test
molecules that appear verbatim (same canonical SMILES) in any other endpoint's training set,
and the fraction whose scaffold does, averaged over the five seeds
(`scaffold_overlap.py`, which reproduces the training splits exactly).

**Table A1.** Fraction of each endpoint's test molecules exposed to the multi-task trunk
through other endpoints' training data: same molecule / same scaffold, mean over five seeds.

| endpoint | f = 0.10 | f = 0.25 | f = 0.50 | f = 1.00 |
|---|---|---|---|---|
| HIA | 0.026 / 0.083 | 0.074 / 0.200 | 0.136 / 0.335 | 0.225 / 0.497 |
| hERG | 0.013 / 0.036 | 0.036 / 0.114 | 0.058 / 0.251 | 0.095 / 0.379 |
| Caco2 | 0.023 / 0.085 | 0.062 / 0.157 | 0.097 / 0.252 | 0.187 / 0.393 |
| BBB | 0.018 / 0.059 | 0.044 / 0.171 | 0.080 / 0.266 | 0.176 / 0.469 |
| AMES | 0.038 / 0.136 | 0.049 / 0.317 | 0.134 / 0.555 | 0.209 / 0.653 |
| Solubility | 0.008 / 0.071 | 0.025 / 0.144 | 0.060 / 0.277 | 0.109 / 0.427 |

Exposure rises with the training fraction because the other endpoints' training sets grow.
The trunk never sees a target endpoint's labels for exposed molecules, but it learns their
structures, and for verbatim molecules it also sees another endpoint's label. The multi-task
gains of Table 1 therefore can't be attributed to transfer alone. A globally scaffold-disjoint
split could show smaller or, in principle, larger gains, so the direction of the bias isn't
known. Building such a split, by assigning scaffolds across the union of endpoints, and
rerunning the transfer and pretraining studies is the direct test.
