"""Regenerable builder for the learned-representations walkthrough notebook.

Authoring the notebook as code keeps it reviewable and lets it be rebuilt whenever the
narrative changes. The notebook is the executable companion to
``MANUSCRIPT_learned_representations_admet.md``: it mirrors the manuscript's results
numbering (4.1-4.7), figure numbers (0-6), and table numbers (1-6). It recomputes paired
summaries and checks test metrics from committed arrays, and reconstructs other tables
from stored metrics (Tier 1, CPU). CPU regeneration and full retraining are separately gated.

Conventions:
    - Every code cell opens with a short docstring stating what it computes and how.
    - Inline citations use bracketed numbers ``[n]`` keyed to the References section, which
      matches the manuscript's reference list.
    - Every ± is a sample standard deviation over seeds (divisor n - 1).

Usage (from the repository root):
    python preprint/dl_forward/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace \\
        preprint/dl_forward/walkthrough_learned_representations_admet.ipynb
"""

from __future__ import annotations

from pathlib import Path
import sys

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE / "walkthrough_learned_representations_admet.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


CELLS: list[nbf.NotebookNode] = []

# --------------------------------------------------------------------------- front matter
CELLS.append(md(r'''
# Molecular graph models for ADMET: multitask comparisons, ensemble probabilities, and retrospective sequential testing

<p style="font-size:1.0em; font-style:italic; opacity:0.8; margin-top:2px; margin-bottom:25px;">Guided walkthrough: model comparisons, probability assessment, and sequential testing from committed results.</p>

<p style="font-size:0.95em; opacity:0.85; margin-top:2px; margin-bottom:25px;">Daniel R. Russell &middot; Autonomous Discovery Systems, SNPTX &middot; October 2026</p>

### Abstract

A companion study [20] built a sequential decision engine around a random-forest oracle.
This walkthrough asks what learned graph models add when they take the oracle's place, and
answers by audit as much as by benchmark. Across six Therapeutics Data Commons endpoints [17]
and five within-endpoint scaffold-split seeds [18], a multi-task Graph Isomorphism Network
[5] lowers Caco2 MAE by 0.11 to 0.17 at every retained training-scaffold fraction (18 of 20
seed-fraction pairs) and lifts HIA AUROC by 0.071 at fraction 0.50 (5 of 5 seeds), while
AMES and Solubility come out worse at full data. We read these as comparisons of training
protocols rather than measurements of transfer: multi-task training recycles the smaller
task loaders, and at full data 10-23% of each endpoint's test records appear as the same
canonical molecule in another endpoint's training set. No pretraining comparison survives
Holm correction [31]. Turning from accuracy to uncertainty, a five-member temperature-scaled
[11] ensemble [10] has lower mean NLL on one fixed split than a single network or Monte
Carlo dropout [12] (0.482 versus 0.505 and 0.532), but not lower ECE, and the RF comparator
leads on accuracy and proper scores. The ensemble's nominal 90% prediction sets [13,14]
cover 0.877 of test labels on average, a figure reported as empirical rather than
guaranteed, since temperature fitting and conformal calibration reuse the same validation
labels and group splitting does not establish molecule-level exchangeability. Wired into
the engine, a retrospective SPRT [1,2] reaches GO on BBB and AMES after 105 measurements
against a continuous fixed-sample requirement of 137.4 under a Gaussian working model;
probability rankings define the subgroups, uncertainty metrics are logged alongside, and
HIA's 31-record budget cannot reach GO at all. Finally, a GAT [4] implementation diagnostic
finds that the proposed incoming-attention sum is constant by normalization, so its stored
rule-agreement results say nothing about interpretability. Every table below is
reconstructed from committed artifacts on CPU; full retraining is a separate tier.

### How to read this document

This notebook is self-contained, and it is also the executable companion to the manuscript
of the same title. Its section numbers follow the manuscript's results sections (4.1-4.7)
and its figure and table numbers are the manuscript's (Figures 0-6, Tables 1-6 and A1), so
the two can be read side by side, but nothing here assumes the manuscript has been read.
Each section opens with the question it answers, defines the quantities involved,
reconstructs the result from committed artifacts, shows the figure with its caption, and
closes with what the result does and doesn't establish. Code cells say which summaries are
recomputed and which metrics are read from stored results. Bracketed numbers refer to the
References at the end.
'''))

CELLS.append(md(r'''
## Overview

<div style="display:flex; gap:48px; line-height:1.5;">
<div style="flex:1;">

<b>4.1 &nbsp;Multitask and single-task performance across endpoints</b><br>
<span style="font-size:0.86em; opacity:0.85;">Does one shared trunk help any of six endpoints? Caco2 yes, at every fraction; others mixed. Unequal update exposure and cross-endpoint structural overlap keep the gains from being called transfer (Table 1, Table A1, Figure 1).</span>
<br><br>

<b>4.2 &nbsp;Ablation: self-supervised pretraining</b><br>
<span style="font-size:0.86em; opacity:0.85;">Would a self-supervised warm start help further? None of 36 paired comparisons survives a Holm correction (Table 2, Figure 2).</span>
<br><br>

<b>4.3 &nbsp;Deep-ensemble uncertainty: scoring rules and calibration</b><br>
<span style="font-size:0.86em; opacity:0.85;">From accuracy to probabilities. A K = 5 ensemble is the best graph model on NLL, Brier score, and AURC on average, but not on ECE; the RF leads overall (Table 3, Figure 3).</span>
<br><br>

<b>4.4 &nbsp;Conformal coverage and selective prediction under scaffold shift</b><br>
<span style="font-size:0.86em; opacity:0.85;">What a nominal 90% set actually covers, what set size costs in coverage, and what selective accuracy owes to class balance (Table 4, Figure 4).</span>

</div>
<div style="flex:1;">

<b>4.5 &nbsp;Sequential go/no-go campaign with the learned oracle</b><br>
<span style="font-size:0.86em; opacity:0.85;">The ensemble goes to work inside the engine: 3 of 4 GO; 105 measurements versus a continuous fixed-sample requirement of 137.4 on BBB and AMES. HIA cannot reach GO within its budget (Table 5, Figure 5).</span>
<br><br>

<b>4.6 &nbsp;Attention-score implementation diagnostic</b><br>
<span style="font-size:0.86em; opacity:0.85;">The result we did not plan to report: the GAT predicts, but its incoming-attention sum is constant to numerical precision, so the salience probe is withdrawn (Figure 6).</span>
<br><br>

<b>4.7 &nbsp;Summary of results</b><br>
<span style="font-size:0.86em; opacity:0.85;">Full-data accuracy for all three model families with the ensemble's decision metrics, per endpoint, in one table (Table 6).</span>

</div>
</div>

Before the results, *Data, models, and protocol* lays out the experimental design and
*System architecture* shows how the pieces connect. The closing *Synopsis* collects what was
shown, what it doesn't establish, and the separate reconstruction and retraining commands.
'''))

CELLS.append(code(r'''
"""Self-locating environment setup.

Finds the repository root by walking up from the working directory to the first ancestor
holding ``src/``, puts the vendored ``src.*`` and ``snptx.*`` modules on the path, locates the
committed artifacts and figures, and applies the SNPTX dark theme. Also defines the display
names, dataset sizes, and formatting helpers used by every table below.
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import Image, display


def _find_repo_root(start: Path) -> Path:
    for cand in (start, *start.parents):
        if (cand / "src").is_dir() and (
            (cand / "src" / "snptx").is_dir() or (cand / "src" / "models").is_dir()
        ):
            return cand.resolve()
    raise FileNotFoundError(f"could not locate repo root from {start}")


ROOT = _find_repo_root(Path.cwd())
sys.path.insert(0, str(ROOT))           # `import src.*`
sys.path.insert(0, str(ROOT / "src"))   # `import snptx.*`

# The study folder sits at preprint/dl_forward (public) or pilot_phd/preprint/dl_forward (private).
DL = next(c for c in (ROOT / "preprint" / "dl_forward",
                      ROOT / "pilot_phd" / "preprint" / "dl_forward") if c.is_dir())
ART = DL / "artifacts"   # committed result JSON / NPZ
FIG = DL / "figures"     # committed PNGs, regenerated by figures.py

from snptx.viz.theme import use_neon_rcparams
use_neon_rcparams(plt)

NAMES = {"bbb": "BBB", "ames": "AMES", "herg": "hERG", "hia": "HIA",
         "solubility": "Solubility", "caco2": "Caco2"}
# Endpoint records after TDC loading and featurization; these are not unique molecules.
SIZES = {"bbb": 2030, "ames": 7278, "herg": 655, "hia": 578, "solubility": 9982, "caco2": 910}
CLF = ("bbb", "ames", "herg", "hia")


def fmt(x: float, signed: bool = True, digits: int = 3) -> str:
    # Fixed decimals with a typographic minus; a value that rounds to zero prints unsigned.
    if signed and round(abs(x), digits) == 0:
        return f"{0:.{digits}f}"
    return (f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}").replace("-", "\u2212")


def show_fig(name: str, width: int = 820) -> None:
    # Display a committed figure; a missing file is reported rather than raised.
    path = FIG / name
    if path.exists():
        display(Image(filename=str(path), width=width))
    else:
        print(f"[missing] {name}: run figures.py to regenerate it")


REGENERATE_CPU = False  # Campaign and figure files; see the closing section
RETRAIN_MODELS = False  # Full retraining; independent, explicit opt-in
print(f"Python: {sys.version.split()[0]}   repository: {ROOT.name}   artifacts: {ART.relative_to(ROOT)}   "
      f"figures: {FIG.relative_to(ROOT)}")
'''))

# --------------------------------------------------------------------------- protocol
CELLS.append(md(r'''
## Data, models, and protocol

**Endpoints.** Six ADMET endpoints from the Therapeutics Data Commons [17]: BBB (Martins) [32],
AMES [33], hERG [34], and HIA (Hou) [35] as binary classification, scored by AUROC, and
Solubility (AqSolDB) [36] and Caco2 (Wang) [37] as regression, scored by MAE on their TDC
target scales, with 2030, 7278, 655, 578, 9982, and 910 labeled records. Those records hold
1975, 7255, 648, 578, 9982, and 906 unique canonical molecules respectively; TDC removes
exact duplicate rows, and we don't go further by collapsing repeated structures or
conflicting labels. Record counts are used throughout the tables. One point of sign
convention matters later: positive labels denote the dataset phenotype, which is BBB
penetration and HIA absorption but AMES mutagenicity and hERG blockade. Enrichment in the
latter two is an adverse phenotype.

**Splits.** Every split is a Murcko scaffold cold-split [18]: unique scaffolds are randomly
permuted and assigned whole to test (20% of scaffolds), validation (20%), and training, so
within an endpoint no evaluation scaffold is seen in training. The word *within* carries
weight. Splits are drawn independently per endpoint, so a model trained on several endpoints
also sees the other endpoints' training molecules, some of which overlap a target's test set
(measured in §4.1). Fractions of 0.10, 0.25, 0.50, and 1.00 retain whole training scaffolds
rather than those fractions of training records, and because scaffold groups vary enormously
in size (one group holds every acyclic structure), record fractions swing substantially
between seeds. The transfer and pretraining studies (§4.1-§4.2) repeat everything over five
seeds, each drawing a new split and initialization; the uncertainty study and the campaign
(§4.3-§4.5) use one fixed split (seed 0). Models train for a fixed 150 epochs with no early
stopping, apart from GAT checkpoint selection by validation AUROC. The same validation labels
fit temperatures and calibrate conformal thresholds, a reuse that departs from the ordinary
held-out split-conformal setup and that §4.4 returns to.

**Models.** Molecules are graphs with nine atom features (atomic number as a scalar, degree,
formal charge, explicit hydrogens, aromatic and ring flags, one-hot sp/sp2/sp3
hybridization) connected by bonds; bond attributes are computed but the GIN ignores them.
The encoder is a four-layer Graph Isomorphism Network (GIN) [5] with hidden width 128,
BatchNorm inside each MLP and after each convolution, sum pooling, per-graph PairNorm [8],
and DropEdge [9], which drops directed message edges independently so that the two
directions of a chemical bond need not go together. The *single-task* model trains one
encoder per endpoint. The *multi-task* model shares one encoder across all six endpoints
with a head per endpoint, visiting endpoints round-robin; each epoch has as many rounds as
the largest task loader, shorter loaders restart, and single-task training traverses its own
loader once per epoch, so target-label exposure is not matched between the two protocols.
Classification losses use inverse-frequency class weights, and regression targets are
standardized to reduce scale differences without ensuring equal gradient scales. The
descriptor baseline is a 300-tree random forest with balanced class weights on ten RDKit
descriptors plus a 1024-bit Morgan fingerprint of radius 2 [30]. It adapts the companion RF
approach [20], whose classification campaign used fingerprints alone, different splits, and
unweighted classes.

**Uncertainty methods.** On the four classification endpoints, the *deep ensemble* [10]
averages five single-task GINs trained with different seeds (initialization, minibatch
order, dropout and DropEdge masks), each temperature-scaled [11] on the validation set. Two
single-model baselines sit beside it. The *single GIN* is member 0 with its own temperature.
*MC dropout* [12] averages 30 stochastic passes of member 0 in training mode, without
temperature scaling; because training mode also activates DropEdge and BatchNorm batch
statistics and updates running statistics, it is a combined stochastic baseline rather than
a controlled dropout-only comparison. RF probabilities are used as produced.

**Statistics.** Paired differences between conditions that share a split are reported as
mean ± sample s.d. over seeds, with the number of seeds favoring one condition. Five seeds
leave a single cell underpowered, so consistency of direction is read alongside the mean.
The single-split uncertainty results carry no seed-level error bars.

<table style="margin-left:0; margin-right:auto;">
<thead><tr><th style="text-align:left">group</th><th style="text-align:left">setting</th><th style="text-align:left">value</th></tr></thead>
<tbody>
<tr><td style="text-align:left">Encoder</td><td style="text-align:left">GIN layers / hidden width / pooling</td><td style="text-align:left">4 / 128 / sum</td></tr>
<tr><td style="text-align:left"></td><td style="text-align:left">dropout / DropEdge / PairNorm scale</td><td style="text-align:left">0.3 / 0.1 / 1.0</td></tr>
<tr><td style="text-align:left">Optimization</td><td style="text-align:left">optimizer</td><td style="text-align:left">Adam, learning rate 5e-4, weight decay 5e-4, batch 128, 150 epochs</td></tr>
<tr><td style="text-align:left">Ensemble</td><td style="text-align:left">members / calibration</td><td style="text-align:left">5 single-task GINs / per-member temperature scaling</td></tr>
<tr><td style="text-align:left">Decision metrics</td><td style="text-align:left">ECE bins / conformal target / selective point</td><td style="text-align:left">10 / 90% / 70% coverage</td></tr>
<tr><td style="text-align:left">SPRT</td><td style="text-align:left">$\alpha$ / $\beta$ / design effect / subgroup</td><td style="text-align:left">0.05 / 0.20 / 0.30 s.d. / top 30%</td></tr>
</tbody>
</table>
'''))

CELLS.append(md(r'''
## System architecture

Figure 0 shows how the pieces connect and, just as usefully, what stays separate. The
learned oracle takes the RF's place at the head of the companion engine, and the campaign
reuses the companion SPRT rule with ensemble probabilities defining its subgroup. Conformal
and selective metrics are reported and logged alongside, but they don't govern acquisition
or stopping. The companion study's abductive rule-discovery component isn't re-run here.
'''))

CELLS.append(code(r'''
"""Display Figure 0, the system schematic (fig0_architecture_v2.png, from architecture_figure_v2.py)."""
show_fig("fig0_architecture_v2.png", width=1100)
'''))

CELLS.append(md(r'''
<sub><strong>Figure 0.</strong> Study architecture. Single-task and multi-task GINs are compared across six endpoints (§4.1); five single-task members form the classification ensemble (§4.3). Probability rankings select campaign subgroups for retrospective SPRT stopping (§4.5). Conformal and selective metrics are assessed and logged separately, rather than used to govern stopping. The GAT score is examined as an implementation diagnostic (§4.6). RF is the comparator; DuckDB records decision summaries. The dashed active-learning loop and companion abductive component are outside this experiment.</sub>
'''))

# --------------------------------------------------------------------------- 4.1
CELLS.append(md(r'''
## 4.1 Multitask and single-task performance across endpoints

**Question.** Does sharing one encoder trunk across six endpoints help any of them, compared
with a single-task GIN trained under the same within-endpoint splits?

**Objective, and why the answer needs qualifying.** Let $\phi_\theta$ be the shared encoder
and $h_t$ the head for endpoint $t$. The multi-task objective is the task-weighted risk

$$
\mathcal{L}(\theta) \;=\; \sum_{t} w_t \, \mathbb{E}_{(x,y)\sim\mathcal{D}_t}
      \big[\ell_t\!\left(h_t(\phi_\theta(x)),\, y\right)\big], \qquad w_t = 1,
$$

optimized by round-robin task batches. Multi-task learning can share information or incur
conflicting task gradients [24], and these experiments don't identify which. Two details of
the implementation matter more. First, smaller task loaders are restarted until the largest
finishes each epoch, whereas a single-task loader is visited once; at full data, seed 0, HIA
receives 7,650 target-task updates against 450 single-task updates, and Caco2 7,650 against
750. Second, the shared trunk also sees auxiliary records, including target test structures
(Table A1). Task size, chemistry, and training exposure all move together, so whatever
pattern appears can't be pinned on endpoint size or on representation transfer alone.

The cell below recomputes Table 1 from per-seed metrics and checks where the RF fails to
lead. The diagnostics that follow reconstruct record counts and update exposure from stored
training counts, and Table A1 displays the committed overlap measurements.
'''))

CELLS.append(code(r'''
"""Table 1: paired multi-task minus single-task differences, from the per-seed curves.

learning_curves.npz stores one test score per (training fraction, seed) for each endpoint,
model family, and metric. Single- and multi-task runs share each seed's split, so the
difference is paired. Each cell is mean ± sample s.d. over seeds, with the number of seeds in
which multi-task is better (higher AUROC, lower MAE). Endpoints are ordered by size.
"""
lc = np.load(ART / "learning_curves.npz")
p1 = json.loads((ART / "multitask_metrics.json").read_text())
fracs = lc["_fractions"]
n_seeds = len(lc["_seeds"])
METRIC = {k: ("auroc" if p1["summary"][k]["task"] == "clf" else "mae") for k in NAMES}

rows, pairs = {}, {}
for k in sorted(NAMES, key=SIZES.get):
    m = METRIC[k]
    d = lc[f"{k}__gnn_multi__{m}"] - lc[f"{k}__gnn_single__{m}"]   # (fractions, seeds)
    better = d > 0 if m == "auroc" else d < 0
    label = f"{NAMES[k]} ({m.upper()} {'↑' if m == 'auroc' else '↓'})"
    rows[label] = {"n": SIZES[k], **{
        f"f = {f:.2f}": f"{fmt(d[i].mean())} ± {fmt(d[i].std(ddof=1), signed=False)} "
                        f"({better[i].sum()}/{n_seeds})"
        for i, f in enumerate(fracs)}}
    pairs[NAMES[k]] = int(better.sum())
display(pd.DataFrame(rows).T)
print(f"seed-fraction pairs in which multi-task is better (of {len(fracs) * n_seeds}): "
      + ", ".join(f"{k} {v}" for k, v in pairs.items()))

# Where does the RF fail to beat both graph models on the seed mean?
exceptions = []
for k, m in METRIC.items():
    for i, f in enumerate(fracs):
        rf, st, mt = (lc[f"{k}__{fam}__{m}"][i].mean() for fam in ("rf", "gnn_single", "gnn_multi"))
        if (m == "auroc" and rf < max(st, mt)) or (m == "mae" and rf > min(st, mt)):
            exceptions.append(f"{NAMES[k]} at f = {f:.2f} (RF {rf:.3f}, single {st:.3f}, multi {mt:.3f})")
print("RF fails to lead at:", "; ".join(exceptions) or "none")
'''))

CELLS.append(code(r'''
"""Training-count diagnostics from committed records; no training or new resampling.

Reconstruct retained record fractions and optimizer update counts using n_train and the
recorded batch size/epochs. A task's single-task updates are epochs * ceil(n_train/batch).
Each multi-task endpoint receives epochs * max_task ceil(n_train/batch) updates.
"""
train_counts = {(r["endpoint"], round(r["fraction"], 2), r["seed"]): r["n_train"]
                for r in p1["records"] if r["method"] == "gnn_single"}
seeds = p1["config"]["seeds"]
rows = {}
for k in NAMES:
    row = {}
    for f in p1["config"]["train_fractions"]:
        counts = np.array([train_counts[k, f, s] for s in seeds])
        retained = np.array([train_counts[k, f, s] / train_counts[k, 1.0, s] for s in seeds])
        row[f"f = {f:.2f}: train records (range)"] = f"{counts.min()}–{counts.max()}"
        row[f"f = {f:.2f}: retained record fraction (range)"] = f"{retained.min():.3f}–{retained.max():.3f}"
    rows[NAMES[k]] = row
print("f denotes retained training scaffolds, not retained records:")
display(pd.DataFrame(rows).T)

batch_size, epochs = p1["config"]["batch_size"], p1["config"]["epochs"]
steps0 = {k: int(np.ceil(train_counts[k, 1.0, 0] / batch_size)) for k in NAMES}
multi_steps0 = epochs * max(steps0.values())
print("Full-data seed 0: optimizer updates on each endpoint's labels")
display(pd.DataFrame({NAMES[k]: {
    "train records": train_counts[k, 1.0, 0],
    "single-task updates": epochs * steps0[k],
    "multi-task target updates": multi_steps0,
    "update ratio": round(multi_steps0 / (epochs * steps0[k]), 1),
} for k in NAMES}).T)
'''))

CELLS.append(md(r'''
**Reading Table 1.** The answer is: some endpoints, and not the ones a dataset-size story
would predict. Caco2 is the clear beneficiary, with lower multi-task MAE at every retained
training-scaffold fraction (18 of 20 seed-fraction pairs) and its largest difference at full
data. HIA is higher consistently only at f = 0.50 (+0.071 ± 0.032, 5 of 5 seeds) and slightly
lower at f = 0.10 and at full data. hERG leans positive at every fraction but by amounts
small against their seed spread (13 of 20 pairs). At full data the shared trunk costs the two
largest endpoints: multi-task AMES AUROC is lower by 0.049 and Solubility MAE higher by
0.145, with no seed improving in either. BBB starts behind at small scaffold fractions
(−0.030 at f = 0.10, 0 of 5 seeds) and is level by full data. These are descriptive protocol
comparisons across five split/initialization seeds, not estimates of a pure sharing effect,
and the fractions themselves are not label budgets: Solubility at f = 0.10 retains 3,330 of
7,938 full-training records in seed 1 (42.0%).
'''))

CELLS.append(code(r'''
"""Table A1: how much of each endpoint's test set the multi-task trunk sees via other endpoints.

Splits are drawn per endpoint, so a model trained on all six endpoints also trains on other
endpoints' molecules. scaffold_overlap.py reproduces the training splits exactly and records,
per target endpoint and fraction, the share of test molecules that appear verbatim (same
canonical SMILES) or share a Murcko scaffold with any other endpoint's training set.
"""
ov = json.loads((ART / "scaffold_overlap.json").read_text())["results"]
display(pd.DataFrame({
    NAMES[k]: {f"f = {f}": f"{ov[k][f]['test_molecule_seen_mean']:.3f} / {ov[k][f]['test_scaffold_seen_mean']:.3f}"
               for f in ("0.10", "0.25", "0.50", "1.00")}
    for k in sorted(NAMES, key=SIZES.get)}).T)
print("cells: same molecule / same scaffold, mean over five seeds")
'''))

CELLS.append(md(r'''
**Reading Table A1.** The single-task baseline never sees these molecules; the multi-task
trunk does, though never with the target endpoint's labels. Exposure grows with the training
fraction, from 1-4% of test molecules seen verbatim at f = 0.10 (4-14% sharing a scaffold) to
10-23% (38-65%) at full data. That is why the gains in Table 1 can't be attributed to transfer
alone and may be optimistic; a globally scaffold-disjoint split could show smaller
differences or, in principle, larger ones. Nor do low overlap fractions settle the matter on
their own: molecule identity, scaffold overlap, auxiliary-task labels, and unequal update
exposure act together, and the present design doesn't separate them.
'''))

CELLS.append(code(r'''
"""Display Figure 1, performance versus retained scaffold fraction (fig_transfer_curves.png)."""
show_fig("fig_transfer_curves.png", width=980)
'''))

CELLS.append(md(r'''
<sub><strong>Figure 1.</strong> Performance versus retained training-scaffold fraction for RF (grey), single-task GIN (yellow), and multi-task GIN (cyan) on six endpoints. Classification uses AUROC and regression uses MAE on the respective TDC target scale. Points are five-seed means and bands are ±1 sample s.d. Table 1 gives paired differences. Record fractions and target-task update counts are not matched between protocols.</sub>

**Against the baseline.** Whatever multi-task training buys, it buys within a graph-model
family that still trails a well-built descriptor model. The RF leads the seed-mean accuracy
comparison at every scaffold fraction on every endpoint except HIA at f = 0.50, where
multi-task (0.904) and the RF (0.900) differ by less than the seed variability. The result
concerns these specified model and training configurations; it doesn't settle the accuracy
ordering for graph representations in general [21,22].
'''))

# --------------------------------------------------------------------------- 4.2
CELLS.append(md(r'''
## 4.2 Ablation: self-supervised attribute-mask pretraining

**Question.** If a shared trunk helps some endpoints, would a self-supervised warm start help
it further? We adapted the node-level attribute-masking objective of Hu et al. [7]: each
atom is masked independently with probability 0.15 by zeroing its feature row, and the trunk
trains for 40 epochs to predict the masked atomic numbers. The unlabeled corpus concatenates
the six endpoints' training graph lists for each seed and fraction, repeated structures and
all. Each endpoint contributes its own training graphs, though those graphs may coincide
with another endpoint's validation or test structures (Table A1). We then fine-tune under
the same supervised settings and compare pretrained against from-scratch training for
single- and multi-task models at the three low-data fractions.

The batch loss averages cross-entropy over all masked atoms in a minibatch $B$, with
$M_B$ the selected atoms and $\mathcal{V}$ the corpus's atomic-number vocabulary:

$$
\mathcal{L}_{\text{mask}}
   (B) = -\,\frac{1}{|M_B|}\sum_{i\in M_B}
       \log p_\theta\!\left(z_i \mid \widetilde{x(i)}\right), \qquad z_i \in \mathcal{V},
$$

where $\widetilde{x(i)}$ is atom $i$'s corrupted graph. Batches with no masked atoms are
skipped, and individual molecules may have none. Larger molecules contribute more masked
atoms in expectation, so this is not an equal-weight per-molecule loss, and the objective is
deliberately narrower than the full pretraining strategy Hu et al. studied. With 36
comparisons, some will look significant by chance, which is why the cell below applies a
Holm correction [31].
'''))

CELLS.append(code(r'''
"""Table 2: pretrained minus from-scratch, per seed, with paired t-tests and a Holm correction.

The ablation artifact stores every run (endpoint, fraction, seed, condition), so the paired
difference and its sample s.d. are recomputed here. Positive AUROC and negative MAE
differences favor pretraining.
"""
abl = json.loads((ART / "pretrain_ablation.json").read_text())
runs = {(r["endpoint"], r["fraction"], r["method"], r["seed"]): r for r in abl["records"]}
afracs = abl["config"]["train_fractions"]


def paired(k, f, arm):
    # Per-seed (pretrained, scratch) primary-metric arrays for one endpoint, fraction, and arm.
    m = METRIC[k]
    seeds = sorted(s for (e, ff, meth, s) in runs if e == k and ff == f and meth == f"{arm}_scratch")
    pre = np.array([runs[(k, f, f"{arm}_pretrained", s)][m] for s in seeds])
    scr = np.array([runs[(k, f, f"{arm}_scratch", s)][m] for s in seeds])
    return pre, scr


rows, pvals = [], []
for k in NAMES:
    for arm in ("single", "multi"):
        row = {"endpoint": NAMES[k], "arm": arm, "metric": METRIC[k].upper()}
        for f in afracs:
            pre, scr = paired(k, f, arm)
            d = pre - scr
            pvals.append(stats.ttest_rel(pre, scr).pvalue)
            row[f"Δ @ f = {f:.2f}"] = f"{fmt(d.mean())} ± {fmt(d.std(ddof=1), signed=False)}"
        rows.append(row)
display(pd.DataFrame(rows).set_index(["endpoint", "arm"]))

# Holm step-down: reject the i-th smallest p-value while p_(i) < 0.05 / (m - i).
p_sorted = np.sort(pvals)
n_tests, n_holm = len(p_sorted), 0
for i, p in enumerate(p_sorted):
    if p >= 0.05 / (n_tests - i):
        break
    n_holm += 1
print(f"{n_tests} paired comparisons: {int((p_sorted < 0.05).sum())} with uncorrected p < 0.05, "
      f"{n_holm} surviving Holm correction")

f0 = min(afracs)
for arm in ("single", "multi"):
    avg = np.mean([np.mean(np.subtract(*paired(k, f0, arm))) for k in CLF])
    print(f"mean AUROC change over classification endpoints at f = {f0:.2f}, {arm}-task: {fmt(avg)}")
'''))

CELLS.append(md(r'''
**Reading Table 2.** The short answer is that nothing survives scrutiny. Seven of the 36
comparisons reach an uncorrected p < 0.05, with mixed signs (pretraining helps hERG
single-task at f = 0.25 and hurts AMES single-task at the same fraction), and none survives
the Holm correction. The most consistent direction is single-task Solubility, where
pretraining lowers MAE at all three fractions (−0.126 ± 0.060 at f = 0.10). Averaged over the
classification endpoints at the lowest fraction, pretraining moves AUROC by +0.002
(single-task) and +0.005 (multi-task). We resist reading this as equivalence or as the
absence of a pretraining effect: five seeds give limited precision, and the conclusion
applies to this masking variant and training protocol.
'''))

CELLS.append(code(r'''
"""Display Figure 2, the pretraining ablation at the lowest fraction (fig_pretrain_ablation.png)."""
show_fig("fig_pretrain_ablation.png", width=1000)
'''))

CELLS.append(md(r'''
<sub><strong>Figure 2.</strong> Pretraining differences at retained training-scaffold fraction f = 0.10, for single-task (yellow) and multi-task (cyan) models. Separate panels show classification ΔAUROC, Solubility −ΔMAE, and Caco2 −ΔMAE on their respective target scales. Positive values favor pretraining; magnitudes across different metrics or target scales are not comparable. Error bars are ±1 sample s.d. over five seeds. No comparison survives Holm correction.</sub>
'''))

# --------------------------------------------------------------------------- 4.3
CELLS.append(md(r'''
## 4.3 Deep-ensemble uncertainty: scoring rules and calibration

**Question.** The second half of the study turns from accuracy to the quality of the
probabilities, which is where a decision engine actually lives. Which graph uncertainty
method produces the most useful probabilities, and how does it compare with the RF? We
compare a single temperature-scaled GIN, Monte Carlo dropout, and a K = 5 deep ensemble on
the four classification endpoints, on one fixed scaffold split.

**The ensemble's predictive distribution.** For temperature-scaled member probabilities
$p^{(k)}(y\mid x)$, the ensemble predicts $\bar p(y\mid x)=\tfrac1K\sum_k p^{(k)}(y\mid x)$,
whose entropy decomposes into mean member entropy and the mutual information between label
and member index, a term that vanishes when the members agree:

$$
\underbrace{\mathcal{H}[\bar p]}_{\text{total}}
  \;=\;
\underbrace{\tfrac1K\textstyle\sum_k \mathcal{H}[p^{(k)}]}_{\text{mean member entropy}}
  \;+\;
\underbrace{\mathcal{H}[\bar p]-\tfrac1K\textstyle\sum_k \mathcal{H}[p^{(k)}]}_{\text{mutual information}} .
$$

These terms are often read as aleatoric and epistemic uncertainty under additional modeling
assumptions. We don't validate that reading here, and member-level arrays aren't committed;
the reported metrics use $\bar p$ directly rather than the mutual information.

**How the probabilities are scored.** Temperature scaling fits one scalar $T>0$ per member on
the validation set to minimize the NLL of $\mathrm{softmax}(z/T)$, changing confidence but not
the predicted class. Calibration is the top-label expected calibration error over $B = 10$
equal-width confidence bins,

$$
\mathrm{ECE} = \sum_{b=1}^{B}\frac{|\mathcal{B}_b|}{N}\,
      \big|\,\mathrm{acc}(\mathcal{B}_b) - \mathrm{conf}(\mathcal{B}_b)\,\big| ,
$$

alongside two proper scoring rules, the negative log-likelihood
$-\frac1N\sum_i\log\hat p(y_i\mid x_i)$ and the Brier score
$\frac1N\sum_i(\hat p(y{=}1\mid x_i)-y_i)^2$. Proper scores reward calibration and
discrimination together. ECE estimates a binned calibration gap, depends on the binning, is
noisy on test sets this small, and, being unsigned, can't tell underconfidence from
overconfidence.
'''))

CELLS.append(code(r'''
"""Table 3: uncertainty methods averaged over the four classification endpoints (fixed split).

All metrics are read from ensemble_uncertainty.json. Coverage is the realized test coverage
of the nominal 90% conformal sets, shown because set sizes are comparable only at matched
coverage. Below the table, the endpoints on which the ensemble is the best graph model.
"""
unc = json.loads((ART / "ensemble_uncertainty.json").read_text())
res = unc["results"]
METHODS = {"rf": "RF (descriptor baseline)", "gnn_single": "single GIN",
           "mc_dropout": "MC dropout", "ensemble": f"deep ensemble (K = {unc['K']})"}
COLS = {"ece": "ECE ↓", "nll": "NLL ↓", "brier": "Brier ↓", "conf_set_size_at_90": "set size @ nominal 90% ↓",
        "conf_coverage_at_90": "coverage", "aurc": "AURC ↓", "sel_acc_at_70": "sel. acc. @ 70% ↑",
        "test_auroc": "test AUROC ↑"}
table3 = pd.DataFrame({METHODS[m]: {c: np.mean([res[k][m][key] for k in CLF]) for key, c in COLS.items()}
                       for m in METHODS}).T
display(table3.round(3))

graph = ("gnn_single", "mc_dropout", "ensemble")
for key in ("nll", "brier", "aurc", "sel_acc_at_70", "ece"):
    pick = max if key == "sel_acc_at_70" else min
    wins = [NAMES[k] for k in CLF if pick(graph, key=lambda m: res[k][m][key]) == "ensemble"]
    print(f"{COLS[key]:<18s} ensemble is the best graph model on: {', '.join(wins) or 'none'}")
print("\ntest AUROC, RF vs ensemble: " + ", ".join(
    f"{NAMES[k]} {res[k]['rf']['test_auroc']:.3f} vs {res[k]['ensemble']['test_auroc']:.3f}" for k in CLF))
'''))

CELLS.append(code(r'''
"""Independently check stored test probability metrics using ensemble_predictions.npz.

This recomputes ECE, NLL, Brier, AURC, selective accuracy, and AUROC from saved positive-class
probabilities and labels. It cannot refit temperatures or reconstruct conformal thresholds:
validation and member-level predictions are not included in these committed arrays.
"""
preds = np.load(ART / "ensemble_predictions.npz")
checked = 0
for k in CLF:
    y = preds[f"{k}___y_true"].astype(int)
    for method in METHODS:
        p1_test = preds[f"{k}__{method}"]
        p = np.column_stack([1 - p1_test, p1_test])
        conf = p.max(axis=1)
        correct = (p.argmax(axis=1) == y).astype(float)
        order = np.argsort(-conf)
        ece = 0.0
        for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            mask = (conf > lo) & (conf <= hi)
            if mask.any():
                ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        ranks = stats.rankdata(p1_test)
        values = {
            "ece": ece,
            "nll": -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean(),
            "brier": np.mean((p1_test - y) ** 2),
            "aurc": np.mean(np.cumsum(1 - correct[order]) / np.arange(1, len(y) + 1)),
            "sel_acc_at_70": correct[order][:max(1, round(0.7 * len(y)))].mean(),
            "test_auroc": (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg),
        }
        for metric, value in values.items():
            assert np.isclose(value, res[k][method][metric], rtol=0, atol=1e-5), (k, method, metric)
            checked += 1
print(f"{checked} test metrics agree with stored values within 1e-5 numerical tolerance.")
print("Temperature fitting and conformal calibration cannot be reconstructed from these test arrays.")
'''))

CELLS.append(md(r'''
**Reading Table 3.** Among the graph models the ensemble is the one to beat: it has the best
mean NLL (0.482 versus 0.505 single and 0.532 MC dropout), Brier score, AURC, selective
accuracy, and conformal set size. Per endpoint it has the best NLL, Brier score, and AURC on
BBB, AMES, and hERG. HIA is the exception: on that smallest test set (106 molecules, 13
negatives) the single network and MC dropout are better on all three. The ensemble is *not*
the best-calibrated graph model by ECE. Its mean ECE (0.068) sits above both the single
temperature-scaled network (0.052) and the RF (0.053), and it has the lowest ECE on no
endpoint. Averaging individually calibrated members can produce underconfidence [26,27], but
a higher unsigned ECE doesn't by itself establish that explanation; inverse-frequency class
weights also change the learning objective, and a scalar temperature can't generally undo a
class-prior shift. Calibrating after averaging is the obvious next comparison, listed as
such rather than as a demonstrated correction of the observed test ECE.

Against the RF the comparison is not close. The RF has the higher test AUROC on every
endpoint and the better mean NLL, Brier score, AURC, and selective accuracy. One split means
no seed-level error bars, so gaps of a few thousandths, such as the ensemble's and RF's mean
set sizes, stay unresolved.
'''))

CELLS.append(code(r'''
"""Display Figure 3, per-endpoint ECE / NLL / Brier (fig_ensemble_calibration.png)."""
show_fig("fig_ensemble_calibration.png", width=1100)
'''))

CELLS.append(md(r'''
<sub><strong>Figure 3.</strong> Per-endpoint ECE, NLL, and Brier score (lower is better) for the RF (grey), single GIN (yellow), MC dropout (green), and K = 5 deep ensemble (cyan) on the fixed scaffold split. The ensemble has the lowest NLL and Brier score among graph models on BBB, AMES, and hERG but not HIA, and it doesn't have the lowest ECE; the RF is lowest on NLL and Brier throughout.</sub>
'''))

# --------------------------------------------------------------------------- 4.4
CELLS.append(md(r'''
## 4.4 Conformal coverage and selective prediction under scaffold shift

**Question.** A prediction set that promises 90% coverage is only as good as the assumptions
behind the promise. How do coverage, set size, and selective accuracy actually behave in the
implemented probability pipeline? The nominal target is 90%; realized coverage has to be read
alongside set size.

**Split-conformal prediction** [13,14] scores each validation molecule by
$s_i = 1-\hat p(y_i\mid x_i)$ and takes $\hat q$, the
$\lceil(1-\alpha)(n+1)\rceil/n$ empirical quantile of the scores. The set
$C(x)=\{y: 1-\hat p(y\mid x)\le \hat q\}$ satisfies the standard marginal guarantee

$$
\Pr\big(y_{\text{test}} \in C(x_{\text{test}})\big) \;\ge\; 1-\alpha
$$

*if* the scoring function is fixed independently of the calibration labels and the
calibration and test scores are exchangeable; here $\alpha = 0.10$. (The implementation
interpolates the quantile, which returns a value at or above the exact order statistic
whenever $\lceil(1-\alpha)(n+1)\rceil \le n$, as for every calibration set here, so it is
slightly conservative under the stated assumptions. If the rank exceeded $n$, exact conformal
calibration would return all labels; the code instead caps the quantile.)

Both assumptions are in doubt here, for different reasons. The graph pipeline fits
temperatures on the same validation labels it uses for conformal scores, so its scoring
function isn't independent of them. A single binary network gets a reprieve: scalar
temperature scaling preserves score order, exact order-statistic conformal sets are
invariant to it, and the interpolation used here contains those exact sets. The ensemble
gets no such reprieve, since individually scaled and averaged members aren't a monotone
transformation of a fixed score. Then there is exchangeability. Random assignment of whole
scaffold groups doesn't establish individual-molecule exchangeability; unequal group sizes
and grouped sampling need their own justification, and correlation alone neither proves nor
disproves the condition. The RF avoids the label-reuse problem but not the group-sampling
one. So we claim no guarantee for any method and report coverage as what it is: an
empirical outcome. Shift-aware methods [28] weren't applied.

**Selective prediction** [15,16] ranks test molecules by $\max_y \hat p(y\mid x)$ and reports
accuracy on the most confident 70%, together with the area under the risk-coverage curve
(AURC, in Table 3). On imbalanced endpoints this has to be read against the majority-class
rate in the retained subset.
'''))

CELLS.append(code(r'''
"""Table 4: realized coverage, set size, and selective accuracy per endpoint (fixed split).

Test-set size and positive rate come from the committed ensemble predictions. The displayed
binomial standard-error scale is a descriptive independence benchmark, not a valid group-aware
significance test. Coverage/set-size values are read from JSON; calibration scores are absent.
"""
preds = np.load(ART / "ensemble_predictions.npz")
rows = {}
for k in CLF:
    y = preds[f"{k}___y_true"]
    n = len(y)
    cov = res[k]["ensemble"]["conf_coverage_at_90"]
    rows[NAMES[k]] = {
        "test n": n,
        "positive rate": f"{y.mean():.2f}",
        "coverage RF / single / MC / ens.": " / ".join(
            f"{res[k][m]['conf_coverage_at_90']:.3f}" for m in METHODS),
        "ens. coverage − 0.90 (naive SE)": fmt((cov - 0.9) / np.sqrt(0.09 / n), digits=1),
        "set size RF / single / MC / ens.": " / ".join(
            f"{res[k][m]['conf_set_size_at_90']:.3f}" for m in METHODS),
        "sel. acc. @ 70% RF / single / MC / ens.": " / ".join(
            f"{res[k][m]['sel_acc_at_70']:.3f}" for m in METHODS),
    }
display(pd.DataFrame(rows).T)

# The relevant majority baseline is the class composition of each retained subset.
retained_rows = {}
for k in CLF:
    y = preds[f"{k}___y_true"].astype(int)
    for method in METHODS:
        p1_test = preds[f"{k}__{method}"]
        probs = np.column_stack([1 - p1_test, p1_test])
        take = np.argsort(-probs.max(axis=1))[:max(1, round(0.7 * len(y)))]
        positives = int(y[take].sum())
        retained_rows[f"{NAMES[k]} / {METHODS[method]}"] = {
            "retained n": len(take),
            "positive labels": positives,
            "majority accuracy on retained subset": max(positives, len(take) - positives) / len(take),
            "model selective accuracy": (probs[take].argmax(axis=1) == y[take]).mean(),
        }
print("Retained-subset majority baselines (descriptive label composition):")
display(pd.DataFrame(retained_rows).T.round(3))
'''))

CELLS.append(md(r'''
**Reading Table 4.** The promise is 90%; the delivery is 0.906, 0.883, 0.830, and 0.887 on
BBB, AMES, hERG, and HIA for the ensemble. hERG is where the promise slips: the shortfall is
about 2.9 binomial standard errors under an independence benchmark, which is a flag rather
than a group-aware test, and it doesn't say which assumption failed. Nominal coverage simply
can't be presumed for this pipeline. The RF, by contrast, covers at or above target on all
four endpoints (mean 0.918). The single network and the ensemble land within 0.01 of each
other on realized coverage everywhere, and the ensemble's sets are smaller on all four at
roughly the same observed coverage, though we read that as descriptive rather than as
matched-coverage efficiency. Against the RF the comparison is confounded: the ensemble's sets
are smaller on AMES, hERG, and HIA, but each time at lower realized coverage, so part of the
reduction is bought with coverage. On BBB, where coverage matches, the RF's sets are much
smaller (1.024 versus 1.299).

Selective accuracy is the mixed result. The ensemble is the best graph model on BBB and AMES,
the worst on hERG (0.785 versus 0.804 single and 0.794 MC dropout), and ties the single
network below MC dropout on HIA; its higher mean in Table 3 rides on the first two endpoints.
BBB and HIA are 83% and 88% positive overall, so retained-subset prevalence is the honest
comparison. For HIA the RF retains 74 records, 73 of them positive, and predicts 73 correctly,
which an always-positive rule on that subset also achieves: 73/74 = 0.986. The ensemble
retains a different subset, 65 positives among 74 records, and predicts 69 correctly (0.932).
Subset composition and prediction quality both move these numbers. The RF has the highest
selective accuracy on all four endpoints.
'''))

CELLS.append(code(r'''
"""Display Figure 4, conformal set size and selective accuracy per endpoint (fig_conformal_efficiency.png)."""
show_fig("fig_conformal_efficiency.png", width=1000)
'''))

CELLS.append(md(r'''
<sub><strong>Figure 4.</strong> Left: mean prediction-set size at a nominal 90% target. Center: realized coverage, with a dashed line at 0.90. Right: accuracy on the 70% most confident test records. RF (grey), single GIN (yellow), MC dropout (green), deep ensemble (cyan). Interpret set sizes alongside coverage (Table 4), and selective accuracy alongside retained-subset class composition. The ensemble's calibration-label reuse and unestablished molecule-level score exchangeability prevent claiming the ordinary guarantee for this protocol.</sub>
'''))

# --------------------------------------------------------------------------- 4.5
CELLS.append(md(r'''
## 4.5 Sequential go/no-go campaign with the learned oracle

**Question.** The companion engine now gets a turn. With the ensemble probabilities defining a
subgroup, how many measurements does its sequential test need to reach a go/no-go call,
compared with a fixed-sample design?

**The test.** For each classification endpoint, test molecules are ranked by the ensemble's
probability of the positive class and the top 30% form the candidate subgroup. Labels are
standardized against the whole test pool, and the subgroup's standardized labels $x_i$ are
revealed one at a time in a random order. Wald's SPRT [1] models $x_i\sim\mathcal{N}(\theta,1)$
and tests $H_0:\theta=0$ against $H_1:\theta=\theta_1=0.30$, accumulating

$$
\Lambda_n = \sum_{i=1}^{n} \theta_1\Big(x_i - \frac{\theta_1}{2}\Big),
\qquad
B=\log\frac{\beta}{1-\alpha} \;<\; \Lambda_n \;<\; A=\log\frac{1-\beta}{\alpha},
$$

and stopping with GO when $\Lambda_n \ge A$ or NO-GO when $\Lambda_n \le B$, at
$(\alpha,\beta)=(0.05,0.20)$. In the simple-hypothesis i.i.d. setting the SPRT carries an
expected-sample-size optimality result [2], which the present campaign does not satisfy. The
comparator is the one-sided fixed-sample $z$-test's continuous sample-size requirement,
$n_{\text{fixed}} = \big((z_{1-\alpha}+z_{1-\beta})/\theta_1\big)^2 \approx 68.7$. Only subgroups of
at least $\lceil n_{\text{fixed}}\rceil=69$ records can run that design. We keep the continuous
approximation, 68.7, for comparison with the stored artifacts; it is not an attainable count of
measurements. For smaller subgroups the stored comparator caps the approximation at subgroup
size, which yields less than nominal power under the working model.

**What a GO means, and what this run is.** A GO means that, in this measurement order, the
working-model likelihood ratio crossed the boundary favoring a 0.30 s.d. shift over no shift,
and nothing more. The subgroup's full-data label mean describes its enrichment relative to the
pool; the boundary crossing is not a validated discovery claim. Positive labels for AMES and
hERG denote mutagenicity and blockade, so a GO there flags adverse-phenotype enrichment, not a
desirable drug-selection outcome. And the run is retrospective through and through: labels are
standardized with the test pool's mean and standard deviation, and the test direction is taken
from the whole subgroup's mean, both of which use labels a real campaign wouldn't yet have.
The measurements are standardized binary labels drawn without replacement, so the Gaussian
likelihood is a working model and Wald's error bounds are nominal. The probability ranking
determines the subgroup; conformal, calibration, and selective metrics are logged separately
and do not determine measurements or stopping. There is no matched campaign comparison between
uncertainty methods. Each stopping count is one realization from one random order, subject to
the HIA feasibility constraint below.
'''))

CELLS.append(code(r'''
"""Table 5: the campaign per endpoint, with the fixed-sample comparator rebuilt from first principles.

Verdicts, sample sizes, and observed shifts come from campaign_learned.json. As a check, the
subgroup size (top 30% of the test set, minimum 8) is recomputed from the committed predictions
and the fixed-sample requirement from the z-test formula, capped at the subgroup.
"""
from scipy.stats import norm

camp = json.loads((ART / "campaign_learned.json").read_text())
cfg = camp["sprt"]
n_fixed = ((norm.ppf(1 - cfg["alpha"]) + norm.ppf(1 - cfg["beta"])) / cfg["meaningful_effect_sigma"]) ** 2
print(f"oracle: {camp['oracle']}")
print(f"H0: theta = 0 vs H1: theta = {cfg['meaningful_effect_sigma']} pool s.d.; alpha = {cfg['alpha']}, "
      f"beta = {cfg['beta']}; continuous fixed-sample requirement = {n_fixed:.1f} "
      f"(integer design: {int(np.ceil(n_fixed))})\n")

VERDICT = {"reject_h0": "GO", "accept_h0": "NO-GO", "continue": "undecided"}
rows = {}
for k, r in camp["per_endpoint"].items():
    d = r["sprt"]
    n_test = len(preds[f"{k}___y_true"])
    subgroup = max(8, int(cfg["top_fraction"] * n_test))
    assert abs(min(n_fixed, subgroup) - d["fixed_sample_n"]) < 0.05, k  # comparator reproduces
    rows[NAMES[k]] = {"test n": n_test, "subgroup": subgroup, "shift (s.d.)": f"{d['effect_observed']:.3f}",
                      "verdict": VERDICT[d["decision"]], "SPRT n": d["n_measurements"],
                      "fixed n (approx.)": f"{d['fixed_sample_n']:.1f}",
                      "difference (approx.)": fmt(d["measurements_saved"], digits=1)}
display(pd.DataFrame(rows).T)

s = camp["summary"]
feasible = [k for k in camp["per_endpoint"]
            if max(8, int(len(preds[f"{k}___y_true"]) * cfg["top_fraction"])) >= np.ceil(n_fixed)]
used_f = sum(camp["per_endpoint"][k]["sprt"]["n_measurements"] for k in feasible)
fixed_f = sum(camp["per_endpoint"][k]["sprt"]["fixed_sample_n"] for k in feasible)
print(f"Where the nominal fixed design is feasible ({', '.join(NAMES[k] for k in feasible)}): "
      f"{used_f} measurements versus {fixed_f:.1f} continuous requirement "
      f"({100 * (1 - used_f / fixed_f):.1f}% below the approximation)")
print(f"with hERG and HIA capped at their subgroup size (lower power): {s['total_measurements_used']} versus "
      f"{s['total_fixed_sample']} capped approximation ({s['pct_saved']}% below); "
      f"{s['n_go_decisions']}/{s['n_endpoints']} GO")
# The stored shift is |subgroup mean|; recompute the signed mean to see which direction was chosen.
signs = {}
for k in camp["per_endpoint"]:
    y = preds[f"{k}___y_true"].astype(float)
    z = (y - y.mean()) / (y.std() + 1e-9)
    top = np.argsort(-preds[f"{k}__ensemble"])[:max(8, int(cfg["top_fraction"] * len(y)))]
    signs[NAMES[k]] = "positive" if z[top].mean() >= 0 else "negative"
print("data-chosen test direction:", ", ".join(f"{k} {v}" for k, v in signs.items()))

# Analytical feasibility from the stored HIA labels, with no order simulations.
hy = preds["hia___y_true"].astype(float)
h_top = np.argsort(-preds["hia__ensemble"])[:max(8, int(cfg["top_fraction"] * len(hy)))]
theta = cfg["meaningful_effect_sigma"]
upper = np.log((1 - cfg["beta"]) / cfg["alpha"])
lower = np.log(cfg["beta"] / (1 - cfg["alpha"]))
positive_z = (1 - hy.mean()) / (hy.std() + 1e-9)
negative_z = -hy.mean() / (hy.std() + 1e-9)
positive_increment = theta * (positive_z - theta / 2)
negative_increment = theta * (negative_z - theta / 2)
max_budget_lr = len(h_top) * positive_increment
lowest_possible_lr = int((hy[h_top] == 0).sum()) * negative_increment
assert max_budget_lr < upper and lowest_possible_lr > lower
print(f"HIA feasibility: even all {len(h_top)} measurements positive give log LR "
      f"≤ {max_budget_lr:.3f} < GO boundary {upper:.3f}; "
      f"at least {int(np.ceil(upper / positive_increment))} positive measurements are needed.")
print(f"The observed HIA subgroup has {int(hy[h_top].sum())} positives and "
      f"{int((hy[h_top] == 0).sum())} negative; its minimum possible log LR "
      f"{lowest_possible_lr:.3f} > NO-GO boundary {lower:.3f}.")
print("\nDuckDB lineage (regenerated by e8_campaign_learned.py; experiment IDs recorded in the JSON):")
for k, r in camp["per_endpoint"].items():
    print(f"  {NAMES[k]:5s} experiment {r['experiment_id'][:8]}  {VERDICT[r['sprt']['decision']]}")
'''))

CELLS.append(md(r'''
**Reading Table 5.** The SPRT returns GO on BBB, AMES, and hERG. On BBB and AMES it spends 105
measurements, 23.6% below their continuous fixed-sample requirement of 137.4, and AMES carries
the reduction (20 against 68.7) while BBB uses 85. Its observed subgroup shift, 0.254, sits
below the design alternative of 0.30, a useful reminder that a GO does not certify an effect
above the design value. hERG reaches GO after 14 of 45 subgroup records.

HIA's indecision isn't bad luck; it is structurally forced under this implementation. With 93
positives among 106 pool records, a positive standardized label is at most $\sqrt{13/93}=0.374$.
Thirty-one positives could accumulate at most 2.082 log likelihood units, below the GO boundary
$\log16=2.773$, and even an all-positive stream needs at least 42 measurements. The actual
subgroup has 30 positives and one negative and cannot reach either boundary in any order. Its
31-record exhaustion therefore says nothing about order-dependent stopping efficiency.

The capped total is 150 measurements against an approximate requirement of 213.4 (29.7% below),
with lower-power hERG and HIA comparators, so it is not a comparison at the same nominal power.
The chosen direction is positive on every endpoint, so these realized decisions equal those
using a pre-specified positive direction; the test-pool standardization still looks ahead.
DuckDB logs summary parameters and experiment IDs. The stored probability spread across pool
records is not ensemble epistemic uncertainty, and the log holds neither member checkpoints nor
a measurement-by-measurement likelihood-ratio trace.
'''))

CELLS.append(code(r'''
"""Display Figure 5, measurements to a decision per endpoint (fig_campaign_efficiency.png)."""
show_fig("fig_campaign_efficiency.png", width=980)
'''))

CELLS.append(md(r'''
<sub><strong>Figure 5.</strong> Retrospective stopping counts (cyan circles) versus the continuous fixed-sample requirement under the Gaussian working model (filled grey squares). The uncapped requirement is 68.7, corresponding to a 69-record integer design; open grey squares mark capped, underpowered hERG and HIA comparators. BBB and AMES use 105 measurements against an approximate requirement of 137.4 (23.6% below). HIA exhausts its budget undecided and cannot reach GO within that budget. Probability rankings define subgroups; uncertainty metrics do not govern stopping.</sub>
'''))

# --------------------------------------------------------------------------- 4.6
CELLS.append(md(r'''
## 4.6 Attention-score implementation diagnostic

**Question.** Does the proposed atom score actually rank anything? The historical probe
trained a separate three-layer GAT [4] on hERG and BBB (hidden width 64; four heads in the
first two layers and one in the last), selected by validation AUROC in an unseeded run. It
defined each atom's score as the sum of incoming final-layer attention coefficients and
compared that score with an aromatic-or-nitrogen rule.

In evaluation mode, GAT normalizes those incoming coefficients over each destination's
neighborhood, including its self-loop [38]. So

$$
a_i = \sum_{j\in\mathcal{N}(i)\cup\{i\}} \alpha_{ji}^{(L)} = 1.
$$

The score is constant by construction; the saved arrays depart from one only at
floating-point precision. The archived rule AUROCs and top-three enrichments therefore
reflect roundoff and tie-breaking, and we withdraw them as evidence about salience or
attention faithfulness [19,29]. None of this touches the separate GAT's predictive AUROC.
The saved atoms come from 148 of 153 hERG and 370 of 458 BBB test records after the original
small-graph and uniform-rule exclusions; no corrected attribution experiment was run.
'''))

CELLS.append(code(r'''
"""Reconstruct the attention-score degeneracy diagnostic from committed atom arrays.

Score spread is the descriptive population spread over saved atoms, not a seed-level
uncertainty estimate. Predictive GAT AUROC is read from the archived metric JSON.
"""
att = json.loads((ART / "attention_attribution.json").read_text())
attention_nodes = np.load(ART / "attention_nodes.npz")
display(pd.DataFrame({NAMES[k]: {
    "eligible test records": r["n_test_molecules"],
    "saved atoms": len(attention_nodes[f"{k}__scores"]),
    "GAT test AUROC": f"{r['test_auroc']:.3f}",
    "minimum score − 1": f"{(attention_nodes[f'{k}__scores'].min() - 1):.3e}",
    "maximum score − 1": f"{(attention_nodes[f'{k}__scores'].max() - 1):.3e}",
    "score population s.d.": f"{attention_nodes[f'{k}__scores'].std(ddof=0):.3e}",
} for k, r in att["results"].items()}).T)
print("Historical rule-agreement metrics remain archived but have no salience interpretation.")
'''))

CELLS.append(md(r'''
**Reading the diagnostic.** The 3,709 saved hERG scores have population standard deviation
$3.58\times10^{-8}$ and the 8,454 BBB scores have $3.33\times10^{-8}$, both pinned near one.
That is numerical dust, not learned atom importance. The GAT's predictive AUROC is untouched,
0.770 on hERG and 0.756 on BBB, from one unseeded split. Predictive performance and
attribution validity are separate questions, and this implementation answers only the first:
it supplies no interpretable atom ranking.
'''))

CELLS.append(code(r'''
"""Display Figure 6, the incoming-attention score diagnostic (fig_attention_probe.png)."""
show_fig("fig_attention_probe.png", width=1000)
'''))

CELLS.append(md(r'''
<sub><strong>Figure 6.</strong> Attention-score implementation diagnostic. Left: histograms of saved incoming-attention sums minus one, displayed in units of 10<sup>−8</sup> (hERG yellow, BBB cyan), with atom counts and descriptive population standard deviations. Right: archived predictive GAT test AUROC (cyan). The proposed atom score equals one by normalization; its numerical spread cannot support a salience ranking. No corrected attribution experiment is reported.</sub>
'''))

# --------------------------------------------------------------------------- 4.7
CELLS.append(md(r'''
## 4.7 Summary of results

Table 6 brings the per-endpoint results together: full-data accuracy for the three model
families over five seeds, the full-data multi-task difference, and the ensemble's decision
metrics on the fixed split. The fraction-resolved transfer differences are in Table 1.
'''))

CELLS.append(code(r'''
"""Table 6: full-data accuracy (mean ± sample s.d. over seeds) and ensemble decision metrics."""
rows = {}
for k in NAMES:
    m = METRIC[k]
    full = {fam: lc[f"{k}__{fam}__{m}"][-1] for fam in ("rf", "gnn_single", "gnn_multi")}
    row = {"records": SIZES[k], "metric": f"{m.upper()} {'↑' if m == 'auroc' else '↓'}"}
    for fam, label in (("rf", "RF"), ("gnn_single", "single-task"), ("gnn_multi", "multi-task")):
        row[label] = f"{full[fam].mean():.3f} ± {full[fam].std(ddof=1):.3f}"
    row["multi−single"] = fmt(full["gnn_multi"].mean() - full["gnn_single"].mean())
    e = res[k]["ensemble"] if k in res else None
    for key, label in (("ece", "ens. ECE ↓"), ("conf_coverage_at_90", "coverage"),
                       ("conf_set_size_at_90", "set @ nominal 90% ↓"), ("sel_acc_at_70", "sel. acc. @ 70% ↑")):
        row[label] = f"{e[key]:.3f}" if e else "–"
    rows[NAMES[k]] = row
display(pd.DataFrame(rows).T)
'''))

CELLS.append(md(r'''
**Reading the summary.** One model wins the accuracy column outright: the RF has the best mean
full-data accuracy on every endpoint here. The multi-task protocol earns lower Caco2 MAE and
some higher HIA AUROCs, but unequal update exposure and cross-endpoint structural overlap keep
us from crediting transfer. The ensemble takes the best mean graph-model proper scores and
AURC on the fixed split, while ECE and selective accuracy swing by endpoint, and its prediction
sets lack the ordinary split-conformal validity conditions. The campaign logs retrospective
SPRT outcomes from probability-ranked subgroups; it does not demonstrate a causal benefit of
calibrated uncertainty for stopping. HIA is analytically undecidable within its subgroup budget.
'''))

# --------------------------------------------------------------------------- synopsis
CELLS.append(md(r'''
## Synopsis, limitations, and how to regenerate

**What was shown.** The results describe specified graph-model training protocols, their
probability metrics, and one retrospective sequential campaign.

- **Model comparisons (4.1).** Multi-task Caco2 MAE is lower at every retained scaffold
  fraction (18 of 20 seed-fraction pairs), and HIA AUROC is higher at f = 0.50 (5 of 5 seeds).
  AMES and Solubility are worse at full data. Update exposure and cross-endpoint overlap
  confound attribution to shared representations.
- **Pretraining (4.2).** None of 36 comparisons for this node-masking variant survives Holm
  correction; that is not evidence of equivalence.
- **Probability metrics (4.3).** The K = 5 ensemble has the best mean graph-model NLL,
  Brier score, and AURC. It does not have the best ECE; RF leads on accuracy and proper scores.
- **Coverage and abstention (4.4).** Ensemble coverage is 0.830-0.906 against a nominal 0.90
  target. Set-size differences require coverage context, and selective accuracy requires
  retained-subset class-composition context; HIA RF's selective accuracy equals the
  always-positive baseline on its retained subset.
- **Stopping (4.5).** BBB and AMES use 105 measurements against a continuous fixed-sample
  requirement of 137.4; the reduction comes from AMES. This is one retrospective order
  under a working likelihood. HIA cannot reach either boundary within its observed budget.
- **Implementation diagnostic (4.6).** Incoming GAT attention sums equal one. Archived
  rule-agreement results are not interpretable; predictive GAT AUROC is a separate result.

**What the evidence does not establish.**

- This is a retrospective simulation over public benchmarks; no wet-lab loop is closed and no
  prospective claim is made.
- Graph models are not more accurate than the descriptor baseline here; the RF leads on every
  endpoint at full data.
- Scaffold splits are drawn per endpoint, and multi-task training restarts smaller loaders.
  Structural exposure through other endpoints (Table A1) and extra target-label updates
  prevent isolating transfer. Retained scaffold fractions are not fractions of labels.
- Five seeds give low power for single cells, and the 24 transfer comparisons are not corrected
  for multiplicity; the uncertainty study and the campaign rest on one split with no seed-level
  error bars, and the hERG and HIA test sets have 153 and 106 molecules.
- Members are temperature-scaled before averaging. Underconfidence is a plausible hypothesis,
  not established by unsigned ECE; calibration after averaging remains an unevaluated
  comparison. A multi-task ensemble was not evaluated.
- The MC dropout baseline is not temperature-scaled, and its training-mode passes also activate
  DropEdge and batch-statistics normalization, so it may understate a tuned dropout baseline.
  Atomic number enters the encoder as a scalar rather than one-hot, and the GIN ignores bond
  attributes.
- Conformal coverage is empirical: temperatures and thresholds reuse validation labels,
  individual-molecule exchangeability has not been established by group splitting, and
  hERG coverage is 0.830.
- The campaign is retrospective: it standardizes with test-pool statistics and picks its
  direction from the whole subgroup, treats standardized binary labels as Gaussian, samples
  without replacement, and reports one random order, so its error rates are nominal. The
  nominal fixed design is feasible only for BBB and AMES. Probability ranking, rather than
  uncertainty metrics, governs subgroup selection. GO denotes a likelihood-ratio crossing;
  AMES/hERG positive-class enrichment is adverse, and HIA is structurally undecidable here.
- The historical attention score is degenerate by normalization. No corrected atom-attribution
  experiment is reported. The companion engine's abductive discovery was not re-run.

**Future studies.** Matched target-update exposure and globally disjoint scaffolds could
clarify the multi-task comparison. Separate temperature-fitting and conformal-calibration
sets, a controlled dropout-only baseline, more evaluation splits, an implementable binary-label
sequential design, and a validated nonconstant atom-attribution score would address specific
limitations. These are prospective experiments, not completed analyses or requirements for
reconstructing the present results.

**CPU reconstruction.** With both regeneration switches off, this notebook reads committed
artifacts on CPU. Tables 1, 2, and 6 recompute summaries from per-seed metrics; Tables 3 and 4
aggregate stored metrics, and a separate cell checks test-probability metrics against the
saved arrays. Table 5 reconstructs logged outcomes and checks its analytic comparator and
HIA budget; it does not rerun the campaign. Figure 6 diagnoses the saved score arrays.
Validation/member predictions, temperatures, conformal thresholds, and checkpoints are not
committed, so calibration and conformal sets cannot be independently regenerated from the
test arrays. Images are displayed from committed files.

**CPU regeneration and full retraining.** Campaign and figure scripts can regenerate files
on CPU. Overlap reconstruction is a separate raw-data step that may fetch TDC data. Full
model retraining is computationally heavier; a GPU is recommended, while the harness also
supports CPU. The historical attention script recreates the invalid score, not a corrected
interpretability experiment. No new model training is needed for this walkthrough.

The repository records Python 3.11.2 and pins PyTDC 1.1.15 plus selected packages;
several other dependencies are version ranges. This host uses Python 3.11.14. The setup cell
prints the executing interpreter so these environments are not presented as identical.

**Reconstruction commands** from the repository root:

```bash
pip install -r requirements.txt                             # CPU dependencies; some version ranges
PYTHONPATH=.:src python preprint/dl_forward/e8_campaign_learned.py # Table 5 + DuckDB lineage (CPU)
python preprint/dl_forward/figures.py                       # Figures 1-6 (CPU)
python preprint/dl_forward/architecture_figure_v2.py        # Figure 0 (CPU)
PYTHONPATH=.:src python preprint/dl_forward/scaffold_overlap.py   # Table A1 (CPU; fetches TDC data)
python preprint/dl_forward/build_notebook.py               # rebuild notebook source
jupyter nbconvert --to notebook --execute --inplace \
    preprint/dl_forward/walkthrough_learned_representations_admet.ipynb

# Optional full retraining; GPU recommended (make install-gpu; PYTHONPATH=.:src)
python preprint/dl_forward/train_multitask_gnn.py           # data for Table 1, Figure 1
python preprint/dl_forward/pretrain_ablation.py --full      # data for Table 2, Figure 2
python preprint/dl_forward/make_ensemble_uncertainty.py     # data for Tables 3-4, Figures 3-4
python preprint/dl_forward/attention_attribution.py         # historical GAT and invalid atom score
```

For the transfer, pretraining, and ensemble scripts, seeds pin NumPy and PyTorch and
`cudnn.deterministic` is set (the attention probe is unseeded); residual GPU nondeterminism
remains. Identical recorded seeds do not ensure identical predictions or guarantee that a
new run falls within the reported seed dispersion. The cell below has independent, default-off
switches for CPU file regeneration and full model retraining.
'''))

CELLS.append(code(r'''
"""Optional file regeneration and full retraining, with separate default-off switches."""
if REGENERATE_CPU or RETRAIN_MODELS:
    import os
    import subprocess
    env = {**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}"}
    if RETRAIN_MODELS:
        for script, args in (("train_multitask_gnn.py", []), ("pretrain_ablation.py", ["--full"]),
                             ("make_ensemble_uncertainty.py", []), ("attention_attribution.py", [])):
            subprocess.run([sys.executable, str(DL / script), *args], cwd=str(ROOT), env=env, check=True)
    if REGENERATE_CPU:
        for script in ("e8_campaign_learned.py", "figures.py", "architecture_figure_v2.py"):
            subprocess.run([sys.executable, str(DL / script)], cwd=str(ROOT), env=env, check=True)
    print("Requested scripts finished; rerun the notebook to read refreshed artifacts.")
else:
    print("Both regeneration switches are False: reading committed artifacts only.")
'''))

CELLS.append(md(r'''
#### Generative AI Disclosure

Generative AI tools, including GitHub Copilot and OpenAI Codex, assisted with code
development, editorial revision, and checks of mathematical and computational consistency.
The author retains responsibility for the manuscript, code, interpretations, and disclosed
limitations. Artifact checks are distinguished from model retraining; identified
implementation problems are reported rather than treated as validated scientific findings.
'''))

CELLS.append(md(r'''
## References

1. Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of Mathematical Statistics*, 16(2), 117-186.
2. Wald, A. and Wolfowitz, J. (1948). Optimum character of the sequential probability ratio test. *Annals of Mathematical Statistics*, 19(3), 326-339.
3. Kipf, T. N. and Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.
4. Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y. (2018). Graph attention networks. *ICLR*.
5. Xu, K., Hu, W., Leskovec, J. and Jegelka, S. (2019). How powerful are graph neural networks? *ICLR*.
6. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O. and Dahl, G. E. (2017). Neural message passing for quantum chemistry. *ICML*.
7. Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V. and Leskovec, J. (2020). Strategies for pre-training graph neural networks. *ICLR*.
8. Zhao, L. and Akoglu, L. (2020). PairNorm: tackling oversmoothing in GNNs. *ICLR*.
9. Rong, Y., Huang, W., Xu, T. and Huang, J. (2020). DropEdge: towards deep graph convolutional networks on node classification. *ICLR*.
10. Lakshminarayanan, B., Pritzel, A. and Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
11. Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
12. Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian approximation: representing model uncertainty in deep learning. *ICML*.
13. Vovk, V., Gammerman, A. and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
14. Angelopoulos, A. N. and Bates, S. (2023). Conformal prediction: a gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494-591.
15. El-Yaniv, R. and Wiener, Y. (2010). On the foundations of noise-free selective classification. *JMLR*, 11, 1605-1641.
16. Geifman, Y. and El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*.
17. Huang, K., Fu, T., Gao, W., et al. (2021). Therapeutics Data Commons: machine learning datasets and tasks for drug discovery and development. *NeurIPS Datasets and Benchmarks*.
18. Bemis, G. W. and Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. *Journal of Medicinal Chemistry*, 39(15), 2887-2893.
19. Jain, S. and Wallace, B. C. (2019). Attention is not explanation. *NAACL-HLT*.
20. Russell, D. R. (2026). Knowing what to measure and when to stop: an autonomous decision engine for molecular property discovery. Preprint, SNPTX. [Manuscript and reproducibility repository](https://github.com/snptx1/snptx-repro-discovery/blob/main/preprint/MANUSCRIPT_calibrated_sequential_discovery.md).
21. Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., Guzman-Perez, A., Hopper, T., Kelley, B., Mathea, M., Palmer, A., Settels, V., Jaakkola, T., Jensen, K. and Barzilay, R. (2019). Analyzing learned molecular representations for property prediction. *Journal of Chemical Information and Modeling*, 59(8), 3370-3388.
22. Jiang, D., Wu, Z., Hsieh, C.-Y., Chen, G., Liao, B., Wang, Z., Shen, C., Cao, D., Wu, J. and Hou, T. (2021). Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptor-based and graph-based models. *Journal of Cheminformatics*, 13, 12.
23. Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K. and Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. *Chemical Science*, 9(2), 513-530.
24. Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.
25. Ramsundar, B., Kearnes, S., Riley, P., Webster, D., Konerding, D. and Pande, V. (2015). Massively multitask networks for drug discovery. *arXiv:1502.02072*.
26. Wu, X. and Gales, M. (2021). Should ensemble members be calibrated? *arXiv:2101.05397*.
27. Rahaman, R. and Thiery, A. H. (2021). Uncertainty quantification and deep ensembles. *NeurIPS*.
28. Tibshirani, R. J., Foygel Barber, R., Candès, E. J. and Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.
29. Wiegreffe, S. and Pinter, Y. (2019). Attention is not not explanation. *EMNLP-IJCNLP*.
30. Rogers, D. and Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742-754.
31. Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
32. Martins, I. F., Teixeira, A. L., Pinheiro, L. and Falcao, A. O. (2012). A Bayesian approach to in silico blood-brain barrier penetration modeling. *Journal of Chemical Information and Modeling*, 52(6), 1686-1697. [doi:10.1021/ci300124c](https://doi.org/10.1021/ci300124c).
33. Xu, C., Cheng, F., Chen, L., Du, Z., Li, W., Liu, G., Lee, P. W. and Tang, Y. (2012). In silico prediction of chemical Ames mutagenicity. *Journal of Chemical Information and Modeling*, 52(11), 2840-2847. [doi:10.1021/ci300400a](https://doi.org/10.1021/ci300400a).
34. Wang, S., Sun, H., Liu, H., Li, D., Li, Y. and Hou, T. (2016a). ADMET evaluation in drug discovery. 16. Predicting hERG blockers by combining multiple pharmacophores and machine learning approaches. *Molecular Pharmaceutics*, 13(8), 2855-2866. [doi:10.1021/acs.molpharmaceut.6b00471](https://doi.org/10.1021/acs.molpharmaceut.6b00471).
35. Hou, T., Wang, J., Zhang, W. and Xu, X. (2007). ADME evaluation in drug discovery. 7. Prediction of oral absorption by correlation and classification. *Journal of Chemical Information and Modeling*, 47(1), 208-218. [doi:10.1021/ci600343x](https://doi.org/10.1021/ci600343x).
36. Sorkun, M. C., Khetan, A. and Er, S. (2019). AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds. *Scientific Data*, 6, 143. [doi:10.1038/s41597-019-0151-1](https://doi.org/10.1038/s41597-019-0151-1).
37. Wang, N.-N., Dong, J., Deng, Y.-H., Zhu, M.-F., Wen, M., Yao, Z.-J., Lu, A.-P., Wang, J.-B. and Cao, D.-S. (2016b). ADME properties evaluation in drug discovery: Prediction of Caco-2 cell permeability using a combination of NSGA-II and Boosting. *Journal of Chemical Information and Modeling*, 56(4), 763-773. [doi:10.1021/acs.jcim.5b00642](https://doi.org/10.1021/acs.jcim.5b00642).
38. PyTorch Geometric contributors (2026). GATConv documentation, version 2.7.0. [Layer implementation](https://pytorch-geometric.readthedocs.io/en/2.7.0/_modules/torch_geometric/nn/conv/gat_conv.html). Accessed 5 October 2026.
'''))


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    nbf.write(nb, str(OUT))
    print(f"wrote {OUT.name} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build()
