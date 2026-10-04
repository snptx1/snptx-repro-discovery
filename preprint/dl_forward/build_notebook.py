"""Regenerable builder for the learned-representations walkthrough notebook.

Authoring the notebook as code keeps it reviewable and lets it be rebuilt whenever the
narrative changes. The notebook is the executable companion to
``MANUSCRIPT_learned_representations_admet.md``: it mirrors the manuscript's results
numbering (4.1-4.7), figure numbers (0-6), and table numbers (1-6), and it recomputes every
table live from the committed artifacts in ``artifacts/`` (Tier 1, CPU). One gated cell
reruns the generating scripts (Tier 2).

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
# Learned molecular representations with calibrated deep-ensemble uncertainty for label-efficient ADMET decisions

<p style="font-size:1.0em; font-style:italic; opacity:0.8; margin-top:2px; margin-bottom:25px;">Guided walkthrough: from a shared graph representation to calibrated go/no-go decisions, with every table recomputed from committed artifacts.</p>

<p style="font-size:0.95em; opacity:0.85; margin-top:2px; margin-bottom:25px;">Daniel R. Russell &middot; Autonomous Discovery Systems, SNPTX &middot; October 2026</p>

### Abstract

Early molecular discovery is limited by the cost of measurement and by a trust gap: models
rarely say how confident they are, or when enough has been measured to make a call. A
companion study [20] built a calibrated sequential decision engine around a random-forest
(RF) descriptor oracle. This walkthrough replaces that oracle with learned graph models and
asks two questions under a pre-specified protocol (Murcko scaffold cold-splits [18] within
each endpoint, every endpoint reported). **First, does a shared graph encoder transfer
across ADMET endpoints?** Across six Therapeutics Data Commons endpoints [17] and five seeds,
a multi-task Graph Isomorphism Network [5] helps the smallest endpoints and hurts the
largest. Caco2 permeability MAE falls by 0.11 to 0.17 at every training fraction (18 of 20
seed-fraction pairs improve), HIA AUROC rises by 0.071 at half the training data (5 of 5
seeds), and hERG gains are directional but within seed variability. On the large AMES and
Solubility endpoints the shared trunk costs accuracy at full data, and BBB shows negative
transfer at low data. Because splits are drawn per endpoint, the shared trunk also sees other
endpoints' molecules, and at full data 10-23% of each endpoint's test molecules appear
verbatim in another endpoint's training set, so these gains may be optimistic and can't be
attributed to transfer alone.
Self-supervised pretraining [7] adds nothing detectable. **Second, does
a deep ensemble turn graph-model uncertainty into better decisions?** On a fixed scaffold
split, a five-member ensemble [10] of temperature-scaled [11] single-task networks gives the
best proper scores of the graph models (mean NLL 0.482 versus 0.505 for a single network and
0.532 for Monte Carlo dropout [12]), the lowest risk-coverage AURC, and the smallest 90%
conformal sets [13,14], though not the lowest expected calibration error. The RF remains the
stronger model on accuracy, NLL, Brier score, AURC, and selective accuracy [15,16]. In a
retrospective run of the engine's sequential probability ratio test [1,2], the ensemble
returns GO on three of four classification endpoints; on the two endpoints where a
fixed-sample design at the same nominal error rates is feasible, the sequential test uses 105
measurements against 137.4. In one unseeded run, graph attention [4] shows near-chance
alignment with a simple atom-salience rule [19].

### How to read this document

This notebook is self-contained. It is also the executable companion to the manuscript of the
same title: its section numbers follow the manuscript's results sections (4.1-4.7) and its
figure and table numbers are the manuscript's (Figures 0-6, Tables 1-6 and A1), so the two can be
cross-referenced, but nothing here assumes the manuscript has been read. Each section states
the question, defines the quantities involved, recomputes the result live from the committed
artifacts, shows the committed figure with its caption, and interprets the result, including
what it does *not* establish. Bracketed numbers refer to the References at the end.
'''))

CELLS.append(md(r'''
## Overview

<div style="display:flex; gap:48px; line-height:1.5;">
<div style="flex:1;">

<b>4.1 &nbsp;Multi-task transfer concentrates on the smallest endpoints</b><br>
<span style="font-size:0.86em; opacity:0.85;">Paired multi-task minus single-task differences over four training fractions and five seeds. Clear gains on Caco2, partial on HIA, directional on hERG; losses on AMES, Solubility, and BBB. Cross-endpoint exposure means the gains may be optimistic (Table 1, Table A1, Figure 1).</span>
<br><br>

<b>4.2 &nbsp;Ablation: self-supervised pretraining</b><br>
<span style="font-size:0.86em; opacity:0.85;">Does attribute-mask pretraining add to supervised sharing? None of 36 paired comparisons survives a Holm correction (Table 2, Figure 2).</span>
<br><br>

<b>4.3 &nbsp;Deep-ensemble uncertainty: scoring rules and calibration</b><br>
<span style="font-size:0.86em; opacity:0.85;">A K = 5 ensemble is the best graph model on NLL, Brier score, and AURC on average, but not on ECE; the RF leads overall (Table 3, Figure 3).</span>
<br><br>

<b>4.4 &nbsp;Conformal coverage and selective prediction under scaffold shift</b><br>
<span style="font-size:0.86em; opacity:0.85;">Realized coverage, set size at matched coverage, and accuracy on the most confident 70%, per endpoint (Table 4, Figure 4).</span>

</div>
<div style="flex:1;">

<b>4.5 &nbsp;Sequential go/no-go campaign with the learned oracle</b><br>
<span style="font-size:0.86em; opacity:0.85;">A retrospective run of Wald's SPRT driven by the ensemble: 3 of 4 GO; 105 versus 137.4 measurements where a fixed design at the same nominal error rates is feasible; every decision logged to DuckDB (Table 5, Figure 5).</span>
<br><br>

<b>4.6 &nbsp;Interpretability probe</b><br>
<span style="font-size:0.86em; opacity:0.85;">In one run, graph attention ranks atoms near chance against an aromatic-or-nitrogen rule although the network is predictive (Figure 6).</span>
<br><br>

<b>4.7 &nbsp;Summary of results</b><br>
<span style="font-size:0.86em; opacity:0.85;">Full-data accuracy for all three model families with the ensemble's decision metrics, per endpoint (Table 6).</span>

</div>
</div>

Before the results, *Data, models, and protocol* summarizes the experimental design and
*System architecture* shows how the pieces connect. The closing *Synopsis* collects what was
shown, what it does not establish, and the commands that regenerate everything.
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
# Molecules per endpoint after featurization (all TDC molecules featurize).
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


RECOMPUTE = False  # Tier 2 switch; see the closing section
print(f"repository: {ROOT.name}   artifacts: {ART.relative_to(ROOT)}   "
      f"figures: {FIG.relative_to(ROOT)}")
'''))

# --------------------------------------------------------------------------- protocol
CELLS.append(md(r'''
## Data, models, and protocol

**Endpoints.** Six ADMET endpoints from the Therapeutics Data Commons [17]: BBB (Martins),
AMES, hERG, and HIA (Hou) as binary classification, scored by AUROC, and Solubility (AqSolDB)
and Caco2 (Wang) as regression, scored by MAE, with 2030, 7278, 655, 578, 9982, and 910
molecules.

**Splits.** Every split is a Murcko scaffold cold-split [18]: unique scaffolds are randomly
permuted and assigned whole to test (20% of scaffolds), validation (20%), and training, so
within an endpoint no evaluation scaffold is seen in training. Splits are drawn independently
per endpoint, so models trained on several endpoints also see other endpoints' training
molecules, some of which overlap a target's test set (measured in §4.1). Training fractions of 0.10, 0.25, 0.50, and 1.00
subsample whole training scaffolds. The transfer and pretraining studies (§4.1-§4.2) repeat
everything over five seeds, each drawing a new split and initialization. The uncertainty
study and the campaign (§4.3-§4.5) use one fixed split (seed 0). Models train for a fixed 150
epochs with no early stopping; the validation set is used for temperature scaling and
conformal calibration.

**Models.** Molecules are graphs with nine atom features (atomic number as a scalar, degree,
formal charge, explicit hydrogens, aromatic and ring flags, one-hot sp/sp2/sp3
hybridization) connected by bonds; bond attributes are computed but the GIN ignores them. The
encoder is a four-layer Graph Isomorphism Network (GIN) [5] with
hidden width 128, batch-normalized MLP updates, sum pooling, PairNorm [8], and DropEdge [9].
The *single-task* model trains one encoder per endpoint; the *multi-task* model shares one
encoder across all six endpoints with a head per endpoint, visiting endpoints round-robin.
Classification losses use inverse-frequency class weights; regression targets are
standardized. The descriptor baseline is a 300-tree random forest on ten RDKit descriptors
plus a 1024-bit Morgan fingerprint of radius 2 [30], the companion study's oracle [20].

**Uncertainty methods.** On the four classification endpoints, the *deep ensemble* [10]
averages five single-task GINs trained with different seeds (initialization, minibatch order,
dropout masks), each temperature-scaled [11] on the validation set. The *single GIN* is member 0 with its own temperature. *MC
dropout* [12] averages 30 stochastic passes of member 0 in training mode, without temperature
scaling. RF probabilities are used as produced.

**Statistics.** Paired differences between conditions that share a split are reported as
mean ± sample s.d. over seeds, with the number of seeds favoring one condition. With five
seeds a single cell is underpowered, so consistency of direction is read alongside the mean.
The single-split uncertainty results carry no seed-level error bars.

| group | setting | value |
|---|---|---|
| Encoder | GIN layers / hidden width / pooling | 4 / 128 / sum |
| | dropout / DropEdge / PairNorm scale | 0.3 / 0.1 / 1.0 |
| Optimization | optimizer | Adam, learning rate 5e-4, weight decay 5e-4, batch 128, 150 epochs |
| Ensemble | members / calibration | 5 single-task GINs / per-member temperature scaling |
| Decision metrics | ECE bins / conformal target / selective point | 10 / 90% / 70% coverage |
| SPRT | $\alpha$ / $\beta$ / design effect / subgroup | 0.05 / 0.20 / 0.30 s.d. / top 30% |
'''))

CELLS.append(md(r'''
## System architecture

Figure 0 places the components in one pipeline. The learned oracle replaces the companion
study's RF oracle, the RF is kept beside it as a descriptor comparator, and the decision
engine is reused unchanged. The engine's abductive rule-discovery component belongs to the
companion study and is not re-run here.
'''))

CELLS.append(code(r'''
"""Display Figure 0, the system schematic (fig0_architecture_v2.png, from architecture_figure_v2.py)."""
show_fig("fig0_architecture_v2.png", width=1100)
'''))

CELLS.append(md(r'''
<sub><strong>Figure 0.</strong> System architecture. Molecular graphs from six TDC ADMET endpoints feed the learned oracle, a K = 5 deep ensemble of temperature-scaled single-task GIN networks; the RF descriptor model sits beside it as a comparator. The decision engine turns the oracle's probabilities into a go/no-go call through sequential stopping (SPRT) and conformal and selective prediction, and every decision is logged to a DuckDB lineage store. The bottom strip shows how the learned models are built: the GIN encoder is trained single-task and multi-task for the transfer study (§4.1), and single-task members form the ensemble (§4.3). The GAT attention probe (§4.6) is a separate network trained on the same graphs. Blue marks inputs and feedback, yellow learned models and calibration, green decisions and outputs, purple interpretability, and grey the descriptor baseline and lineage. The dashed active-learning loop is future work.</sub>
'''))

# --------------------------------------------------------------------------- 4.1
CELLS.append(md(r'''
## 4.1 Multi-task transfer concentrates on the smallest endpoints

**Question.** Does one GIN encoder trained jointly on all six endpoints beat an identical
encoder trained on each endpoint alone, and if so, where?

**Why sharing could help, and why it could hurt.** Let $\phi_\theta$ be the shared encoder
and $h_t$ the head for endpoint $t$. The multi-task objective is the task-weighted risk

$$
\mathcal{L}(\theta) \;=\; \sum_{t} w_t \, \mathbb{E}_{(x,y)\sim\mathcal{D}_t}
      \big[\ell_t\!\left(h_t(\phi_\theta(x)),\, y\right)\big], \qquad w_t = 1,
$$

optimized by round-robin task batches so the largest endpoint doesn't dominate the trunk.
When $|\mathcal{D}_t|$ is small, a single-task estimate of $\phi$ is high-variance, and
gradients from the other endpoints regularize it. When an endpoint has ample labels of its
own, or its signal conflicts with the others, the shared trunk splits its capacity and the
endpoint can lose accuracy (negative transfer [24]). The design separates the two
possibilities: a pure low-data effect would shrink as the training fraction grows, whereas an
endpoint-dependent effect would track dataset size.

The cell below recomputes Table 1 from the per-seed learning curves and checks where the RF
fails to lead. The next one measures a confound that the per-endpoint splits introduce.
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

CELLS.append(md(r'''
**Reading Table 1.** The pattern follows endpoint size more than training fraction. Caco2,
the third-smallest endpoint, benefits at every fraction (18 of 20 seed-fraction pairs), with
its largest gain at full data, which is not what a pure low-data effect would produce. HIA
benefits consistently only at half the data (+0.071 ± 0.032, 5 of 5 seeds) and is slightly
worse at 10% and at full data. hERG is positive on average at every fraction but small
relative to its seed spread (13 of 20 pairs). On the two largest endpoints sharing costs
accuracy at full data (AMES −0.049, Solubility MAE +0.145, 0 of 5 seeds improve in either),
and BBB shows negative transfer at low data (−0.030 at 10%, 0 of 5 seeds) that fades by full
data. With five seeds no single cell is decisive; the robust signals are the direction counts
for Caco2 and the large-endpoint losses.
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
10-23% (38-65%) at full data. The multi-task gains in Table 1 therefore can't be attributed
to transfer alone and may be optimistic; a globally scaffold-disjoint split could show smaller
or, in principle, larger gains. Exposure can't explain everything, since
Caco2 already improves at f = 0.10 (4 of 5 seeds) with only 2% of its test molecules exposed,
but its largest gain, at full data, coincides with its largest exposure (19%), and the two
can't be separated with this design.
'''))

CELLS.append(code(r'''
"""Display Figure 1, the data-efficiency curves (fig_transfer_curves.png, from figures.py)."""
show_fig("fig_transfer_curves.png", width=980)
'''))

CELLS.append(md(r'''
<sub><strong>Figure 1.</strong> Data-efficiency curves for the RF descriptor baseline (grey), single-task GIN (yellow), and multi-task GIN (cyan) on six ADMET endpoints; AUROC for classification (higher is better), MAE for regression (lower is better). Points are means over five scaffold-split seeds and bands are ±1 sample s.d. Multi-task improves on single-task for Caco2 at every fraction and for HIA at f = 0.50; it is worse than single-task on AMES and Solubility at full data and on BBB at low data. Table 1 gives the paired differences.</sub>

**Against the baseline.** Neither graph model approaches the RF on raw accuracy. The RF
leads at every fraction on every endpoint except HIA at f = 0.50, where multi-task (0.904)
and the RF (0.900) are level within seed variability. What the shared trunk offers is a
transferable representation that helps some small endpoints; it is not a more accurate
predictor than descriptors and fingerprints, consistent with earlier comparisons [21,22].
'''))

# --------------------------------------------------------------------------- 4.2
CELLS.append(md(r'''
## 4.2 Ablation: self-supervised attribute-mask pretraining

**Question.** Having seen where supervised sharing helps, does self-supervised pretraining
of the same trunk add more? Following the attribute-masking objective of Hu et al. [7], we
mask 15% of atoms per molecule by zeroing their feature rows and train the trunk for 40
epochs to predict each masked atom's element. The unlabeled corpus is the union of all six
endpoints' training molecules for each seed and fraction: no labels and none of an endpoint's
own validation or test scaffolds enter pretraining, though, as Table A1 shows, other
endpoints' training molecules can overlap a target's test set. We then fine-tune under the identical protocol and compare
pretrained against from-scratch training for single- and multi-task models at the three
low-data fractions.

The masked-atom loss is cross-entropy over the corpus's atomic-number vocabulary
$\mathcal{V}$:

$$
\mathcal{L}_{\text{mask}}
   = -\,\mathbb{E}_{x}\,\frac{1}{|M(x)|}\sum_{i\in M(x)}
       \log p_\theta\!\left(z_i \mid \tilde{x}\right), \qquad z_i \in \mathcal{V},
$$

where $M(x)$ is the masked atom set, $\tilde x$ the corrupted graph, and $z_i$ the true
element. Hu et al. found that node-level pretraining alone gives limited improvement and can
transfer negatively, so a null result is plausible. With 36 comparisons, some will look
significant by chance, which is why the cell below applies a Holm correction [31].
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
**Reading Table 2.** Pretraining has no detectable effect. Seven of the 36 comparisons reach
an uncorrected p < 0.05, with mixed signs (pretraining helps hERG single-task at f = 0.25 and
hurts AMES single-task at the same fraction), and none survives the Holm correction. The most
consistent direction is single-task Solubility, where pretraining lowers MAE at all three
fractions (−0.126 ± 0.060 at f = 0.10). Averaged over the classification endpoints at the
lowest fraction, pretraining moves AUROC by +0.002 (single-task) and +0.005 (multi-task).
Supervised multi-task training (§4.1) therefore remains the operative sharing mechanism in
this label regime.
'''))

CELLS.append(code(r'''
"""Display Figure 2, the pretraining ablation at the lowest fraction (fig_pretrain_ablation.png)."""
show_fig("fig_pretrain_ablation.png", width=1000)
'''))

CELLS.append(md(r'''
<sub><strong>Figure 2.</strong> Attribute-mask pretraining minus from-scratch training at the lowest training fraction (f = 0.10), in the beneficial direction (AUROC difference, or the negative of the MAE difference), for single-task (yellow) and multi-task (cyan) models. Bars are means and error bars ±1 sample s.d. over five seeds. Single-task Solubility is the largest and most consistent effect; no difference survives multiple-comparison correction.</sub>
'''))

# --------------------------------------------------------------------------- 4.3
CELLS.append(md(r'''
## 4.3 Deep-ensemble uncertainty: scoring rules and calibration

**Question.** The second half of the study turns from accuracy to uncertainty. Which graph
uncertainty method produces the most useful probabilities, and how does it compare with the
RF? We compare a single temperature-scaled GIN, Monte Carlo dropout, and a K = 5 deep
ensemble on the four classification endpoints, on one fixed scaffold split.

**The ensemble's predictive distribution.** For temperature-scaled member probabilities
$p^{(k)}(y\mid x)$, the ensemble predicts $\bar p(y\mid x)=\tfrac1K\sum_k p^{(k)}(y\mid x)$,
whose entropy splits into an aleatoric part and an epistemic part, the mutual information
between label and member, which vanishes when members agree:

$$
\underbrace{\mathcal{H}[\bar p]}_{\text{total}}
  \;=\;
\underbrace{\tfrac1K\textstyle\sum_k \mathcal{H}[p^{(k)}]}_{\text{aleatoric}}
  \;+\;
\underbrace{\mathcal{H}[\bar p]-\tfrac1K\textstyle\sum_k \mathcal{H}[p^{(k)}]}_{\text{epistemic (mutual information)}} .
$$

The decision metrics below use $\bar p$ directly, so member disagreement enters them by
flattening $\bar p$ rather than through this term explicitly.

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
$\frac1N\sum_i(\hat p(y{=}1\mid x_i)-y_i)^2$. Proper scores reward both calibration and
discrimination; ECE measures calibration alone and is noisy on test sets of a few hundred
molecules.
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
COLS = {"ece": "ECE ↓", "nll": "NLL ↓", "brier": "Brier ↓", "conf_set_size_at_90": "set size @ 90% ↓",
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

CELLS.append(md(r'''
**Reading Table 3.** Among the graph models the ensemble has the best mean NLL (0.482 versus
0.505 single and 0.532 MC dropout), Brier score, AURC, selective accuracy, and conformal set
size. Per endpoint, it has the best NLL, Brier score, and AURC on BBB, AMES, and hERG; on HIA,
the smallest test set (106 molecules, 13 negatives), the single network and MC dropout are
better on all three. The ensemble is *not* the best-calibrated graph model by ECE: its mean
ECE (0.068) is above both the single temperature-scaled network (0.052) and the RF (0.053), and
it has the lowest ECE on no endpoint. That is the expected consequence of averaging members
that were each calibrated first, which flattens probabilities and tends to make an ensemble
underconfident [26,27]. Inverse-frequency class weights may add to this by shifting
probabilities toward a balanced prior that a single temperature can't undo. Calibrating the
averaged ensemble instead of each member is the natural fix.

Against the RF the comparison is not close: the RF has the higher test AUROC on every endpoint
and better mean NLL, Brier score, AURC, and selective accuracy. All of these values come from
one split with no seed-level error bars, so gaps of a few thousandths, such as the ensemble's
and RF's mean set sizes, are unresolved.
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

**Question.** Calibrated probabilities matter because they feed decisions. Two of them are
set-valued prediction and abstention: how large must a prediction set be to cover the true
label 90% of the time, and how accurate is the model on the cases it is most sure of?

**Split-conformal prediction** [13,14] scores each validation molecule by
$s_i = 1-\hat p(y_i\mid x_i)$ and takes $\hat q$, the
$\lceil(1-\alpha)(n+1)\rceil/n$ empirical quantile of the scores. The set
$C(x)=\{y: 1-\hat p(y\mid x)\le \hat q\}$ then satisfies

$$
\Pr\big(y_{\text{test}} \in C(x_{\text{test}})\big) \;\ge\; 1-\alpha
$$

marginally, *if* validation and test molecules are exchangeable; here $\alpha = 0.10$. (The
implementation interpolates the quantile, which returns a value at or above the exact order
statistic whenever $\lceil(1-\alpha)(n+1)\rceil \le n$, as for every calibration set here, so
it is slightly conservative and the guarantee still holds under exchangeability.) Under
that assumption validity is automatic and efficiency, the mean set size, is the informative
comparison, but only between methods at the same realized coverage. A scaffold split is built
to break exchangeability: scaffolds are assigned to validation and test at random, so the two
sets are exchangeable as scaffold groups, but molecules within a scaffold are correlated, so
realized coverage can depart from 90% [28].

**Selective prediction** [15,16] ranks test molecules by $\max_y \hat p(y\mid x)$ and reports
accuracy on the most confident 70%, together with the area under the risk-coverage curve
(AURC, in Table 3). On imbalanced endpoints this should be read against the majority-class
rate.
'''))

CELLS.append(code(r'''
"""Table 4: realized coverage, set size, and selective accuracy per endpoint (fixed split).

Test-set size and positive rate come from the committed ensemble predictions. The ensemble's
coverage shortfall is expressed in binomial standard errors, sqrt(0.9 * 0.1 / n), to show
whether a departure from the 90% target exceeds sampling noise.
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
        "coverage RF / single / ens.": " / ".join(
            f"{res[k][m]['conf_coverage_at_90']:.3f}" for m in ("rf", "gnn_single", "ensemble")),
        "ens. coverage − 0.90 (SE)": fmt((cov - 0.9) / np.sqrt(0.09 / n), digits=1),
        "set size RF / single / MC / ens.": " / ".join(
            f"{res[k][m]['conf_set_size_at_90']:.3f}" for m in METHODS),
        "sel. acc. @ 70% RF / single / MC / ens.": " / ".join(
            f"{res[k][m]['sel_acc_at_70']:.3f}" for m in METHODS),
    }
display(pd.DataFrame(rows).T)
'''))

CELLS.append(md(r'''
**Reading Table 4.** Realized coverage of the ensemble's nominal 90% sets is 0.906, 0.883,
0.830, and 0.887 on BBB, AMES, hERG, and HIA. On hERG the shortfall is about 2.9 binomial
standard errors; molecules within a scaffold are correlated, so that standard error
understates the uncertainty, but the gap is large enough that nominal coverage shouldn't be
assumed under scaffold shift. The RF covers at or above target on
all four endpoints (mean 0.918).
The single network and the ensemble agree in realized coverage within 0.01 on every endpoint,
and at that matched coverage the ensemble's sets are smaller on all four. Against the RF the
comparison is confounded: the ensemble's sets are smaller on AMES, hERG, and HIA, but each
time at lower realized coverage, so part of the reduction is bought with coverage. On BBB,
where coverage matches, the RF's sets are much smaller (1.024 versus 1.299).

Selective accuracy is mixed. The ensemble is the best graph model on BBB and AMES, the worst on
hERG (0.785 versus 0.804 single and 0.794 MC dropout), and ties the single network below MC
dropout on HIA; its higher mean in Table 3 comes from the first two endpoints. BBB and HIA are
83% and 88% positive, so selective accuracy there sits near ceiling. The RF is highest on all
four endpoints.
'''))

CELLS.append(code(r'''
"""Display Figure 4, conformal set size and selective accuracy per endpoint (fig_conformal_efficiency.png)."""
show_fig("fig_conformal_efficiency.png", width=1000)
'''))

CELLS.append(md(r'''
<sub><strong>Figure 4.</strong> Left: mean split-conformal set size at a nominal 90% target (lower is sharper; the dashed line marks a single-label set). Right: accuracy on the 70% most confident test molecules (higher is better). RF (grey), single GIN (yellow), MC dropout (green), deep ensemble (cyan). Set sizes are comparable only at matched realized coverage, which holds between the single GIN and the ensemble but not against the RF (Table 4).</sub>
'''))

# --------------------------------------------------------------------------- 4.5
CELLS.append(md(r'''
## 4.5 Sequential go/no-go campaign with the learned oracle

**Question.** The third decision is when to stop measuring. With the ensemble as oracle, how
many measurements does the companion engine's sequential test need to reach a go/no-go call,
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
$(\alpha,\beta)=(0.05,0.20)$. Among tests with these error rates the SPRT minimizes the
expected sample size [2]. The comparator is the one-sided fixed-sample $z$-test,
$n_{\text{fixed}} = \big((z_{1-\alpha}+z_{1-\beta})/\theta_1\big)^2 \approx 68.7$. Only subgroups of
at least that size can run this design; for smaller subgroups we cap it at the subgroup size,
which leaves the capped design with less than the nominal power.

**What a GO means, and what this run is.** A GO means that, in this measurement order, the
working-model likelihood ratio crossed the boundary favoring a 0.30 s.d. shift over no shift.
It is evidence that the oracle's top-ranked subgroup is enriched for positives relative to the
pool, consistent with its ranking ability in §4.3, and validates no particular chemistry. The
run is retrospective: labels are standardized with the test pool's mean and standard
deviation, and the test direction is taken from the whole subgroup's mean, both of which use
labels a real campaign wouldn't yet have. The measurements are standardized binary labels
drawn without replacement, so the Gaussian likelihood is a working model and Wald's error
bounds are nominal. Each sample size below is one realization from one random order.
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
      f"beta = {cfg['beta']}; uncapped fixed-sample requirement = {n_fixed:.1f}\n")

VERDICT = {"reject_h0": "GO", "accept_h0": "NO-GO", "continue": "undecided"}
rows = {}
for k, r in camp["per_endpoint"].items():
    d = r["sprt"]
    n_test = len(preds[f"{k}___y_true"])
    subgroup = max(8, int(cfg["top_fraction"] * n_test))
    assert abs(min(n_fixed, subgroup) - d["fixed_sample_n"]) < 0.05, k  # comparator reproduces
    rows[NAMES[k]] = {"test n": n_test, "subgroup": subgroup, "shift (s.d.)": f"{d['effect_observed']:.3f}",
                      "verdict": VERDICT[d["decision"]], "SPRT n": d["n_measurements"],
                      "fixed n": f"{d['fixed_sample_n']:.1f}", "saved": fmt(d["measurements_saved"], digits=1)}
display(pd.DataFrame(rows).T)

s = camp["summary"]
feasible = [k for k in camp["per_endpoint"] if len(preds[f"{k}___y_true"]) * cfg["top_fraction"] >= n_fixed]
used_f = sum(camp["per_endpoint"][k]["sprt"]["n_measurements"] for k in feasible)
fixed_f = sum(camp["per_endpoint"][k]["sprt"]["fixed_sample_n"] for k in feasible)
print(f"same nominal error rates, where the fixed design is feasible ({', '.join(NAMES[k] for k in feasible)}): "
      f"{used_f} versus {fixed_f:.1f} measurements ({100 * (1 - used_f / fixed_f):.1f}% fewer)")
print(f"with hERG and HIA capped at their subgroup size (lower power): {s['total_measurements_used']} versus "
      f"{s['total_fixed_sample']} ({s['pct_saved']}% fewer); {s['n_go_decisions']}/{s['n_endpoints']} GO")
# The stored shift is |subgroup mean|; recompute the signed mean to see which direction was chosen.
signs = {}
for k in camp["per_endpoint"]:
    y = preds[f"{k}___y_true"].astype(float)
    z = (y - y.mean()) / (y.std() + 1e-9)
    top = np.argsort(-preds[f"{k}__ensemble"])[:max(8, int(cfg["top_fraction"] * len(y)))]
    signs[NAMES[k]] = "positive" if z[top].mean() >= 0 else "negative"
print("data-chosen test direction:", ", ".join(f"{k} {v}" for k, v in signs.items()))
print("\nDuckDB lineage (regenerated by e8_campaign_learned.py; experiment IDs recorded in the JSON):")
for k, r in camp["per_endpoint"].items():
    print(f"  {NAMES[k]:5s} experiment {r['experiment_id'][:8]}  {VERDICT[r['sprt']['decision']]}")
'''))

CELLS.append(md(r'''
**Reading Table 5.** The SPRT returns GO on BBB, AMES, and hERG. Only the BBB and AMES
subgroups can run the fixed design at the same nominal error rates; there the SPRT used 105
measurements against 137.4 (24% fewer), and the saving comes entirely from AMES (0.64 s.d.
shift; 20 versus 68.7). On BBB the shift (0.25) lies below the design effect, and the test
needed 85 measurements before crossing the GO boundary, 16 more than the fixed design; this is
how an SPRT behaves when the true effect sits between $\theta_0$ and $\theta_1$. hERG reached GO
after 14 of its 45 subgroup molecules. HIA's subgroup has only 31 molecules, and its shift
(0.28) never crossed either boundary, so the test ended undecided when the subgroup ran out.
Capping the fixed design at the subgroup size gives 150 versus 213.4 (29.7% fewer), but the
capped hERG and HIA designs have less than nominal power, so that total isn't a
like-for-like comparison. The data-chosen direction was positive on all four endpoints, the
direction one would pre-specify, so the decisions equal those of a one-sided test; the
test-pool standardization remains a look-ahead. Each decision
is logged to the DuckDB lineage store with its experiment ID, parameters, and the oracle's
summary prediction, so every call can be traced back to the oracle that made it.
'''))

CELLS.append(code(r'''
"""Display Figure 5, measurements to a decision per endpoint (fig_campaign_efficiency.png)."""
show_fig("fig_campaign_efficiency.png", width=980)
'''))

CELLS.append(md(r'''
<sub><strong>Figure 5.</strong> Measurements to a decision per endpoint: SPRT driven by the deep-ensemble oracle (cyan circles) against the fixed-sample test at the same nominal error rates (grey squares; 68.7 measurements, capped at the subgroup size for hERG and HIA, where the capped design has less than nominal power). AMES and hERG reach GO well below the fixed design, BBB reaches GO above it, and HIA exhausts its 31-molecule subgroup undecided. Retrospective run with test-pool standardization (§4.5).</sub>
'''))

# --------------------------------------------------------------------------- 4.6
CELLS.append(md(r'''
## 4.6 Interpretability probe: attention against a descriptor rule

**Question.** A common hope is that a graph-attention network's weights highlight the atoms a
chemist would. We test that directly. A separate three-layer GAT [4] (hidden width 64; four
heads in the first two layers, one in the last) is trained on hERG and BBB, with the
checkpoint chosen by validation AUROC; its training isn't seeded. Each atom's score is the sum
of its incoming attention in the single-head final layer. The reference is
a transparent rule: an atom is salient if it is aromatic or a nitrogen, a coarse proxy for the
lipophilic-ring and basic-amine motifs associated with hERG binding and CNS penetration.
Agreement is the AUROC of attention for predicting salience over all test atoms, and the
enrichment of salient atoms among each molecule's three most-attended atoms relative to the
molecule's base rate. Molecules with fewer than four atoms, or with all atoms salient or all
non-salient, are excluded (5 of 153 hERG and 88 of 458 BBB test molecules). An AUROC of 0.5
and an enrichment of 0 mean attention carries no information about the rule.
'''))

CELLS.append(code(r'''
"""Attention-versus-rule agreement per endpoint (attention_attribution.json)."""
att = json.loads((ART / "attention_attribution.json").read_text())
print(f"rule: {att['rule']}\n")
display(pd.DataFrame({NAMES[k]: {
    "test molecules": r["n_test_molecules"],
    "GAT test AUROC": f"{r['test_auroc']:.3f}",
    "attention vs rule AUROC": f"{r['node_attention_auroc']:.3f}",
    "top-3 enrichment": fmt(r["topk_enrichment"]),
    "salient base rate": f"{r['salient_base_rate']:.2f}",
} for k, r in att["results"].items()}).T)
'''))

CELLS.append(md(r'''
**Reading the probe.** The GAT is predictive (test AUROC 0.77 on hERG and 0.76 on BBB), yet in
this run its attention ranks atoms close to chance against the rule (AUROC 0.52 and 0.49), and the top three
attended atoms are barely richer or poorer in salient atoms than the molecule as a whole
(+0.04 and −0.04).
The probe can't separate two readings: attention weights may not be faithful explanations
[19], or the rule may be too coarse a target, since it marks nearly half of all atoms (47% on
hERG, 44% on BBB) as salient. Whether attention can serve as an explanation depends on how it
is tested [29]; here it simply isn't used as one. The probe is one unseeded run on one split
with no confidence intervals, so "near chance" describes this run rather than a tested
null.
'''))

CELLS.append(code(r'''
"""Display Figure 6, the attention probe (fig_attention_probe.png)."""
show_fig("fig_attention_probe.png", width=1000)
'''))

CELLS.append(md(r'''
<sub><strong>Figure 6.</strong> Attention probe on hERG and BBB. Left: GAT test AUROC (cyan) against the AUROC of final-layer node attention for the aromatic-or-nitrogen salience rule (yellow); the dashed line marks chance. Right: salient-atom enrichment among each molecule's three most-attended atoms relative to the molecule's base rate (zero means no enrichment).</sub>
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
    row = {"n": SIZES[k], "metric": f"{m.upper()} {'↑' if m == 'auroc' else '↓'}"}
    for fam, label in (("rf", "RF"), ("gnn_single", "single-task"), ("gnn_multi", "multi-task")):
        row[label] = f"{full[fam].mean():.3f} ± {full[fam].std(ddof=1):.3f}"
    row["multi−single"] = fmt(full["gnn_multi"].mean() - full["gnn_single"].mean())
    e = res[k]["ensemble"] if k in res else None
    for key, label in (("ece", "ens. ECE ↓"), ("conf_coverage_at_90", "coverage"),
                       ("conf_set_size_at_90", "set @ 90% ↓"), ("sel_acc_at_70", "sel. acc. @ 70% ↑")):
        row[label] = f"{e[key]:.3f}" if e else "–"
    rows[NAMES[k]] = row
display(pd.DataFrame(rows).T)
'''))

CELLS.append(md(r'''
**Reading the summary.** The RF is the most accurate model on every endpoint. Sharing a graph
encoder helps some small endpoints (clearly Caco2, partly HIA) and hurts the large ones. A deep
ensemble is the strongest graph-model uncertainty method on proper scoring rules and decision
metrics on average, without being the best calibrated by ECE or uniformly best per endpoint.
Driving the sequential test in a retrospective run, it reaches traceable calls, and where a
fixed design at the same nominal error rates is feasible it needs fewer measurements, with the saving
concentrated where its top-ranked subgroup is strongly enriched.
'''))

# --------------------------------------------------------------------------- synopsis
CELLS.append(md(r'''
## Synopsis, limitations, and how to regenerate

**What was shown.** Replacing a descriptor oracle with learned graph models changes less about
accuracy than one might hope, and the gains it does bring are specific to particular endpoints
and decisions.

- **Transfer (4.1).** A shared GIN encoder improves Caco2 MAE at every training fraction (18 of
  20 seed-fraction pairs) and HIA AUROC at half the data (5 of 5 seeds), is directional on
  hERG, and costs accuracy on AMES and Solubility at full data and on BBB at low data. Because
  the trunk sees some of each target's test structures through other endpoints, these gains
  may be optimistic.
- **Pretraining (4.2).** Attribute-mask pretraining adds nothing detectable: none of 36 paired
  comparisons survives a Holm correction.
- **Uncertainty (4.3).** The K = 5 ensemble has the best mean NLL, Brier score, and AURC of the
  graph models, but not the best ECE; the RF is stronger on every accuracy and scoring metric.
- **Coverage and abstention (4.4).** Realized coverage under scaffold shift is 0.830-0.906
  against a 0.90 target; at matched coverage the ensemble's sets are smaller than a single
  network's, and its selective accuracy leads the graph models on two of four endpoints.
- **Stopping (4.5).** In a retrospective run with the ensemble as oracle, the SPRT reaches 3 GO
  calls; where a fixed design at the same nominal error rates is feasible (BBB, AMES) it uses
  105 versus 137.4
  measurements, driven by AMES, and every call is logged to DuckDB.
- **Interpretability (4.6).** In one unseeded run, graph attention shows near-chance alignment with an
  aromatic-or-nitrogen rule, although the attention network is predictive.

**What the evidence does not establish.**

- This is a retrospective simulation over public benchmarks; no wet-lab loop is closed and no
  prospective claim is made.
- Graph models are not more accurate than the descriptor baseline here; the RF leads on every
  endpoint at full data.
- Scaffold splits are drawn per endpoint, so the multi-task trunk and the pretraining corpus
  see other endpoints' molecules that overlap a target's test set (Table A1); the transfer
  gains can't be attributed to transfer alone, and a globally scaffold-disjoint split is
  needed to resolve this.
- Five seeds give low power for single cells, and the 24 transfer comparisons are not corrected
  for multiplicity; the uncertainty study and the campaign rest on one split with no seed-level
  error bars, and the hERG and HIA test sets have 153 and 106 molecules.
- The ensemble's members are calibrated before averaging, which likely explains its higher ECE;
  a multi-task ensemble was not evaluated.
- The MC dropout baseline is not temperature-scaled, and its training-mode passes also activate
  DropEdge and batch-statistics normalization, so it may understate a tuned dropout baseline.
  Atomic number enters the encoder as a scalar rather than one-hot, and the GIN ignores bond
  attributes.
- Conformal coverage is measured, not guaranteed: scaffold shift breaks exchangeability, and
  coverage falls to 0.830 on hERG.
- The campaign is retrospective: it standardizes with test-pool statistics and picks its
  direction from the whole subgroup, treats standardized binary labels as Gaussian, samples
  without replacement, and reports one random order, so its error rates are nominal. The
  fixed design at the same nominal error rates is feasible only for BBB and AMES. A GO is evidence of ranking
  ability, not a discovery.
- The attention probe is one unseeded run with a coarse rule, two endpoints, one split, and no
  confidence intervals. The companion engine's
  abductive rule discovery was not re-run with the learned oracle.

**What would change the conclusions.** A globally scaffold-disjoint split for the transfer and
pretraining studies; multi-seed uncertainty and campaign runs with error bars; calibrating the averaged ensemble rather than each member; a temperature-scaled,
evaluation-mode MC dropout baseline; shift-aware conformal calibration [28]; a pre-specified
test direction, standardization from training data, and Monte Carlo over measurement orders
for the SPRT; one-hot atom features; and
a prospective loop in which the GO subgroups are the first molecules measured.

**CPU versus GPU.** Everything in this notebook runs on CPU in seconds from the committed
artifacts. Training the encoders, the ensemble, the pretraining ablation, and the attention
probe needs a GPU; no number here depends on rerunning them.

**Regenerate everything** from the repository root (Python 3.11.2, PyTDC 1.1.15):

```bash
pip install -r requirements.txt                             # pinned CPU environment
python preprint/dl_forward/e8_campaign_learned.py           # Table 5 + DuckDB lineage (CPU)
python preprint/dl_forward/figures.py                       # Figures 1-6 (CPU)
python preprint/dl_forward/architecture_figure_v2.py        # Figure 0 (CPU)
PYTHONPATH=.:src python preprint/dl_forward/scaffold_overlap.py   # Table A1 (CPU; fetches TDC data)
jupyter nbconvert --to notebook --execute --inplace \
    preprint/dl_forward/walkthrough_learned_representations_admet.ipynb

# GPU retraining (make install-gpu; PYTHONPATH=.:src)
python preprint/dl_forward/train_multitask_gnn.py           # data for Table 1, Figure 1
python preprint/dl_forward/pretrain_ablation.py --full      # data for Table 2, Figure 2
python preprint/dl_forward/make_ensemble_uncertainty.py     # data for Tables 3-4, Figures 3-4
python preprint/dl_forward/attention_attribution.py         # data for Figure 6
```

For the transfer, pretraining, and ensemble scripts, seeds pin NumPy and PyTorch and
`cudnn.deterministic` is set (the attention probe is unseeded); residual GPU nondeterminism
remains, so retrained numbers should match within seed variability rather than
bitwise. The cell below runs the CPU tier when `RECOMPUTE = True` is set in the setup cell.
'''))

CELLS.append(code(r'''
"""Optional recompute: the same scripts the manuscript cites, on the same code path."""
if RECOMPUTE:
    import os
    import subprocess
    env = {**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'src'}"}
    # CPU, seconds: the campaign (with its DuckDB lineage) and every figure.
    for script in ("e8_campaign_learned.py", "figures.py", "architecture_figure_v2.py"):
        subprocess.run([sys.executable, str(DL / script)], cwd=str(ROOT), env=env, check=True)
    # GPU, hours: uncomment on a GPU host to retrain from scratch.
    # for script, args in (("train_multitask_gnn.py", []), ("pretrain_ablation.py", ["--full"]),
    #                      ("make_ensemble_uncertainty.py", []), ("attention_attribution.py", [])):
    #     subprocess.run([sys.executable, str(DL / script), *args], cwd=str(ROOT), env=env, check=True)
    print("recompute done; rerun the notebook to read the refreshed artifacts")
else:
    print("RECOMPUTE is False: reading committed artifacts only.")
'''))

CELLS.append(md(r'''
#### Generative AI Disclosure

Some assertions and model development steps within this document were developed with
reference to Generative AI tools (Copilot; 2026 version). AI assistance was used for
clarifying concepts, validating code logic, identifying potential errors, and generating
some code segments. All AI-generated material was independently reviewed, debugged, and
validated for correctness before inclusion.
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
20. Russell, D. R. (2026). Knowing what to measure and when to stop: an autonomous decision engine for molecular property discovery. Preprint, SNPTX. https://github.com/snptx1/snptx-repro-discovery
21. Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., Guzman-Perez, A., Hopper, T., Kelley, B., Mathea, M., Palmer, A., Settels, V., Jaakkola, T., Jensen, K. and Barzilay, R. (2019). Analyzing learned molecular representations for property prediction. *Journal of Chemical Information and Modeling*, 59(8), 3370-3388.
22. Jiang, D., Wu, Z., Hsieh, C.-Y., Chen, G., Liao, B., Wang, Z., Shen, C., Cao, D., Wu, J. and Hou, T. (2021). Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptor-based and graph-based models. *Journal of Cheminformatics*, 13, 12.
23. Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K. and Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. *Chemical Science*, 9(2), 513-530.
24. Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.
25. Ramsundar, B., Kearnes, S., Riley, P., Webster, D., Konerding, D. and Pande, V. (2015). Massively multitask networks for drug discovery. *arXiv:1502.02072*.
26. Wu, X. and Gales, M. (2021). Should ensemble members be calibrated? *arXiv:2101.05397*.
27. Rahaman, R. and Thiery, A. H. (2021). Uncertainty quantification and deep ensembles. *NeurIPS*.
28. Tibshirani, R. J., Foygel Barber, R., Candès, E. and Ramdas, A. (2019). Conformal prediction under covariate shift. *NeurIPS*.
29. Wiegreffe, S. and Pinter, Y. (2019). Attention is not not explanation. *EMNLP-IJCNLP*.
30. Rogers, D. and Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742-754.
31. Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65-70.
'''))


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, str(OUT))
    print(f"wrote {OUT.name} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build()
