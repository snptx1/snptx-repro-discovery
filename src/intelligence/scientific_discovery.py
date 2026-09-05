"""B.6.10 Automated Scientific Discovery.

Abductive reasoning for surprise-driven hypothesis generation
(Peirce 1903; Josephson & Josephson 1996).  Novelty search deliberately
exploring unusual configurations (Lehman & Stanley 2011).  Knowledge
distillation across experiments into compact meta-models (Hinton et al.
2015).  Symbolic regression for interpretable meta-rules: e.g.
optimal_n_estimators ~ f(n_samples, n_features) (Schmidt & Lipson 2009;
Cranmer et al. 2020).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------------------------
# Abductive reasoning / surprise-driven hypothesis generation
# (Peirce 1903; Josephson & Josephson 1996)
# ---------------------------------------------------------------------------

@dataclass
class Surprise:
    """A surprising experimental result that warrants abductive explanation."""

    experiment_id: str
    metric: str
    observed: float
    expected: float
    z_score: float
    explanation_candidates: list[str] = field(default_factory=list)


def detect_surprises(
    results: pd.DataFrame,
    metric_col: str = "f1",
    z_threshold: float = 2.0,
) -> list[Surprise]:
    """Identify experiments whose metric deviates significantly from the mean.

    Parameters
    ----------
    results : DataFrame
        Must contain columns ``experiment_id`` and *metric_col*.
    metric_col : str
        Column name for the metric to analyse.
    z_threshold : float
        Absolute z-score threshold for flagging surprises.

    Returns list of Surprise objects sorted by abs(z_score) descending.
    """
    vals = results[metric_col].to_numpy(dtype=np.float64)
    mu = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
    if sigma <= 0:
        return []

    surprises: list[Surprise] = []
    for _, row in results.iterrows():
        z = (float(row[metric_col]) - mu) / sigma
        if abs(z) >= z_threshold:
            surprises.append(
                Surprise(
                    experiment_id=str(row["experiment_id"]),
                    metric=metric_col,
                    observed=float(row[metric_col]),
                    expected=mu,
                    z_score=z,
                )
            )
    surprises.sort(key=lambda s: abs(s.z_score), reverse=True)
    return surprises


def generate_abductive_hypotheses(
    surprise: Surprise,
    feature_deltas: dict[str, float],
    top_k: int = 3,
) -> list[str]:
    """Generate candidate explanations for a surprising result.

    Uses the largest absolute feature deltas (difference from population
    mean) to propose abductive hypotheses.

    Parameters
    ----------
    surprise : Surprise
        The surprising observation to explain.
    feature_deltas : dict
        Feature name -> signed delta from population mean for this experiment.
    top_k : int
        Number of top explanations to generate.

    Returns list of natural-language hypothesis strings.
    """
    sorted_feats = sorted(feature_deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)
    direction = "above" if surprise.z_score > 0 else "below"
    hypotheses: list[str] = []
    for feat_name, delta in sorted_feats[:top_k]:
        sign = "high" if delta > 0 else "low"
        hypotheses.append(
            f"{surprise.metric} is {direction} expected because "
            f"{feat_name} is unusually {sign} (delta={delta:.3f})"
        )
    surprise.explanation_candidates = hypotheses
    return hypotheses


# ---------------------------------------------------------------------------
# Novelty search (Lehman & Stanley 2011)
# ---------------------------------------------------------------------------

def behaviour_distance(
    a: NDArray[np.floating[Any]],
    b: NDArray[np.floating[Any]],
) -> float:
    """Euclidean distance between two behaviour characterisation vectors."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


@dataclass
class NoveltyArchive:
    """Archive for novelty search.

    Maintains a bounded archive of behaviourally diverse configurations.
    """

    max_size: int = 500
    k_nearest: int = 15
    archive: list[NDArray[np.floating[Any]]] = field(default_factory=list)

    def novelty_score(self, behaviour: NDArray[np.floating[Any]]) -> float:
        """Average distance to k nearest neighbours in archive + current pop."""
        if len(self.archive) == 0:
            return float("inf")
        b = np.asarray(behaviour, dtype=np.float64)
        dists = sorted(behaviour_distance(b, a) for a in self.archive)
        k = min(self.k_nearest, len(dists))
        return float(np.mean(dists[:k]))

    def maybe_add(self, behaviour: NDArray[np.floating[Any]], threshold: float = 0.0) -> bool:
        """Add to archive if novelty score exceeds threshold."""
        score = self.novelty_score(behaviour)
        if score >= threshold:
            self.archive.append(np.asarray(behaviour, dtype=np.float64))
            if len(self.archive) > self.max_size:
                # Remove the element closest to its nearest neighbour
                min_novelty_idx = 0
                min_novelty = float("inf")
                for i, a in enumerate(self.archive):
                    dists = sorted(
                        behaviour_distance(a, self.archive[j])
                        for j in range(len(self.archive))
                        if j != i
                    )
                    nn_dist = dists[0] if dists else 0.0
                    if nn_dist < min_novelty:
                        min_novelty = nn_dist
                        min_novelty_idx = i
                self.archive.pop(min_novelty_idx)
            return True
        return False


def novelty_search_select(
    candidates: Sequence[NDArray[np.floating[Any]]],
    archive: NoveltyArchive,
    n_select: int = 5,
) -> list[int]:
    """Select the most novel candidates from a list.

    Returns indices of the top-n_select most novel candidates.
    """
    scores = [(i, archive.novelty_score(c)) for i, c in enumerate(candidates)]
    scores.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in scores[:n_select]]


# ---------------------------------------------------------------------------
# Knowledge distillation (Hinton et al. 2015)
# ---------------------------------------------------------------------------

def distill_soft_targets(
    teacher_predictions: NDArray[np.floating[Any]],
    temperature: float = 3.0,
) -> NDArray[np.floating[Any]]:
    """Compute softened probability targets from teacher logits.

    soft_i = exp(z_i / T) / sum_j exp(z_j / T)
    """
    logits = np.asarray(teacher_predictions, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    scaled = logits / temperature
    # Numerically stable softmax
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def distillation_loss(
    student_logits: NDArray[np.floating[Any]],
    teacher_logits: NDArray[np.floating[Any]],
    temperature: float = 3.0,
    alpha: float = 0.7,
    hard_labels: NDArray[np.integer[Any]] | None = None,
) -> float:
    """Combined distillation loss.

    L = alpha * T^2 * KL(soft_teacher || soft_student)
      + (1-alpha) * CE(hard_labels, student)

    When hard_labels is None, only the soft loss is used (alpha=1).
    """
    s_logits = np.asarray(student_logits, dtype=np.float64)
    t_logits = np.asarray(teacher_logits, dtype=np.float64)
    if s_logits.ndim == 1:
        s_logits = s_logits.reshape(1, -1)
        t_logits = t_logits.reshape(1, -1)

    soft_t = distill_soft_targets(t_logits, temperature)
    soft_s = distill_soft_targets(s_logits, temperature)

    # KL divergence (natural log for loss, not bits)
    mask = soft_t > 0
    kl = float(np.sum(soft_t[mask] * np.log(soft_t[mask] / np.clip(soft_s[mask], 1e-12, None))))
    soft_loss = temperature * temperature * kl

    if hard_labels is None:
        return soft_loss

    # Cross-entropy with hard labels
    labels = np.asarray(hard_labels).ravel()
    # Standard softmax on student (T=1)
    student_probs = distill_soft_targets(s_logits, temperature=1.0)
    ce = 0.0
    for i, lab in enumerate(labels):
        ce -= math.log(max(float(student_probs[i, lab]), 1e-12))
    ce /= len(labels)

    return alpha * soft_loss + (1.0 - alpha) * ce


@dataclass
class MetaModelDistiller:
    """Distill knowledge from many experiments into a compact meta-model.

    Fits a small model (Ridge regression) that predicts performance
    from meta-features, using experiment history as training data.
    """

    alpha: float = 1.0
    model: Ridge | None = None

    def fit(
        self,
        meta_features: pd.DataFrame,
        performance: NDArray[np.floating[Any]],
    ) -> None:
        """Train compact meta-model on (meta-features -> performance)."""
        x = meta_features.to_numpy(dtype=np.float64)
        y = np.asarray(performance, dtype=np.float64).ravel()
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(x, y)

    def predict(self, meta_features: pd.DataFrame) -> NDArray[np.floating[Any]]:
        """Predict performance for new datasets."""
        if self.model is None:
            raise RuntimeError("MetaModelDistiller not fitted yet")
        x = meta_features.to_numpy(dtype=np.float64)
        return np.asarray(self.model.predict(x), dtype=np.float64)

    def feature_importance(self, feature_names: list[str]) -> list[tuple[str, float]]:
        """Return feature importances sorted by absolute weight."""
        if self.model is None:
            raise RuntimeError("MetaModelDistiller not fitted yet")
        coefs = np.asarray(self.model.coef_).ravel()
        pairs = list(zip(feature_names, [float(c) for c in coefs]))
        pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        return pairs


# ---------------------------------------------------------------------------
# Symbolic regression for interpretable meta-rules
# (Schmidt & Lipson 2009; Cranmer et al. 2020)
# ---------------------------------------------------------------------------

@dataclass
class SymbolicRule:
    """An interpretable symbolic meta-rule."""

    expression: str
    r_squared: float
    complexity: int  # number of nodes in expression tree
    feature_names: list[str] = field(default_factory=list)


def _tree_to_expression(
    tree: DecisionTreeRegressor,
    feature_names: list[str],
) -> str:
    """Convert a shallow decision tree into a human-readable rule string."""
    t: Any = tree.tree_
    if t.feature[0] < 0:
        return f"{t.value[0].ravel()[0]:.4f}"

    feat = feature_names[t.feature[0]] if t.feature[0] < len(feature_names) else f"x{t.feature[0]}"
    thresh = t.threshold[0]
    left_val = t.value[t.children_left[0]].ravel()[0]
    right_val = t.value[t.children_right[0]].ravel()[0]
    return f"if {feat} <= {thresh:.4f} then {left_val:.4f} else {right_val:.4f}"


def symbolic_regression_tree(
    features: pd.DataFrame,
    target: NDArray[np.floating[Any]],
    max_depth: int = 3,
    min_samples_leaf: int = 5,
) -> SymbolicRule:
    """Fit a shallow decision tree as a symbolic regression proxy.

    True symbolic regression (genetic programming) is expensive; a shallow
    tree provides interpretable rules with bounded complexity.

    Parameters
    ----------
    features : DataFrame
        Predictor meta-features.
    target : 1-D array
        Target metric to model.
    max_depth : int
        Maximum tree depth (controls complexity).
    min_samples_leaf : int
        Minimum samples per leaf.

    Returns SymbolicRule with expression, R^2, and complexity.
    """
    x = features.to_numpy(dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).ravel()
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    tree.fit(x, y)
    y_pred = tree.predict(x)

    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    feat_names = [str(c) for c in features.columns]
    expr = _tree_to_expression(tree, feat_names)
    n_nodes = int(tree.tree_.node_count)

    return SymbolicRule(
        expression=expr,
        r_squared=r2,
        complexity=n_nodes,
        feature_names=feat_names,
    )


def polynomial_regression_rule(
    features: pd.DataFrame,
    target: NDArray[np.floating[Any]],
    degree: int = 2,
    alpha: float = 1.0,
) -> SymbolicRule:
    """Fit a polynomial regression as a symbolic meta-rule.

    Generates polynomial features up to ``degree`` and fits Ridge regression
    for interpretable coefficients.
    """
    from sklearn.preprocessing import PolynomialFeatures

    x = features.to_numpy(dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).ravel()
    feat_names = [str(c) for c in features.columns]

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly.fit_transform(x)
    poly_names = poly.get_feature_names_out(feat_names)

    model = Ridge(alpha=alpha)
    model.fit(x_poly, y)
    y_pred = model.predict(x_poly)

    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Build expression from non-negligible coefficients
    coefs = model.coef_.ravel()
    intercept = float(model.intercept_)
    terms: list[str] = []
    if abs(intercept) > 1e-6:
        terms.append(f"{intercept:.4f}")
    for name, coef in zip(poly_names, coefs):
        if abs(coef) > 1e-6:
            terms.append(f"{coef:+.4f}*{name}")
    expr = " ".join(terms) if terms else "0.0"

    return SymbolicRule(
        expression=expr,
        r_squared=r2,
        complexity=len(terms),
        feature_names=feat_names,
    )


# ---------------------------------------------------------------------------
# Discovery pipeline: orchestrates all components
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryReport:
    """Summary of an automated discovery cycle."""

    surprises: list[Surprise] = field(default_factory=list)
    novel_indices: list[int] = field(default_factory=list)
    meta_model_r2: float = 0.0
    symbolic_rules: list[SymbolicRule] = field(default_factory=list)


def run_discovery_cycle(
    results: pd.DataFrame,
    meta_features: pd.DataFrame,
    metric_col: str = "f1",
    z_threshold: float = 2.0,
    n_novel: int = 5,
) -> DiscoveryReport:
    """Run one full automated discovery cycle.

    1. Detect surprising results (abductive reasoning).
    2. Identify novel configurations (novelty search).
    3. Distill meta-model (knowledge distillation).
    4. Extract symbolic rules (symbolic regression).

    Parameters
    ----------
    results : DataFrame
        Must have ``experiment_id`` and *metric_col* columns.
    meta_features : DataFrame
        Same row order as results; meta-feature columns.
    metric_col : str
        Performance metric column.
    z_threshold : float
        Z-score threshold for surprise detection.
    n_novel : int
        Number of novel configurations to highlight.

    Returns DiscoveryReport.
    """
    report = DiscoveryReport()

    # 1. Surprise detection
    report.surprises = detect_surprises(results, metric_col, z_threshold)

    # 2. Novelty search on meta-feature space
    archive = NoveltyArchive(max_size=max(len(meta_features), 50))
    mf_array = meta_features.to_numpy(dtype=np.float64)
    candidates = [mf_array[i] for i in range(len(mf_array))]
    # Seed archive with first half
    mid = max(len(candidates) // 2, 1)
    for c in candidates[:mid]:
        archive.maybe_add(c, threshold=0.0)
    report.novel_indices = novelty_search_select(candidates, archive, n_novel)

    # 3. Knowledge distillation (meta-model)
    target = results[metric_col].to_numpy(dtype=np.float64)
    distiller = MetaModelDistiller(alpha=1.0)
    distiller.fit(meta_features, target)
    y_pred = distiller.predict(meta_features)
    ss_res = float(np.sum((target - y_pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    report.meta_model_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # 4. Symbolic regression
    tree_rule = symbolic_regression_tree(meta_features, target, max_depth=3)
    report.symbolic_rules.append(tree_rule)
    if meta_features.shape[1] <= 10:  # poly only tractable for low-dim
        poly_rule = polynomial_regression_rule(meta_features, target, degree=2)
        report.symbolic_rules.append(poly_rule)

    return report
