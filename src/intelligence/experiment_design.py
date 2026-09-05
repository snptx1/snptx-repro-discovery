"""Optimal experimental design for pipeline experiment selection.

Recommends which experiments to run next for maximum information gain,
computes Value of Information before committing resources, and provides
adaptive stopping rules via sequential hypothesis testing.

References
----------
- Chaloner & Verdinelli, "Bayesian Experimental Design: A Review",
  Stat. Sci. 1995.
- Foster, Ivanova, Malik & Rainforth, "Deep Adaptive Design", NeurIPS 2021.
- Howard, "Information Value Theory", IEEE Trans. Systems 1966.
- Rainforth et al., "Modern Bayesian Experimental Design", Stat. Sci. 2024.
- Settles, "Active Learning", Synthesis Lectures on AI & ML, 2012.
- Wald, "Sequential Tests of Statistical Hypotheses", Ann. Math. Stat. 1945.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Information Gain / Bayesian Experimental Design
# ═══════════════════════════════════════════════════════════════════════


def information_gain(
    mu: NDArray[np.floating],
    sigma: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Expected information gain for candidate experiments.

    Under a Gaussian surrogate posterior, the differential entropy of
    N(mu, sigma^2) is 0.5 * log(2 * pi * e * sigma^2). The information
    gain from observing a point is proportional to the posterior variance
    -- higher uncertainty = more information (Chaloner & Verdinelli 1995).

    IG(x) = 0.5 * log(1 + sigma(x)^2 / noise_var)

    Simplified form (noise_var absorbed): IG proportional to log(sigma).

    Parameters
    ----------
    mu : predicted means from surrogate model.
    sigma : predicted standard deviations.

    Returns
    -------
    Information gain scores for each candidate (non-negative).
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    # Use differential entropy as proxy: higher sigma = more info
    return np.maximum(0.5 * np.log1p(sigma**2), 0.0)


def recommend_experiments(
    catalog: Any,
    dataset: str,
    model_type: str,
    param_bounds: dict[str, tuple[float, float]],
    n_recommend: int = 5,
    n_candidates: int = 1000,
    random_state: int | None = None,
) -> list[dict[str, Any]]:
    """Recommend maximally informative next experiments (Foster et al. 2021).

    Fits a GP surrogate to historical results, then ranks candidate
    configs by expected information gain. Returns the top-k most
    informative experiments to run next.

    Parameters
    ----------
    catalog : ExperimentCatalog with historical results.
    dataset : dataset name.
    model_type : model type.
    param_bounds : dict mapping param name -> (low, high).
    n_recommend : number of experiments to recommend.
    n_candidates : random candidates to evaluate.
    random_state : seed for reproducibility.

    Returns
    -------
    List of dicts with 'config', 'information_gain', 'predicted_metric',
    'predicted_uncertainty'.
    """
    import json

    from src.intelligence.surrogate import SurrogateModel

    rng = np.random.default_rng(random_state)
    param_names = sorted(param_bounds.keys())

    experiments = catalog.get_experiments(dataset=dataset, model_type=model_type)

    X_obs_list: list[list[float]] = []
    y_obs_list: list[float] = []

    for _, row in experiments.iterrows():
        hp_str = row.get("hyperparams")
        f1 = row.get("f1_score")
        if hp_str is None or f1 is None:
            continue
        hp = json.loads(hp_str) if isinstance(hp_str, str) else hp_str
        try:
            vec = [float(hp[p]) for p in param_names]
        except (KeyError, TypeError):
            continue
        X_obs_list.append(vec)
        y_obs_list.append(float(f1))

    # Generate candidate configs
    candidates = np.column_stack([
        rng.uniform(param_bounds[p][0], param_bounds[p][1], size=n_candidates)
        for p in param_names
    ])

    n_historical = len(X_obs_list)

    # If insufficient history, rank by diversity (maximin distance)
    if n_historical < 2:
        recommendations: list[dict[str, Any]] = []
        indices = rng.choice(n_candidates, size=min(n_recommend, n_candidates), replace=False)
        for idx in indices:
            config = {p: float(candidates[idx, i]) for i, p in enumerate(param_names)}
            recommendations.append({
                "config": config,
                "information_gain": None,
                "predicted_metric": None,
                "predicted_uncertainty": None,
            })
        return recommendations

    X_obs = np.array(X_obs_list, dtype=np.float64)
    y_obs = np.array(y_obs_list, dtype=np.float64)

    surrogate = SurrogateModel(random_state=random_state)
    surrogate.fit(X_obs, y_obs)

    mu, sigma = surrogate.predict(candidates)
    ig = information_gain(mu, sigma)

    # Select top-k by information gain
    top_indices = np.argsort(ig)[::-1][:n_recommend]

    recommendations = []
    for idx in top_indices:
        config = {p: float(candidates[idx, i]) for i, p in enumerate(param_names)}
        recommendations.append({
            "config": config,
            "information_gain": float(ig[idx]),
            "predicted_metric": float(mu[idx]),
            "predicted_uncertainty": float(sigma[idx]),
        })

    return recommendations


# ═══════════════════════════════════════════════════════════════════════
# Value of Information (VoI)
# ═══════════════════════════════════════════════════════════════════════


def value_of_information(
    mu: NDArray[np.floating],
    sigma: NDArray[np.floating],
    best_f: float,
    cost: NDArray[np.floating] | float = 1.0,
) -> NDArray[np.floating]:
    """Compute Value of Information for candidate experiments (Howard 1966).

    VoI(x) = E[max(0, Y(x) - best_f)] / cost(x)

    Under a Gaussian posterior N(mu, sigma^2), this is the Expected
    Improvement normalized by experiment cost. Experiments with
    VoI < 0 are not worth running at any cost.

    Parameters
    ----------
    mu : predicted means.
    sigma : predicted standard deviations.
    best_f : current best observed metric.
    cost : per-experiment cost (scalar or array). Higher cost reduces VoI.

    Returns
    -------
    VoI scores for each candidate (can be negative = not worth running).
    """
    from src.intelligence.surrogate import expected_improvement

    ei = expected_improvement(mu, sigma, best_f, xi=0.0)
    cost_arr = np.broadcast_to(
        np.asarray(cost, dtype=np.float64), ei.shape
    )
    # Avoid division by zero
    safe_cost = np.maximum(cost_arr, 1e-12)
    return np.asarray(ei / safe_cost, dtype=np.float64)


def should_run_experiment(
    voi: float,
    cost: float,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Decision rule: should an experiment be run based on VoI (Howard 1966).

    Parameters
    ----------
    voi : Value of Information for the experiment.
    cost : cost of running the experiment.
    threshold : minimum VoI to justify running.

    Returns
    -------
    dict with 'run' (bool), 'voi', 'cost', 'net_value'.
    """
    net_value = voi - threshold
    return {
        "run": net_value > 0,
        "voi": voi,
        "cost": cost,
        "net_value": net_value,
        "threshold": threshold,
    }


# ═══════════════════════════════════════════════════════════════════════
# Active Learning: Uncertainty Sampling & Query-by-Committee
# ═══════════════════════════════════════════════════════════════════════


def uncertainty_sampling(
    sigma: NDArray[np.floating],
    n_select: int = 1,
) -> NDArray[np.intp]:
    """Select experiments with highest predictive uncertainty (Settles 2012).

    The simplest active learning strategy: query the instance where the
    model is least confident. For regression with a GP surrogate, this
    is the point with highest posterior variance.

    Parameters
    ----------
    sigma : predicted standard deviations for candidate experiments.
    n_select : number of experiments to select.

    Returns
    -------
    Indices of the selected candidates, sorted by uncertainty (descending).
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    n_select = min(n_select, len(sigma))
    return np.argsort(sigma)[::-1][:n_select]


def query_by_committee(
    predictions: NDArray[np.floating],
    n_select: int = 1,
) -> NDArray[np.intp]:
    """Query-by-Committee active learning (Settles 2012, Seung et al. 1992).

    Select instances where a committee of models disagrees most.
    Disagreement is measured by the variance across committee predictions.

    Parameters
    ----------
    predictions : (n_committee, n_candidates) array of predictions from
        different models or bootstrap samples.
    n_select : number of experiments to select.

    Returns
    -------
    Indices of the selected candidates, sorted by disagreement (descending).
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 2:
        raise ValueError("predictions must be 2D (n_committee, n_candidates)")
    disagreement = np.var(predictions, axis=0)
    n_select = min(n_select, predictions.shape[1])
    return np.argsort(disagreement)[::-1][:n_select]


def expected_model_change(
    mu: NDArray[np.floating],
    sigma: NDArray[np.floating],
    gradient_norms: NDArray[np.floating],
    n_select: int = 1,
) -> NDArray[np.intp]:
    """Expected Model Change active learning (Settles 2012, Ch. 2.4).

    Selects instances that would cause the largest expected change to
    the current model. Approximated as |gradient| * sigma.

    Parameters
    ----------
    mu : predicted means (unused, kept for API consistency).
    sigma : predicted standard deviations.
    gradient_norms : norm of the loss gradient at each candidate.
    n_select : number of experiments to select.

    Returns
    -------
    Indices of the selected candidates.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    gradient_norms = np.asarray(gradient_norms, dtype=np.float64)
    emc = sigma * gradient_norms
    n_select = min(n_select, len(emc))
    return np.argsort(emc)[::-1][:n_select]


# ═══════════════════════════════════════════════════════════════════════
# Sequential Probability Ratio Test (SPRT)
# ═══════════════════════════════════════════════════════════════════════


class SPRT:
    """Wald's Sequential Probability Ratio Test (Wald 1945).

    Tests H0: theta = theta_0 vs H1: theta = theta_1 sequentially,
    observing one data point at a time. Stops as soon as enough evidence
    accumulates, using far fewer samples than fixed-sample tests.

    For Gaussian observations with known variance:
        Lambda_n = sum_{i=1}^n (x_i - (theta_0 + theta_1)/2) * (theta_1 - theta_0) / sigma^2

    Decision boundaries:
        Accept H0 if Lambda_n <= log(beta / (1 - alpha))
        Accept H1 if Lambda_n >= log((1 - beta) / alpha)
        Continue otherwise.

    Parameters
    ----------
    theta_0 : null hypothesis value (e.g., baseline metric).
    theta_1 : alternative hypothesis value (e.g., improved metric).
    alpha : Type I error rate (false positive). Default 0.05.
    beta : Type II error rate (false negative). Default 0.20.
    sigma : known standard deviation of observations.
    """

    def __init__(
        self,
        theta_0: float,
        theta_1: float,
        alpha: float = 0.05,
        beta: float = 0.20,
        sigma: float = 1.0,
    ) -> None:
        if theta_0 == theta_1:
            raise ValueError("theta_0 and theta_1 must differ")
        if not (0 < alpha < 1) or not (0 < beta < 1):
            raise ValueError("alpha and beta must be in (0, 1)")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

        # Log-likelihood ratio boundaries
        self._lower = np.log(beta / (1 - alpha))
        self._upper = np.log((1 - beta) / alpha)

        self._log_lr: float = 0.0
        self._observations: list[float] = []
        self._decision: str | None = None

    def update(self, observation: float) -> str:
        """Process one observation and return the current decision.

        Parameters
        ----------
        observation : new data point (e.g., metric value from an experiment).

        Returns
        -------
        'continue', 'reject_h0' (accept H1), or 'accept_h0'.
        """
        if self._decision is not None:
            return self._decision

        self._observations.append(observation)

        # Incremental log-likelihood ratio for Gaussian
        midpoint = (self.theta_0 + self.theta_1) / 2
        self._log_lr += (
            (observation - midpoint) * (self.theta_1 - self.theta_0)
            / self.sigma**2
        )

        if self._log_lr >= self._upper:
            self._decision = "reject_h0"
        elif self._log_lr <= self._lower:
            self._decision = "accept_h0"
        else:
            self._decision = None

        return self._decision or "continue"

    def update_batch(self, observations: NDArray[np.floating] | list[float]) -> str:
        """Process multiple observations sequentially.

        Returns the decision after the last observation, or 'continue'.
        Stops early if a decision boundary is crossed.
        """
        for obs in observations:
            result = self.update(float(obs))
            if result != "continue":
                return result
        return "continue"

    @property
    def decision(self) -> str:
        """Current decision: 'continue', 'reject_h0', or 'accept_h0'."""
        return self._decision or "continue"

    @property
    def log_likelihood_ratio(self) -> float:
        """Current cumulative log-likelihood ratio."""
        return self._log_lr

    @property
    def n_observations(self) -> int:
        """Number of observations processed so far."""
        return len(self._observations)

    @property
    def boundaries(self) -> dict[str, float]:
        """Decision boundaries."""
        return {"lower": self._lower, "upper": self._upper}

    def status(self) -> dict[str, Any]:
        """Full status report."""
        return {
            "decision": self.decision,
            "n_observations": self.n_observations,
            "log_likelihood_ratio": self._log_lr,
            "lower_boundary": self._lower,
            "upper_boundary": self._upper,
            "theta_0": self.theta_0,
            "theta_1": self.theta_1,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def reset(self) -> None:
        """Reset for a new sequence of observations."""
        self._log_lr = 0.0
        self._observations.clear()
        self._decision = None


# ═══════════════════════════════════════════════════════════════════════
# Adaptive Stopping Rule
# ═══════════════════════════════════════════════════════════════════════


def adaptive_stopping(
    metrics_so_far: NDArray[np.floating] | list[float],
    target_metric: float,
    min_experiments: int = 3,
    alpha: float = 0.05,
    beta: float = 0.20,
) -> dict[str, Any]:
    """Should we stop running more experiments? (Wald 1945 + heuristics).

    Combines SPRT with practical convergence checks:
    1. SPRT tests whether the mean metric exceeds the target.
    2. Convergence check: is the running mean stabilizing?

    Parameters
    ----------
    metrics_so_far : observed metrics from experiments run so far.
    target_metric : target to test against (H1: mean > target).
    min_experiments : minimum experiments before stopping.
    alpha : SPRT Type I error.
    beta : SPRT Type II error.

    Returns
    -------
    dict with 'stop' (bool), 'reason', 'n_experiments', 'mean_metric',
    'sprt_decision', 'converged'.
    """
    metrics = np.asarray(metrics_so_far, dtype=np.float64).ravel()
    n = len(metrics)

    if n < min_experiments:
        return {
            "stop": False,
            "reason": f"Need at least {min_experiments} experiments (have {n})",
            "n_experiments": n,
            "mean_metric": float(np.mean(metrics)) if n > 0 else None,
            "sprt_decision": "continue",
            "converged": False,
        }

    mean_metric = float(np.mean(metrics))
    sigma_est = float(np.std(metrics, ddof=1)) if n > 1 else 0.1

    # SPRT: H0 = target_metric, H1 = target_metric + delta
    delta = max(sigma_est * 0.5, 0.01)
    sprt = SPRT(
        theta_0=target_metric,
        theta_1=target_metric + delta,
        alpha=alpha,
        beta=beta,
        sigma=max(sigma_est, 1e-6),
    )
    sprt_decision = sprt.update_batch(metrics)

    # Convergence: coefficient of variation of last half
    converged = False
    if n >= 4:
        recent = metrics[n // 2:]
        cv = float(np.std(recent, ddof=1) / abs(np.mean(recent))) if abs(np.mean(recent)) > 1e-12 else 0.0
        converged = cv < 0.05

    stop = False
    reason = "continue"
    if sprt_decision == "reject_h0":
        stop = True
        reason = "SPRT: metric significantly exceeds target"
    elif sprt_decision == "accept_h0":
        stop = True
        reason = "SPRT: metric does not exceed target"
    elif converged:
        stop = True
        reason = "Converged: metric variance below threshold"

    return {
        "stop": stop,
        "reason": reason,
        "n_experiments": n,
        "mean_metric": mean_metric,
        "sprt_decision": sprt_decision,
        "converged": converged,
    }
