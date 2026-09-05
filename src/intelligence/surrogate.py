"""Bayesian optimization and sequential decision-making for HPO.

Gaussian Process surrogates model the config-to-metric response surface,
enabling principled acquisition functions to guide hyperparameter search.

References
----------
- Snoek, Larochelle & Adams, "Practical Bayesian Optimization of Machine
  Learning Algorithms", NeurIPS 2012.
- Swersky, Snoek & Adams, "Multi-Task Bayesian Optimization", NeurIPS 2013.
- Wilson, Hutter & Deisenroth, "Maximizing Acquisition Functions for
  Bayesian Optimization", NeurIPS 2018.
- Russo et al., "A Tutorial on Thompson Sampling", Foundations & Trends
  in ML, 2018.
- Li et al., "A Contextual-Bandit Approach to Personalized News Article
  Recommendation", WWW 2010 (LinUCB).
- Agrawal & Goyal, "Thompson Sampling for Contextual Bandits with Linear
  Payoffs", ICML 2013.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# GP Surrogate Model
# ═══════════════════════════════════════════════════════════════════════


class SurrogateModel:
    """Gaussian Process surrogate for hyperparameter response surfaces.

    Wraps ``sklearn.gaussian_process.GaussianProcessRegressor`` with an
    RBF + WhiteKernel covariance function, suitable for modeling smooth
    metric landscapes over continuous hyperparameter spaces.

    Parameters
    ----------
    length_scale : initial RBF length scale (auto-tuned during fit).
    noise_level : initial noise variance for WhiteKernel.
    n_restarts : number of optimizer restarts for kernel hyperparameters.
    random_state : seed for reproducibility.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        noise_level: float = 1e-5,
        n_restarts: int = 5,
        random_state: int | None = None,
    ) -> None:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            random_state=random_state,
            normalize_y=True,
        )
        self._fitted = False

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> None:
        """Fit surrogate to observed (config, metric) pairs.

        Parameters
        ----------
        X : (n_samples, n_features) array of hyperparameter configurations.
        y : (n_samples,) array of observed metric values.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(X) < 2:
            raise ValueError("Need at least 2 observations to fit GP surrogate")
        self._gp.fit(X, y)
        self._fitted = True

    def predict(
        self, X: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return posterior mean and standard deviation.

        Parameters
        ----------
        X : (n_candidates, n_features) array of candidate configs.

        Returns
        -------
        mu : (n_candidates,) predicted means.
        sigma : (n_candidates,) predicted standard deviations.
        """
        if not self._fitted:
            raise RuntimeError("SurrogateModel must be fit before predict")
        X = np.asarray(X, dtype=np.float64)
        result: Any = self._gp.predict(X, return_std=True)
        mu, sigma = result[0], result[1]
        return np.asarray(mu, dtype=np.float64), np.asarray(sigma, dtype=np.float64)

    @property
    def fitted(self) -> bool:
        return self._fitted


# ═══════════════════════════════════════════════════════════════════════
# Acquisition Functions
# ═══════════════════════════════════════════════════════════════════════


def expected_improvement(
    mu: NDArray[np.floating],
    sigma: NDArray[np.floating],
    best_f: float,
    xi: float = 0.01,
) -> NDArray[np.floating]:
    """Expected Improvement acquisition function (Snoek et al. 2012).

    EI(x) = (mu(x) - f* - xi) * Phi(Z) + sigma(x) * phi(Z)
    where Z = (mu(x) - f* - xi) / sigma(x).

    Parameters
    ----------
    mu : predicted means.
    sigma : predicted standard deviations.
    best_f : best observed value so far.
    xi : exploration-exploitation trade-off (>0 = more exploration).

    Returns
    -------
    EI values for each candidate (non-negative).
    """
    from scipy.stats import norm

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    mask = sigma > 1e-12
    ei = np.zeros_like(mu)

    improvement = mu[mask] - best_f - xi
    Z = improvement / sigma[mask]
    ei[mask] = improvement * norm.cdf(Z) + sigma[mask] * norm.pdf(Z)

    return np.maximum(ei, 0.0)


def upper_confidence_bound(
    mu: NDArray[np.floating],
    sigma: NDArray[np.floating],
    beta: float = 2.0,
) -> NDArray[np.floating]:
    """GP-UCB acquisition function (Srinivas et al. 2010).

    UCB(x) = mu(x) + beta * sigma(x)

    Parameters
    ----------
    mu : predicted means.
    sigma : predicted standard deviations.
    beta : exploration parameter. Higher = more exploration.
           Theory: beta_t = 2 log(|D| t^2 pi^2 / 6 delta).

    Returns
    -------
    UCB values for each candidate.
    """
    return np.asarray(mu + beta * sigma, dtype=np.float64)


def knowledge_gradient(
    mu: NDArray[np.floating],
    sigma: NDArray[np.floating],
    best_f: float,
) -> NDArray[np.floating]:
    """One-step Knowledge Gradient (Frazier et al. 2009).

    KG(x) = sigma(x) * [Z * Phi(Z) + phi(Z)]
    where Z = (mu(x) - best_f) / sigma(x).

    Measures the expected improvement in the best predicted mean
    after one additional observation at x.

    Parameters
    ----------
    mu : predicted means.
    sigma : predicted standard deviations.
    best_f : current best predicted mean.

    Returns
    -------
    KG values for each candidate (non-negative).
    """
    from scipy.stats import norm

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    mask = sigma > 1e-12
    kg = np.zeros_like(mu)

    Z = (mu[mask] - best_f) / sigma[mask]
    kg[mask] = sigma[mask] * (Z * norm.cdf(Z) + norm.pdf(Z))

    return np.maximum(kg, 0.0)


def thompson_sample(
    mu: NDArray[np.floating],
    sigma: NDArray[np.floating],
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """Thompson Sampling from GP posterior (Russo et al. 2018).

    Draws one sample per candidate from N(mu, sigma^2) and returns the
    sampled values. The argmax gives the Thompson-optimal next point.

    Parameters
    ----------
    mu : predicted means.
    sigma : predicted standard deviations.
    rng : numpy random Generator for reproducibility.

    Returns
    -------
    Sampled function values for each candidate.
    """
    if rng is None:
        rng = np.random.default_rng()
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    return np.asarray(rng.normal(mu, np.maximum(sigma, 1e-12)), dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════
# Multi-Task Surrogate
# ═══════════════════════════════════════════════════════════════════════


class MultiTaskSurrogate:
    """Multi-task Bayesian optimization via augmented feature space.

    Models correlated response surfaces across datasets by appending
    dataset meta-features to the config feature vector, enabling
    knowledge transfer between related tasks (Swersky et al. 2013).

    Parameters
    ----------
    n_restarts : optimizer restarts for GP kernel fitting.
    random_state : seed for reproducibility.
    """

    def __init__(
        self,
        n_restarts: int = 5,
        random_state: int | None = None,
    ) -> None:
        self._surrogate = SurrogateModel(
            n_restarts=n_restarts, random_state=random_state
        )
        self._task_features: dict[str, NDArray[np.floating]] = {}

    def register_task(
        self, task_id: str, meta_features: NDArray[np.floating]
    ) -> None:
        """Register a dataset/task with its meta-feature vector."""
        self._task_features[task_id] = np.asarray(meta_features, dtype=np.float64)

    @property
    def registered_tasks(self) -> list[str]:
        return list(self._task_features.keys())

    def fit(
        self,
        task_ids: list[str],
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> None:
        """Fit multi-task surrogate on pooled observations.

        Parameters
        ----------
        task_ids : task identifier per row of X.
        X : (n_samples, n_config_features) config arrays.
        y : (n_samples,) observed metrics.
        """
        X_aug = self._augment(task_ids, X)
        self._surrogate.fit(X_aug, y)

    def predict(
        self, task_id: str, X: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Predict metric for a specific task/dataset.

        Parameters
        ----------
        task_id : which dataset to predict for.
        X : (n_candidates, n_config_features) candidate configs.

        Returns
        -------
        mu, sigma arrays.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = X.shape[0]
        task_feat = self._task_features[task_id]
        X_aug = np.hstack([X, np.tile(task_feat, (n, 1))])
        return self._surrogate.predict(X_aug)

    def _augment(
        self, task_ids: list[str], X: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Append task meta-features to config feature vectors."""
        X = np.asarray(X, dtype=np.float64)
        rows = []
        for i, tid in enumerate(task_ids):
            if tid not in self._task_features:
                raise KeyError(f"Task {tid!r} not registered; call register_task first")
            rows.append(np.concatenate([X[i], self._task_features[tid]]))
        return np.array(rows, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════
# Contextual Bandit (LinUCB)
# ═══════════════════════════════════════════════════════════════════════


class ContextualBandit:
    """LinUCB contextual bandit for config recommendation.

    Maintains a separate ridge regression model per arm (config template).
    Context vector encodes dataset meta-features so the bandit learns
    which configs work best for which dataset types.

    Regret bound: O(d sqrt(T log T)) where d = context dimension,
    T = number of rounds (Li et al. 2010, Agrawal & Goyal 2013).

    Parameters
    ----------
    n_arms : number of config templates (arms).
    context_dim : dimensionality of the context vector.
    alpha : exploration parameter controlling UCB width.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        alpha: float = 1.0,
    ) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")
        if context_dim < 1:
            raise ValueError("context_dim must be >= 1")
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha

        # Per-arm ridge regression: A_a = I_d, b_a = 0_d
        self._A = [np.eye(context_dim, dtype=np.float64) for _ in range(n_arms)]
        self._b = [np.zeros(context_dim, dtype=np.float64) for _ in range(n_arms)]
        self._counts = [0] * n_arms

    def select_arm(self, context: NDArray[np.floating]) -> int:
        """Select the arm with highest LinUCB score for the given context.

        Parameters
        ----------
        context : (context_dim,) feature vector for the current dataset.

        Returns
        -------
        Index of the selected arm.
        """
        context = np.asarray(context, dtype=np.float64).ravel()
        if context.shape[0] != self.context_dim:
            raise ValueError(
                f"Context dim mismatch: expected {self.context_dim}, got {context.shape[0]}"
            )

        ucb_values = np.zeros(self.n_arms, dtype=np.float64)
        for a in range(self.n_arms):
            A_inv = np.linalg.solve(self._A[a], np.eye(self.context_dim))
            theta_a = A_inv @ self._b[a]
            pred = float(theta_a @ context)
            uncertainty = float(np.sqrt(context @ A_inv @ context))
            ucb_values[a] = pred + self.alpha * uncertainty

        return int(np.argmax(ucb_values))

    def update(
        self, arm: int, context: NDArray[np.floating], reward: float
    ) -> None:
        """Update the ridge regression model for the pulled arm.

        Parameters
        ----------
        arm : index of the arm that was pulled.
        context : (context_dim,) feature vector.
        reward : observed reward (e.g. f1_score).
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Invalid arm index: {arm}")
        context = np.asarray(context, dtype=np.float64).ravel()
        self._A[arm] += np.outer(context, context)
        self._b[arm] += reward * context
        self._counts[arm] += 1

    def arm_counts(self) -> list[int]:
        """Return pull counts per arm."""
        return list(self._counts)

    def estimated_rewards(self, context: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return estimated mean reward per arm (no exploration bonus).

        Parameters
        ----------
        context : (context_dim,) feature vector.

        Returns
        -------
        (n_arms,) array of estimated rewards.
        """
        context = np.asarray(context, dtype=np.float64).ravel()
        rewards = np.zeros(self.n_arms, dtype=np.float64)
        for a in range(self.n_arms):
            A_inv = np.linalg.solve(self._A[a], np.eye(self.context_dim))
            theta_a = A_inv @ self._b[a]
            rewards[a] = float(theta_a @ context)
        return rewards


# ═══════════════════════════════════════════════════════════════════════
# Suggest Next Configuration (end-to-end BO)
# ═══════════════════════════════════════════════════════════════════════


def suggest_next_config(
    catalog: Any,
    dataset: str,
    model_type: str,
    param_bounds: dict[str, tuple[float, float]],
    acquisition: str = "ei",
    n_candidates: int = 1000,
    xi: float = 0.01,
    beta: float = 2.0,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Propose the next hyperparameter config via Bayesian optimization.

    Fits a GP surrogate to historical experiments from the catalog, then
    maximizes the chosen acquisition function over random candidates.

    Parameters
    ----------
    catalog : ExperimentCatalog with historical results.
    dataset : dataset name to optimize for.
    model_type : model type to optimize.
    param_bounds : dict mapping param name -> (low, high) bounds.
    acquisition : 'ei', 'ucb', 'kg', or 'thompson'.
    n_candidates : random candidates for acquisition maximization.
    xi : EI exploration parameter.
    beta : UCB exploration parameter.
    random_state : seed for reproducibility.

    Returns
    -------
    dict with 'config', 'acquisition_value', 'predicted_metric',
    'predicted_uncertainty', and 'n_historical'.
    """
    rng = np.random.default_rng(random_state)

    experiments = catalog.get_experiments(dataset=dataset, model_type=model_type)
    param_names = sorted(param_bounds.keys())

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

    n_historical = len(X_obs_list)

    # Fall back to random if insufficient history
    if n_historical < 2:
        config = {
            p: float(rng.uniform(lo, hi)) for p, (lo, hi) in param_bounds.items()
        }
        return {
            "config": config,
            "acquisition_value": None,
            "predicted_metric": None,
            "predicted_uncertainty": None,
            "n_historical": n_historical,
        }

    X_obs = np.array(X_obs_list, dtype=np.float64)
    y_obs = np.array(y_obs_list, dtype=np.float64)

    surrogate = SurrogateModel(random_state=random_state)
    surrogate.fit(X_obs, y_obs)

    candidates = np.column_stack([
        rng.uniform(param_bounds[p][0], param_bounds[p][1], size=n_candidates)
        for p in param_names
    ])

    mu, sigma = surrogate.predict(candidates)
    best_f = float(y_obs.max())

    acq_fn: dict[str, Any] = {
        "ei": lambda: expected_improvement(mu, sigma, best_f, xi=xi),
        "ucb": lambda: upper_confidence_bound(mu, sigma, beta=beta),
        "kg": lambda: knowledge_gradient(mu, sigma, best_f),
        "thompson": lambda: thompson_sample(mu, sigma, rng=rng),
    }
    if acquisition not in acq_fn:
        raise ValueError(
            f"Unknown acquisition function: {acquisition!r}. "
            f"Choose from {list(acq_fn)}"
        )

    acq_values = acq_fn[acquisition]()
    best_idx = int(np.argmax(acq_values))

    config = {
        p: float(candidates[best_idx, i]) for i, p in enumerate(param_names)
    }

    return {
        "config": config,
        "acquisition_value": float(acq_values[best_idx]),
        "predicted_metric": float(mu[best_idx]),
        "predicted_uncertainty": float(sigma[best_idx]),
        "n_historical": n_historical,
    }
