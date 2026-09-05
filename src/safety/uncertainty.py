"""Prediction uncertainty quantification for clinical safety.

Implements conformal prediction intervals, Monte Carlo dropout,
and calibration assessment.  The ``UncertaintyQuantifier`` wraps
any sklearn-compatible or PyTorch model and adds distribution-free
coverage guarantees via conformal prediction (Vovk et al. 2005)
plus optional MC-dropout for deep learning models.

References:
    - Vovk, Gammerman & Shafer, "Algorithmic Learning in a Random World", 2005
    - Romano, Sesia & Candès, "Conformalized Quantile Regression", NeurIPS 2019
    - Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    """Container for uncertainty estimates on a single prediction."""

    prediction: int
    probabilities: np.ndarray
    confidence: float
    entropy: float
    conformal_set: list[int] = field(default_factory=list)
    mc_mean: np.ndarray | None = None
    mc_std: np.ndarray | None = None
    calibrated_confidence: float | None = None
    should_refuse: bool = False
    refusal_reason: str | None = None


class UncertaintyQuantifier:
    """Uncertainty quantification with conformal prediction and calibration.

    Parameters
    ----------
    confidence_threshold : float
        Minimum confidence to accept a prediction.
    alpha : float
        Conformal significance level (1 - coverage). E.g. 0.1 for 90% coverage.
    mc_samples : int
        Number of MC-dropout forward passes (only for PyTorch models).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        alpha: float = 0.1,
        mc_samples: int = 20,
    ):
        self.confidence_threshold = confidence_threshold
        self.alpha = alpha
        self.mc_samples = mc_samples

        self._calibration_scores: np.ndarray | None = None
        self._conformal_threshold: float | None = None
        self._calibration_bins: dict[str, np.ndarray] | None = None

    # ───────────────────────────────────────────────────────────────
    # Conformal prediction
    # ───────────────────────────────────────────────────────────────

    def calibrate_conformal(
        self, cal_probabilities: np.ndarray, cal_labels: np.ndarray
    ) -> float:
        """Calibrate conformal predictor on held-out calibration data.

        Uses the softmax scores as non-conformity measure:
        score_i = 1 - p(y_true | x_i)

        Parameters
        ----------
        cal_probabilities : (n, C) predicted probability matrix on calibration set.
        cal_labels : (n,) true labels.

        Returns
        -------
        float : conformal threshold q_hat.
        """
        n = len(cal_labels)
        scores = 1.0 - cal_probabilities[np.arange(n), cal_labels.astype(int)]
        self._calibration_scores = np.sort(scores)

        # Quantile at ceil((n+1)(1-alpha)) / n
        q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self._conformal_threshold = float(np.quantile(scores, q_level))
        logger.info(
            "Conformal calibration: n=%d, alpha=%.2f, q_hat=%.4f",
            n, self.alpha, self._conformal_threshold,
        )
        return self._conformal_threshold

    def conformal_set(self, probabilities: np.ndarray) -> list[int]:
        """Return the conformal prediction set for a single sample.

        The set contains all classes whose score (1-p_k) <= q_hat.

        Parameters
        ----------
        probabilities : (C,) predicted probabilities for one sample.

        Returns
        -------
        List of class indices in the prediction set.
        """
        if self._conformal_threshold is None:
            return [int(np.argmax(probabilities))]
        scores = 1.0 - probabilities
        return [int(k) for k in np.where(scores <= self._conformal_threshold)[0]]

    # ───────────────────────────────────────────────────────────────
    # Calibration assessment
    # ───────────────────────────────────────────────────────────────

    def compute_calibration(
        self, probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
    ) -> dict[str, Any]:
        """Compute Expected Calibration Error (ECE) and reliability diagram data.

        Parameters
        ----------
        probabilities : (n, C) probability matrix.
        labels : (n,) true labels.
        n_bins : int - number of bins for ECE.

        Returns
        -------
        Dict with 'ece', 'bin_confidences', 'bin_accuracies', 'bin_counts'.
        """
        confidences = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        accuracies = (predictions == labels).astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_confidences = np.zeros(n_bins)
        bin_accuracies = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
            count = int(np.sum(mask))
            bin_counts[i] = count
            if count > 0:
                bin_confidences[i] = float(np.mean(confidences[mask]))
                bin_accuracies[i] = float(np.mean(accuracies[mask]))

        total = len(labels)
        ece = float(np.sum(bin_counts / max(total, 1) * np.abs(bin_accuracies - bin_confidences)))

        self._calibration_bins = {
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies,
        }

        return {
            "ece": ece,
            "bin_confidences": bin_confidences.tolist(),
            "bin_accuracies": bin_accuracies.tolist(),
            "bin_counts": bin_counts.tolist(),
        }

    # ───────────────────────────────────────────────────────────────
    # Monte Carlo dropout (PyTorch only)
    # ───────────────────────────────────────────────────────────────

    def mc_dropout_predict(
        self, model: object, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run MC-dropout forward passes to estimate predictive uncertainty.

        Requires a PyTorch model with dropout layers.

        Parameters
        ----------
        model : nn.Module with dropout layers.
        X : (n, d) input features.

        Returns
        -------
        (mean_probs, std_probs) each of shape (n, C).
        """
        import torch

        model.train()  # type: ignore[union-attr]
        tensor_x = torch.from_numpy(X).float()

        all_probs = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                logits = model(tensor_x)  # type: ignore[operator]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        stacked = np.stack(all_probs, axis=0)
        mean_probs = np.mean(stacked, axis=0)
        std_probs = np.std(stacked, axis=0)

        model.eval()  # type: ignore[union-attr]
        return mean_probs, std_probs

    # ───────────────────────────────────────────────────────────────
    # Combined quantification
    # ───────────────────────────────────────────────────────────────

    def quantify(self, probabilities: np.ndarray) -> UncertaintyResult:
        """Full uncertainty quantification for a single sample.

        Parameters
        ----------
        probabilities : (C,) predicted probabilities.

        Returns
        -------
        UncertaintyResult with all uncertainty estimates.
        """
        pred = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
        conf_set = self.conformal_set(probabilities)

        should_refuse = confidence < self.confidence_threshold
        reason = None
        if should_refuse:
            reason = (
                f"Prediction confidence ({confidence:.3f}) below threshold "
                f"({self.confidence_threshold:.3f}). Clinical review recommended."
            )

        return UncertaintyResult(
            prediction=pred,
            probabilities=probabilities,
            confidence=confidence,
            entropy=entropy,
            conformal_set=conf_set,
            should_refuse=should_refuse,
            refusal_reason=reason,
        )
