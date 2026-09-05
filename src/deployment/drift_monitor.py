"""Formal drift detection & monitoring with theoretical guarantees.

Implements sequential changepoint detection (CUSUM, Page-Hinkley), population
stability index with chi-squared tests, optimal-transport Wasserstein drift,
conformal prediction monitors, and Maximum Mean Discrepancy two-sample testing.

References:
    Page (1954) — CUSUM
    Basseville & Nikiforov (1993) — Sequential Analysis
    Ramdas et al. (2017) — Wasserstein-based drift
    Vovk et al. (2005) — Conformal prediction
    Gretton et al. (2012) — MMD two-sample test
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ChangePointResult:
    """Result from sequential changepoint detection."""

    detected: bool
    statistic: float
    threshold: float
    change_index: int | None = None
    method: str = "cusum"


@dataclass
class WassersteinResult:
    """Result from Wasserstein (earth-mover) distance drift test."""

    distance: float
    drifted: bool
    threshold: float
    feature: str = ""


@dataclass
class MMDResult:
    """Result from Maximum Mean Discrepancy two-sample test."""

    mmd_squared: float
    threshold: float
    reject_null: bool
    p_value: float


@dataclass
class ConformalViolation:
    """Conformal coverage monitor violation record."""

    timestamp: float
    expected_coverage: float
    observed_coverage: float
    n_samples: int
    violated: bool


@dataclass
class DriftMonitorResult:
    """Aggregated result across all formal drift methods."""

    changepoint_results: list[ChangePointResult] = field(default_factory=list)
    wasserstein_results: list[WassersteinResult] = field(default_factory=list)
    mmd_result: MMDResult | None = None
    conformal_violation: ConformalViolation | None = None
    psi_chi2_pvalue: float | None = None
    overall_drifted: bool = False
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        from dataclasses import asdict

        return asdict(self)


# ---------------------------------------------------------------------------
# CUSUM & Page-Hinkley changepoint detectors
# ---------------------------------------------------------------------------
class ChangePointDetector:
    """Sequential changepoint detection with theoretical ARL guarantees.

    CUSUM (Page 1954): Detects shifts in the mean of a sequential stream.
    The Average Run Length (ARL) to false alarm is ≈ exp(h) / δ for threshold h
    and allowance δ.

    Page-Hinkley: A variant that accumulates deviations from the running mean
    with a tolerance parameter.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        allowance: float = 0.5,
        min_observations: int = 30,
    ) -> None:
        self.threshold = threshold
        self.allowance = allowance
        self.min_observations = min_observations

    def cusum(self, values: np.ndarray) -> ChangePointResult:
        """Run CUSUM on a 1-D stream.

        Maintains upper/lower cumulative sums S+ and S-.  A changepoint is
        flagged when either exceeds *threshold*.  The *allowance* δ controls
        sensitivity (smaller δ → more sensitive, shorter ARL).
        """
        s_pos = 0.0
        s_neg = 0.0
        target = float(np.mean(values[: self.min_observations])) if len(values) >= self.min_observations else 0.0

        for i, x in enumerate(values):
            s_pos = max(0.0, s_pos + (x - target) - self.allowance)
            s_neg = max(0.0, s_neg - (x - target) - self.allowance)
            if (s_pos > self.threshold or s_neg > self.threshold) and i >= self.min_observations:
                logger.info("CUSUM changepoint detected at index %d (stat=%.4f)", i, max(s_pos, s_neg))
                return ChangePointResult(
                    detected=True,
                    statistic=max(s_pos, s_neg),
                    threshold=self.threshold,
                    change_index=i,
                    method="cusum",
                )

        return ChangePointResult(
            detected=False,
            statistic=max(s_pos, s_neg),
            threshold=self.threshold,
            method="cusum",
        )

    def page_hinkley(self, values: np.ndarray, delta: float = 0.005) -> ChangePointResult:
        """Page-Hinkley test for change in mean.

        Accumulates the deviation of observations from the running mean,
        minus a tolerance *delta*.  A changepoint is signalled when the
        difference (max_cumulated − current_cumulated) exceeds *threshold*.
        """
        cumulated = 0.0
        min_cumulated = 0.0
        running_mean = 0.0

        for i, x in enumerate(values):
            running_mean = running_mean + (x - running_mean) / (i + 1)
            cumulated += x - running_mean - delta
            min_cumulated = min(min_cumulated, cumulated)

            if (cumulated - min_cumulated > self.threshold) and i >= self.min_observations:
                logger.info("Page-Hinkley changepoint at index %d (stat=%.4f)", i, cumulated - min_cumulated)
                return ChangePointResult(
                    detected=True,
                    statistic=cumulated - min_cumulated,
                    threshold=self.threshold,
                    change_index=i,
                    method="page_hinkley",
                )

        return ChangePointResult(
            detected=False,
            statistic=cumulated - min_cumulated,
            threshold=self.threshold,
            method="page_hinkley",
        )

    def theoretical_arl(self) -> float:
        """Approximate Average Run Length to false alarm for CUSUM.

        ARL₀ ≈ exp(h) / δ  where h = threshold, δ = allowance.
        """
        if self.allowance <= 0:
            return float("inf")
        return math.exp(self.threshold) / self.allowance


# ---------------------------------------------------------------------------
# Wasserstein (optimal transport) drift
# ---------------------------------------------------------------------------
class WassersteinDrift:
    """Optimal-transport drift detection via 1-D Wasserstein distance.

    Uses the closed-form solution for 1-D distributions:
        W₁(P, Q) = ∫|F_P⁻¹(t) − F_Q⁻¹(t)| dt
    approximated by sorting and averaging absolute differences.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold

    def compute(self, reference: np.ndarray, production: np.ndarray) -> float:
        """Compute 1-D Wasserstein distance between two samples."""
        ref_sorted = np.sort(reference)
        prod_sorted = np.sort(production)
        # Interpolate to common grid for unequal sizes
        n = max(len(ref_sorted), len(prod_sorted))
        grid = np.linspace(0, 1, n, endpoint=False)
        ref_interp = np.interp(grid, np.linspace(0, 1, len(ref_sorted), endpoint=False), ref_sorted)
        prod_interp = np.interp(grid, np.linspace(0, 1, len(prod_sorted), endpoint=False), prod_sorted)
        return float(np.mean(np.abs(ref_interp - prod_interp)))

    def test(self, reference: np.ndarray, production: np.ndarray, feature: str = "") -> WassersteinResult:
        """Test whether Wasserstein distance exceeds threshold."""
        dist = self.compute(reference, production)
        return WassersteinResult(
            distance=dist,
            drifted=dist > self.threshold,
            threshold=self.threshold,
            feature=feature,
        )

    def test_multivariate(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> list[WassersteinResult]:
        """Per-feature Wasserstein drift detection."""
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
        if production.ndim == 1:
            production = production.reshape(-1, 1)

        n_features = reference.shape[1]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]
        return [self.test(reference[:, i], production[:, i], feature=names[i]) for i in range(n_features)]


# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy (MMD) two-sample test
# ---------------------------------------------------------------------------
class MMDTest:
    """Kernel Maximum Mean Discrepancy two-sample test (Gretton et al. 2012).

    Under H₀ (same distribution), n·MMD² converges to a weighted sum of χ²
    random variables.  We approximate the null via a permutation test.
    """

    def __init__(self, kernel_bandwidth: float | None = None, alpha: float = 0.05, n_permutations: int = 200) -> None:
        self.kernel_bandwidth = kernel_bandwidth
        self.alpha = alpha
        self.n_permutations = n_permutations

    @staticmethod
    def _rbf_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
        """Gaussian RBF kernel."""
        sq_dist = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
        return np.exp(-sq_dist / (2.0 * bandwidth**2))

    def _median_heuristic(self, x: np.ndarray, y: np.ndarray) -> float:
        """Median heuristic for kernel bandwidth."""
        combined = np.vstack([x, y])
        # Sub-sample for efficiency when large
        if len(combined) > 500:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(combined), 500, replace=False)
            combined = combined[idx]
        dists = np.sqrt(np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=2))
        median_dist = float(np.median(dists[dists > 0]))
        return max(median_dist, 1e-8)

    def _compute_mmd_squared(self, x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
        """Unbiased estimate of MMD²."""
        k_xx = self._rbf_kernel(x, x, bandwidth)
        k_yy = self._rbf_kernel(y, y, bandwidth)
        k_xy = self._rbf_kernel(x, y, bandwidth)

        m = len(x)
        n = len(y)

        # Zero diagonals for unbiased estimate
        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)

        term1 = np.sum(k_xx) / (m * (m - 1)) if m > 1 else 0.0
        term2 = np.sum(k_yy) / (n * (n - 1)) if n > 1 else 0.0
        term3 = 2.0 * np.sum(k_xy) / (m * n)

        return float(term1 + term2 - term3)

    def test(self, x: np.ndarray, y: np.ndarray) -> MMDResult:
        """Perform MMD two-sample test with permutation-based p-value."""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        bandwidth = self.kernel_bandwidth or self._median_heuristic(x, y)
        observed_mmd = self._compute_mmd_squared(x, y, bandwidth)

        # Permutation test
        combined = np.vstack([x, y])
        m = len(x)
        rng = np.random.default_rng(42)
        count_ge = 0
        for _ in range(self.n_permutations):
            perm = rng.permutation(len(combined))
            x_perm = combined[perm[:m]]
            y_perm = combined[perm[m:]]
            perm_mmd = self._compute_mmd_squared(x_perm, y_perm, bandwidth)
            if perm_mmd >= observed_mmd:
                count_ge += 1

        p_value = (count_ge + 1) / (self.n_permutations + 1)

        return MMDResult(
            mmd_squared=observed_mmd,
            threshold=self.alpha,
            reject_null=p_value < self.alpha,
            p_value=p_value,
        )


# ---------------------------------------------------------------------------
# Conformal prediction coverage monitor
# ---------------------------------------------------------------------------
class ConformalMonitor:
    """Monitors conformal prediction coverage in production.

    Flags violations when observed coverage falls below the guaranteed
    1 − α level beyond a statistical tolerance (Vovk et al. 2005).
    """

    def __init__(self, alpha: float = 0.1, violation_threshold: float = 0.03, window_size: int = 500) -> None:
        self.alpha = alpha
        self.expected_coverage = 1.0 - alpha
        self.violation_threshold = violation_threshold
        self.window_size = window_size
        self._labels: list[int] = []
        self._conformal_sets: list[list[int]] = []

    def update(self, label: int, conformal_set: list[int]) -> None:
        """Record a single observation."""
        self._labels.append(label)
        self._conformal_sets.append(conformal_set)
        # Keep window
        if len(self._labels) > self.window_size:
            self._labels = self._labels[-self.window_size :]
            self._conformal_sets = self._conformal_sets[-self.window_size :]

    def check(self) -> ConformalViolation:
        """Check current coverage against expected level."""
        import time

        n = len(self._labels)
        if n == 0:
            return ConformalViolation(
                timestamp=time.time(),
                expected_coverage=self.expected_coverage,
                observed_coverage=1.0,
                n_samples=0,
                violated=False,
            )

        covered = sum(1 for lbl, cs in zip(self._labels, self._conformal_sets) if lbl in cs)
        observed = covered / n
        violated = (self.expected_coverage - observed) > self.violation_threshold

        if violated:
            logger.warning(
                "Conformal coverage violation: expected %.3f, observed %.3f (n=%d)",
                self.expected_coverage,
                observed,
                n,
            )

        return ConformalViolation(
            timestamp=time.time(),
            expected_coverage=self.expected_coverage,
            observed_coverage=observed,
            n_samples=n,
            violated=violated,
        )


# ---------------------------------------------------------------------------
# Aggregated drift monitor
# ---------------------------------------------------------------------------
class DriftMonitor:
    """Unified formal drift monitoring combining all methods.

    Provides a single ``run()`` method that executes CUSUM, Page-Hinkley,
    Wasserstein, MMD, and conformal coverage checks.
    """

    def __init__(
        self,
        cusum_threshold: float = 5.0,
        cusum_allowance: float = 0.5,
        wasserstein_threshold: float = 0.1,
        mmd_alpha: float = 0.05,
        conformal_alpha: float = 0.1,
    ) -> None:
        self.changepoint = ChangePointDetector(threshold=cusum_threshold, allowance=cusum_allowance)
        self.wasserstein = WassersteinDrift(threshold=wasserstein_threshold)
        self.mmd = MMDTest(alpha=mmd_alpha)
        self.conformal = ConformalMonitor(alpha=conformal_alpha)

    def run(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        feature_names: list[str] | None = None,
        stream: np.ndarray | None = None,
    ) -> DriftMonitorResult:
        """Execute all drift tests and return unified result.

        Args:
            reference: Reference dataset (n_ref, d) or 1-D.
            production: Production dataset (n_prod, d) or 1-D.
            feature_names: Optional feature labels.
            stream: Optional 1-D stream for CUSUM/Page-Hinkley.  If None,
                    the first feature column of production is used.
        """
        result = DriftMonitorResult()

        # --- Changepoint on stream ---
        if stream is None:
            stream = production[:, 0] if production.ndim > 1 else production
        result.changepoint_results = [
            self.changepoint.cusum(stream),
            self.changepoint.page_hinkley(stream),
        ]

        # --- Per-feature Wasserstein ---
        result.wasserstein_results = self.wasserstein.test_multivariate(reference, production, feature_names)

        # --- MMD two-sample test ---
        result.mmd_result = self.mmd.test(reference, production)

        # --- Conformal coverage ---
        result.conformal_violation = self.conformal.check()

        # --- PSI + chi-squared (lightweight) ---
        result.psi_chi2_pvalue = self._psi_chi2(reference, production)

        # --- Aggregate decision ---
        any_changepoint = any(cp.detected for cp in result.changepoint_results)
        any_wasserstein = any(w.drifted for w in result.wasserstein_results)
        mmd_drift = result.mmd_result.reject_null if result.mmd_result else False
        conformal_drift = result.conformal_violation.violated if result.conformal_violation else False

        result.overall_drifted = any_changepoint or any_wasserstein or mmd_drift or conformal_drift
        result.summary = {
            "changepoint_detected": any_changepoint,
            "wasserstein_drift": any_wasserstein,
            "mmd_reject_null": mmd_drift,
            "conformal_violated": conformal_drift,
            "psi_chi2_pvalue": result.psi_chi2_pvalue,
        }

        return result

    @staticmethod
    def _psi_chi2(reference: np.ndarray, production: np.ndarray, n_bins: int = 10) -> float:
        """PSI with chi-squared distributional test (p-value returned)."""
        ref_1d = reference.ravel() if reference.ndim > 1 else reference
        prod_1d = production.ravel() if production.ndim > 1 else production

        edges = np.histogram_bin_edges(ref_1d, bins=n_bins)
        ref_counts = np.histogram(ref_1d, bins=edges)[0].astype(float)
        prod_counts = np.histogram(prod_1d, bins=edges)[0].astype(float)

        # Laplace smoothing
        ref_counts += 1
        prod_counts += 1
        ref_freq = ref_counts / ref_counts.sum()
        prod_freq = prod_counts / prod_counts.sum()

        # Chi-squared statistic
        chi2_stat = float(np.sum((prod_freq - ref_freq) ** 2 / ref_freq) * prod_counts.sum())
        dof = max(n_bins - 1, 1)
        # Approximate p-value via survival function of chi-squared
        # Using the regularised incomplete gamma function approximation
        p_value = _chi2_survival(chi2_stat, dof)
        return p_value


def _chi2_survival(x: float, k: int) -> float:
    """Approximate chi-squared survival function P(X > x) for k dof.

    Uses the Wilson-Hilferty normal approximation for tractability without scipy.
    """
    if x <= 0:
        return 1.0
    # Wilson-Hilferty approximation
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
    # Standard normal survival (erfc-based)
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return max(0.0, min(1.0, p))
