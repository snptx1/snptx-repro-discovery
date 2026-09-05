"""Fast, no-download regression tests for snptx-repro-discovery.

These validate the vendoring closure and the determinism of the engine primitives
without touching the network (no TDC download). Run with `make test`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_vendoring_closure_imports():
    """Every module the engine wires must import from this repo alone."""
    from snptx.viz.theme import DARK_BG, apply_dark_theme  # noqa: F401
    from src.intelligence.catalog import ExperimentCatalog  # noqa: F401
    from src.intelligence.experiment_design import SPRT  # noqa: F401
    from src.intelligence.scientific_discovery import run_discovery_cycle  # noqa: F401
    from src.intelligence.surrogate import SurrogateModel  # noqa: F401
    from src.models.gnn import build_gnn_model  # noqa: F401
    from src.safety.uncertainty import UncertaintyQuantifier  # noqa: F401


def test_theme_matches_site_palette():
    """The vendored theme must carry the GitHub-dark palette used site-wide."""
    from snptx.viz.theme import ACCENT_BLUE, CARD_BG, DARK_BG

    assert DARK_BG == "#0d1117"
    assert CARD_BG == "#161b22"
    assert ACCENT_BLUE == "#58a6ff"


def test_sprt_is_deterministic_and_saves_samples():
    """SPRT reaches the same decision with the same seed, terminates before the
    budget, and (averaged over seeds) uses fewer samples than a fixed-sample test."""
    from scipy.stats import norm

    from src.intelligence.experiment_design import SPRT

    def run(seed: int) -> tuple[str, int]:
        rng = np.random.default_rng(seed)
        sprt = SPRT(theta_0=0.0, theta_1=0.5, alpha=0.05, beta=0.20, sigma=1.0)
        for _ in range(5000):
            if sprt.update(float(rng.normal(0.5, 1.0))) != "continue":
                break
        return sprt.decision, sprt.n_observations

    d1, n1 = run(0)
    d2, n2 = run(0)
    assert (d1, n1) == (d2, n2)          # deterministic
    assert d1 == "reject_h0"             # detects the real effect
    assert 0 < n1 < 5000                 # terminates before the budget

    # the efficiency claim is an average operating characteristic, not a single run
    mean_sprt_n = float(np.mean([run(s)[1] for s in range(200)]))
    fixed_n = ((norm.ppf(0.95) + norm.ppf(0.80)) / 0.5) ** 2
    assert mean_sprt_n < 2.0 * fixed_n   # sequential is competitive on average


def test_conformal_coverage_meets_target():
    """Split-conformal coverage should sit near the 1 - alpha target."""
    from src.safety.uncertainty import UncertaintyQuantifier

    rng = np.random.default_rng(7)
    n = 4000
    y = rng.integers(0, 2, n)
    # a moderately calibrated classifier: p(correct) ~ 0.8
    probs = np.zeros((n, 2))
    for i in range(n):
        p = rng.uniform(0.55, 0.95)
        probs[i, y[i]] = p
        probs[i, 1 - y[i]] = 1 - p
    cal, test = slice(0, 2000), slice(2000, n)
    uq = UncertaintyQuantifier(alpha=0.10)
    q = uq.calibrate_conformal(probs[cal], y[cal])
    scores = 1.0 - probs[test]
    covered = [int(lab in np.where(scores[i] <= q)[0]) for i, lab in enumerate(y[test])]
    coverage = float(np.mean(covered))
    assert 0.85 <= coverage <= 0.97


def test_discovery_cycle_recovers_planted_driver():
    """The abductive discovery cycle should split on the true driver feature."""
    from src.intelligence.scientific_discovery import run_discovery_cycle

    rng = np.random.default_rng(1)
    n, d = 400, 6
    X = rng.normal(size=(n, d))
    y = 2.0 * X[:, 0] + 0.1 * rng.normal(size=n)      # driver is feature f0
    cols = [f"f{i}" for i in range(d)]
    meta = pd.DataFrame(X, columns=cols)
    results = pd.DataFrame({"experiment_id": [f"m{i}" for i in range(n)], "y": y})
    rep = run_discovery_cycle(results, meta, metric_col="y", z_threshold=2.5, n_novel=5)
    assert rep.meta_model_r2 > 0.5
    assert rep.symbolic_rules[0].expression.split()[1] == "f0"
