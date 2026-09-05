"""Phase D.6 — Theoretical Hardening of Deployment & Compliance.

Formally grounded methods for distribution shift detection, algorithmic fairness,
adversarial robustness, differential privacy, regulatory science, and uncertainty
quantification in production.
"""

from __future__ import annotations

from src.deployment.drift_monitor import (  # noqa: I001
    ChangePointDetector,
    ConformalMonitor,
    DriftMonitor,
    DriftMonitorResult,
    MMDTest,
    WassersteinDrift,
)
from src.deployment.fairness_audit import (
    CausalFairnessAuditor,
    EqualizedOddsPostProcessor,
    FairnessAuditReport,
    FairnessHardeningAuditor,
    IncompatibilityAnalysis,
)
from src.deployment.privacy import (
    DPAccountant,
    DPSGDTrainer,
    FederatedAggregator,
    PrivacyBudget,
    SecureAggregator,
)
from src.deployment.regulatory_docs import (
    ComplianceReport,
    GMP5RiskAssessment,
    ModelCardGenerator,
    RegulatoryDocGenerator,
    SoftwareLifecycleRecord,
)
from src.deployment.robustness import (
    CertifiedRadius,
    ManifoldDistanceChecker,
    PGDAttacker,
    RandomizedSmoother,
    RobustnessReport,
)
from src.deployment.selective_predict import (
    BayesianMonitor,
    CalibrationAnalyzer,
    PredictionPoweredInference,
    RejectOptionClassifier,
    SelectivePredictionResult,
    SelectivePredictor,
)

__all__ = [
    # drift_monitor
    "ChangePointDetector",
    "ConformalMonitor",
    "DriftMonitor",
    "DriftMonitorResult",
    "MMDTest",
    "WassersteinDrift",
    # fairness_audit
    "CausalFairnessAuditor",
    "EqualizedOddsPostProcessor",
    "FairnessAuditReport",
    "FairnessHardeningAuditor",
    "IncompatibilityAnalysis",
    # privacy
    "DPAccountant",
    "DPSGDTrainer",
    "FederatedAggregator",
    "PrivacyBudget",
    "SecureAggregator",
    # regulatory_docs
    "ComplianceReport",
    "GMP5RiskAssessment",
    "ModelCardGenerator",
    "RegulatoryDocGenerator",
    "SoftwareLifecycleRecord",
    # robustness
    "CertifiedRadius",
    "ManifoldDistanceChecker",
    "PGDAttacker",
    "RandomizedSmoother",
    "RobustnessReport",
    # selective_predict
    "BayesianMonitor",
    "CalibrationAnalyzer",
    "PredictionPoweredInference",
    "RejectOptionClassifier",
    "SelectivePredictor",
    "SelectivePredictionResult",
]
