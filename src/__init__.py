"""
Enterprise Financial Fraud Intelligence & Risk Forensics ML Platform Core Package.
"""

from src.config.settings import PlatformSettings
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.features.transformer import FeatureTransformer
from src.models.inference import ChampionModelEngine
from src.models.calibrator import ProbabilityCalibrator
from src.policy.engine import AdaptiveDecisionEngine
from src.explainability.shap_engine import TreeSHAPExplainer
from src.explainability.adverse_action import AdverseActionEngine
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.performance import LatencyProfiler

__version__ = "2.0.0"

__all__ = [
    "PlatformSettings",
    "DataLoader",
    "DataValidator",
    "FeatureTransformer",
    "ChampionModelEngine",
    "ProbabilityCalibrator",
    "AdaptiveDecisionEngine",
    "TreeSHAPExplainer",
    "AdverseActionEngine",
    "DriftDetector",
    "LatencyProfiler"
]
