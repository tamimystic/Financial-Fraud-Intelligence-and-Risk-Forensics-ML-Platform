"""
Pydantic v2 Request, Response, and Compliance Schemas.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SingleTransactionInput(BaseModel):
    Time: float = Field(default=0.0, description="Elapsed seconds from initial transaction in dataset.")
    Amount: float = Field(..., gt=0.0, description="Transaction monetary amount.")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0

class BatchTransactionInput(BaseModel):
    transactions: List[SingleTransactionInput] = Field(..., max_length=5000)

class StepUpVerificationInput(BaseModel):
    transaction_id: str
    entered_otp: str

class AdverseActionCode(BaseModel):
    code: str
    feature: str
    description: str
    shap_attribution: float

class PredictionResponse(BaseModel):
    status: str
    transaction_id: str
    fraud_probability: float
    fraud_percentage: float
    decision: str
    decision_tier: str
    expected_dollar_loss: float
    friction_cost: float
    requires_otp_challenge: bool
    requires_manual_review: bool
    is_high_value: bool
    explanation: str
    recommendation: str
    top_adverse_action_codes: List[AdverseActionCode]
    inference_latency_ms: float

class BatchItemPrediction(BaseModel):
    index: int
    amount: float
    fraud_probability: float
    decision: str
    decision_tier: str
    expected_dollar_loss: float

class BatchPredictionResponse(BaseModel):
    status: str
    total_transactions: int
    total_fraud_flagged: int
    total_step_up_challenges: int
    total_manual_reviews: int
    total_approved: int
    projected_loss_prevented: float
    processing_time_seconds: float
    results: List[BatchItemPrediction]

class DriftHealthResponse(BaseModel):
    status: str
    model_version: str
    stability_tier: str
    psi_score: float
    action_required: str
    decile_drift_analysis: Dict[str, Any]
