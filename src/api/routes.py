"""
FastAPI Microservice Endpoints and Request Handlers.
"""

import time
import uuid
import numpy as np
import pandas as pd
from typing import List
from fastapi import APIRouter

from src.config.settings import TRAIN_FEATURES_PATH, TEST_FEATURES_PATH
from src.features.transformer import FeatureTransformer
from src.models.inference import ChampionModelEngine
from src.policy.engine import AdaptiveDecisionEngine
from src.explainability.adverse_action import AdverseActionEngine
from src.monitoring.drift_detector import DriftDetector
from src.api.schemas import (
    SingleTransactionInput,
    BatchTransactionInput,
    StepUpVerificationInput,
    PredictionResponse,
    BatchPredictionResponse,
    BatchItemPrediction,
    DriftHealthResponse,
    AdverseActionCode
)

router = APIRouter()

model_engine = ChampionModelEngine()
model_engine.load_artifacts()

feature_transformer = FeatureTransformer()
decision_engine = AdaptiveDecisionEngine()
adverse_action_engine = AdverseActionEngine(model_engine)
drift_detector = DriftDetector()

@router.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "HEALTHY",
        "service": "Financial Fraud Intelligence Platform",
        "version": "2.0.0",
        "model_loaded": model_engine.model is not None,
        "calibrator_loaded": model_engine.calibrator is not None
    }

@router.post("/predict/single", response_model=PredictionResponse, tags=["Scoring"])
def score_single_transaction(payload: SingleTransactionInput):
    t0 = time.perf_counter()
    tx_id = f"TX-{uuid.uuid4().hex[:8].upper()}"
    
    raw_df = pd.DataFrame([payload.model_dump()])
    transformed_df = feature_transformer.transform(raw_df)
    
    feature_matrix = np.ascontiguousarray(transformed_df[model_engine.feature_names].values)
    calibrated_prob = float(model_engine.predict_risk_probability(feature_matrix)[0])
    
    policy_eval = decision_engine.evaluate_policy(calibrated_prob, payload.Amount)
    
    codes_data = adverse_action_engine.generate_adverse_action_codes(transformed_df[model_engine.feature_names])
    
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    codes = [
        AdverseActionCode(
            code=c["code"],
            feature=c["feature"],
            description=c["description"],
            shap_attribution=c["shap_attribution"]
        )
        for c in codes_data
    ]
    
    return PredictionResponse(
        status="SUCCESS",
        transaction_id=tx_id,
        fraud_probability=policy_eval["fraud_probability"],
        fraud_percentage=policy_eval["fraud_percentage"],
        decision=policy_eval["action"],
        decision_tier=policy_eval["action_tier"],
        expected_dollar_loss=policy_eval["expected_dollar_loss"],
        friction_cost=policy_eval["friction_cost"],
        requires_otp_challenge=policy_eval["requires_otp"],
        requires_manual_review=policy_eval["requires_manual_ops"],
        is_high_value=policy_eval["is_high_value"],
        explanation=policy_eval["explanation"],
        recommendation=policy_eval["recommendation"],
        top_adverse_action_codes=codes,
        inference_latency_ms=round(elapsed_ms, 3)
    )

@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Batch Auditing"])
def score_batch_transactions(payload: BatchTransactionInput):
    t0 = time.perf_counter()
    raw_df = pd.DataFrame([t.model_dump() for t in payload.transactions])
    
    transformed_df = feature_transformer.transform(raw_df)
    feature_matrix = np.ascontiguousarray(transformed_df[model_engine.feature_names].values)
    calibrated_probs = model_engine.predict_risk_probability(feature_matrix)
    
    results: List[BatchItemPrediction] = []
    total_blocked = 0
    total_step_up = 0
    total_manual = 0
    total_approved = 0
    total_loss_prevented = 0.0
    
    for idx, (p, amt) in enumerate(zip(calibrated_probs, raw_df["Amount"])):
        policy = decision_engine.evaluate_policy(p, amt)
        action = policy["action"]
        
        if action == "HARD_BLOCK":
            total_blocked += 1
            total_loss_prevented += amt
        elif action == "CHALLENGE_3DS":
            total_step_up += 1
        elif action == "MANUAL_REVIEW":
            total_manual += 1
        else:
            total_approved += 1
            
        results.append(BatchItemPrediction(
            index=idx,
            amount=round(float(amt), 2),
            fraud_probability=policy["fraud_probability"],
            decision=policy["action"],
            decision_tier=policy["action_tier"],
            expected_dollar_loss=policy["expected_dollar_loss"]
        ))
        
    elapsed_sec = time.perf_counter() - t0
    
    return BatchPredictionResponse(
        status="SUCCESS",
        total_transactions=len(raw_df),
        total_fraud_flagged=total_blocked,
        total_step_up_challenges=total_step_up,
        total_manual_reviews=total_manual,
        total_approved=total_approved,
        projected_loss_prevented=round(total_loss_prevented, 2),
        processing_time_seconds=round(elapsed_sec, 3),
        results=results
    )

@router.post("/decision/step-up-verify", tags=["Interventions"])
def verify_step_up_challenge(payload: StepUpVerificationInput):
    res = decision_engine.verify_otp_challenge(payload.entered_otp)
    return {
        "status": "SUCCESS",
        "transaction_id": payload.transaction_id,
        "verification_result": res["verification_status"],
        "final_action": res["final_decision"],
        "message": res["message"]
    }

@router.get("/mlops/drift-health", response_model=DriftHealthResponse, tags=["Governance"])
def check_mlops_drift_status():
    val_df = pd.read_parquet(TRAIN_FEATURES_PATH)
    test_df = pd.read_parquet(TEST_FEATURES_PATH)
    
    drop_cols = [c for c in ["Class", "Time"] if c in val_df.columns]
    feats = [c for c in val_df.columns if c not in drop_cols]
    
    base_probs = model_engine.predict_risk_probability(np.ascontiguousarray(val_df[feats].values))
    curr_probs = model_engine.predict_risk_probability(np.ascontiguousarray(test_df[feats].values))
    
    psi_report = drift_detector.compute_psi(base_probs, curr_probs)
    
    return DriftHealthResponse(
        status="SUCCESS",
        model_version="2.0.0-Champion-XGBoost",
        stability_tier=psi_report["stability_tier"],
        psi_score=psi_report["psi_score"],
        action_required=psi_report["operational_action"],
        decile_drift_analysis={
            "baseline_deciles": psi_report["decile_baseline_percentages"],
            "current_deciles": psi_report["decile_current_percentages"]
        }
    )
