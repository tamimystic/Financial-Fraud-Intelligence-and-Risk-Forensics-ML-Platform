"""
4-Tier Adaptive Risk Policy Engine with Dynamic Escalation.
"""

from typing import Dict, Any
from src.config.settings import (
    THRESHOLD_FRICTIONLESS,
    THRESHOLD_STEP_UP_MAX,
    THRESHOLD_MANUAL_REVIEW_MAX,
    HIGH_VALUE_THRESHOLD,
    COST_CHARGEBACK,
    COST_FRICTION,
    COST_REVIEW
)
from src.policy.interventions import StepUpInterventionEngine

class AdaptiveDecisionEngine:
    def __init__(
        self,
        theta_frictionless: float = THRESHOLD_FRICTIONLESS,
        theta_step_up_max: float = THRESHOLD_STEP_UP_MAX,
        theta_manual_max: float = THRESHOLD_MANUAL_REVIEW_MAX,
        high_value_limit: float = HIGH_VALUE_THRESHOLD
    ):
        self.theta_frictionless = theta_frictionless
        self.theta_step_up_max = theta_step_up_max
        self.theta_manual_max = theta_manual_max
        self.high_value_limit = high_value_limit
        self.intervention_engine = StepUpInterventionEngine()

    def evaluate_policy(self, calibrated_prob: float, transaction_amount: float) -> Dict[str, Any]:
        p = float(np_clip(calibrated_prob, 0.0, 1.0))
        amt = float(transaction_amount)
        
        is_high_value = amt >= self.high_value_limit
        expected_dollar_loss = p * (amt + COST_CHARGEBACK)
        
        if p < self.theta_frictionless:
            action = "APPROVE"
            action_tier = "TIER_1_FRICTIONLESS"
            explanation = "Extremely low probability of fraud; automated frictionless approval."
            recommendation = "Authorize transaction immediately."
            friction_cost = 0.0
            requires_otp = False
            requires_manual_ops = False
        elif p < self.theta_step_up_max:
            action = "CHALLENGE_3DS"
            action_tier = "TIER_2_STEP_UP_CHALLENGE"
            explanation = "Moderate statistical anomaly detected; dispatching 3D-Secure 2.2 / SMS OTP challenge."
            recommendation = "Hold settlement pending cardholder multi-factor authentication."
            friction_cost = COST_FRICTION * 0.20
            requires_otp = True
            requires_manual_ops = False
        elif p < self.theta_manual_max:
            action = "MANUAL_REVIEW"
            action_tier = "TIER_3_MANUAL_OPERATIONS_REVIEW"
            explanation = "Elevated risk index; transaction routed to Fraud Investigation Desk."
            recommendation = "Analyst inspection required prior to cardholder authorization."
            friction_cost = COST_REVIEW
            requires_otp = False
            requires_manual_ops = True
        else:
            action = "HARD_BLOCK"
            action_tier = "TIER_4_HARD_BLOCK"
            explanation = "Severe fraud risk score; automated decline to protect cardholder and merchant."
            recommendation = "Decline authorization and log incident for forensic registry."
            friction_cost = 0.0
            requires_otp = False
            requires_manual_ops = False
            
        if is_high_value and action == "APPROVE":
            action = "MANUAL_REVIEW"
            action_tier = "TIER_3_HIGH_VALUE_SAFEGUARD"
            explanation = f"Monetary amount (${amt:,.2f}) exceeds high-value threshold; safety review triggered."
            recommendation = "Perform senior risk desk approval."
            requires_manual_ops = True

        return {
            "fraud_probability": round(p, 6),
            "fraud_percentage": round(p * 100.0, 4),
            "action": action,
            "action_tier": action_tier,
            "expected_dollar_loss": round(expected_dollar_loss, 2),
            "friction_cost": round(friction_cost, 2),
            "requires_otp": requires_otp,
            "requires_manual_ops": requires_manual_ops,
            "is_high_value": is_high_value,
            "explanation": explanation,
            "recommendation": recommendation
        }

    def verify_otp_challenge(self, entered_otp: str) -> Dict[str, Any]:
        return self.intervention_engine.verify_otp(entered_otp)

def np_clip(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))
