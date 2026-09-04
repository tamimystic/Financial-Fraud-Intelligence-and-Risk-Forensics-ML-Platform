"""
3D-Secure 2.2 and Multi-Factor Step-Up Challenge Interventions.
"""

from typing import Dict, Any

class StepUpInterventionEngine:
    def __init__(self, valid_mock_otp: str = "749201"):
        self.valid_mock_otp = valid_mock_otp

    def verify_otp(self, entered_otp: str) -> Dict[str, Any]:
        is_success = (str(entered_otp).strip() == self.valid_mock_otp)
        return {
            "verification_status": "PASSED" if is_success else "FAILED",
            "final_decision": "APPROVE" if is_success else "HARD_BLOCK",
            "message": "Cardholder successfully authenticated via SMS OTP" if is_success else "Invalid OTP code provided. Transaction blocked."
        }
