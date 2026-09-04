"""
Enterprise Financial Fraud Intelligence & Risk Forensics 3D Platform Dashboard.
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.features.transformer import FeatureTransformer
from src.models.inference import ChampionModelEngine
from src.policy.engine import AdaptiveDecisionEngine
from src.explainability.shap_engine import TreeSHAPExplainer
from src.monitoring.drift_detector import DriftDetector

from src.dashboard.views.live_simulator import render_live_simulator_view
from src.dashboard.views.batch_audit import render_batch_audit_view
from src.dashboard.views.roi_simulator import render_roi_simulator_view
from src.dashboard.views.adverse_action_view import render_adverse_action_view
from src.dashboard.views.mlops_health import render_mlops_health_view

st.set_page_config(
    page_title="Enterprise 3D Fraud Forensics Operations Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

clean_custom_css = """<style>
.stApp {
    background-color: #030712;
    color: #F8FAFC;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #0B0F19;
    padding: 8px;
    border-radius: 12px;
    border: 1px solid #1E293B;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94A3B8;
    font-weight: 600;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background-color: #1E293B !important;
    color: #38BDF8 !important;
    border: 1px solid #38BDF844 !important;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
}
div[data-testid="stMetricValue"] {
    font-family: 'Courier New', monospace;
    color: #38BDF8;
    text-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
}
.stButton>button {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #38BDF855;
    color: #F8FAFC;
    border-radius: 8px;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
.stButton>button:hover {
    border-color: #38BDF8;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    transform: translateY(-2px);
}
</style>"""

st.markdown(clean_custom_css, unsafe_allow_html=True)

@st.cache_resource
def load_platform_engines():
    model_eng = ChampionModelEngine()
    model_eng.load_artifacts()
    feat_trans = FeatureTransformer()
    dec_eng = AdaptiveDecisionEngine()
    shap_eng = TreeSHAPExplainer(model_eng)
    drift_det = DriftDetector()
    return model_eng, feat_trans, dec_eng, shap_eng, drift_det

model_engine, feature_transformer, decision_engine, shap_explainer, drift_detector = load_platform_engines()

st.sidebar.title("3D Operations Command")
st.sidebar.markdown("**Architecture**: Tier-1 Payment Gateway")
st.sidebar.markdown("**3D Engine**: WebGL GPU-Accelerated")
st.sidebar.markdown("**Model Engine**: XGBoost + Isotonic Calibration")
st.sidebar.markdown("**Adaptive Policy**: 4-Tier Zero-Miss Routing")
st.sidebar.markdown("**Compliance**: FCRA / ECOA / SR 11-7")
st.sidebar.markdown("---")

st.title("Enterprise 3D Financial Fraud Intelligence & Forensics Center")
st.markdown("Sub-5ms High-Throughput Risk Scoring, 3D Latent Manifold Topologies, 3D Loss Terrain Modeling, and FCRA/ECOA Regulatory Compliance Suite.")

tabs = st.tabs([
    "1. 3D Live Risk Simulator",
    "2. 3D Batch Forensics Audit",
    "3. 3D Cost-Utility Terrain",
    "4. FCRA Adverse Action",
    "5. 3D MLOps Drift Waterfall"
])

with tabs[0]:
    render_live_simulator_view(model_engine, feature_transformer, decision_engine, shap_explainer)

with tabs[1]:
    render_batch_audit_view(model_engine, decision_engine)

with tabs[2]:
    render_roi_simulator_view()

with tabs[3]:
    render_adverse_action_view()

with tabs[4]:
    render_mlops_health_view(model_engine, drift_detector)
