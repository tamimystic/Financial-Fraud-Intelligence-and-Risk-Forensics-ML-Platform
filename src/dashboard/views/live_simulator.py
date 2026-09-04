"""
Live Risk Operations & 3D Interactive Holographic HUD.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

def render_3d_interactive_card_component(card_token: str, amount: float, tier: str, fraud_pct: float):
    if tier == "APPROVE":
        glow_color = "#10B981"
        badge_bg = "rgba(16, 185, 129, 0.25)"
        badge_border = "#10B981"
        badge_text = "TIER 1: FRICTIONLESS APPROVE"
    elif tier == "CHALLENGE_3DS":
        glow_color = "#F59E0B"
        badge_bg = "rgba(245, 158, 11, 0.25)"
        badge_border = "#F59E0B"
        badge_text = "TIER 2: 3DS 2.2 OTP CHALLENGE"
    elif tier == "MANUAL_REVIEW":
        glow_color = "#EC4899"
        badge_bg = "rgba(236, 72, 153, 0.25)"
        badge_border = "#EC4899"
        badge_text = "TIER 3: MANUAL REVIEW DESK"
    else:
        glow_color = "#EF4444"
        badge_bg = "rgba(239, 68, 68, 0.25)"
        badge_border = "#EF4444"
        badge_text = "TIER 4: HARD FRAUD DECLINE"

    raw_html = f"""<!DOCTYPE html>
<html>
<head>
<style>
body {{
    margin: 0;
    padding: 0;
    background: transparent;
    display: flex;
    justify-content: center;
    align-items: center;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    overflow: hidden;
}}
.perspective-wrapper {{
    perspective: 1200px;
    padding: 8px;
}}
.card-3d {{
    width: 360px;
    height: 200px;
    background: linear-gradient(135deg, #0B132B 0%, #1C2541 50%, #0B132B 100%);
    border-radius: 16px;
    padding: 18px;
    box-sizing: border-box;
    border: 1px solid {glow_color}88;
    box-shadow: 0 20px 40px -10px {glow_color}44, inset 0 1px 1px rgba(255,255,255,0.2), inset 0 -1px 2px rgba(0,0,0,0.8);
    color: #FFFFFF;
    transform-style: preserve-3d;
    transition: transform 0.15s ease-out;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}}
.card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.shield-label {{
    font-size: 10px;
    letter-spacing: 2px;
    color: #94A3B8;
    font-weight: 600;
}}
.badge {{
    background: {badge_bg};
    border: 1px solid {badge_border};
    color: {glow_color};
    font-size: 8px;
    padding: 3px 6px;
    border-radius: 4px;
    font-weight: bold;
    letter-spacing: 1px;
}}
.chip-container {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 4px;
}}
.emv-chip {{
    width: 38px;
    height: 26px;
    background: linear-gradient(135deg, #FFD700 0%, #B8860B 100%);
    border-radius: 4px;
    border: 1px solid #FFF8DC;
    box-shadow: inset 0 0 3px rgba(0,0,0,0.6);
}}
.contactless-text {{
    font-size: 10px;
    color: #64748B;
    letter-spacing: 1px;
}}
.card-number {{
    font-size: 16px;
    letter-spacing: 3px;
    color: #F8FAFC;
    text-shadow: 0 2px 4px rgba(0,0,0,0.8);
    font-family: 'Courier New', monospace;
}}
.card-footer {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}}
.field-label {{
    font-size: 8px;
    color: #64748B;
    letter-spacing: 1px;
}}
.field-value {{
    font-size: 11px;
    color: #E2E8F0;
    font-weight: 600;
}}
.amount-value {{
    font-size: 15px;
    color: {glow_color};
    font-weight: bold;
    text-align: right;
}}
.risk-value {{
    font-size: 14px;
    color: #F8FAFC;
    font-weight: bold;
    text-align: right;
}}
</style>
</head>
<body>
<div class="perspective-wrapper">
<div id="targetCard" class="card-3d">
<div class="card-header">
<span class="shield-label">ENTERPRISE 3D SHIELD</span>
<span class="badge">{badge_text}</span>
</div>
<div class="chip-container">
<div class="emv-chip"></div>
<span class="contactless-text">EMV 3DS 2.2 SECURED</span>
</div>
<div class="card-number">4532  8901  2384  9482</div>
<div class="card-footer">
<div>
<div class="field-label">CARD TOKEN</div>
<div class="field-value">{card_token}</div>
</div>
<div>
<div class="field-label" style="text-align: right;">AMOUNT</div>
<div class="amount-value">${amount:,.2f}</div>
</div>
<div>
<div class="field-label" style="text-align: right;">RISK INDEX</div>
<div class="risk-value">{fraud_pct:.2f}%</div>
</div>
</div>
</div>
</div>
<script>
const card = document.getElementById('targetCard');
const wrapper = document.querySelector('.perspective-wrapper');
wrapper.addEventListener('mousemove', (e) => {{
    const rect = card.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    const rotX = -y / 8;
    const rotY = x / 8;
    card.style.transform = `rotateX(${{rotX}}deg) rotateY(${{rotY}}deg)`;
}});
wrapper.addEventListener('mouseleave', () => {{
    card.style.transform = 'rotateX(4deg) rotateY(-4deg)';
}});
card.style.transform = 'rotateX(4deg) rotateY(-4deg)';
</script>
</body>
</html>"""
    components.html(raw_html, height=220)

def render_3d_risk_manifold(current_v14: float, current_v10: float, current_v12: float, prob: float):
    np.random.seed(42)
    n_norm = 250
    norm_v14 = np.random.normal(0.0, 1.0, n_norm)
    norm_v10 = np.random.normal(0.0, 1.0, n_norm)
    norm_v12 = np.random.normal(0.0, 1.0, n_norm)

    n_fraud = 40
    fraud_v14 = np.random.normal(-6.0, 1.8, n_fraud)
    fraud_v10 = np.random.normal(-4.5, 1.5, n_fraud)
    fraud_v12 = np.random.normal(-3.5, 1.4, n_fraud)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=norm_v14,
        y=norm_v10,
        z=norm_v12,
        mode='markers',
        name='Legitimate Baseline Manifold',
        marker=dict(
            size=3,
            color='#10B981',
            opacity=0.35
        ),
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter3d(
        x=fraud_v14,
        y=fraud_v10,
        z=fraud_v12,
        mode='markers',
        name='Fraud Attacks Cluster',
        marker=dict(
            size=4,
            color='#EF4444',
            opacity=0.60
        ),
        hoverinfo='skip'
    ))

    marker_color = '#EF4444' if prob >= 0.30 else ('#F59E0B' if prob >= 0.02 else '#10B981')

    fig.add_trace(go.Scatter3d(
        x=[current_v14],
        y=[current_v10],
        z=[current_v12],
        mode='markers+text',
        name='Current Live Scored Transaction',
        text=['LIVE TX'],
        textposition='top center',
        textfont=dict(color='#FFFFFF', size=11),
        marker=dict(
            size=9,
            color=marker_color,
            symbol='diamond',
            line=dict(color='#FFFFFF', width=2),
            opacity=1.0
        ),
        hovertemplate="<b>Current Scored Transaction</b><br>V14: %{x:.2f}<br>V10: %{y:.2f}<br>V12: %{z:.2f}<br>Risk: " + f"{prob*100:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title="<b>3D Latent Risk Manifold (V14 vs V10 vs V12 Space)</b>",
        scene=dict(
            xaxis=dict(title='V14 (Behavioral Dim)', backgroundcolor="#0B0F19", gridcolor="#1E293B", showbackground=True),
            yaxis=dict(title='V10 (Merchant Dim)', backgroundcolor="#0B0F19", gridcolor="#1E293B", showbackground=True),
            zaxis=dict(title='V12 (Velocity Dim)', backgroundcolor="#0B0F19", gridcolor="#1E293B", showbackground=True),
            bgcolor="#0B0F19",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2)
            )
        ),
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19",
        font=dict(color="#E2E8F0"),
        height=480,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9)
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def init_live_session_state():
    if "tx_amount" not in st.session_state:
        st.session_state.tx_amount = 120.50
    if "tx_time" not in st.session_state:
        st.session_state.tx_time = 45000.0
    if "tx_card_token" not in st.session_state:
        st.session_state.tx_card_token = "CARD-EU-948210"
    for i in range(1, 29):
        key = f"tx_v{i}"
        if key not in st.session_state:
            st.session_state[key] = 0.0
    if "live_scored" not in st.session_state:
        st.session_state.live_scored = True
    if "otp_status" not in st.session_state:
        st.session_state.otp_status = "IDLE"
    if "otp_result_msg" not in st.session_state:
        st.session_state.otp_result_msg = ""

def apply_preset(preset_type: str):
    if preset_type == "legit":
        st.session_state.tx_amount = 45.50
        st.session_state.tx_time = 45000.0
        st.session_state.tx_v14 = 0.10
        st.session_state.tx_v10 = 0.05
        st.session_state.tx_v12 = 0.02
        st.session_state.tx_v4 = -0.08
        st.session_state.tx_v17 = 0.04
        st.session_state.tx_v11 = 0.01
        for i in [1,2,3,5,6,7,8,9,13,15,16,18,19,20,21,22,23,24,25,26,27,28]:
            st.session_state[f"tx_v{i}"] = 0.0
    elif preset_type == "fraud":
        st.session_state.tx_amount = 890.00
        st.session_state.tx_time = 14200.0
        st.session_state.tx_v14 = -6.20
        st.session_state.tx_v10 = -4.50
        st.session_state.tx_v12 = -3.80
        st.session_state.tx_v4 = 4.10
        st.session_state.tx_v17 = -5.10
        st.session_state.tx_v11 = 3.50
        st.session_state.tx_v16 = -3.20
        st.session_state.tx_v3 = -4.10
    elif preset_type == "high_val":
        st.session_state.tx_amount = 12500.00
        st.session_state.tx_time = 54000.0
        st.session_state.tx_v14 = 0.20
        st.session_state.tx_v10 = -0.10
        st.session_state.tx_v12 = 0.05
        st.session_state.tx_v4 = -0.15
        st.session_state.tx_v17 = 0.10
        st.session_state.tx_v11 = 0.05
        for i in [1,2,3,5,6,7,8,9,13,15,16,18,19,20,21,22,23,24,25,26,27,28]:
            st.session_state[f"tx_v{i}"] = 0.0
    st.session_state.otp_status = "IDLE"
    st.session_state.otp_result_msg = ""
    st.session_state.live_scored = True

def render_live_simulator_view(model_engine, feature_transformer, decision_engine, shap_explainer):
    init_live_session_state()

    st.markdown("### 3D Live Payment Gateway & Real-Time Risk Operations")
    st.markdown("Enter transaction telemetry below and click the **EXECUTE 3D REAL-TIME RISK SCORING** button to perform full-spectrum forensic inference, 3D manifold projection, and 4-tier policy evaluation.")

    st.markdown("#### Quick Load Production Presets")
    col_p1, col_p2, col_p3 = st.columns(3)
    if col_p1.button("Load Legitimate Transaction Sample", use_container_width=True):
        apply_preset("legit")
    if col_p2.button("Load High-Risk Fraud Attack Sample", use_container_width=True):
        apply_preset("fraud")
    if col_p3.button("Load High-Value Safe Transaction ($12,500)", use_container_width=True):
        apply_preset("high_val")

    st.markdown("#### Transaction Telemetry & Key Risk Dimensions")
    with st.container():
        c1, c2, c3 = st.columns(3)
        st.session_state.tx_amount = c1.number_input("Transaction Amount ($)", min_value=0.01, max_value=100000.0, value=float(st.session_state.tx_amount), step=10.0)
        st.session_state.tx_time = c2.number_input("Transaction Time (Seconds)", min_value=0.0, max_value=200000.0, value=float(st.session_state.tx_time), step=100.0)
        st.session_state.tx_card_token = c3.text_input("Card Token / Reference ID", value=st.session_state.tx_card_token)

        st.markdown("**Top Forensic Risk Components (V14, V10, V12, V4, V17, V11)**")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        st.session_state.tx_v14 = k1.number_input("V14 (Behavioral #1)", value=float(st.session_state.tx_v14), step=0.5)
        st.session_state.tx_v10 = k2.number_input("V10 (Merchant #2)", value=float(st.session_state.tx_v10), step=0.5)
        st.session_state.tx_v12 = k3.number_input("V12 (Velocity #3)", value=float(st.session_state.tx_v12), step=0.5)
        st.session_state.tx_v4 = k4.number_input("V4 (Acceleration #4)", value=float(st.session_state.tx_v4), step=0.5)
        st.session_state.tx_v17 = k5.number_input("V17 (Cross-Channel #5)", value=float(st.session_state.tx_v17), step=0.5)
        st.session_state.tx_v11 = k6.number_input("V11 (Usage Pattern #6)", value=float(st.session_state.tx_v11), step=0.5)

    with st.expander("Full 28 PCA Forensic Vectors Fine-Tuning Lab", expanded=False):
        st.markdown("**Behavioral Dimension Vectors (V1 - V10)**")
        cols_b = st.columns(5)
        for idx, i in enumerate(range(1, 11)):
            if i not in [10, 4]:
                st.session_state[f"tx_v{i}"] = cols_b[idx % 5].number_input(f"V{i}", value=float(st.session_state[f"tx_v{i}"]), step=0.5, key=f"inp_v{i}")

        st.markdown("**Terminal & Merchant Interaction Vectors (V11 - V20)**")
        cols_m = st.columns(5)
        for idx, i in enumerate(range(11, 21)):
            if i not in [14, 12, 17, 11]:
                st.session_state[f"tx_v{i}"] = cols_m[idx % 5].number_input(f"V{i}", value=float(st.session_state[f"tx_v{i}"]), step=0.5, key=f"inp_v{i}")

        st.markdown("**Velocity & Network Acceleration Vectors (V21 - V28)**")
        cols_v = st.columns(4)
        for idx, i in enumerate(range(21, 29)):
            st.session_state[f"tx_v{i}"] = cols_v[idx % 4].number_input(f"V{i}", value=float(st.session_state[f"tx_v{i}"]), step=0.5, key=f"inp_v{i}")

    st.markdown("<br>", unsafe_allow_html=True)
    score_btn = st.button("EXECUTE 3D REAL-TIME RISK SCORING & POLICY ROUTING", type="primary", use_container_width=True)

    if score_btn:
        st.session_state.live_scored = True
        st.session_state.otp_status = "IDLE"
        st.session_state.otp_result_msg = ""

    if st.session_state.live_scored:
        t0 = time.perf_counter()

        raw_dict = {"Time": st.session_state.tx_time, "Amount": st.session_state.tx_amount}
        for i in range(1, 29):
            raw_dict[f"V{i}"] = float(st.session_state[f"tx_v{i}"])

        raw_df = pd.DataFrame([raw_dict])
        transformed_df = feature_transformer.transform(raw_df)

        X_mat = np.ascontiguousarray(transformed_df[model_engine.feature_names].values)
        prob = float(model_engine.predict_risk_probability(X_mat)[0])

        policy = decision_engine.evaluate_policy(prob, st.session_state.tx_amount)
        top_drivers = shap_explainer.get_top_contributors(transformed_df[model_engine.feature_names], top_k=6)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        st.markdown("---")
        st.markdown("### Real-Time Forensic Results & 3D Intelligence")

        render_3d_interactive_card_component(st.session_state.tx_card_token, st.session_state.tx_amount, policy["action"], policy["fraud_percentage"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calibrated Fraud Probability", f"{policy['fraud_percentage']:.2f}%")
        m2.metric("4-Tier Policy Action", policy["action"])
        m3.metric("Expected Monetary Loss", f"${policy['expected_dollar_loss']:,.2f}")
        m4.metric("Sub-5ms Inference SLA", f"{latency_ms:.2f} ms")

        if policy["action"] == "APPROVE":
            st.success(f"**{policy['action_tier']}**: {policy['explanation']} ({policy['recommendation']})")
        elif policy["action"] == "CHALLENGE_3DS":
            st.info(f"**{policy['action_tier']}**: {policy['explanation']} ({policy['recommendation']})")

            with st.container():
                st.markdown("#### 3D-Secure 2.2 / SMS OTP Interactive Authentication Modal")
                st.markdown("A dynamic step-up verification code has been dispatched to cardholder mobile ending in **XXXX-8921**.")
                c_otp_in, c_otp_btn = st.columns([3, 1])
                otp_in = c_otp_in.text_input("Enter 6-Digit SMS OTP Code (Sample Valid Code: 749201)", key="otp_verify_input")
                if c_otp_btn.button("Submit OTP Verification", key="otp_sub_btn", use_container_width=True):
                    ver_res = decision_engine.verify_otp_challenge(otp_in)
                    st.session_state.otp_status = ver_res["verification_status"]
                    st.session_state.otp_result_msg = f"{ver_res['message']} (Final Action: {ver_res['final_decision']})"

                if st.session_state.otp_status == "PASSED":
                    st.success(f"OTP Verified: {st.session_state.otp_result_msg}")
                elif st.session_state.otp_status == "FAILED":
                    st.error(f"OTP Failed: {st.session_state.otp_result_msg}")
        elif policy["action"] == "MANUAL_REVIEW":
            st.warning(f"**{policy['action_tier']}**: {policy['explanation']} ({policy['recommendation']})")
        else:
            st.error(f"**{policy['action_tier']}**: {policy['explanation']} ({policy['recommendation']})")

        col_3d, col_shap = st.columns([1.2, 1.0])

        with col_3d:
            render_3d_risk_manifold(float(st.session_state.tx_v14), float(st.session_state.tx_v10), float(st.session_state.tx_v12), prob)

        with col_shap:
            st.markdown("#### Real-Time TreeSHAP Risk Drivers")
            shap_df = pd.DataFrame(top_drivers)
            fig_shap = px.bar(
                shap_df,
                x="shap_value",
                y="feature",
                orientation="h",
                title="<b>Feature Impact on Fraud Log-Odds</b>",
                labels={"shap_value": "SHAP Attribution (Positive = Increases Risk)", "feature": "Feature Vector"},
                color="shap_value",
                color_continuous_scale="Reds"
            )
            fig_shap.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="#0B0F19",
                plot_bgcolor="#0B0F19",
                font=dict(color="#E2E8F0"),
                height=480
            )
            st.plotly_chart(fig_shap, use_container_width=True)
