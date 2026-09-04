"""
Batch Fraud Auditing with Custom 3D Manifold Cluster Topology.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.config.settings import TEST_FEATURES_PATH

def render_3d_batch_manifold(merged_df: pd.DataFrame, x_axis: str, y_axis: str, z_axis: str):
    color_map = {
        "APPROVE": "#10B981",
        "CHALLENGE_3DS": "#F59E0B",
        "MANUAL_REVIEW": "#EC4899",
        "HARD_BLOCK": "#EF4444"
    }

    fig = go.Figure()

    for action, color in color_map.items():
        subset = merged_df[merged_df["Action_Decision"] == action]
        if len(subset) > 0:
            fig.add_trace(go.Scatter3d(
                x=subset[x_axis],
                y=subset[y_axis],
                z=subset[z_axis],
                mode='markers',
                name=f"{action} ({len(subset)} tx)",
                marker=dict(
                    size=4 if action == "APPROVE" else 6,
                    color=color,
                    opacity=0.65 if action == "APPROVE" else 0.95,
                    line=dict(width=0.5, color='#FFFFFF')
                ),
                hovertemplate=(
                    "<b>Decision: " + action + "</b><br>" +
                    "Amount: $%{customdata[0]:,.2f}<br>" +
                    "Risk Prob: %{customdata[1]:.4f}<br>" +
                    f"{x_axis}: " + "%{x:.2f}<br>" +
                    f"{y_axis}: " + "%{y:.2f}<br>" +
                    f"{z_axis}: " + "%{z:.2f}<extra></extra>"
                ),
                customdata=np.stack((subset["Amount ($)"], subset["Fraud_Probability"]), axis=-1)
            ))

    fig.update_layout(
        title=f"<b>3D Batch Forensics Manifold ({x_axis} vs {y_axis} vs {z_axis})</b>",
        scene=dict(
            xaxis=dict(title=x_axis, backgroundcolor="#0B0F19", gridcolor="#1E293B", showbackground=True),
            yaxis=dict(title=y_axis, backgroundcolor="#0B0F19", gridcolor="#1E293B", showbackground=True),
            zaxis=dict(title=z_axis, backgroundcolor="#0B0F19", gridcolor="#1E293B", showbackground=True),
            bgcolor="#0B0F19",
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.3)
            )
        ),
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19",
        font=dict(color="#E2E8F0"),
        height=580,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def render_batch_audit_view(model_engine, decision_engine):
    st.markdown("### 3D Batch Transaction Forensics Lab & Cluster Topology")
    st.markdown("Select your audit batch volume and click **RUN BATCH FORENSIC AUDIT STREAM** to execute high-speed multi-record classification, 3D point cloud topology mapping, and export forensic CSV audit manifests.")

    col_count, col_btn = st.columns([1, 2])
    sample_size = col_count.selectbox("Select Audit Batch Volume", [500, 1000, 2000, 5000], index=1)
    run_batch_btn = col_btn.button(f"RUN BATCH FORENSIC AUDIT STREAM ({sample_size:,} TRANSACTIONS)", type="primary", use_container_width=True)

    if "batch_executed" not in st.session_state:
        st.session_state.batch_executed = False

    if run_batch_btn:
        with st.spinner("Executing GPU-accelerated batch forensics audit..."):
            test_df = pd.read_parquet(TEST_FEATURES_PATH).head(sample_size)

            X_batch = np.ascontiguousarray(test_df[model_engine.feature_names].values)
            probs = model_engine.predict_risk_probability(X_batch)

            batch_results = []
            for i, (p, amt) in enumerate(zip(probs, test_df["Amount"])):
                pol = decision_engine.evaluate_policy(p, amt)
                batch_results.append({
                    "Transaction_Index": i + 1,
                    "Amount ($)": round(amt, 2),
                    "Fraud_Probability": pol["fraud_probability"],
                    "Risk_Percentage": f"{pol['fraud_percentage']:.2f}%",
                    "Action_Decision": pol["action"],
                    "Decision_Tier": pol["action_tier"],
                    "Expected_Loss ($)": pol["expected_dollar_loss"]
                })

            b_df = pd.DataFrame(batch_results)
            st.session_state.b_df = b_df
            st.session_state.test_batch_df = test_df
            st.session_state.batch_executed = True

    if st.session_state.batch_executed:
        b_df = st.session_state.b_df
        test_df = st.session_state.test_batch_df

        st.markdown("---")
        st.markdown("### Batch Audit Summary KPIs")

        b1, b2, b3, b4 = st.columns(4)
        total_flagged = len(b_df[b_df["Action_Decision"] == "HARD_BLOCK"])
        total_challenged = len(b_df[b_df["Action_Decision"] == "CHALLENGE_3DS"])
        total_app = len(b_df[b_df["Action_Decision"] == "APPROVE"])
        total_prevented = b_df[b_df["Action_Decision"] == "HARD_BLOCK"]["Amount ($)"].sum()

        b1.metric("Total Audited", f"{len(b_df):,}")
        b2.metric("Hard Fraud Blocks", total_flagged)
        b3.metric("3DS Step-Up Challenges", total_challenged)
        b4.metric("Direct Dollar Savings", f"${total_prevented:,.2f}")

        st.markdown("#### 3D Manifold Topological Projection")
        c_x, c_y, c_z = st.columns(3)
        available_dims = ["V14", "V10", "V12", "V4", "V17", "V11", "Amount", "Hour_of_Day", "iForest_Anomaly_Score"]
        x_axis = c_x.selectbox("3D X-Axis Dimension", available_dims, index=0)
        y_axis = c_y.selectbox("3D Y-Axis Dimension", available_dims, index=1)
        z_axis = c_z.selectbox("3D Z-Axis Dimension", available_dims, index=2)

        merged_df = pd.concat([b_df, test_df.reset_index(drop=True)], axis=1)
        render_3d_batch_manifold(merged_df, x_axis, y_axis, z_axis)

        st.markdown("#### Detailed Transaction Forensics Table")
        st.dataframe(b_df, use_container_width=True)

        csv_data = b_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Certified Forensic Audit Report (CSV)",
            data=csv_data,
            file_name="enterprise_fraud_audit_report.csv",
            mime="text/csv",
            use_container_width=True
        )
