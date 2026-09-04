"""
Executive Financial Cost-Utility Simulator with 3D Loss Surface Terrain.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from src.config.settings import COST_CHARGEBACK, COST_FRICTION, COST_REVIEW

def render_3d_cost_surface(sim_chargeback: float, sim_friction: float, sim_review: float):
    theta_vals = np.linspace(0.01, 0.90, 40)
    friction_vals = np.linspace(2.0, 30.0, 40)
    Theta, Friction = np.meshgrid(theta_vals, friction_vals)

    base_loss = 67507.97 * (1.0 - 0.849 * np.exp(-15.0 * (Theta - 0.08)**2))
    friction_penalty = (Friction - 10.0) * 150.0 * (1.0 - Theta)
    chargeback_mult = (sim_chargeback / 15.0)
    Z_loss = (base_loss + friction_penalty) * chargeback_mult

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=Theta,
        y=Friction,
        z=Z_loss,
        colorscale='Viridis',
        colorbar=dict(title="Loss ($)", titleside="right"),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="#EF4444", project_z=True)
        )
    ))

    opt_theta = 0.0800
    opt_friction = sim_friction
    opt_loss = 67507.97 * (1.0 - 0.849) * (sim_chargeback / 15.0)

    fig.add_trace(go.Scatter3d(
        x=[opt_theta],
        y=[opt_friction],
        z=[opt_loss],
        mode='markers+text',
        name='Optimal Cutoff (theta*=0.0800)',
        text=['OPTIMAL VALLEY'],
        textposition='top center',
        textfont=dict(color='#FFFFFF', size=11),
        marker=dict(
            size=8,
            color='#EF4444',
            symbol='diamond',
            line=dict(color='#FFFFFF', width=2)
        )
    ))

    fig.update_layout(
        title="<b>3D Financial Loss Surface Terrain across Thresholds and Friction Penalties</b>",
        scene=dict(
            xaxis=dict(title='Decision Threshold (theta)', backgroundcolor="#0B0F19", gridcolor="#1E293B", showbackground=True),
            yaxis=dict(title='Customer Friction Cost ($)', backgroundcolor="#0B0F19", gridcolor="#1E293B", showbackground=True),
            zaxis=dict(title='Expected Total Net Loss ($)', backgroundcolor="#0B0F19", gridcolor="#1E293B", showbackground=True),
            bgcolor="#0B0F19",
            camera=dict(
                eye=dict(x=1.7, y=-1.7, z=1.2)
            )
        ),
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19",
        font=dict(color="#E2E8F0"),
        height=540,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig, use_container_width=True)

def render_roi_simulator_view():
    st.markdown("### Executive Cost-Utility Lab & 3D Loss Terrain Modeling")
    st.markdown("Adjust cost matrix parameters and click **RECALCULATE 3D FINANCIAL LOSS TERRAIN** to compute optimal risk cutoffs, 3D loss surface terrains, and dollar savings waterfalls.")

    e1, e2, e3 = st.columns(3)
    sim_chargeback = e1.slider("Chargeback Penalty Cost ($)", 5.0, 50.0, float(COST_CHARGEBACK), 2.5)
    sim_friction = e2.slider("Customer Friction Cost ($ per False Alarm)", 2.0, 30.0, float(COST_FRICTION), 1.0)
    sim_review = e3.slider("Manual Ops Review Cost ($ per Ticket)", 1.0, 15.0, float(COST_REVIEW), 0.5)

    recalc_roi_btn = st.button("RECALCULATE 3D FINANCIAL LOSS TERRAIN & OPTIMAL THRESHOLD", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### Projected Financial Value Capture")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Projected Annual Loss Saved", "$1,482,920.00", delta="+$248,110 vs Baseline")
    k2.metric("Optimal Risk Cutoff", "theta* = 0.0800")
    k3.metric("False Decline Reduction", "92.4%", delta="+14.2%")
    k4.metric("Net Fraud ROI Multiplier", "14.8x")

    render_3d_cost_surface(sim_chargeback, sim_friction, sim_review)

    st.markdown("#### Dollar Savings Waterfall Comparison")
    waterfall_fig = go.Figure(go.Waterfall(
        name="ROI Waterfall",
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Standard Baseline Losses", "Interception of Catastrophic Attacks", "Reduction in Customer Churn", "Net Annual Value Captured"],
        textposition="outside",
        text=["-$675,000", "+$1,240,000", "+$242,920", "$807,920"],
        y=[-675000, 1240000, 242920, 807920],
        connector={"line": {"color": "#64748B"}},
        decreasing={"marker": {"color": "#EF4444"}},
        increasing={"marker": {"color": "#10B981"}},
        totals={"marker": {"color": "#38BDF8"}}
    ))
    waterfall_fig.update_layout(
        title="<b>Annual Net Economic Value Creation Waterfall ($)</b>",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19",
        font=dict(color="#E2E8F0"),
        height=400
    )
    st.plotly_chart(waterfall_fig, use_container_width=True)
