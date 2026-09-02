# Financial Fraud Intelligence & Risk Forensics ML Platform

An enterprise-grade, end-to-end Machine Learning, Deep Representation Learning, Graph Forensics, and Adaptive MLOps platform analyzing 284,807 financial transactions for extreme class-imbalance fraud detection and asymmetric cost-utility risk management.

---

## Executive Architecture

This platform implements a 7-squad corporate risk engineering framework across 18 specialized research notebooks:
1. **Forensic Statistical Physics**: Heavy-tail transaction physics, power-law Pareto distributions, and KS-divergence.
2. **Financial Graph Forensics**: Transaction-card-merchant bipartite graphs and Louvain community fraud rings.
3. **Asymmetric Cost-Utility Engineering**: Dollar loss optimization against customer friction and chargeback costs.
4. **Temporal Feature Store**: Rolling acceleration velocity windows (5m, 1h, 24h) with strict leak-free temporal splitting.
5. **Cost-Sensitive Supervised Ensembles**: XGBoost, LightGBM, CatBoost with optimal threshold search and probability calibration.
6. **Deep Anomaly Autoencoders**: PyTorch Variational Autoencoders (VAEs) and TabNet sequential attention.
7. **Explainability & Compliance**: TreeSHAP, LIME surrogates, Adverse Action reason codes (FCRA/ECOA), and MLOps drift monitoring.

---

## Master 18-Notebook Research Suite

| # | Notebook Name | Focus Domain & Methodology | Status |
| :---: | :--- | :--- | :---: |
| **01** | `01_Comprehensive_Univariate_Transaction_Analysis.ipynb` | 31-Feature Profiling Matrix, Amount Power-Law Pareto Decay, 48-Hour Diurnal Cyclicity, 28 PCA KDE Distributions, 577.87:1 Class Imbalance Geometry | 100% Executed (0 Errors) |
| **02** | `02_Comprehensive_Bivariate_and_Class_Separability_EDA.ipynb` | Legitimate vs Fraud Separability, Kolmogorov-Smirnov Tests, Point Biserial Correlations | Scheduled |
| **03** | `03_High_Dimensional_Multivariate_and_Manifold_Topology_EDA.ipynb` | Covariance Geometry, PCA Orthogonality, t-SNE 2D/3D, UMAP Manifold Separability | Scheduled |
| **04** | `04_Temporal_Velocity_and_Cyclic_Periodicity_EDA.ipynb` | Fast Fourier Transform (FFT) Periodicity, Velocity Spikes, Time-Delta Dynamics | Scheduled |
| **05** | `05_Outlier_Forensics_and_Unsupervised_Anomaly_Profiling.ipynb` | Isolation Forest, Local Outlier Factor (LOF), Elliptic Envelope, Mahalanobis Distance | Scheduled |
| **06** | `06_Cardholder_Behavioral_and_Transaction_Graph_EDA.ipynb` | NetworkX Bipartite Graphs, Degree Centrality, Organized Fraud Ring Clustering | Scheduled |
| **07** | `07_Weight_of_Evidence_and_Information_Value_Analysis.ipynb` | Financial Risk Scoring with Weight of Evidence (WoE) and Information Value (IV) | Scheduled |
| **08** | `08_Financial_Cost_Utility_and_Asymmetric_Risk_Formulation.ipynb` | Expected Value Cost Matrix (Net Loss = FN * Amount + FP * FrictionCost) | Scheduled |
| **09** | `09_Advanced_Feature_Engineering_and_Leak_Free_Pipeline.ipynb` | Rolling Window Velocity Aggregations, Yeo-Johnson Power Transforms, Temporal Splits | Scheduled |
| **10** | `10_Imbalance_Mitigation_and_Sampling_Strategy_Benchmark.ipynb` | SMOTE, Borderline-SMOTE, ADASYN, Tomek Links, SMOTE-ENN, Focal Loss Arena | Scheduled |
| **11** | `11_Supervised_Cost_Sensitive_Classification_Benchmark.ipynb` | 12-Model Benchmark Matrix (XGBoost, LightGBM, CatBoost, ExtraTrees, Balanced RF) | Scheduled |
| **12** | `12_Ensemble_Stacking_and_Cost_Optimal_Thresholding.ipynb` | Stacking Meta-Learners and Financial Cost-Optimal Decision Threshold Tuning | Scheduled |
| **13** | `13_Probability_Calibration_and_Brier_Score_Optimization.ipynb` | Isotonic Regression & Platt Scaling Reliability Calibration (Brier Score Minimization) | Scheduled |
| **14** | `14_Deep_Learning_Variational_Autoencoders_Anomaly_Architecture.ipynb` | PyTorch Deep Variational Autoencoders (VAEs) with ELBO Loss Latent Reconstruction | Scheduled |
| **15** | `15_Deep_Tabular_Representation_TabNet_Architecture.ipynb` | TabNet Architecture with Sequential Attention Masking for Tabular Deep Learning | Scheduled |
| **16** | `16_Explainable_AI_SHAP_LIME_and_Adverse_Action_Compliance.ipynb` | TreeSHAP, LIME Local Surrogates, Adverse Action Regulatory Reason Codes | Scheduled |
| **17** | `17_Counterfactual_Policy_Simulations_and_Intervention_Engine.ipynb` | Counterfactual What-If Simulations and Step-Up Authentication Thresholds | Scheduled |
| **18** | `18_MLOps_Model_Validation_Drift_and_Latency_SLAs.ipynb` | PSI, KS Tests, Wasserstein Drift Monitoring, P50/P95/P99 Latency SLAs, Model Registry | Scheduled |

---

## Dataset Overview

- **Source**: Anonymized credit card transactions by European cardholders in September 2013.
- **Volume**: 284,807 transactions across 48 continuous hours.
- **Class Imbalance**: 284,315 legitimate transactions (99.8273%) vs. 492 fraudulent transactions (0.1727%).
- **Features**: 28 PCA transformed components ($V_1 \dots V_{28}$), `Time` (seconds elapsed), `Amount` (transaction value), and `Class` (ground truth label).

---

## Installation & Setup

```bash
git clone https://github.com/tamimystic/Financial-Fraud-Intelligence-and-Risk-Forensics-ML-Platform.git
cd Financial-Fraud-Intelligence-and-Risk-Forensics-ML-Platform

python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/macOS

pip install -r requirements.txt
```
