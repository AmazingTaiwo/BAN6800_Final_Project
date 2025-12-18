# BAN6800_Final_Project
📦 Databricks-Enabled Procurement SLA Prediction Platform

Author: Taiwo Babalola
Program: BAN6800 – Business Analytics (Final Project)
Platform: Databricks (Delta Lake, MLflow, Model Serving)
Data Layer: Gold (abc.abc_dw_gold)

📌 Overview

This project delivers an end-to-end, production-ready AI solution that predicts:

SLA breach risk (classification)

PR → PO cycle time in business days (regression)

The solution is built on Databricks Lakehouse architecture, using Gold Delta tables, MLflow Model Registry, and a single real-time serving endpoint consumable by Power BI, APIs, and downstream systems.

🏗️ Architecture
SAP PR + PO (Silver Layer)
        ↓
Gold KPI Transformation (Delta)
        ↓
Machine Learning (sklearn)
        ↓
MLflow Model Registry
        ↓
Databricks Serving Endpoint (ONE)
        ↓
Power BI / APIs / Dashboards

🥇 Gold Table Transformation
Source Tables

abc.abc_dw_silver.abc_dw_sl_pr_req (PR)

abc.abc_dw_silver.abc_dw_sl_pur_ord (PO)

Gold Table
abc.abc_dw_gold.abc_dw_gl_pr_po_kpi

KPIs (Business Days: Sun–Thu)
KPI	Description
pr_to_po_ageing	PR approved → PO created
pr_approval_ageing	PR created → PR approved
po_approval_ageing	PO created → PO approved
SLA Flags
Flag	Rule
sla_breach_flag	PR→PO > 5 business days
pr_cycle_sla_breach_flag	PR approval > 2 days
po_cycle_sla_breach_flag	PO approval > 2 days
Record Classification
Type
PR_PO_MATCHED
PR_ONLY
PO_ONLY
🤖 Machine Learning Models
1️⃣ SLA Breach Classifier

Algorithm: Random Forest (balanced with SMOTE)

Output:

pred_sla_breach_probability

pred_sla_breach_label (YES / NO)

Performance (Typical):

Accuracy ≈ 0.83

ROC AUC ≈ 0.93

2️⃣ PR→PO Cycle Time Regressor

Algorithm: Random Forest Regressor

Output:

pred_pr_to_po_ageing (business days)

🧠 Feature Engineering (Leak-Free)
Numeric
pr_orderqty
po_orderquantity
po_netamount
pr_approval_ageing
po_approval_ageing

Categorical
pr_companycode
po_companycode
pr_plant
po_plant
pr_documenttype
po_purchasingdoctypedesc
po_purchasinggroupdesc
po_countrykey
materialgroupdesc
materialtypedesc
record_type


❗ Engineered KPIs are never used as inputs to predict themselves.

🚀 Model Deployment
MLflow Experiment
/Shared/Procurement_SLA_Models

Registered Models (Workspace Registry)
abc.abc_dw_gold.sla_breach_classifier
abc.abc_dw_gold.pr_to_po_regressor

Serving Strategy

Single Databricks Serving Endpoint

PyFunc model combining:

Classifier

Regressor

Threshold logic

Preprocessing

🔌 Serving Endpoint – Input & Output
Input (JSON)
{
  "dataframe_records": [
    {
      "pr_orderqty": 10,
      "po_orderquantity": 10,
      "po_netamount": 12500,
      "pr_companycode": "1000",
      "po_companycode": "1000",
      "pr_plant": "P001",
      "po_plant": "P001",
      "pr_documenttype": "NB",
      "po_purchasingdoctypedesc": "Standard PO",
      "po_purchasinggroupdesc": "LOCAL",
      "po_countrykey": "QA",
      "materialgroupdesc": "ELECTRICAL",
      "materialtypedesc": "ROH",
      "record_type": "PR_PO_MATCHED",
      "pr_approval_ageing": 2,
      "po_approval_ageing": 1
    }
  ]
}

Output
{
  "predictions": [
    {
      "pred_sla_breach_probability": 0.82,
      "pred_sla_breach_bin": 1,
      "pred_sla_breach_label": "YES",
      "pred_pr_to_po_ageing": 6.4
    }
  ]
}

🧾 Batch Scoring Output (Gold Schema)

Predictions are written back to Delta:

abc.abc_dw_gold.abc_dw_gl_pr_po_kpi_predictions

Columns
pred_sla_breach_probability
pred_sla_breach_label
pred_pr_to_po_ageing
scoring_timestamp

📊 Power BI Integration
Recommended Pattern
Power BI → Databricks SQL Warehouse
        → Gold Predictions Table

Benefits

No API throttling

Full governance & lineage

High performance

Audit-ready

🔐 Governance & Best Practices

✔ Gold-layer only consumption
✔ MLflow model versioning
✔ Reproducible training
✔ Feature leakage prevention
✔ Business-day logic aligned with Qatar working calendar
✔ Enterprise-ready deployment

📈 Business Value

Proactive SLA risk detection

Cycle-time forecasting before delays occur

Reduced escalations & penalties

Improved procurement planning

Executive-ready KPIs for decision-making
