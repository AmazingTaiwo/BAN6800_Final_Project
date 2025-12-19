# 📘 README
# BAN6800 - Data Analytics Capstone
# Final Project: End-to-End ML Model Training, Registration, Deployment & Scoring
# Project Name – Databricks-Enabled Procurement Analytics Optimization

# Author: Taiwo Babalola
# Learner ID: 162894
# Submitted to: DR Raphael Wanjiku


#  1. Project Overview
        This project implements a production-ready business analytics solution for procurement performance monitoring and optimizayion using below Key Technologies such as
         - Databricks
         - Apache Spark
         - Unity Catalog
         - MLflow (UC Registry)
         - Scikit-learn
         - Imbalanced-learn (SMOTE)
         - Databricks Model Serving

#  1.1  The solution design follows below methodology
        - Trained two machine learning models on curated Gold-layer procurement data
        - Combines them into one unified PyFunc model
        - Registers the model in the Unity Catalog GOLD schema
        - Deploys a single Databricks Model Serving endpoint
        - Scores the full Gold dataset
        - Writes predictions back into the Gold schema for analytics and Power BI consumption
        This architecture supports predictive SLA monitoring, cycle-time forecasting, SLA prediction and data-driven decision-making for procurement stakeholders.

#  2. Business Use Case
        ABC Procurement teams face challenges such as:
                - Late PR→PO processing
                - SLA breaches impacting supplier performance
                - Limited forward-looking visibility into delays
        This solution addresses these issues by:
                - Predicting SLA breaches before they occur
                - Forecasting PR→PO cycle times in business days
                - Providing a single, scalable inference endpoint
                - Persisting predictions directly in the Gold layer
#  3. Data Source (Gold Layer)
      Gold Table: abc.abc_dw_gold.abc_dw_gl_pr_po_kpi
      This table is produced through a controlled Bronze → Silver → Gold transformation, including:
        - PR–PO matching logic
        - Business-day ageing calculations (Sun–Thu)
        - SLA breach flag derivation
        - Record type classification (PR_ONLY, PO_ONLY, PR_PO_MATCHED)
      Only PR_PO_MATCHED records are used for model training and scoring to ensure business relevance.

#  4. Models Implemented
#  4.1 SLA Breach Classification Model
        - Type: Random Forest Classifier
        - Target: sla_breach_flag (YES / NO)
        - Techniques:
                - SMOTE for class imbalance
                - Leak-free feature selection
                - Probability-based thresholding (default: 0.55)
        - Key Metrics Logged
                - Accuracy
                - Precision
                - Recall
                - F1-Score
                - ROC-AUC

#  4.2 PR→PO Cycle Time Regression Model
        - Type: Random Forest Regressor
        - Target: pr_to_po_ageing (business days)
        - Purpose: Forecast procurement cycle duration
#      Key Metrics Logged
        - MAE
        - RMSE
        - R²
#  5. Combined PyFunc Model (Single Endpoint)
        Both models are wrapped into one MLflow PyFunc model:
#        Endpoint Outputs are
                - pred_sla_breach_probability
                - pred_sla_breach_bin
                - pred_sla_breach_label
                - pred_pr_to_po_ageing
#        Why we use one model & one endpoint?
                - Simplifies deployment
                - Reduces infrastructure cost
                - Enables atomic predictions for dashboards and APIs
                - Aligns with Databricks serving best practices

#  6. Model Registry (Unity Catalog – GOLD Schema)
      The model was registered in the Unity Catalog Model Registry, ensuring governance, lineage, and access control.

#  6.1   Registered Model on Unity Catalog
          - Modela Name: **abc.abc_dw_gold.abc_dw_gd_model_Procurement_sla_Combined_Model_BAN6800**
        The model is saved within the GOLD schema and governed with Databricks Unity Catalog.

#  7. Model Serving Endpoint
      A single Databricks Model Serving endpoint was created.
        - ban6800-procurement-sla-combined
#       Characteristics
        - One endpoint
        - One combined model
        - 100% traffic routed to the active version
        - Ready for REST, Power BI, or application integration

#  8. Prediction Output (Written Back to Gold)
        Output Table: **abc.abc_dw_gold.abc_dw_gl_pr_po_kpi_predictions**
#        Contents
         - Business identifiers (PR, PO, company, plant, document type)
         - SLA breach probability & classification
         - Predicted PR→PO cycle time
         - Scoring timestamp
         
This table can be:
         - Consumed directly by Power BI
         - Joined back to procurement fact tables
         - Used for SLA dashboards and alerts

# 9. How to Run
         - Ensure the Gold table exists and is up to date
         - Run the script (BAN6800_Final_Project_Procurement_Model_and_endpoint.ipynb) in a Databricks notebook (Python)
         - The script will automatically:
                - Train both models.
                - Register the combined model in UC.
                - Write predictions to Gold.
                - Deploy/update the endpoint.
The Endpoint Start Automatically, No manual steps are required once execution starts.
