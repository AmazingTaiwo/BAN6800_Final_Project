# 📘 README
# BAN6800 - Data Analytics Capstone
# Final Project: End-to-End ML Model Training, Registration, Deployment & Scoring
# Project Name – Databricks-Enabled Procurement Analytics Optimization

# Author: Taiwo Babalola
# Learner ID: 162894
# Submitted to: DR Raphael Wanjiku
# Date: 19th Of December, 2025 

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

#  4.3 Repository Structure
        GOLD SCHEMA (Unity Catalog)
                │
                ├── Gold Tables
                │   ├── abc_dw_gl_pr_po_kpi
                │   └── abc_dw_gl_pr_po_kpi_predictions
                │
                ├── images/
                │     └── charts_and_visuals/                 # Feature importance, ROC, confusion matrix, etc.
                ├── Registered Model (UC)
                │   └── abc.abc_dw_gold.abc_dw_gd_model_Procurement_sla_Combined_Model_BAN6800
                │
                └── Model Serving Endpoint (Workspace-level)
                │   └── ban6800-procurement-sla-combined
                │       └── references UC model above
                └── README.md                                # This file
                
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

#  10. Result
#  10.1 Endpoint
                Below is the endpoint invocation URL
                https://dbc-a08452d4-da56.cloud.databricks.com/serving-endpoints/ban6800-procurement-sla-combined/invocations
#  10.1.1 Curl URL Endpoint
                curl \
                -u token:$DATABRICKS_TOKEN \
                -X POST \
                -H "Content-Type: application/json" \
                -d@data.json \
                  https://dbc-a08452d4-da56.cloud.databricks.com/serving-endpoints/ban6800-procurement-sla-combined/invocations
                  
#  10.1.2  Python Endpoint
                import os
                import requests
                import numpy as np
                import pandas as pd
                import json

                def create_tf_serving_json(data):
                    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

                def score_model(dataset):
                    url = 'https://dbc-a08452d4-da56.cloud.databricks.com/serving-endpoints/ban6800-procurement-sla-combined/invocations'
                    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
                    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else                              create_tf_serving_json(dataset)
                    data_json = json.dumps(ds_dict, allow_nan=True)
                    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
                    if response.status_code != 200:
                            raise Exception(f'Request failed with status {response.status_code}, {response.text}')
                    return response.json()
                    
# 10.1.3  SQL Endpoint Query
                SELECT ai_query('ban6800-procurement-sla-combined',
                    request => {
                  "dataframe_split": {
                    "columns": [
                      "pr_orderqty",
                      "po_orderquantity",
                      "po_netamount",
                      "pr_companycode",
                      "po_purchasingorgdesc",
                      "po_companycode",
                      "po_companycodedesc",
                      "pr_plant",
                      "po_plant",
                      "pr_documenttype",
                      "po_purchasingdoctypedesc",
                      "po_purchasinggroupdesc",
                      "po_countrykey",
                      "materialgroupdesc",
                      "materialtypedesc",
                      "record_type",
                      "pr_approval_ageing",
                      "po_approval_ageing",
                      "po_purchasingdoctype"
                    ],
                    "data": [
                      [
                        "2",
                        "2",
                        "62.72",
                        "9900",
                        "ABC logistics Limited Porg",
                        "9900",
                        "ABC Logistics Limited",
                        "9900",
                        "9900",
                        "ZNPR",
                        "Local PO",
                        "Tobi Emmaunel",
                        "GH",
                        "Stationery Expenses",
                        "Non-Stock Materials",
                        "PR_PO_MATCHED",
                        2.0,
                        8,
                        "ZLPO"
                      ],
                      [
                        "50",
                        "50",
                        "710.5",
                        "9900",
                        "ABC logistics Limited Porg",
                        "9900",
                        "ABC Logistics Limited",
                        "9900",
                        "9900",
                        "ZNPR",
                        "Local PO",
                        "Christian PAT",
                        "GH",
                        "Accessories & Consumables",
                        "Consumables",
                        "PR_PO_MATCHED",
                        5.0,
                        2,
                        "ZLPO"
                      ],
                      [
                        "20",
                        "20",
                        "764.4",
                        "9900",
                        "ABC logistics Limited Porg",
                        "9900",
                        "ABC Logistics Limited",
                        "9900",
                        "9900",
                        "ZNPR",
                        "Local PO",
                        "Christian PAT",
                        "GH",
                        "Safety Shoes",
                        "Consumables",
                        "PR_PO_MATCHED",
                        3.0,
                        2,
                        "ZLPO"
                      ],
                      [
                        "24",
                        "24",
                        "41.16",
                        "9900",
                        "ABC logistics Limited Porg",
                        "9900",
                        "ABC Logistics Limited",
                        "9900",
                        "9900",
                        "ZNPR",
                        "Local PO",
                        "Tobi Emmaunel",
                        "GH",
                        "Stationery Expenses",
                        "Non-Stock Materials",
                        "PR_PO_MATCHED",
                        2.0,
                        8,
                        "ZLPO"
                      ],
                      [
                        "100",
                        "25",
                        "266.78",
                        "9900",
                        "ABC logistics Limited Porg",
                        "9900",
                        "ABC Logistics Limited",
                        "9900",
                        "9900",
                        "ZNPR",
                        "Local PO",
                        "Christian PAT",
                        "GH",
                        "Accessories & Consumables",
                        "Consumables",
                        "PR_PO_MATCHED",
                        14.0,
                        5,
                        "ZLPO"
                      ]
                    ]
                  }
                })
#  10.2 Endpoint Template
#  10.2.1 Request Template 
          Below is the endpoint invocation format.
                {
                  "dataframe_split": {
                    "columns": [
                      "pr_orderqty",
                      "po_orderquantity",
                      "po_netamount",
                      "pr_companycode",
                      "po_purchasingorgdesc",
                      "po_companycode",
                      "po_companycodedesc",
                      "pr_plant",
                      "po_plant",
                      "pr_documenttype",
                      "po_purchasingdoctypedesc",
                      "po_purchasinggroupdesc",
                      "po_countrykey",
                      "materialgroupdesc",
                      "materialtypedesc",
                      "record_type",
                      "pr_approval_ageing",
                      "po_approval_ageing",
                      "po_purchasingdoctype"
                    ],
                    "data": [
                      [
                        "2",
                        "2",
                        "62.72",
                        "9900",
                        "ABC logistics Limited Porg",
                        "9900",
                        "ABC Logistics Limited",
                        "9900",
                        "9900",
                        "ZNPR",
                        "Local PO",
                        "Tobi Emmaunel",
                        "GH",
                        "Stationery Expenses",
                        "Non-Stock Materials",
                        "PR_PO_MATCHED",
                        2,
                        8,
                        "ZLPO"
                      ],
                      [
                        "50",
                        "50",
                        "710.5",
                        "9900",
                        "ABC logistics Limited Porg",
                        "9900",
                        "ABC Logistics Limited",
                        "9900",
                        "9900",
                        "ZNPR",
                        "Local PO",
                        "Christian PAT",
                        "GH",
                        "Accessories & Consumables",
                        "Consumables",
                        "PR_PO_MATCHED",
                        5,
                        2,
                        "ZLPO"
                      ],
                      [
                        "20",
                        "20",
                        "764.4",
                        "9900",
                        "ABC logistics Limited Porg",
                        "9900",
                        "ABC Logistics Limited",
                        "9900",
                        "9900",
                        "ZNPR",
                        "Local PO",
                        "Christian PAT",
                        "GH",
                        "Safety Shoes",
                        "Consumables",
                        "PR_PO_MATCHED",
                        3,
                        2,
                        "ZLPO"
                      ],
                      [
                        "24",
                        "24",
                        "41.16",
                        "9900",
                        "ABC logistics Limited Porg",
                        "9900",
                        "ABC Logistics Limited",
                        "9900",
                        "9900",
                        "ZNPR",
                        "Local PO",
                        "Tobi Emmaunel",
                        "GH",
                        "Stationery Expenses",
                        "Non-Stock Materials",
                        "PR_PO_MATCHED",
                        2,
                        8,
                        "ZLPO"
                      ],
                      [
                        "100",
                        "25",
                        "266.78",
                        "9900",
                        "ABC logistics Limited Porg",
                        "9900",
                        "ABC Logistics Limited",
                        "9900",
                        "9900",
                        "ZNPR",
                        "Local PO",
                        "Christian PAT",
                        "GH",
                        "Accessories & Consumables",
                        "Consumables",
                        "PR_PO_MATCHED",
                        14,
                        5,
                        "ZLPO"
                      ]
                    ]
                  }
                }
#  10.2.2 Endpoint Response
                Below is the endpoint response format
                        {
                          "predictions": [
                            {
                              "pred_sla_breach_probability": 0.9826907011809315,
                              "pred_sla_breach_bin": 1,
                              "pred_sla_breach_label": "YES",
                              "pred_pr_to_po_ageing": 41
                            },
                            {
                              "pred_sla_breach_probability": 0.32494237041334567,
                              "pred_sla_breach_bin": 0,
                              "pred_sla_breach_label": "NO",
                              "pred_pr_to_po_ageing": 24
                            },
                            {
                              "pred_sla_breach_probability": 0.42864690499593183,
                              "pred_sla_breach_bin": 0,
                              "pred_sla_breach_label": "NO",
                              "pred_pr_to_po_ageing": 20
                            },
                            {
                              "pred_sla_breach_probability": 0.9395665085827828,
                              "pred_sla_breach_bin": 1,
                              "pred_sla_breach_label": "YES",
                              "pred_pr_to_po_ageing": 41
                            },
                            {
                              "pred_sla_breach_probability": 0.19873059422359807,
                              "pred_sla_breach_bin": 0,
                              "pred_sla_breach_label": "NO",
                              "pred_pr_to_po_ageing": 1
                            }
                          ]
                        }
                
        
