#===============================================
# BAN6800 - Data Analytics Capstone
# Final Project: End-to-End ML Model Training, Registration, Deployment & Scoring
# Project Name – Databricks-Enabled Procurement Analytics Optimization
#===============================================
# Author: Taiwo Babalola
# Learner ID: 162894
# Submitted to: DR Raphael Wanjiku
# Date: 19th Of December, 2025 
# Model Test Script
#===============================================
import json
import requests
import pandas as pd

DATABRICKS_HOST = "https://dbc-a08452d4-da56.cloud.databricks.com"
ENDPOINT_NAME = "ban6800-procurement-sla-combined"
TOKEN = "*******************"  # <-- your PAT

url = f"{DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations"
headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# load Model into dataframe (use your real columns)
df = pd.DataFrame(
    [
        [2, 2, 62.72, "9900", "ABC logistics Limited Porg", "9900", "ABC Logistics Limited", "9900", "9900",
         "ZNPR", "Local PO", "Tobi Emmaunel", "GH", "Stationery Expenses", "Non-Stock Materials",
         "PR_PO_MATCHED", 2, 8, "ZLPO"]
    ],
    columns=[
        "pr_orderqty","po_orderquantity","po_netamount","pr_companycode","po_purchasingorgdesc","po_companycode",
        "po_companycodedesc","pr_plant","po_plant","pr_documenttype","po_purchasingdoctypedesc",
        "po_purchasinggroupdesc","po_countrykey","materialgroupdesc","materialtypedesc","record_type",
        "pr_approval_ageing","po_approval_ageing","po_purchasingdoctype"
    ],
)

payload = {"dataframe_split": df.to_dict(orient="split")}

resp = requests.post(url, headers=headers, data=json.dumps(payload, allow_nan=True), timeout=60)
print(resp.status_code)
print(resp.text)
resp.raise_for_status()
print(resp.json())

