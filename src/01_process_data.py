import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

trans_file = RAW_DIR / "transactions_data.csv"
labels_file = RAW_DIR / "train_fraud_labels.json"

if not trans_file.exists() or not labels_file.exists():
    raise SystemExit(f"Error: Could not find required files in {RAW_DIR}")

# Load transactions data
print("\n[Step 1/5] Loading transactions_data.csv...")
df_trans = pd.read_csv(trans_file)
print(f"- Loaded {len(df_trans):,} transaction rows")

# Load the fraud labels (JSON) and convert to a clean DataFrame
print("\n[Step 2/5] Loading fraud labels...")
with labels_file.open("r") as f:
    fraud_labels = json.load(f)

if "target" in fraud_labels:
    print("- Nested JSON structure detected. Unpacking 'target' layer")
    data_items = fraud_labels["target"].items()
else:
    data_items = fraud_labels.items()

df_labels = pd.DataFrame(list(data_items), columns=["id", "is_fraud"])

df_labels = df_labels[df_labels["id"].astype(str).str.isnumeric()]

# Change ID to integers
df_labels["id"] = df_labels["id"].astype(int)

# Convert No to 0 and Yes to 1
df_labels["is_fraud"] = df_labels["is_fraud"].replace({"No": 0, "Yes": 1, "0": 0, "1": 1})
print("- Text-based labels ('No'/'Yes') detected. Encoding to binary numbers (0/1)")

# Remove incomplete rows
df_labels = df_labels.dropna(subset=["is_fraud"])
df_labels["is_fraud"] = df_labels["is_fraud"].astype(int)
print(f"- Loaded {len(df_labels):,} fraud label mappings successfully.")

# Merge transactions & label
print("\n[Step 3/5] Merging transactions with fraud labels...")
df_merged = pd.merge(df_trans, df_labels, on="id", how="inner")
print(f"- Merged dataset shape: {df_merged.shape}")

# Clean data columns
print("\n[Step 4/5] Preprocessing columns...")

# Convert transaction amounts such as "$14.57" to floats.
if df_merged["amount"].dtype == "O":
    df_merged["amount"] = df_merged["amount"].str.replace("$", "", regex=False).astype(float)
    print("- Successfully removed '$' and converted amount to decimal float numbers.")

# Standardize date time to pandas format.
date_col = "date" if "date" in df_merged.columns else "timestamp"
df_merged[date_col] = pd.to_datetime(df_merged[date_col])
print(f"- Converted '{date_col}' column to datetime formatting.")

# Drop columns with sensitive/personal information (privacy)
sensitive_cols = ["card_number", "cvv", "full_address", "client_name", "zip"]
df_merged = df_merged.drop(columns=sensitive_cols, errors="ignore")
print("- Removed sensitive personal data columns for identity protection.")

# Save processed data.
print("\n[Step 5/5] Saving cleaned dataset...")
output_file = PROCESSED_DIR / "cleaned_transactions.csv"
df_merged.to_csv(output_file, index=False)
print(f"Processed data saved safely at: {output_file}")
