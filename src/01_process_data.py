import os
import json
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

trans_file = os.path.join(RAW_DIR, "transactions_data.csv")
labels_file = os.path.join(RAW_DIR, "train_fraud_labels.json")

if not os.path.exists(trans_file) or not os.path.exists(labels_file):
    print("Error: Could not find files in data/raw/")
    exit()

# Load transactions data
print("\n[Step 1/5] Loading transactions_data.csv...")
df_trans = pd.read_csv(trans_file)
print(f"- Loaded {len(df_trans):,} transaction rows")

# Load the fraud labels (JSON) and convert to a clean DataFrame
print("\n[Step 2/5] Loading fraud labels...")
with open(labels_file, "r") as f:
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

# Convert fraud labels to binary numbers
print("- Encoding fraud labels to binary numbers (No=0, Yes=1)")

df_labels["is_fraud"] = (
    df_labels["is_fraud"]
    .astype(str)
    .str.strip()
    .map({
        "No": 0,
        "Yes": 1,
        "no": 0,
        "yes": 1,
        "0": 0,
        "1": 1
    })
)

# Remove rows that could not be mapped
df_labels = df_labels.dropna(subset=["is_fraud"])

# Convert to integer
df_labels["is_fraud"] = df_labels["is_fraud"].astype(int)

print(df_labels["is_fraud"].value_counts())
print(f"- Loaded {len(df_labels):,} fraud label mappings successfully.")

# Merge transactions & label
print("\n[Step 3/5] Merging transactions with fraud labels...")
df_merged = pd.merge(df_trans, df_labels, on="id", how="inner")
print(f"- Merged dataset shape: {df_merged.shape}")

# Clean data columns
print("\n[Step 4/5] Preprocessing columns...")

# Clean empty values: change $ to O
if df_merged["amount"].dtype == "O":
    df_merged["amount"] = df_merged["amount"].str.replace("$", "", regex=False).astype(float)
    print("- Successfully removed '$' and converted amount to decimal float numbers.")

# Standardize data time to pandas format
date_col = "date" if "date" in df_merged.columns else "timestamp"
df_merged[date_col] = pd.to_datetime(df_merged[date_col])
print(f"- Converted '{date_col}' column to datetime formatting.")

# Drop columns with sensitive/personal information (privacy)
sensitive_cols = ["card_number", "cvv", "full_address", "client_name", "zip"]
df_merged = df_merged.drop(columns=sensitive_cols, errors="ignore")
print("- Removed sensitive personal data columns for identity protection.")

# Save pocessed data 
print("\n[Step 5/5] Saving cleaned dataset...")
output_file = os.path.join(PROCESSED_DIR, "cleaned_transactions.csv")
df_merged.to_csv(output_file, index=False)
print(f"Processed data saved safely at: {output_file}")