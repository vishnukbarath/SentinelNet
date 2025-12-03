import os
import glob
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------------
# CONFIGURATION
# ------------------------
MODEL_DIR = r"C:\Users\vishn\Documents\SentinelNet\lane_segmentation"
NEW_DATA_DIR = r"../data/new_samples"  # folder with new traffic CSVs
OUTPUT_DIR = r"../data/predictions"  # folder to save predictions

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------
# LOAD TRAINED MODEL
# ------------------------
model = joblib.load(os.path.join(MODEL_DIR, 'sentinel_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

print("[+] Model, scaler, and label encoder loaded successfully.")

# ------------------------
# LOAD NEW DATA
# ------------------------
csv_files = glob.glob(os.path.join(NEW_DATA_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {NEW_DATA_DIR}")

print(f"[+] Found {len(csv_files)} new CSV files. Processing...")

all_preds_summary = {}

for file in csv_files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()  # remove extra spaces in column names
    
    # Features only (ignore 'Label' if present)
    X = df.drop(columns=['Label'], errors='ignore')
    
    # Replace infinite values and missing values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    preds_encoded = model.predict(X_scaled)
    preds_labels = le.inverse_transform(preds_encoded)
    
    # Add predictions to DataFrame
    df['Predicted_Label'] = preds_labels
    
    # Save predictions
    output_file = os.path.join(OUTPUT_DIR, os.path.basename(file).replace(".csv", "_predictions.csv"))
    df.to_csv(output_file, index=False)
    print(f"[+] Predictions saved to: {output_file}")
    
    # Summarize predictions
    summary = pd.Series(preds_labels).value_counts()
    all_preds_summary[os.path.basename(file)] = summary

# ------------------------
# VISUALIZE PREDICTIONS
# ------------------------
for fname, summary in all_preds_summary.items():
    summary.plot(kind='bar', color=['green', 'red'])
    plt.title(f"Predicted Traffic Distribution - {fname}")
    plt.xlabel("Traffic Type")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.show()

print("[+] All predictions processed and visualized successfully.")
