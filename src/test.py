import os
import glob
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------------
# CONFIGURATION
# ------------------------

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Model folder (absolute path)
MODEL_DIR = os.path.join(script_dir, '..', 'lane_segmentation')

# New data folder (absolute path)
NEW_DATA_DIR = os.path.join(script_dir, '..', 'data', 'new_samples')

# Output folder for predictions
OUTPUT_DIR = os.path.join(script_dir, '..', 'data', 'predictions')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[+] Model directory: {MODEL_DIR}")
print(f"[+] New data directory: {NEW_DATA_DIR}")
print(f"[+] Predictions will be saved to: {OUTPUT_DIR}")

# ------------------------
# LOAD TRAINED MODEL
# ------------------------
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'sentinel_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
except FileNotFoundError as e:
    raise FileNotFoundError(f"Model, scaler, or label encoder not found in {MODEL_DIR}") from e

print("[+] Model, scaler, and label encoder loaded successfully.")

# ------------------------
# FIND CSV FILES
# ------------------------
csv_files = glob.glob(os.path.join(NEW_DATA_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {NEW_DATA_DIR}")

print(f"[+] Found {len(csv_files)} CSV files. Processing...")

# ------------------------
# PROCESS EACH CSV
# ------------------------
all_preds_summary = {}

for file in csv_files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()  # clean column names
    
    # Features only (ignore 'Label' if present)
    X = df.drop(columns=['Label'], errors='ignore')
    
    # Handle missing or infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    preds_encoded = model.predict(X_scaled)
    preds_labels = le.inverse_transform(preds_encoded)
    
    # Add predictions to DataFrame
    df['Predicted_Label'] = preds_labels
    
    # Save predictions CSV
    output_file = os.path.join(OUTPUT_DIR, os.path.basename(file).replace(".csv", "_predictions.csv"))
    df.to_csv(output_file, index=False)
    print(f"[+] Predictions saved to: {output_file}")
    
    # Summary for visualization
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
