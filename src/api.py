import os
import io
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List

# ------------------------
# FASTAPI APP
# ------------------------
app = FastAPI(title="SentinelNet Traffic Analyzer")

# ------------------------
# PATH CONFIGURATION
# ------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(script_dir, '..', 'model')  # model folder
print(f"[+] Loading model from {MODEL_DIR}")

# ------------------------
# LOAD TRAINED MODEL
# ------------------------
model = joblib.load(os.path.join(MODEL_DIR, 'sentinel_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
print("[+] Model, scaler, and label encoder loaded successfully.")

# ------------------------
# HELPER FUNCTION: preprocess and predict
# ------------------------
def predict_df(df: pd.DataFrame):
    df.columns = df.columns.str.strip()  # clean column names
    X = df.drop(columns=['Label'], errors='ignore')  # drop label if present
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)  # handle missing/infinite values
    X_scaled = scaler.transform(X)
    preds_encoded = model.predict(X_scaled)
    preds_labels = le.inverse_transform(preds_encoded)
    df['Predicted_Label'] = preds_labels
    summary = pd.Series(preds_labels).value_counts().to_dict()
    return df, summary

# ------------------------
# ROUTE 1: POST CSV file
# ------------------------
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return JSONResponse(content={"error": "Only CSV files are supported"}, status_code=400)
    
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to read CSV: {str(e)}"}, status_code=400)
    
    df_pred, summary = predict_df(df)
    
    # Convert predictions to a list of dicts
    predictions = df_pred.to_dict(orient="records")
    
    return {"filename": file.filename, "summary": summary, "predictions": predictions}

# ------------------------
# ROUTE 2: POST JSON data
# ------------------------
@app.post("/predict_json")
async def predict_json(data: List[dict]):
    try:
        df = pd.DataFrame(data)
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to convert JSON to DataFrame: {str(e)}"}, status_code=400)
    
    df_pred, summary = predict_df(df)
    predictions = df_pred.to_dict(orient="records")
    
    return {"summary": summary, "predictions": predictions}

# ------------------------
# HEALTH CHECK
# ------------------------
@app.get("/health")
def health_check():
    return {"status": "SentinelNet API is running"}
