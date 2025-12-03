import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

#############################################
# 1. Load & Merge CIC-IDS2017 Dataset
#############################################

def load_dataset(folder_path="data"):
    all_files = [folder_path + "/" + f for f in os.listdir(folder_path) if f.endswith(".csv")]
    data_list = []

    print(f"[+] Found {len(all_files)} CSV files. Loading...")

    for file in all_files:
        print("Loading:", file)
        df = pd.read_csv(file, low_memory=False)
        data_list.append(df)

    dataset = pd.concat(data_list, ignore_index=True)
    print(f"[+] Dataset Loaded. Total shape: {dataset.shape}")
    return dataset

#############################################
# 2. Preprocess
#############################################

def preprocess(df):
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df = df.rename(columns={'Label':'label'})
    df['label'] = df['label'].apply(lambda x: 'Normal' if x=='BENIGN' else 'Attack')

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    X = df.drop('label', axis=1)
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

#############################################
# 3. Train RandomForest Model (Baseline)
#############################################

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, max_depth=20)
    model.fit(X_train, y_train)
    return model

#############################################
# 4. PyTorch Deep Learning Model
#############################################

class IDSNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.layers(x)

def train_dl(X_train, y_train, input_dim, epochs=10, batch_size=512, lr=0.001):

    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train.values).float())
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = IDSNet(input_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            pred = model(data).squeeze()
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    return model

#############################################
# 5. Evaluate
#############################################

def evaluate_model(model, X_test, y_test, deep=False):
    if deep:
        with torch.no_grad():
            preds = model(torch.tensor(X_test).float()).numpy().squeeze()
            preds = (preds>0.5).astype(int)
    else:
        preds = model.predict(X_test)

    print("\n===== Evaluation Report =====")
    print("Accuracy:", accuracy_score(y_test,preds))
    print("ROC-AUC:", roc_auc_score(y_test,preds))
    print("\nClassification Report:\n", classification_report(y_test,preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test,preds))

#############################################
# Main Execution
#############################################

if __name__ == "__main__":
    df = load_dataset()
    X, y, scaler = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print("\n=== Training RandomForest Model ===")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model,X_test,y_test)

    print("\n=== Training Deep Learning Model (PyTorch) ===")
    dl_model = train_dl(X_train, y_train, input_dim=X_train.shape[1])
    evaluate_model(dl_model,X_test,y_test,deep=True)

    # Save models
    import joblib
    joblib.dump(rf_model,"rf_ids_model.pkl")
    torch.save(dl_model.state_dict(),"ids_dl_model.pth")

    print("\n[+] Models Saved Successfully!")
