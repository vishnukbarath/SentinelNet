import os
import glob
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib  # for saving the model

# ------------------------
# DATA LOADING FUNCTION
# ------------------------
def load_dataset(data_dir):
    data_dir = os.path.abspath(data_dir)
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"[+] Found {len(csv_files)} CSV files. Loading...")

    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV files found in the data directory!")

    df_list = []
    for file in csv_files:
        print(f"Loading: {file}")
        df_list.append(pd.read_csv(file))

    df = pd.concat(df_list, ignore_index=True)
    print(f"[+] Dataset Loaded. Total shape: {df.shape}")
    return df

# ------------------------
# PREPROCESSING FUNCTION
# ------------------------
def preprocess(df):
    # Strip spaces from columns
    df.columns = df.columns.str.strip()

    # Standardize label column
    if 'Label' not in df.columns:
        raise ValueError("Label column not found in the dataset!")
    df['Label'] = df['Label'].apply(lambda x: 'Normal' if x == 'BENIGN' else 'Attack')

    # Split features and target
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Replace infinities with NaN and fill with 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X_scaled, y_encoded, scaler, le

# ------------------------
# TRAINING FUNCTION
# ------------------------
def train_model(X_train, y_train, X_val, y_val, epochs=10):
    # Using simple MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(128, 64),
                          activation='relu',
                          solver='adam',
                          max_iter=1,  # we will manually loop for visualization
                          warm_start=True,
                          verbose=False)

    train_acc = []
    val_acc = []
    epoch_times = []

    for epoch in range(epochs):
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()

        # Record epoch time
        epoch_times.append(end_time - start_time)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f} | Time: {epoch_times[-1]:.2f}s")

    print(f"[+] Total training time: {sum(epoch_times):.2f}s")
    return model, train_acc, val_acc, epoch_times

# ------------------------
# VISUALIZATION FUNCTION
# ------------------------
def plot_training(train_acc, val_acc, epoch_times):
    epochs = range(1, len(train_acc)+1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_acc, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, epoch_times, 'g-o')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Time per Epoch')

    plt.tight_layout()
    plt.show()

# ------------------------
# MAIN FUNCTION
# ------------------------
def main():
    DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '../lane_segmentation')

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_dataset(DATA_DIR)
    X, y, scaler, le = preprocess(df)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train
    model, train_acc, val_acc, epoch_times = train_model(X_train, y_train, X_val, y_val, epochs=10)

    # Visualize
    plot_training(train_acc, val_acc, epoch_times)

    # Save model, scaler, label encoder
    joblib.dump(model, os.path.join(MODEL_DIR, 'sentinel_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    print(f"[+] Model and preprocessing saved in {MODEL_DIR}")

if __name__ == "__main__":
    main()
