import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def load_dataset(data_folder):
    # Make absolute path safe for Windows
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), data_folder))
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

    if not csv_files:
        print("[!] No CSV files found in", data_folder)
        return None

    print(f"[+] Found {len(csv_files)} CSV files. Loading...")
    df_list = []
    for file in csv_files:
        print("Loading:", file)
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)
    print("[+] Dataset Loaded. Total shape:", df.shape)
    return df

def preprocess(df):
    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Standardize label column
    if 'Label' not in df.columns:
        raise ValueError("Label column not found in the dataset!")

    df['Label'] = df['Label'].apply(lambda x: 'Normal' if x == 'BENIGN' else 'Attack')

    # Features and target
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Fill missing values
    X = X.fillna(0)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def main():
    df = load_dataset("../data")
    if df is None:
        return

    X, y, scaler = preprocess(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    import joblib
    os.makedirs("../lane_segmentation", exist_ok=True)
    joblib.dump(clf, "../lane_segmentation/sentinel_model.pkl")
    print("[+] Model saved to ../lane_segmentation/sentinel_model.pkl")

if __name__ == "__main__":
    main()
