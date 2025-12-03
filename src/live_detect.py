import os
import time
import joblib
import numpy as np
import pandas as pd
from scapy.all import sniff
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# ------------------------
# LOAD MODEL AND PREPROCESSORS
# ------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(script_dir, '..', 'model')

model = joblib.load(os.path.join(MODEL_DIR, 'sentinel_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

print("[+] Model and preprocessors loaded successfully.")

# ------------------------
# FUNCTION: Extract features from a packet
# ------------------------
def packet_to_features(pkt):
    """
    Extract features from a scapy packet.
    You need to add all features your model expects.
    Example: Flow Duration, Total Fwd Packets, Total Back Packets...
    """
    # Placeholder example features (modify according to your trained dataset)
    features = {}
    features['Flow Duration'] = getattr(pkt, 'time', 0)  # Use timestamp as placeholder
    features['Total Fwd Packets'] = 1
    features['Total Back Packets'] = 0
    features['Fwd Packet Length'] = len(pkt)
    # Add more features as per your training dataset
    return features

# ------------------------
# FUNCTION: Predict flow
# ------------------------
def predict_flow(features_dict):
    df = pd.DataFrame([features_dict])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = scaler.transform(df)
    pred_encoded = model.predict(X_scaled)
    pred_label = le.inverse_transform(pred_encoded)[0]
    return pred_label

# ------------------------
# REAL-TIME PACKET SNIFFING
# ------------------------
def live_sniffer(interface=None, packet_count=0):
    """
    Capture live packets and predict on-the-fly.
    interface: network interface to sniff (e.g., 'Ethernet', 'Wi-Fi')
    packet_count: number of packets to capture (0 = infinite)
    """
    normal_count = 0
    attack_count = 0

    def process_packet(pkt):
        nonlocal normal_count, attack_count
        features = packet_to_features(pkt)
        prediction = predict_flow(features)
        if prediction == 'Normal':
            normal_count += 1
        else:
            attack_count += 1

        print(f"[+] Prediction: {prediction} | Total Normal: {normal_count} | Total Attack: {attack_count}")

    print("[+] Starting live packet capture...")
    sniff(iface=interface, prn=process_packet, store=False, count=packet_count)

# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    # If you want, specify your network interface here, e.g., "Wi-Fi" or "Ethernet"
    live_sniffer(interface=None, packet_count=0)  # 0 = infinite capture
