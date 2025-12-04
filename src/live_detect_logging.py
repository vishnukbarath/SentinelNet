import os
import time
import joblib
import numpy as np
import pandas as pd
from scapy.all import sniff
from pathlib import Path
from datetime import datetime

# =========================================================
# LOAD MODEL + SCALER + LABEL ENCODER
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

model = joblib.load(MODEL_DIR / "sentinel_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
le = joblib.load(MODEL_DIR / "label_encoder.pkl")

print("[+] Sentinel Model Loaded")
print("[+] Logging directory:", LOG_DIR)


# =========================================================
# FEATURE EXTRACTION (Modify according to your trained dataset)
# =========================================================
def packet_to_features(pkt):
    """Extract features required by the model."""
    try:
        features = {
            "Flow Duration": getattr(pkt, "time", 0),
            "Total Fwd Packets": 1,
            "Total Back Packets": 0,
            "Fwd Packet Length": len(pkt),
        }
        return features
    except:
        return None


# =========================================================
# PREDICT PACKET CLASS
# =========================================================
def predict_flow(features):
    df = pd.DataFrame([features])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaled = scaler.transform(df)
    pred = model.predict(scaled)
    return le.inverse_transform(pred)[0]


# =========================================================
# LOGGING SYSTEM
# =========================================================
def write_log(prediction, pkt_len, src, dst):
    log_file = LOG_DIR / f"{datetime.now():%Y-%m-%d}.log"

    with open(log_file, "a") as f:
        f.write(f"{datetime.now()} | {prediction} | Size:{pkt_len}B | {src} -> {dst}\n")


# =========================================================
# PACKET HANDLER
# =========================================================
normal = 0
attack = 0

def handle_packet(pkt):
    global normal, attack

    features = packet_to_features(pkt)
    if features is None:
        return

    prediction = predict_flow(features)

    src = pkt[0][1].src if hasattr(pkt[0][1], "src") else "Unknown"
    dst = pkt[0][1].dst if hasattr(pkt[0][1], "dst") else "Unknown"

    if prediction == "Normal":
        normal += 1
    else:
        attack += 1

    print(f"[+] {prediction} | Normal:{normal} | Attack:{attack} | {src} -> {dst}")

    write_log(prediction, len(pkt), src, dst)


# =========================================================
# START SNIFFER
# =========================================================
def start(interface="Ethernet", packet_limit=0):
    """
    interface = your network adapter name (Wi-Fi/Ethernet)
    packet_limit=0 means infinite sniff
    """
    print("\n#####################################################")
    print("#   SENTINEL - REAL TIME INTRUSION DETECTION ACTIVE #")
    print("#####################################################\n")

    print(f"[+] Listening on interface: {interface}")
    print("[+] Press CTRL + C to stop\n")

    sniff(iface=interface, prn=handle_packet, store=False, count=packet_limit)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    # Change interface to what you use ("Wi-Fi" or "Ethernet")
    start(interface="Wi-Fi", packet_limit=0)
