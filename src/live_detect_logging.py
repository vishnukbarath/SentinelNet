import joblib
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, Raw
from datetime import datetime
import threading

# ============================
# Load trained ML model
# ============================
model = joblib.load("../model/model.pkl")   # path adjust if needed

# Feature columns MUST MATCH TRAINING ORDER
FEATURE_COLUMNS = ['Destination Port','Flow Duration','Total Fwd Packets','Total Backward Packets',
                   'Fwd Packet Length Max','Fwd Packet Length Min','Fwd Packet Length Mean',
                   'Bwd Packet Length Max','Bwd Packet Length Min','Bwd Packet Length Mean','Flow Bytes/s']

# ============================
# Extract features from packets
# ============================
flow_cache = {}

def extract_features(pkt):
    if IP in pkt:
        key = (pkt[IP].src, pkt[IP].dst, pkt.sport if hasattr(pkt,'sport') else 0)

        if key not in flow_cache:
            flow_cache[key] = {
                "start_time": datetime.now(),
                "fwd":0,"bwd":0,"fwd_len":[],"bwd_len":[]
            }

        flow = flow_cache[key]

        # Counting direction
        if pkt[IP].src < pkt[IP].dst:
            flow["fwd"] += 1
            if Raw in pkt: flow["fwd_len"].append(len(pkt[Raw]))
        else:
            flow["bwd"] += 1
            if Raw in pkt: flow["bwd_len"].append(len(pkt[Raw]))

        # Convert to feature row
        duration = (datetime.now()-flow["start_time"]).total_seconds()*1000

        row = {
            'Destination Port': pkt.dport if hasattr(pkt,'dport') else 0,
            'Flow Duration': duration,
            'Total Fwd Packets': flow["fwd"],
            'Total Backward Packets': flow["bwd"],
            'Fwd Packet Length Max': max(flow["fwd_len"]) if flow["fwd_len"] else 0,
            'Fwd Packet Length Min': min(flow["fwd_len"]) if flow["fwd_len"] else 0,
            'Fwd Packet Length Mean': sum(flow["fwd_len"])/len(flow["fwd_len"]) if flow["fwd_len"] else 0,
            'Bwd Packet Length Max': max(flow["bwd_len"]) if flow["bwd_len"] else 0,
            'Bwd Packet Length Min': min(flow["bwd_len"]) if flow["bwd_len"] else 0,
            'Bwd Packet Length Mean': sum(flow["bwd_len"])/len(flow["bwd_len"]) if flow["bwd_len"] else 0,
            'Flow Bytes/s': (sum(flow["fwd_len"])+sum(flow["bwd_len"])) / duration if duration>0 else 0,
        }

        return pd.DataFrame([row])[FEATURE_COLUMNS]

# ============================
# Live Detection Callback
# ============================
def detect_packet(pkt):
    try:
        data = extract_features(pkt)
        if data is not None:
            pred = model.predict(data)[0]
            label = "ATTACK" if pred==1 else "NORMAL"

            log = f"[{datetime.now()}] {pkt[IP].src} -> {pkt[IP].dst}  STATUS:{label}"
            print(log)

            # save attacks
            if label=="ATTACK":
                with open("../logs/detections.log","a") as f:
                    f.write(log+"\n")

    except Exception as e:
        print("Error:",e)

# ============================
# Start Live Sniffer
# ============================
def start_sniff():
    print("🔍 Live Network Intrusion Detection Running...")
    sniff(prn=detect_packet, store=0)

if __name__=="__main__":
    start_sniff()
