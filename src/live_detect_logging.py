"""
live_detect_logging.py

Real-time packet sniffing + model prediction + SQLite logging for SentinelNet.

Place this file in: SentinelNet/src/live_detect_logging.py
Database will be created at: SentinelNet/sentinel_logs.db

Requirements:
    pip install scapy joblib pandas numpy

Run (Windows as Admin after installing Npcap):
    cd C:\Users\vishn\Documents\SentinelNet\src
    python live_detect_logging.py

This script:
 - captures live packets (L2 if Npcap installed, falls back to L3)
 - extracts fields into a features dict
 - scales + predicts using your saved model
 - stores each prediction and metadata in sentinel_logs.db
 - commits in batches for performance
"""

import os
import time
import json
import sqlite3
import joblib
import socket
import traceback
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
from scapy.all import sniff, conf, L3RawSocket, get_if_list
from scapy.layers.inet import IP, TCP, UDP, ICMP

# ------------------------
# CONFIG
# ------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../SentinelNet/src
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))    # .../SentinelNet
MODEL_DIR = os.path.join(PROJECT_DIR, 'model')                   # .../SentinelNet/model
DB_PATH = os.path.join(PROJECT_DIR, 'sentinel_logs.db')          # DB in project root

BATCH_SIZE = 256   # number of inserts per transaction commit
VERBOSE = True     # print to console
USE_L3_FALLBACK = True  # if L2 fails and no npcap, fallback to L3RawSocket

# ------------------------
# LOAD MODEL + PREPROCESSORS
# ------------------------
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'sentinel_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
except Exception as e:
    raise RuntimeError(f"Failed to load model/scaler/encoder from {MODEL_DIR}: {e}")

if VERBOSE:
    print("[+] Model, scaler, and label encoder loaded successfully.")

# ------------------------
# SQLITE HELPERS
# ------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    src_ip TEXT,
    dst_ip TEXT,
    src_port INTEGER,
    dst_port INTEGER,
    protocol TEXT,
    length INTEGER,
    prediction TEXT,
    features_json TEXT
);
"""

INSERT_SQL = """
INSERT INTO detections (ts, src_ip, dst_ip, src_port, dst_port, protocol, length, prediction, features_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, timeout=30)
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn, cur

# ------------------------
# FEATURE EXTRACTION
# ------------------------
def packet_to_features(pkt):
    """
    Convert a scapy packet into a dictionary of features.
    IMPORTANT: Expand this to match the exact features used for training.
    Right now we extract common network features that are useful for IDS logging.
    """
    features = {}

    # Timestamp
    features['timestamp'] = getattr(pkt, 'time', time.time())

    # Packet length (raw)
    try:
        pkt_len = len(pkt)
    except Exception:
        pkt_len = 0
    features['pkt_len'] = pkt_len

    # Default values
    src_ip = dst_ip = src_port = dst_port = None
    proto_name = 'OTHER'

    # IP layer
    if pkt.haslayer(IP):
        ip = pkt[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        proto_num = ip.proto
        # simple name mapping
        if proto_num == 6:
            proto_name = 'TCP'
        elif proto_num == 17:
            proto_name = 'UDP'
        elif proto_num == 1:
            proto_name = 'ICMP'
        else:
            proto_name = str(proto_num)
    else:
        # Non-IP packet: attempt to read raw
        src_ip = getattr(pkt, 'src', None)
        dst_ip = getattr(pkt, 'dst', None)
        proto_name = pkt.name if hasattr(pkt, 'name') else 'NON-IP'

    # Transport layer ports
    if pkt.haslayer(TCP):
        tcp = pkt[TCP]
        src_port = tcp.sport
        dst_port = tcp.dport
        # flags, window, etc.
        features['tcp_flags'] = int(tcp.flags)
        features['tcp_window'] = int(tcp.window)
    elif pkt.haslayer(UDP):
        udp = pkt[UDP]
        src_port = udp.sport
        dst_port = udp.dport
    elif pkt.haslayer(ICMP):
        # ICMP type/code
        features['icmp_type'] = int(pkt[ICMP].type)
        features['icmp_code'] = int(pkt[ICMP].code if hasattr(pkt[ICMP], 'code') else 0)

    # Basic counts we can compute per-packet (for flow-level you'd aggregate)
    features['src_ip'] = src_ip
    features['dst_ip'] = dst_ip
    features['src_port'] = src_port if src_port is not None else 0
    features['dst_port'] = dst_port if dst_port is not None else 0
    features['protocol'] = proto_name

    # Additional raw metadata (can be used for more features later)
    features['has_ip'] = int(pkt.haslayer(IP))
    features['has_tcp'] = int(pkt.haslayer(TCP))
    features['has_udp'] = int(pkt.haslayer(UDP))
    features['has_icmp'] = int(pkt.haslayer(ICMP))

    # NOTE:
    # Your scaler/model expects a fixed set of numerical features in a specific order.
    # If your model was trained on aggregated flow-level CSV, you must implement
    # an aggregator. For now, we will produce a consistent set of features for per-packet prediction.
    return features

# ------------------------
# MODEL PREDICTION WRAPPER
# ------------------------
def predict_from_features(features_dict):
    """
    Convert features dict -> DataFrame -> scale -> predict -> label
    Must ensure order and columns same as model training or scaler shape matches.
    If scaler was fitted on N columns, we must supply same number of columns.
    Here we rely on the scaler expecting the keys we provide or we pad zero columns.
    """
    # Build DataFrame from features. Order of columns matters if scaler expects it.
    # Best practice is to store the training columns order when saving scaler/model.
    # For now, use features keys sorted by name (deterministic).
    keys = sorted(features_dict.keys())
    df = pd.DataFrame([[features_dict[k] for k in keys]], columns=keys)

    # Ensure numeric columns only for scaler
    df_numeric = df.select_dtypes(include=[np.number]).fillna(0)

    # If scaler expects more features than present, try to pad zeros to match scaler.mean_.shape
    try:
        expected_n = scaler.mean_.shape[0]
        provided_n = df_numeric.shape[1]
        if provided_n < expected_n:
            # pad additional zero columns
            for i in range(expected_n - provided_n):
                df_numeric[f'_pad_{i}'] = 0.0
        elif provided_n > expected_n:
            # trim extra columns (last ones)
            df_numeric = df_numeric.iloc[:, :expected_n]
    except Exception:
        # If scaler doesn't have mean_ (unlikely), just proceed
        pass

    # Final replace infs/nans
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scale and predict
    try:
        X_scaled = scaler.transform(df_numeric.values)
        pred_encoded = model.predict(X_scaled)
        pred_label = le.inverse_transform(pred_encoded)[0]
    except Exception as e:
        # If prediction fails (mismatch), log and return 'Unknown'
        if VERBOSE:
            print("[!] Prediction error:", e)
            traceback.print_exc()
        pred_label = 'Unknown'

    return pred_label

# ------------------------
# SNIFFING + BATCH INSERT
# ------------------------
def live_sniffer_and_log(iface=None, packet_count=0):
    """
    Capture packets and log predictions to sqlite DB in batches.
    """
    # DB init
    conn, cur = init_db(DB_PATH)
    insert_buffer = deque()
    total_seen = 0
    total_logged = 0

    # helper to flush buffer
    def flush_buffer():
        nonlocal total_logged
        if not insert_buffer:
            return
        try:
            cur.executemany(INSERT_SQL, list(insert_buffer))
            conn.commit()
            total_logged += len(insert_buffer)
            if VERBOSE:
                print(f"[DB] Committed {len(insert_buffer)} rows. Total logged: {total_logged}")
            insert_buffer.clear()
        except Exception as e:
            print("[DB] Failed to commit batch:", e)
            conn.rollback()

    def process_packet(pkt):
        nonlocal total_seen
        total_seen += 1

        # Extract features and metadata
        features = packet_to_features(pkt)
        ts = datetime.utcfromtimestamp(features.get('timestamp', time.time())).isoformat() + 'Z'
        src_ip = features.get('src_ip')
        dst_ip = features.get('dst_ip')
        src_port = int(features.get('src_port') or 0)
        dst_port = int(features.get('dst_port') or 0)
        proto = features.get('protocol') or 'UNK'
        length = int(features.get('pkt_len') or 0)

        # Prediction
        pred = predict_from_features(features)

        # Insert record tuple
        features_json = json.dumps(features, default=str)
        row = (ts, src_ip, dst_ip, src_port, dst_port, proto, length, pred, features_json)
        insert_buffer.append(row)

        # Periodically flush
        if len(insert_buffer) >= BATCH_SIZE:
            flush_buffer()

        # Console output
        if VERBOSE and total_seen % 100 == 0:
            print(f"[+] Seen: {total_seen} packets | Buffer: {len(insert_buffer)} | Last pred: {pred}")

    # If iface is None, use scapy default but handle L2 error
    if iface is None:
        iface = conf.iface

    if VERBOSE:
        print(f"[+] Starting live capture on interface: {iface} (count={packet_count if packet_count>0 else 'infinite'})")

    # Attempt sniff; if layer2 fails on Windows, fallback to L3 socket if enabled
    try:
        sniff(iface=iface, prn=process_packet, store=False, count=packet_count)
    except RuntimeError as e:
        print("[!] RuntimeError during sniff:", e)
        if USE_L3_FALLBACK:
            print("[*] Falling back to layer 3 raw socket sniffing (conf.L2socket = L3RawSocket).")
            conf.L2socket = L3RawSocket
            sniff(iface=iface, prn=process_packet, store=False, count=packet_count)
        else:
            raise

    # After sniffing ends or user interrupts, flush remaining
    flush_buffer()
    conn.close()
    if VERBOSE:
        print(f"[+] Sniffing finished. Total seen: {total_seen}. Total logged: {total_logged}")

# ------------------------
# UTIL: export logs to CSV
# ------------------------
def export_db_to_csv(db_path=DB_PATH, csv_path=None, limit=None):
    if csv_path is None:
        csv_path = os.path.join(PROJECT_DIR, f'sentinel_logs_export_{int(time.time())}.csv')
    conn = sqlite3.connect(db_path)
    query = "SELECT id, ts, src_ip, dst_ip, src_port, dst_port, protocol, length, prediction, features_json FROM detections"
    if limit:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(query, conn)
    # Optionally parse features_json into columns (skipped here)
    df.to_csv(csv_path, index=False)
    conn.close()
    print(f"[+] Exported logs to {csv_path}")
    return csv_path

# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    try:
        # If you want to pick a specific interface, uncomment below:
        # print("Available interfaces:", get_if_list())
        # iface_name = "Wi-Fi"  # or "Ethernet"
        iface_name = None  # None lets scapy choose default
        pkt_count = 0       # 0 = infinite
        live_sniffer_and_log(iface=iface_name, packet_count=pkt_count)
    except KeyboardInterrupt:
        print("\n[+] Stopped by user (KeyboardInterrupt). Exiting gracefully.")
    except Exception as e:
        print("[!] Fatal error:", e)
        traceback.print_exc()
