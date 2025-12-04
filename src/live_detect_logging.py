# save as: SentinelNet/src/live_detect_logging_full.py

import os
import time
import json
import sqlite3
import joblib
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw, Ether, conf, get_if_list

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent    # SentinelNet/
MODEL_DIR = BASE_DIR / "model"
DB_PATH = BASE_DIR / "sentinel_logs.db"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

FLOW_TIMEOUT = 60.0        # seconds of inactivity to finalize a flow
BATCH_SIZE = 128           # DB commits per batch
VERBOSE = True

# -------------------------
# FEATURE COLUMNS (strip whitespace)
# (Derived from your CSV header; order matters)
# -------------------------
FEATURE_COLUMNS = [
"Destination Port","Flow Duration","Total Fwd Packets","Total Backward Packets",
"Total Length of Fwd Packets","Total Length of Bwd Packets","Fwd Packet Length Max",
"Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std",
"Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std",
"Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
"Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
"Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
"Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Fwd Header Length","Bwd Header Length",
"Fwd Packets/s","Bwd Packets/s","Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std",
"Packet Length Variance","FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count",
"URG Flag Count","CWE Flag Count","ECE Flag Count","Down/Up Ratio","Average Packet Size","Avg Fwd Segment Size",
"Avg Bwd Segment Size","Fwd Header Length.1","Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate",
"Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate","Subflow Fwd Packets","Subflow Fwd Bytes",
"Subflow Bwd Packets","Subflow Bwd Bytes","Init_Win_bytes_forward","Init_Win_bytes_backward","act_data_pkt_fwd",
"min_seg_size_forward","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max",
"Idle Min","Label"
]

# remove trailing Label because model expects features only (we'll not include Label)
if FEATURE_COLUMNS and FEATURE_COLUMNS[-1] == "Label":
    FEATURE_COLUMNS = FEATURE_COLUMNS[:-1]

# -------------------------
# LOAD MODEL & SCALER & ENCODER
# -------------------------
try:
    model = joblib.load(MODEL_DIR / "sentinel_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    le = joblib.load(MODEL_DIR / "label_encoder.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model/scaler/encoder from {MODEL_DIR}: {e}")

if VERBOSE:
    print("[+] Model, scaler, encoder loaded.")
    print(f"[+] Expecting {len(FEATURE_COLUMNS)} features.")

# -------------------------
# SQLITE INIT
# -------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    flow_key TEXT,
    src_ip TEXT,
    dst_ip TEXT,
    src_port INTEGER,
    dst_port INTEGER,
    protocol TEXT,
    duration REAL,
    total_packets INTEGER,
    total_bytes INTEGER,
    prediction TEXT,
    features_json TEXT
);
"""
INSERT_SQL = """
INSERT INTO detections (ts, flow_key, src_ip, dst_ip, src_port, dst_port, protocol, duration, total_packets, total_bytes, prediction, features_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn, cur

conn, cur = init_db(DB_PATH)

# -------------------------
# FLOW AGGREGATOR STRUCTURES
# -------------------------
# Key: (src, dst, sport, dport, proto)
flows = {}

# flow structure:
# {
#   'first_ts': float, 'last_ts': float,
#   'pkts': int, 'bytes': int,
#   'fwd_pkts': int, 'bwd_pkts': int,
#   'fwd_lens': [], 'bwd_lens': [],
#   'times': [], 'fwd_times': [], 'bwd_times': [],
#   'flags_counts': {'FIN':0,'SYN':0,...},
#   'init_win_fwd': None, 'init_win_bwd': None,
#   'active_gaps': [durations], 'idle_gaps': [durations],
#   ...
# }
def make_flow_entry(first_pkt, direction_is_forward):
    return {
        'first_ts': first_pkt.time,
        'last_ts': first_pkt.time,
        'pkts': 0,
        'bytes': 0,
        'fwd_pkts': 0,
        'bwd_pkts': 0,
        'fwd_lens': [],
        'bwd_lens': [],
        'times': [],
        'fwd_times': [],
        'bwd_times': [],
        'flags': defaultdict(int),
        'init_win_fwd': None,
        'init_win_bwd': None,
        'last_pkt_ts': first_pkt.time,
        'segments': [],  # for active/idle segmentation: list of (start,end)
        'last_segment_start': first_pkt.time,
    }

# -------------------------
# UTILS: flag checks
# -------------------------
def tcp_flags_to_counts(tcp_flags):
    # tcp_flags may be an int or object; convert to int
    try:
        f = int(tcp_flags)
    except Exception:
        f = 0
    return {
        'FIN': 1 if (f & 0x01) else 0,
        'SYN': 1 if (f & 0x02) else 0,
        'RST': 1 if (f & 0x04) else 0,
        'PSH': 1 if (f & 0x08) else 0,
        'ACK': 1 if (f & 0x10) else 0,
        'URG': 1 if (f & 0x20) else 0,
        'ECE': 1 if (f & 0x40) else 0,
        'CWR': 1 if (f & 0x80) else 0
    }

# -------------------------
# FEATURE COMPUTATION ON FINALIZE
# -------------------------
def finalize_flow_and_predict(key, flow):
    """
    Build features dict in order FEATURE_COLUMNS, scale, predict, and log result.
    """
    # basic metadata
    src, dst, sport, dport, proto = key
    duration_s = max( (flow['last_ts'] - flow['first_ts']), 0.000001 )
    duration_ms = duration_s * 1000.0

    total_packets = flow['pkts']
    total_bytes = flow['bytes']
    fwd_pkts = flow['fwd_pkts']
    bwd_pkts = flow['bwd_pkts']

    all_lengths = flow['fwd_lens'] + flow['bwd_lens']
    min_pkt_len = int(min(all_lengths)) if all_lengths else 0
    max_pkt_len = int(max(all_lengths)) if all_lengths else 0
    mean_pkt_len = float(np.mean(all_lengths)) if all_lengths else 0.0
    std_pkt_len = float(np.std(all_lengths, ddof=0)) if all_lengths else 0.0
    var_pkt_len = float(np.var(all_lengths, ddof=0)) if all_lengths else 0.0

    # Flow rates
    flow_bytes_s = (total_bytes / duration_s) if duration_s>0 else 0.0
    flow_pkts_s = (total_packets / duration_s) if duration_s>0 else 0.0
    fwd_pkts_s = (fwd_pkts / duration_s) if duration_s>0 else 0.0
    bwd_pkts_s = (bwd_pkts / duration_s) if duration_s>0 else 0.0

    # IATs
    times = sorted(flow['times'])
    iats = np.diff(times) if len(times)>1 else np.array([0.0])
    flow_iat_mean = float(np.mean(iats)) if len(iats)>0 else 0.0
    flow_iat_std = float(np.std(iats, ddof=0)) if len(iats)>0 else 0.0
    flow_iat_max = float(np.max(iats)) if len(iats)>0 else 0.0
    flow_iat_min = float(np.min(iats)) if len(iats)>0 else 0.0

    def iat_stats(ts_list):
        ts_list = sorted(ts_list)
        if len(ts_list) <= 1:
            arr = np.array([0.0])
        else:
            arr = np.diff(ts_list)
        return (float(np.sum(arr)), float(np.mean(arr)), float(np.std(arr, ddof=0)), float(np.max(arr)), float(np.min(arr)))

    fwd_iat_total, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = iat_stats(flow['fwd_times'])
    bwd_iat_total, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = iat_stats(flow['bwd_times'])

    # forward/back lengths stats
    def len_stats(l):
        if not l:
            return (0,0,0.0,0.0)
        a = np.array(l)
        return (int(np.max(a)), int(np.min(a)), float(np.mean(a)), float(np.std(a, ddof=0)))
    fwd_max, fwd_min, fwd_mean, fwd_std = len_stats(flow['fwd_lens'])
    bwd_max, bwd_min, bwd_mean, bwd_std = len_stats(flow['bwd_lens'])

    # flags counts aggregated
    flags_counts = flow['flags']
    fin = flags_counts.get('FIN',0)
    syn = flags_counts.get('SYN',0)
    rst = flags_counts.get('RST',0)
    psh = flags_counts.get('PSH',0)
    ack = flags_counts.get('ACK',0)
    urg = flags_counts.get('URG',0)
    ece = flags_counts.get('ECE',0)
    cwr = flags_counts.get('CWR',0)

    # PSH/URG per direction approximate using stored counts per flow (we counted per packet)
    fwd_psh = flow.get('fwd_psh',0)
    bwd_psh = flow.get('bwd_psh',0)
    fwd_urg = flow.get('fwd_urg',0)
    bwd_urg = flow.get('bwd_urg',0)

    # header lengths: use TCP.dataofs*4 if present else IP.ihl*4
    fwd_hdr = flow.get('fwd_hdr_len',0)
    bwd_hdr = flow.get('bwd_hdr_len',0)

    # Down/Up ratio bytes
    bytes_fwd = sum(flow['fwd_lens']) if flow['fwd_lens'] else 0
    bytes_bwd = sum(flow['bwd_lens']) if flow['bwd_lens'] else 0
    down_up_ratio = (bytes_bwd/bytes_fwd) if bytes_fwd>0 else 0.0

    avg_packet_size = (total_bytes/total_packets) if total_packets>0 else 0.0
    avg_fwd_seg = (bytes_fwd / fwd_pkts) if fwd_pkts>0 else 0.0
    avg_bwd_seg = (bytes_bwd / bwd_pkts) if bwd_pkts>0 else 0.0

    # init win bytes from first packet's TCP.window if noted
    init_win_fwd = flow.get('init_win_fwd') or 0
    init_win_bwd = flow.get('init_win_bwd') or 0

    # active/idle segmentation: compute durations of active segments (gaps < 1s)
    segs = []  # list of (start, end)
    ts_sorted = sorted(flow['times'])
    if ts_sorted:
        seg_start = ts_sorted[0]
        prev = seg_start
        for t in ts_sorted[1:]:
            gap = t - prev
            if gap <= 1.0:
                prev = t
                continue
            else:
                # segment end
                segs.append((seg_start, prev))
                seg_start = t
                prev = t
        segs.append((seg_start, prev))
    active_durations = [ (end - start) for (start,end) in segs ] if segs else []
    idle_gaps = []
    if segs:
        # compute gaps between segments
        for i in range(1,len(segs)):
            idle_gaps.append(segs[i][0] - segs[i-1][1])

    def stats_list(lst):
        if not lst:
            return (0.0, 0.0, 0.0, 0.0)
        arr = np.array(lst)
        return (float(np.mean(arr)), float(np.std(arr, ddof=0)), float(np.max(arr)), float(np.min(arr)))

    active_mean, active_std, active_max, active_min = stats_list(active_durations)
    idle_mean, idle_std, idle_max, idle_min = stats_list(idle_gaps)

    # Bulk / Subflow features placeholders (advanced logic required) — set to 0
    fwd_avg_bytes_bulk = 0.0
    fwd_avg_pkts_bulk = 0.0
    fwd_avg_bulk_rate = 0.0
    bwd_avg_bytes_bulk = 0.0
    bwd_avg_pkts_bulk = 0.0
    bwd_avg_bulk_rate = 0.0
    subflow_fwd_pkts = 0
    subflow_fwd_bytes = 0
    subflow_bwd_pkts = 0
    subflow_bwd_bytes = 0
    act_data_pkt_fwd = sum(1 for l in flow['fwd_lens'] if l>0)
    min_seg_size_forward = min(flow['fwd_lens']) if flow['fwd_lens'] else 0

    # Build feature dict in exact FEATURE_COLUMNS order
    feat = {
        "Destination Port": int(dport or 0),
        "Flow Duration": duration_ms,
        "Total Fwd Packets": int(fwd_pkts),
        "Total Backward Packets": int(bwd_pkts),
        "Total Length of Fwd Packets": int(bytes_fwd),
        "Total Length of Bwd Packets": int(bytes_bwd),
        "Fwd Packet Length Max": int(fwd_max),
        "Fwd Packet Length Min": int(fwd_min),
        "Fwd Packet Length Mean": float(fwd_mean),
        "Fwd Packet Length Std": float(fwd_std),
        "Bwd Packet Length Max": int(bwd_max),
        "Bwd Packet Length Min": int(bwd_min),
        "Bwd Packet Length Mean": float(bwd_mean),
        "Bwd Packet Length Std": float(bwd_std),
        "Flow Bytes/s": float(flow_bytes_s),
        "Flow Packets/s": float(flow_pkts_s),
        "Flow IAT Mean": float(flow_iat_mean),
        "Flow IAT Std": float(flow_iat_std),
        "Flow IAT Max": float(flow_iat_max),
        "Flow IAT Min": float(flow_iat_min),
        "Fwd IAT Total": float(fwd_iat_total),
        "Fwd IAT Mean": float(fwd_iat_mean),
        "Fwd IAT Std": float(fwd_iat_std),
        "Fwd IAT Max": float(fwd_iat_max),
        "Fwd IAT Min": float(fwd_iat_min),
        "Bwd IAT Total": float(bwd_iat_total),
        "Bwd IAT Mean": float(bwd_iat_mean),
        "Bwd IAT Std": float(bwd_iat_std),
        "Bwd IAT Max": float(bwd_iat_max),
        "Bwd IAT Min": float(bwd_iat_min),
        "Fwd PSH Flags": int(fwd_psh),
        "Bwd PSH Flags": int(bwd_psh),
        "Fwd URG Flags": int(fwd_urg),
        "Bwd URG Flags": int(bwd_urg),
        "Fwd Header Length": int(fwd_hdr),
        "Bwd Header Length": int(bwd_hdr),
        "Fwd Packets/s": float(fwd_pkts_s),
        "Bwd Packets/s": float(bwd_pkts_s),
        "Min Packet Length": int(min_pkt_len),
        "Max Packet Length": int(max_pkt_len),
        "Packet Length Mean": float(mean_pkt_len),
        "Packet Length Std": float(std_pkt_len),
        "Packet Length Variance": float(var_pkt_len),
        "FIN Flag Count": int(fin),
        "SYN Flag Count": int(syn),
        "RST Flag Count": int(rst),
        "PSH Flag Count": int(psh),
        "ACK Flag Count": int(ack),
        "URG Flag Count": int(urg),
        "CWE Flag Count": int(cwr),   # approximate
        "ECE Flag Count": int(ece),
        "Down/Up Ratio": float(down_up_ratio),
        "Average Packet Size": float(avg_packet_size),
        "Avg Fwd Segment Size": float(avg_fwd_seg),
        "Avg Bwd Segment Size": float(avg_bwd_seg),
        "Fwd Header Length.1": int(fwd_hdr),
        "Fwd Avg Bytes/Bulk": float(fwd_avg_bytes_bulk),
        "Fwd Avg Packets/Bulk": float(fwd_avg_pkts_bulk),
        "Fwd Avg Bulk Rate": float(fwd_avg_bulk_rate),
        "Bwd Avg Bytes/Bulk": float(bwd_avg_bytes_bulk),
        "Bwd Avg Packets/Bulk": float(bwd_avg_pkts_bulk),
        "Bwd Avg Bulk Rate": float(bwd_avg_bulk_rate),
        "Subflow Fwd Packets": int(subflow_fwd_pkts),
        "Subflow Fwd Bytes": int(subflow_fwd_bytes),
        "Subflow Bwd Packets": int(subflow_bwd_pkts),
        "Subflow Bwd Bytes": int(subflow_bwd_bytes),
        "Init_Win_bytes_forward": int(init_win_fwd),
        "Init_Win_bytes_backward": int(init_win_bwd),
        "act_data_pkt_fwd": int(act_data_pkt_fwd),
        "min_seg_size_forward": int(min_seg_size_forward),
        "Active Mean": float(active_mean),
        "Active Std": float(active_std),
        "Active Max": float(active_max),
        "Active Min": float(active_min),
        "Idle Mean": float(idle_mean),
        "Idle Std": float(idle_std),
        "Idle Max": float(idle_max),
        "Idle Min": float(idle_min)
    }

    # Ensure all FEATURE_COLUMNS exist in feat (fill zeros if missing)
    row = [feat.get(col, 0.0) for col in FEATURE_COLUMNS]

    # Create DataFrame for scaler/model
    df_row = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    # numeric cleanup
    df_row = df_row.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Try scaling + prediction
    try:
        Xs = scaler.transform(df_row.values)
        pred_enc = model.predict(Xs)
        pred_label = le.inverse_transform(pred_enc)[0]
    except Exception as e:
        if VERBOSE:
            print("[!] Prediction failed:", e)
            traceback.print_exc()
        pred_label = "Unknown"

    # Log to sqlite
    ts = datetime.utcfromtimestamp(flow['last_ts']).isoformat() + "Z"
    flow_key = f"{src}:{sport}->{dst}:{dport}/{proto}"
    features_json = json.dumps(feat, default=str)
    row_insert = (ts, flow_key, src, dst, int(sport or 0), int(dport or 0), str(proto), float(duration_s), int(total_packets), int(total_bytes), str(pred_label), features_json)

    buff.append(row_insert)

    if len(buff) >= BATCH_SIZE:
        flush_db()

    # console
    if VERBOSE:
        print(f"[PREDICT] {ts} {flow_key} -> {pred_label} (pkts={total_packets} bytes={total_bytes} dur={duration_s:.2f}s)")

# -------------------------
# DB buffer & flush
# -------------------------
buff = deque()
def flush_db():
    global buff
    if not buff:
        return
    try:
        cur.executemany(INSERT_SQL, list(buff))
        conn.commit()
        if VERBOSE:
            print(f"[DB] committed {len(buff)} rows")
    except Exception as e:
        print("[DB] commit failed:", e)
        conn.rollback()
    buff.clear()

# -------------------------
# PACKET PROCESSOR
# -------------------------
def process_packet(pkt):
    # Only process IP packets
    if not pkt.haslayer(IP):
        return
    ip = pkt[IP]
    src = ip.src
    dst = ip.dst
    proto = ip.proto
    sport = None
    dport = None

    if pkt.haslayer(TCP):
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
    elif pkt.haslayer(UDP):
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
    else:
        sport = 0
        dport = 0

    key = (src, dst, int(sport or 0), int(dport or 0), int(proto or 0))
    rev_key = (dst, src, int(dport or 0), int(sport or 0), int(proto or 0))

    now = float(pkt.time)
    pkt_len = len(pkt)
    flags = {}
    if pkt.haslayer(TCP):
        tcp = pkt[TCP]
        flags = tcp_flags_to_counts(tcp.flags)
        hdr_len = int(getattr(tcp, "dataofs", 0) * 4)
        win = int(getattr(tcp, "window", 0))
    else:
        hdr_len = int(getattr(ip, "ihl", 0) * 4) if hasattr(ip, "ihl") else 0
        win = 0

    # Check if flow exists in forward direction or reverse (we maintain flows keyed by observed direction)
    if key in flows:
        flow = flows[key]
        direction_forward = True
    elif rev_key in flows:
        flow = flows[rev_key]
        direction_forward = False
    else:
        # new flow, create
        flow = make_flow_entry(pkt, True)
        flows[key] = flow
        direction_forward = True

    # Update flow
    flow['last_ts'] = now
    flow['pkts'] += 1
    flow['bytes'] += pkt_len
    flow['times'].append(now)

    # detect direction relative to flow's initial direction: if key is original key stored in flows,
    # we can infer forward packets are those arriving from stored src->dst; we check stored first packet src
    # we handle by comparing (src,dst, sport,dport) with flow's first direction
    # Determine if this packet is forward: if (src,dst,sport,dport) equals the flow key (the stored dict key)
    # For simplicity, treat packets whose source equals the flow's stored src as forward
    stored_src = None
    # find stored flow key
    found_key = None
    for k,v in flows.items():
        if v is flow:
            found_key = k
            break
    if found_key:
        stored_src = found_key[0]
        if src == stored_src:
            is_forward = True
        else:
            is_forward = False
    else:
        is_forward = direction_forward

    if is_forward:
        flow['fwd_pkts'] += 1
        flow['fwd_lens'].append(pkt_len)
        flow['fwd_times'].append(now)
        # PSH/URG counters
        flow['fwd_psh'] = flow.get('fwd_psh',0) + flags.get('PSH',0)
        flow['fwd_urg'] = flow.get('fwd_urg',0) + flags.get('URG',0)
        if flow.get('init_win_fwd') is None:
            flow['init_win_fwd'] = win
        flow['fwd_hdr_len'] = flow.get('fwd_hdr_len',0) or hdr_len
    else:
        flow['bwd_pkts'] += 1
        flow['bwd_lens'].append(pkt_len)
        flow['bwd_times'].append(now)
        flow['bwd_psh'] = flow.get('bwd_psh',0) + flags.get('PSH',0)
        flow['bwd_urg'] = flow.get('bwd_urg',0) + flags.get('URG',0)
        if flow.get('init_win_bwd') is None:
            flow['init_win_bwd'] = win
        flow['bwd_hdr_len'] = flow.get('bwd_hdr_len',0) or hdr_len

    # aggregate flags overall
    for f_name, val in flags.items():
        flow['flags'][f_name] += val

    # active/idle segmentation: update last_segment_start/last_pkt_ts
    gap = now - flow['last_pkt_ts'] if flow.get('last_pkt_ts') else 0
    if gap <= 1.0:
        # continue segment
        pass
    else:
        # end previous segment
        prev_start = flow.get('last_segment_start')
        if prev_start is not None:
            flow['segments'].append((prev_start, flow.get('last_pkt_ts', prev_start)))
        flow['last_segment_start'] = now
    flow['last_pkt_ts'] = now

    # Update flow's first/last timestamps
    flow['first_ts'] = min(flow.get('first_ts', now), now)
    flow['last_ts'] = max(flow.get('last_ts', now), now)

# -------------------------
# FLOW FINALIZER THREAD
# -------------------------
import threading

def flow_finalizer_loop():
    while True:
        now = time.time()
        keys_to_close = []
        for key, flow in list(flows.items()):
            last_seen = flow.get('last_ts', 0)
            if now - last_seen > FLOW_TIMEOUT:
                keys_to_close.append(key)
        for key in keys_to_close:
            try:
                flow = flows.pop(key)
                # finalize predict + log
                finalize_flow_and_predict(key, flow)
            except Exception as e:
                print("[!] finalize error:", e)
                traceback.print_exc()
        time.sleep(1.0)

# -------------------------
# START SNIFFING
# -------------------------
def start_sniff(iface=None, packet_count=0):
    # default iface from scapy
    if iface is None:
        iface = conf.iface
    if VERBOSE:
        print(f"[+] Starting sniff on iface={iface}, FLOW_TIMEOUT={FLOW_TIMEOUT}s")
    # start finalizer thread
    t = threading.Thread(target=flow_finalizer_loop, daemon=True)
    t.start()

    try:
        sniff(iface=iface, prn=process_packet, store=False, count=packet_count)
    except Exception as e:
        print("[!] sniff error:", e)
        traceback.print_exc()
    finally:
        # finalize remaining
        for key, flow in list(flows.items()):
            try:
                flows.pop(key)
                finalize_flow_and_predict(key, flow)
            except:
                pass
        flush_db()
        conn.close()

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    try:
        # print available interfaces if needed
        # print("Interfaces:", get_if_list())
        iface_name = None   # None => scapy default active interface
        pkt_cnt = 0         # 0 => infinite
        start_sniff(iface=iface_name, packet_count=pkt_cnt)
    except KeyboardInterrupt:
        print("\n[+] Stopped by user")
    except Exception as e:
        print("[FATAL]", e)
        traceback.print_exc()
