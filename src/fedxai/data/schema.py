"""Column canonicalisation and label mapping for CICIDS2017 and CSE-CIC-IDS2018.

All column names are mapped to the CICIDS2017 (MachineLearningCVE) naming, then
normalised with `norm()`. The 2018 -> 2017 map below covers the original CIC
CSV releases. Corrected re-releases may use different headers: run
`scripts/check_columns.py` and extend COLUMN_MAP_2018 for anything it reports.
"""
from __future__ import annotations

import re

import pandas as pd


def norm(name: str) -> str:
    """Lowercase, keep only [a-z0-9]. 'Flow Bytes/s' -> 'flowbytess'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Columns removed in every protocol (identifiers / leakage-prone).
ID_COLUMNS = {norm(c) for c in [
    "Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
    "Source Port", "Src Port", "Timestamp", "Protocol",
    "Fwd Header Length.1",  # exact duplicate column in CICIDS2017
    "id", "Attempted Category",  # improved (Distrinet) releases: row id, label metadata
]}
DST_PORT = norm("Destination Port")
LABEL = "label"

# CSE-CIC-IDS2018 header -> CICIDS2017 header (verify with scripts/check_columns.py)
COLUMN_MAP_2018 = {
    "Dst Port": "Destination Port",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "CWE Flag Count": "CWE Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}
_MAP_2018_NORM = {norm(k): norm(v) for k, v in COLUMN_MAP_2018.items()}


def canonicalize_columns(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    # The map is applied to BOTH datasets: the improved (Distrinet) 2017 release
    # also uses abbreviated headers ('Dst Port', 'Fwd Seg Size Min', ...).
    # Original 2017 headers are not keys of the map, so they pass unchanged.
    cols = [_MAP_2018_NORM.get(norm(c), norm(c)) for c in df.columns]
    df = df.copy()
    df.columns = cols
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


DROP = "__DROP__"


def map_label(raw: str) -> str:
    """Raw CIC label (2017 or 2018) -> family label, or DROP. Raises on unknown."""
    s = norm(raw)
    # improved releases mark failed/partial attacks as '... - Attempted': dropped (locked)
    if "attempted" in s:
        return DROP
    if s == "benign":
        return "Benign"
    if s.startswith("infilt") or s == "heartbleed":
        return DROP
    # web attacks first: 2018 'Brute Force -Web' must not become BruteForce
    if s.startswith("webattack") or "xss" in s or "sql" in s or s.endswith("web"):
        return "WebAttack"
    if s.startswith("ddos"):
        return "DDoS"
    if s.startswith("dos"):
        return "DoS"
    if s == "portscan":
        return "PortScan"
    if s in {"ftppatator", "sshpatator", "ftpbruteforce", "sshbruteforce"}:
        return "BruteForce"
    if s.startswith("bot"):  # 'Bot', 'Botnet', 'Botnet Ares'
        return "Bot"
    raise ValueError(f"Unknown label: {raw!r}")
