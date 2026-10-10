"""Synthetic CICIDS-like CSVs (real header names, messy labels, Inf/NaN,
duplicates, label conflicts, repeated 2018 header row) for smoke tests.
Usage: python scripts/make_synthetic.py data/synthetic
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.data.schema import COLUMN_MAP_2018

SAME = ["Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Down/Up Ratio",
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]
L17 = {"BENIGN": 3000, "DoS Hulk": 400, "DoS slowloris": 100, "DDoS": 400, "PortScan": 400,
       "FTP-Patator": 150, "SSH-Patator": 150, "Web Attack \x96 Brute Force": 60,
       "Web Attack \x96 XSS": 40, "Web Attack \x96 Sql Injection": 20, "Bot": 120,
       "Infiltration": 10, "Heartbleed": 5}
L18 = {"Benign": 3000, "DoS attacks-Hulk": 400, "DDOS attack-HOIC": 400, "FTP-BruteForce": 150,
       "SSH-Bruteforce": 150, "Brute Force -Web": 40, "Brute Force -XSS": 30, "SQL Injection": 20,
       "Bot": 120, "Infilteration": 30}
FAM = {"benign": 0, "dos": 1, "ddos": 2, "portscan": 3, "ftp": 4, "ssh": 4, "web": 5, "brute force -": 5,
       "sql": 5, "bot": 6, "inf": 7, "heart": 7}


def fam_id(lbl):
    l = lbl.lower()
    for k in ["ddos", "brute force -", "web", "sql", "ftp", "ssh", "portscan", "dos", "bot", "inf", "heart", "benign"]:
        if l.startswith(k) or k in l:
            return FAM[k]
    return 0


def make(labels, cols, rng, extra=None):
    rows = []
    for lbl, n in labels.items():
        f = fam_id(lbl)
        base = rng.lognormal(mean=1 + 0.4 * f, sigma=1.0, size=(n, len(cols)))
        base[:, f % len(cols)] *= 20  # class signal
        df = pd.DataFrame(base, columns=cols)
        df["__lbl"] = lbl
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df.iloc[:5, 0] = np.inf; df.iloc[5:10, 1] = np.nan          # Inf / NaN
    df = pd.concat([df, df.iloc[20:60]], ignore_index=True)     # 40 exact duplicates
    conf = df.iloc[[100]].copy(); conf["__lbl"] = "Bot"          # 1 label conflict pair
    df = pd.concat([df, conf], ignore_index=True)
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df.sample(frac=1, random_state=0).reset_index(drop=True)


def main(out):
    rng = np.random.default_rng(0)
    c17 = list(COLUMN_MAP_2018.values()) + SAME
    c18 = list(COLUMN_MAP_2018.keys()) + SAME
    for d in ["cicids2017", "csecicids2018"]:
        os.makedirs(os.path.join(out, d), exist_ok=True)
    d17 = make(L17, c17, rng)
    d17["Fwd Header Length.1"] = d17["Fwd Header Length"]
    d17["Bwd URG Flags"] = 0.0                                  # constant column
    d17 = d17.rename(columns=lambda c: " " + c if c != "Flow Bytes/s" else c)  # CIC leading spaces
    d17 = d17.rename(columns={" __lbl": " Label"})
    h = len(d17) // 2
    d17.iloc[:h].to_csv(os.path.join(out, "cicids2017", "Monday.csv"), index=False)
    d17.iloc[h:].to_csv(os.path.join(out, "cicids2017", "Tuesday.csv"), index=False)
    d18 = make(L18, c18, rng, {"Protocol": 6, "Timestamp": "14/02/2018 08:31:01"})
    d18 = d18.rename(columns={"__lbl": "Label"})
    hdr = pd.DataFrame([{c: c for c in d18.columns}])          # repeated header row
    d18 = pd.concat([d18.iloc[:100], hdr, d18.iloc[100:]], ignore_index=True)
    d18.to_csv(os.path.join(out, "csecicids2018", "Wednesday-14-02-2018.csv"), index=False)
    print("synthetic CSVs written to", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
