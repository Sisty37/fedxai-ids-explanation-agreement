"""Locked cleaning pipeline (research plan, 'Datasets and preprocessing').

Order: canonicalise -> drop header-repeat rows -> map labels / drop classes ->
drop ID columns (+ Dst Port unless sensitivity) -> to numeric, +-Inf -> NaN ->
remove NaN rows -> dedupe (drop conflicting-label groups) -> stratified
70/10/20 split -> drop constant features (decided on TRAIN only).
Every step's removed count goes into the lock record.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .schema import DROP, DST_PORT, ID_COLUMNS, LABEL, canonicalize_columns, map_label


def load_raw(folder: str, dataset: str, benign_fraction: float = 1.0, seed: int = 0,
             chunksize: int = 1_000_000) -> pd.DataFrame:
    """Read all CSVs in `folder`. With benign_fraction < 1, files are streamed in
    chunks and Benign rows are subsampled (all attack rows kept) -- needed for
    CSE-CIC-IDS2018 (tens of GB). Header-repeat rows are removed later in clean()."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    rng = np.random.default_rng(seed)
    parts = []
    for f in files:
        reader = pd.read_csv(f, low_memory=False, encoding="latin-1",
                             chunksize=chunksize if benign_fraction < 1 else None)
        for df in ([reader] if isinstance(reader, pd.DataFrame) else reader):
            df = canonicalize_columns(df, dataset)
            if benign_fraction < 1:
                ben = df[LABEL].astype(str).str.strip().str.lower() == "benign"
                keep = ~ben | (rng.random(len(df)) < benign_fraction)
                df = df.loc[keep]
            parts.append(df)
    return pd.concat(parts, ignore_index=True, join="outer")


def clean(df: pd.DataFrame, include_dst_port: bool = False):
    rep: dict = {"rows_raw": int(len(df))}
    if LABEL not in df.columns:
        raise KeyError("No label column after canonicalisation")

    # 2018 CSVs repeat the header row inside the file
    hdr = df[LABEL].astype(str).str.strip().str.lower() == "label"
    df = df.loc[~hdr]
    rep["rows_removed_header_repeat"] = int(hdr.sum())

    fam = df[LABEL].astype(str).map(map_label)
    rep["raw_label_counts"] = df[LABEL].astype(str).value_counts().to_dict()
    keep = fam != DROP
    rep["rows_removed_dropped_classes"] = int((~keep).sum())
    df, fam = df.loc[keep], fam.loc[keep]

    drop_cols = [c for c in df.columns if c in ID_COLUMNS or c == LABEL]
    if not include_dst_port and DST_PORT in df.columns:
        drop_cols.append(DST_PORT)
    rep["columns_removed_id"] = sorted(set(drop_cols) - {LABEL})
    X = df.drop(columns=drop_cols).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    nan_rows = X.isna().any(axis=1)
    rep["rows_removed_inf_nan"] = int(nan_rows.sum())
    X, y = X.loc[~nan_rows], fam.loc[~nan_rows]

    # exact duplicates on (features, label); then groups with conflicting labels
    Xy = X.assign(_y=y.values)
    dup = Xy.duplicated(keep="first")
    rep["rows_removed_duplicates"] = int(dup.sum())
    Xy = Xy.loc[~dup]
    conflict = Xy.drop(columns="_y").duplicated(keep=False)
    rep["rows_removed_label_conflicts"] = int(conflict.sum())
    Xy = Xy.loc[~conflict]

    y = Xy.pop("_y").reset_index(drop=True)
    X = Xy.reset_index(drop=True).astype(np.float64)
    rep["rows_final"] = int(len(X))
    return X, y, rep


def stratified_split(y: pd.Series, fracs=(0.70, 0.10, 0.20), seed: int = 0):
    assert abs(sum(fracs) - 1) < 1e-9
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=fracs[0], stratify=y, random_state=seed)
    a_share = fracs[1] / (fracs[1] + fracs[2])
    ta, tb = train_test_split(rest, train_size=a_share, stratify=y.iloc[rest], random_state=seed)
    return np.sort(tr), np.sort(ta), np.sort(tb)


def constant_columns(X_train: pd.DataFrame) -> list[str]:
    return [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
