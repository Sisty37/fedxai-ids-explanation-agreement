# fedxai-ids — Week 1 (data pipeline + agreement metrics)

Code for the TAIIS 2026 study *cross-client explanation agreement in federated IDS*.
Every choice here follows the locked protocol in `configs/protocol.yaml`.

## Layout

| Path | What it does |
| --- | --- |
| `configs/protocol.yaml` | Locked protocol (split, R, transform, partitions, EED settings) |
| `src/fedxai/data/schema.py` | 2018→2017 column map, label → family mapping (Infiltration/Heartbleed dropped) |
| `src/fedxai/data/prepare.py` | Cleaning: header-repeat rows, labels, ID cols + Dst Port, ±Inf→NaN→drop, dedupe, label conflicts, 70/10/20 split, constant cols (train only) |
| `src/fedxai/data/partition.py` | Dirichlet + attack-skew partitions; Test-B split with the same per-class client proportions |
| `src/fedxai/data/fed_norm.py` | Federated normalisation statistics (min → log1p mask; n, Σx, Σx² → mean/std); reference set R + SHAP background |
| `src/fedxai/metrics/agreement.py` | RBO, Spearman divergence, J₀ / J\*, bootstrap noise floor, signed EED, EED⁺, sig. rate, factorized 2×2, t-based CI |
| `scripts/` | `check_columns.py`, `prepare_data.py`, `make_partitions.py`, `make_synthetic.py` |
| `tests/` | 48 unit tests (run `python -m pytest -q`) |

## Download (recommended: improved Distrinet releases)

Use the error-corrected versions from Liu et al., IEEE CNS 2022
(https://intrusion-detection.distrinet-research.be/CNS2022/index.html).
Both were regenerated with the same fixed CICFlowMeter, so 2017 and 2018 share
one column schema.

```bash
mkdir -p data/raw && cd data/raw
curl -L -O https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip   # ~328 MB
curl -L -O https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip # ~9.9 GB zip, ~36 GB CSV
unzip CICIDS2017_improved.zip   -d cicids2017
unzip CSECICIDS2018_improved.zip -d csecicids2018
```

Make sure each folder directly contains the day-wise `*.csv` files (move them up
if the zip adds a sub-folder). Record the download date in the lock record.

- `... - Attempted` flows are dropped (locked); the `Attempted Category` column is removed.
- 2018 is streamed in chunks: all attack rows kept, Benign sampled at
  `cicids2018_benign_fraction` (default 0.05). Needs ~40 GB free disk for the CSVs.

## Week 1 steps

1. `pip install -r requirements.txt` then `python -m pytest -q` (all should pass).
2. Put the CSVs in `data/raw/cicids2017/` and `data/raw/csecicids2018/`.
3. `python scripts/check_columns.py data/raw/cicids2017 data/raw/csecicids2018`
   — anything listed as "only 2017/only 2018" needs an entry in `COLUMN_MAP_2018`
   (`schema.py`). Corrected dataset releases may use different headers.
4. `python scripts/prepare_data.py` — if it stops with `Unknown label`, add the
   label to `map_label()`; never silently drop it.
5. `python scripts/make_partitions.py` — 35 partitions (6 α + attack-skew × 5 seeds).
6. Fill the dataset lock record below from `data/processed/lock_record.json`
   and commit both. After this, the data protocol does not change.

## Dataset lock record

```text
CICIDS2017      — exact source/version:
CSE-CIC-IDS2018 — exact source/version:
Removed rows (header repeat / dropped classes / Inf-NaN / duplicates / label conflicts):
Removed columns (ID / constant):
Final features (2017 full / 2017∩2018 common):
Final class counts (2017 / 2018):
```

## Notes

- Normalisation stats are additive, so federated stats equal pooled train
  stats for every partition (tested). Cross-dataset uses the 2017 stats only.
- J₀ reproduces FedSHAP-WSN Remark 1 (k = 3, 5, 10, 15; F = 18) — tested.
- Bootstrap reuses the SHAP matrix (no recomputation): B = 1000 takes ~0.3 s per client.
- R and background indices are stored in `cicids2017.npz`; R never enters
  training, tuning or early stopping.
