# Measuring Explanation Agreement in Federated Intrusion Detection

Code for the TAIIS 2026 paper *"Measuring Explanation Agreement in Federated Intrusion
Detection: A Noise-Aware Evaluation Protocol"* (A. J. Sisty, N. Roy, A. Bhowmik,
S. R. Ahona, J. A. Noshin; American International University-Bangladesh).

The protocol compares SHAP explanations across federated clients and models with the
**excess explanation divergence (EED)**: a top-weighted rank divergence (1 - RBO, p = 0.9)
minus a bootstrap noise floor, read against retraining-seed and IID-federated baselines.

## Repository layout

| Path | Content |
|---|---|
| `configs/protocol.yaml` | Locked protocol: split, reference set, background, partitions, RBO p, bootstrap, seeds |
| `src/fedxai/data/` | Cleaning, label mapping, stratified split, federated normalisation, partitions |
| `src/fedxai/train/` | MLP / 1D-CNN, centralized, FedAvg, FedProx, local-only training, evaluation |
| `src/fedxai/explain/eg.py` | Expected gradients (estimator behind SHAP's GradientExplainer) |
| `src/fedxai/metrics/agreement.py` | Importance vectors, RBO, noise floor, EED, chance-corrected Jaccard, t-intervals |
| `scripts/` | One script per experiment (see below) |
| `tests/` | Unit tests for metrics, data handling and models (`pytest`) |
| `docs/fig1_methodology.drawio` | Editable source of the methodology diagram (Fig. 1) |
| `docs/supplement.pdf` | Supplement: EED pseudocode, tables and figures moved from the paper, sensitivity analyses, full UNSW-NB15 results |

Datasets are **not** included (see below).

## Setup

Python 3.10+, PyTorch 2.x (CUDA optional), then

```
pip install -r requirements.txt
pip install shap          # only for the KernelSHAP checks
pytest -q                 # optional
```

All results in the paper were produced on a laptop (RTX 4050, 6 GB; 16 GB RAM) with Python 3.11.9, PyTorch 2.5.1 (CUDA 12.1), NumPy 2.4.6, scikit-learn 1.9.0 and SHAP 0.51.0.

## Data

| Dataset | Source | Put the CSVs in |
|---|---|---|
| CICIDS2017, error-corrected (Engelen et al. 2021; Liu et al. 2022) | Distrinet, KU Leuven: https://intrusion-detection.distrinet-research.be/CNS2022/ | `data/raw/cicids2017/` |
| CSE-CIC-IDS2018, error-corrected (Liu et al. 2022) | same page | `data/raw/csecicids2018/` |
| UNSW-NB15 (training-set + testing-set CSVs) | UNSW Canberra: https://research.unsw.edu.au/projects/unsw-nb15-dataset | `data/raw/unswnb15/` |

## Reproducing the paper (run from the repository root)

| Step | Command | Paper |
|---|---|---|
| 1 | `python scripts/prepare_data.py --skip-2018` | Sect. 4 (cleaning counts in `data/processed/lock_record.json`) |
| 2 | `python scripts/extend_seeds.py` then `python scripts/make_partitions.py` | 10 seeds, client partitions |
| 3 | `python scripts/train_all.py` and `python scripts/train_all.py --fedprox` | centralized, FedAvg, FedProx models |
| 4 | `python scripts/train_local.py` | local-only models (factorized analysis) |
| 5 | `python scripts/analyze_main.py` | seed baseline, fidelity, detection, class mix, trajectories |
| 6 | `python scripts/analyze_extra.py` | heterogeneity drift, factorized analysis (Table 2) |
| 7 | `python scripts/analyze_fedprox_rq2.py` | FedProx vs FedAvg (Table 3), run-level coupling |
| 8 | `python scripts/e7_ablations.py`, `python scripts/e7_dstport.py` | Jaccard, background, KernelSHAP, destination port |
| 9 | `python scripts/prepare_2018.py`, `analyze_e6.py`, `sanity_e6.py`, `e6_mechanism.py` | cross-dataset shift (Fig. 4) |
| 10 | `python scripts/cnn_validation.py` | 1D-CNN |
| 11 | `python scripts/sensitivity.py` | sensitivity: RBO p, background size, explained flows, bootstrap, KernelSHAP per class, cost, per-class F1 |
| 12 | `python scripts/unsw.py all` | replication on UNSW-NB15 |
| 13 | `python scripts/make_figures.py`, `python scripts/check_numbers.py` | supplementary figures; checks every number quoted in the paper |
| 14 | `python scripts/make_paper_figures.py` | paper Fig. 2 (CICIDS2017 vs. UNSW-NB15) and Fig. 3 (detection, distance over rounds) |

Every training script is resumable (finished runs are skipped). Random seeds: data split
and reference set use seed 0; training seeds are 0-9 (0-4 for the CNN); partitions use
the training seed.

## EED in a few lines

```python
from fedxai.metrics.agreement import importance, d_rbo, bootstrap_importances, within_divergences
import numpy as np

def noise_floor(phi, B=1000, pairs=5000, p=0.9, rng=np.random.default_rng(0)):
    # phi: SHAP matrix (n_flows, n_features) of ONE class, shared background
    return within_divergences(bootstrap_importances(phi, B, rng), pairs, rng, p).mean()

def eed(phi_a, phi_b, p=0.9):
    D = d_rbo(importance(phi_a), importance(phi_b), p)
    return D - 0.5 * (noise_floor(phi_a) + noise_floor(phi_b))
```

## Citation

If you use this code, please cite the TAIIS 2026 paper (Springer LNEE, to appear).
