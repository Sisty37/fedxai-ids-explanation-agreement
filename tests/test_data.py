import numpy as np, pandas as pd, pytest
from fedxai.data.schema import map_label, DROP, canonicalize_columns, norm
from fedxai.data.prepare import load_raw, clean, stratified_split, constant_columns
from fedxai.data.partition import dirichlet_partition, attack_skew_partition, local_test_split
from fedxai.data.fed_norm import federated_norm_stats, build_reference
import make_synthetic


@pytest.mark.parametrize("raw,fam", [
    ("BENIGN", "Benign"), ("Benign", "Benign"), ("DoS Hulk", "DoS"), ("DoS slowloris", "DoS"),
    ("DoS Slowhttptest", "DoS"), ("DoS attacks-SlowHTTPTest", "DoS"), ("DDoS", "DDoS"),
    ("DDOS attack-HOIC", "DDoS"), ("DDoS attacks-LOIC-HTTP", "DDoS"), ("DDOS attack-LOIC-UDP", "DDoS"),
    ("PortScan", "PortScan"), ("FTP-Patator", "BruteForce"), ("SSH-Patator", "BruteForce"),
    ("FTP-BruteForce", "BruteForce"), ("SSH-Bruteforce", "BruteForce"),
    ("Web Attack \x96 Brute Force", "WebAttack"), ("Web Attack \ufffd XSS", "WebAttack"),
    ("Web Attack – Sql Injection", "WebAttack"), ("Brute Force -Web", "WebAttack"),
    ("Brute Force -XSS", "WebAttack"), ("SQL Injection", "WebAttack"), ("Bot", "Bot"),
    ("Infiltration", DROP), ("Infilteration", DROP), ("Heartbleed", DROP),
    ("Botnet", "Bot"), ("Botnet Ares", "Bot"), ("Portscan", "PortScan"),
    ("DoS Hulk - Attempted", DROP), ("FTP-Patator - Attempted", DROP),
    ("Infiltration - Portscan", DROP), ("Web Attack - SQL Injection", "WebAttack")])
def test_label_map(raw, fam):
    assert map_label(raw) == fam


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        map_label("Mystery attack")


def test_2018_columns_map_to_2017():
    df = pd.DataFrame(columns=["Dst Port", "Flow Byts/s", "Init Fwd Win Byts", "Label"])
    assert list(canonicalize_columns(df, "2018").columns) == [
        norm("Destination Port"), norm("Flow Bytes/s"), norm("Init_Win_bytes_forward"), "label"]


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    make_synthetic.main(str(d))
    return d


def test_clean_pipeline(syn):
    X, y, rep = clean(load_raw(str(syn / "cicids2017"), "2017"))
    assert rep["rows_removed_duplicates"] == 40
    assert rep["rows_removed_label_conflicts"] == 2
    assert rep["rows_removed_inf_nan"] == 5          # NaN rows; Inf sits in dropped Dst Port
    assert rep["rows_removed_dropped_classes"] == 15
    assert norm("Destination Port") not in X.columns
    assert np.isfinite(X.to_numpy()).all()
    assert not X.assign(_y=y.values).duplicated().any()
    assert set(y) == {"Benign", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot"}


def test_clean_2018_header_and_dst_port_sensitivity(syn):
    X, y, rep = clean(load_raw(str(syn / "csecicids2018"), "2018"), include_dst_port=True)
    assert rep["rows_removed_header_repeat"] == 1
    assert norm("Destination Port") in X.columns
    assert "timestamp" not in X.columns and "protocol" not in X.columns


def test_split_and_reference_no_leakage(syn):
    X, y, _ = clean(load_raw(str(syn / "cicids2017"), "2017"))
    tr, ta, tb = stratified_split(y, (0.7, 0.1, 0.2), seed=0)
    assert len(set(tr) | set(ta) | set(tb)) == len(y)
    assert not (set(tr) & set(ta)) and not (set(tr) & set(tb)) and not (set(ta) & set(tb))
    R, bg = build_reference(y.to_numpy()[ta], per_class=20, background_size=100, seed=0)
    assert set(bg) <= set(R)
    assert len(bg) == 100
    assert pd.Series(y.to_numpy()[ta][R]).value_counts().max() <= 20
    assert "bwdurgflags" in constant_columns(X.iloc[tr])


def _labels(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array(list("ABCDEFG")), size=n, p=[.55, .1, .1, .08, .08, .05, .04])


@pytest.mark.parametrize("alpha", [100, 1, 0.1])
def test_dirichlet_partition(alpha):
    y = _labels()
    parts, P = dirichlet_partition(y, 10, alpha, seed=1, min_size=50)
    allidx = np.concatenate(parts)
    assert len(allidx) == len(y) == len(set(allidx))
    assert np.allclose(P.sum(axis=0), 1)
    assert min(map(len, parts)) >= 50


def test_local_test_split_follows_train_proportions():
    y = _labels(20000)
    parts, P = dirichlet_partition(y, 10, 0.3, seed=2, min_size=50)
    yt = _labels(8000, seed=9)
    classes = list(np.unique(y))
    tparts = local_test_split(yt, P, classes, seed=2)
    assert sorted(np.concatenate(tparts)) == list(range(len(yt)))
    for k in range(10):
        for ci, c in enumerate(classes):
            exp = P[k, ci] * (yt == c).sum()
            assert abs((yt[tparts[k]] == c).sum() - exp) <= 1


def test_attack_skew():
    y = _labels()
    parts, P = attack_skew_partition(y, 10, "A", seed=0, families_per_client=2)
    for k, p in enumerate(parts):
        fams = set(y[p]) - {"A"}
        assert "A" in set(y[p]) and len(fams) == 2
    for c in "BCDEFG":
        assert sum(c in set(y[p]) for p in parts) >= 2


def test_federated_norm_equals_pooled():
    rng = np.random.default_rng(0)
    X = np.c_[rng.lognormal(size=(3000, 3)), rng.normal(size=(3000, 2))]  # 3 nonneg, 2 signed
    clients = np.array_split(X[rng.permutation(len(X))], 7)
    ns = federated_norm_stats(clients)
    assert ns.log1p_mask.tolist() == [True, True, True, False, False]
    Z = X.copy(); Z[:, :3] = np.log1p(Z[:, :3])
    assert np.allclose(ns.mean, Z.mean(0)) and np.allclose(ns.std, Z.std(0))
    T = ns.transform(X)
    assert np.allclose(T.mean(0), 0, atol=1e-5) and np.allclose(T.std(0), 1, atol=1e-4)
    shifted = X.copy(); shifted[0, 0] = -5          # out-of-range in transfer data
    assert np.isfinite(ns.transform(shifted)).all()


def test_benign_subsampling_keeps_all_attacks(syn):
    full = load_raw(str(syn / "csecicids2018"), "2018")
    sub = load_raw(str(syn / "csecicids2018"), "2018", benign_fraction=0.1, chunksize=500)
    is_b = lambda d: d["label"].astype(str).str.lower() == "benign"
    assert (~is_b(sub)).sum() == (~is_b(full)).sum()
    assert is_b(sub).sum() < 0.2 * is_b(full).sum()


def test_improved_2017_headers_are_canonicalised():
    df = pd.DataFrame(columns=["Dst Port", "Fwd Act Data Pkts", "Fwd Seg Size Min", "Flow Duration", "Label"])
    cols = list(canonicalize_columns(df, "2017").columns)
    assert cols == [norm("Destination Port"), norm("act_data_pkt_fwd"), norm("min_seg_size_forward"),
                    norm("Flow Duration"), "label"]
    X, _, _ = clean(pd.DataFrame({"Dst Port": [80, 443], "Flow Duration": [1.0, 2.0], "Label": ["BENIGN", "DDoS"]}).pipe(canonicalize_columns, "2017"))
    assert norm("Destination Port") not in X.columns   # Dst Port must be dropped in the main protocol
