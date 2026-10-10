"""Print 2017 vs 2018 column alignment after canonicalisation. Run BEFORE
prepare_data.py; extend COLUMN_MAP_2018 in schema.py for anything unmatched.
Usage: python scripts/check_columns.py data/raw/cicids2017 data/raw/csecicids2018
"""
import glob, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedxai.data.schema import canonicalize_columns, ID_COLUMNS, LABEL

def cols(folder, ds):
    f = sorted(glob.glob(os.path.join(folder, "*.csv")))[0]
    return set(canonicalize_columns(pd.read_csv(f, nrows=5, encoding="latin-1"), ds).columns) - ID_COLUMNS - {LABEL}

a, b = cols(sys.argv[1], "2017"), cols(sys.argv[2], "2018")
print(f"common: {len(a & b)}")
print("only 2017:", sorted(a - b))
print("only 2018:", sorted(b - a))
