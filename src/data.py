"""Single source of truth for loading the data and making THE split.

Every model trains on exactly the split this module returns, so the comparison is fair.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "inventario_julio_2025_limpio.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns that must never become features: the label is built from Stock and Punto reorden,
# and the other four are functions of the same columns (see CLAUDE.md / data/README.md).
# Price columns are excluded separately as a design choice (markup x cost, nothing beyond Costo).
LEAKY_COLUMNS = [
    "Stock", "Estado stock", "Valor inventario", "Punto reorden", "Cant. reorden", "Alertas",
]

# Business cost of each kind of mistake, in DOP. ASSUMPTIONS, not measured figures:
# a false positive discontinues a live item; a false negative keeps a dead one listed.
# Used by train.py to pick the threshold and by app.py to show the live cost figure.
FP_COST = 1500.0
FN_COST = 300.0


def load_raw() -> pd.DataFrame:
    """Load the cleaned snapshot and keep only physical products with a usable cost."""
    df = pd.read_csv(DATA_PATH)
    df = df[(df["Tipo"] == "Producto") & (df["Costo"] > 0)].copy()
    return df.reset_index(drop=True)


def make_target(df: pd.DataFrame) -> pd.Series:
    """Dead stock = nothing on hand AND no reorder point, i.e. the system does not intend
    to restock it. Returned as int (1 = dead, 0 = live)."""
    return ((df["Stock"] <= 0) & (df["Punto reorden"] == 0)).astype(int)


def split(df: pd.DataFrame):
    """One split for the whole project. Returns (train_df, test_df)."""
    return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)


if __name__ == "__main__":
    df = load_raw()
    tr, te = split(df)
    print(f"{len(df)} rows -> train {len(tr)} / test {len(te)}")
    y = make_target(df)
    print(f"dead: {int(y.sum())} of {len(y)} ({y.mean():.1%})")
