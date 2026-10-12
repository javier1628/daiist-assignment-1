"""Single source of truth for loading the data and making THE split.

Every model trains on exactly the split this module returns, so the comparison is fair.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "inventario_julio_2025_limpio.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns that must never become features (see CLAUDE.md / data/README.md).
LEAKY_COLUMNS = [
    "Precio detalle", "Precio mayor 1", "Precio mayor 2", "Precio mayor 3", "Precio mínimo",
    "Margen %", "Stock", "Estado stock", "Valor inventario", "Punto reorden", "Cant. reorden",
    "Alertas",
]


def load_raw() -> pd.DataFrame:
    """Load the cleaned snapshot and keep only physical products with a usable cost."""
    df = pd.read_csv(DATA_PATH)
    df = df[(df["Tipo"] == "Producto") & (df["Costo"] > 0)].copy()
    return df.reset_index(drop=True)


def make_target(df: pd.DataFrame) -> pd.Series:
    """log(Costo): cost spans 6 orders of magnitude, so model the log, not the raw value."""
    return np.log(df["Costo"])


def split(df: pd.DataFrame):
    """One split for the whole project. Returns (train_df, test_df)."""
    return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)


if __name__ == "__main__":
    df = load_raw()
    tr, te = split(df)
    print(f"{len(df)} rows -> train {len(tr)} / test {len(te)}")
    print(make_target(df).describe())
