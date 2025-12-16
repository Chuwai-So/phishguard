from __future__ import annotations
#Better handling of paths
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

#Tidy up return types with dataclass
#frozen=True prevents unwanted writing
@dataclass(frozen=True)
class DataSplits:
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame


def load_csv(csv_path: str | Path, url_col: str = "url", label_col: str = "label") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist")

    df = pd.read_csv(csv_path)

    #Column presence checks:
    if url_col not in df.columns:
        raise ValueError(f"Missing url column '{url_col}'")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'")

    #Data preparing
    df = df[[url_col, label_col]].dropna()

    #Normalizing types
    df[url_col] = df[url_col].astype(str)

    df[label_col] = df[label_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(int)

    #Enforce labels
    allowed = {0,1}
    found = set(df[label_col].unique())
    if not found.issubset(allowed):
        raise ValueError(f"Labels must be 0/1. Found: {sorted(found)}'")

    return df

def split_df(
        df: pd.DataFrame,
        label_col: str = "label",
        seed: int = 42,
        test_size: float = 0.15,
        val_size: float = 0.15,
) -> DataSplits:
    #First Split between train&validation sets and testing sets:
    train_val, test = train_test_split(df,
                                       test_size=test_size,
                                       random_state=seed,
                                       stratify=df[label_col]
                                       )

    val_ratio = val_size/(1.0 - test_size)

    #Second Splint between train and validation sets:
    train, val = train_test_split(train_val,
                                  test_size=val_ratio,
                                  random_state=seed,
                                  stratify=train_val[label_col]
                                  )
    return DataSplits(
        train_frame = train.reset_index(drop = True),
        val_frame = val.reset_index(drop = True),
        test_frame = test.reset_index(drop = True),
    )

