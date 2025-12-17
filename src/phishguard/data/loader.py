from __future__ import annotations
#Better handling of paths
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from urllib.parse import urlparse

#Tidy up return types with dataclass
#frozen=True prevents unwanted writing
@dataclass(frozen=True)
class DataSplits:
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame

def url_group(u: str) -> str:
    return (urlparse(u).hostname or "").lower()

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

#Deal with Domain leakage
def split_df_grouped(df: pd.DataFrame,
                     url_col: str = "url",
                     label_col: str = "label",
                     train_size: float = 0.7,
                     val_size: float =  0.15,
                     test_size: float =  0.15,
                     seed: int = 42,
                     ) -> DataSplits:
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9
    #Grouping all urls with the same hostname and never split the same group across sets
    groups = df[url_col].map(url_group)

    #Split base on groups
    #Train & Temp(Vali + Test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    #This returns indices on which group goes to training, which group goes to temp
    train_idx, temp_idx = next(
        gss1.split(df, df[label_col], groups = groups)
    )
    #Temp dataframe
    temp = df.iloc[temp_idx].copy()
    temp_groups = groups.iloc[temp_idx]

    #Temp & Test
    val_prop = val_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_prop, random_state=seed)
    val_rel, test_rel = next(gss2.split(temp, y=temp[label_col], groups=temp_groups))

    train_frame = df.iloc[train_idx].reset_index(drop=True)
    val_frame = temp.iloc[val_rel].reset_index(drop=True)
    test_frame = temp.iloc[test_rel].reset_index(drop=True)

    # Leakage checks (hostname disjoint)
    train_g = set(train_frame[url_col].map(url_group))
    val_g = set(val_frame[url_col].map(url_group))
    test_g = set(test_frame[url_col].map(url_group))
    assert train_g.isdisjoint(val_g), "Leakage: train and val share hosts"
    assert train_g.isdisjoint(test_g), "Leakage: train and test share hosts"
    assert val_g.isdisjoint(test_g), "Leakage: val and test share hosts"

    return DataSplits(train_frame=train_frame, val_frame=val_frame, test_frame=test_frame)





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

