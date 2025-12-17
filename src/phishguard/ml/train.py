import joblib
import numpy as np
import argparse
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

from phishguard.data.loader import load_csv, split_df, split_df_grouped
from phishguard.ml.features import extract_features

#Stacking feature vectors into matrix
def _featurize(urls) -> np.ndarray:
    return np.vstack([extract_features(u) for u in urls])

#hardcoded paths
def train(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train phishguard model")
    p.add_argument(
        "--dataset",
        type = Path,
        default = Path("data/processed/urls_clean.csv"),
        help = "Path to the dataset",
    )
    p.add_argument(
        "--artifacts-dir",
        type = Path,
        default = Path("artifacts"),
        help = "Where to save artifacts",
    )
    args = p.parse_args(argv)
    dataset_path = Path(args.dataset)

    df = load_csv(dataset_path)
    splits = split_df_grouped(df)

    #Preperation
    x_train = _featurize(splits.train_frame["url"])
    y_train = splits.train_frame["label"].to_numpy(dtype=np.int64)

    x_val = _featurize(splits.val_frame["url"])
    y_val = splits.val_frame["label"].to_numpy(dtype=np.int64)

    x_test = _featurize(splits.test_frame["url"])
    y_test = splits.test_frame["label"].to_numpy(dtype=np.int64)

    print("[train] Loaded rows:", len(df))
    print("[train] Splits (train/val/test):", len(splits.train_frame), len(splits.val_frame), len(splits.test_frame))
    print("[train] X_train shape:", x_train.shape, "y_train shape:", y_train.shape)

    #Training
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    #Evaluate on validation
    val_p = model.predict_proba(x_val)[:, 1]
    #fine tunning model with higher FN rate:
    t = 0.2
    val_pred = (val_p >= t).astype(int)
    print("\n[train] Validation confusion matrix:\n", confusion_matrix(y_val, val_pred))
    print("[train] Validation report:\n", classification_report(y_val, val_pred, digits=4))

    #Evaluate on test
    test_p = model.predict_proba(x_test)[:, 1]
    t = 0.2
    test_pred = (test_p >= t).astype(int)
    print("\n[test] Test confusion matrix:\n", confusion_matrix(y_test, test_pred))
    print("[test] Test report:\n", classification_report(y_test, test_pred, digits=4))

    #Save artifact
    artifact_dir = Path(args.artifacts_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "threshhold": t,
            "features_dim": x_train.shape[1],
            "features_version": 1,
        },
        artifact_dir / "logreg.joblib",
    )
    print(f"\n[train] Saved model to {artifact_dir / 'logreg.joblib'}")


