import joblib
import numpy as np

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

from phishguard.data.loader import load_csv, split_df
from phishguard.ml.features import extract_features

#Stacking feature vectors into matrix
def _featurize(urls) -> np.ndarray:
    return np.vstack([extract_features(u) for u in urls])

#hardcoded paths
def train() -> None:
    dataset_path = Path("data/raw/dataset.csv")

    df = load_csv(dataset_path)
    splits = split_df(df)

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
    val_pred = model.predict(x_val)
    print("\n[train] Validation confusion matrix:\n", confusion_matrix(y_val, val_pred))
    print("[train] Validation report:\n", classification_report(y_val, val_pred, digits=4))

    #Save artifact
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features_dim": x_train.shape[1],
            "features_version": 1,
        },
        artifact_dir / "logreg.joblib",
    )
    print(f"\n[train] Saved model to {artifact_dir / 'logreg.joblib'}")


