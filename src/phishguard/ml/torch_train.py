import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from phishguard.data.loader import load_csv, split_df_grouped
from phishguard.ml.features import extract_features
from phishguard.ml.torch_mlp import MLPBinary

def eval_metrics(logits: torch.Tensor,
                 y_true: torch.Tensor,
                 t: float = 0.2):
    p = torch.sigmoid(logits)
    pred = (p >= t).to(torch.int64)
    y = y_true.to(torch.int64)

    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())

    recall1 = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    return tn, fp, fn, tp, recall1, fpr

def _featureize(urls) -> np.ndarray:
    return np.vstack([extract_features(u) for u in urls]).astype(np.float32)

def train_step(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        Xb: torch.Tensor,
        yb: torch.Tensor,
        ) -> float:
    model.train()
    optimizer.zero_grad()

    logits = model(Xb)
    loss = criterion(logits, yb)

    loss.backward()
    optimizer.step()

    return float(loss.item())

def train_torch(
       dataset_path: Path = Path("data/processed/urls_clean.csv"),
       epochs: int = 5,
        lr: float = 1e-3,
        seed: int = 42,
) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

        df = load_csv(dataset_path)
        splits = split_df_grouped(df)

        X_train = _featureize(splits.train_frame["url"])
        y_train = splits.train_frame["label"].to_numpy(dtype=np.float32).reshape(-1, 1)

        X_val = _featureize(splits.val_frame["url"])
        y_val = splits.val_frame["label"].to_numpy(dtype=np.float32).reshape(-1, 1)

        Xtr = torch.tensor(X_train, dtype=torch.float32)
        ytr = torch.tensor(y_train, dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(Xtr, ytr),
            batch_size=1024,
            shuffle=True,
        )
        Xva = torch.tensor(X_val, dtype=torch.float32)
        yva = torch.tensor(y_val, dtype=torch.float32)

        model = MLPBinary(in_dim=Xtr.shape[1])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        print("[torch] X_train:", Xtr.shape, "y_train:", ytr.shape)

        #Early Stopping
        best_fn = 10**18
        best_state = None
        patience = 3
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            #Train over batches
            total_loss = 0
            for Xb, yb in train_loader:
                total_loss += train_step(model, optimizer, criterion, Xb, yb)

            train_loss = total_loss / len(train_loader)

            model.eval()
            with torch.no_grad():
                val_logits = model(Xva)
                val_loss = criterion(val_logits, yva).item()
                tn, fp, fn, tp, recall1, fpr = eval_metrics(val_logits, yva, t=0.2)



            if fn < best_fn:
                best_fn = fn
                bad_epochs = 0
                #Save best model snapshot
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"[torch] Early stopping at epoch {epoch}")
                    break

            print(f"[torch] epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
                  f"FN={fn} FP={fp} recall1={recall1:.4f} FPR={fpr:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"[torch] Restored best model (best FN = {best_fn})")

        model.eval()

        artifact_dir = Path("artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        chosen_t = 0.15  # assistant mode baseline
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_dim": Xtr.shape[1],
                "threshold": chosen_t,
                "features_version": 1,
            },
            artifact_dir / "mlp.pt",
        )
        print(f"[torch] Saved artifacts to {artifact_dir / 'mlp.pt'} (threshold={chosen_t})")

