"""Train and compare RNN, GRU and LSTM on a local UCI dataset."""

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_preparation import prepare_data
from models import RecurrentRegressor

ROOT = Path(__file__).resolve().parents[1]


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mse = np.mean((actual - predicted) ** 2)
    return {"MSE": float(mse), "RMSE": float(np.sqrt(mse)),
            "R2": float(1 - np.sum((actual - predicted) ** 2)
                        / np.sum((actual - actual.mean()) ** 2))}


def predict(model, loader):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(x).numpy() for x, _ in loader])


def train(kind, loaders, args, out, info):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = RecurrentRegressor(kind, hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    best_loss, stale, best_state = float("inf"), 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loaders["train"]:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(y)
        val_prediction = predict(model, loaders["val"])
        val_y = loaders["val"].dataset.targets[
            loaders["val"].dataset.start:loaders["val"].dataset.stop].numpy()
        val_loss = float(np.mean((val_prediction - val_y) ** 2))
        train_loss = total / len(loaders["train"].dataset)
        history.append({"epoch": epoch, "train_mse_scaled": train_loss,
                        "val_mse_scaled": val_loss})
        if epoch == 1 or epoch % getattr(args, "log_every", 1) == 0:
            print(f"{kind} {epoch:03d}: train={train_loss:.5f}, val={val_loss:.5f}", flush=True)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            stale += 1
        if epoch >= getattr(args, "min_epochs", 0) and stale >= args.patience:
            break
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    torch.save({"state_dict": best_state, "kind": kind, "hidden_size": args.hidden_size,
                "input_size": 5, "preprocessing": info}, out / f"{kind.lower()}.pt")
    pd.DataFrame(history).to_csv(out / f"{kind.lower()}_history.csv", index=False)
    return model, history, {"best_epoch": best_epoch, "epochs_run": len(history),
                            "best_val_mse_scaled": best_loss,
                            "training_seconds": elapsed,
                            "parameters": sum(p.numel() for p in model.parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/Steel_industry_data.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.epochs, args.hidden_size, args.batch_size, args.patience, args.threads) < 1:
        parser.error("Epochs, hidden size, batch size, patience and threads must be positive")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive")
    if args.min_epochs < 0 or args.min_epochs > args.epochs:
        parser.error("min-epochs must be between zero and epochs")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    datasets, frame, info = prepare_data(args.data, args.window)
    args.output.mkdir(parents=True, exist_ok=True)
    loaders = {key: DataLoader(ds, batch_size=args.batch_size, shuffle=key == "train")
               for key, ds in datasets.items()}
    test_start = datasets["test"].start
    actual = frame.Usage_kWh.to_numpy()[test_start:]
    predictions = pd.DataFrame({"date": frame.date.iloc[test_start:].to_numpy(),
                                "actual": actual})
    persistence = frame.Usage_kWh.to_numpy()[test_start - 1:-1]
    predictions["Persistence"] = persistence
    scores = {"Persistence": metrics(actual, persistence)}
    histories, training = {}, {}
    for kind in ("RNN", "GRU", "LSTM"):
        model, histories[kind], training[kind] = train(kind, loaders, args, args.output, info)
        values = predict(model, loaders["test"]) * info["std"] + info["mean"]
        predictions[kind] = values
        scores[kind] = metrics(actual, values)
    table = pd.DataFrame(scores).T.rename_axis("model")
    table.to_csv(args.output / "metrics.csv")
    predictions.to_csv(args.output / "predictions.csv", index=False)
    metadata = {"config": {k: str(v) if isinstance(v, Path) else v
                            for k, v in vars(args).items()},
                "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "data": info, "training": training, "metrics": scores,
                "versions": {"torch": torch.__version__, "numpy": np.__version__,
                             "pandas": pd.__version__}}
    (args.output / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (kind, history) in zip(axes, histories.items()):
        h = pd.DataFrame(history)
        ax.plot(h.epoch, h.train_mse_scaled, label="Train")
        ax.plot(h.epoch, h.val_mse_scaled, label="Validation")
        ax.set(title=kind, xlabel="Epoch", ylabel="MSE (standardized target)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "loss.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 4))
    for column in ("actual", "RNN", "GRU", "LSTM"):
        ax.plot(predictions.date[:384], predictions[column][:384], label=column, alpha=0.8)
    ax.set(xlabel="Date", ylabel="Usage (kWh)", title="First four test days: rolling one-step forecast")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.output / "predictions.png", dpi=150)
    plt.close(fig)
    print("\nTest metrics (original kWh scale):\n" + table.to_string(float_format="%.6f"))


if __name__ == "__main__":
    main()
