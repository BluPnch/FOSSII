"""Train and compare RNN, GRU and LSTM on a local UCI dataset."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# BLAS читает лимит потоков при импорте NumPy.
if __name__ == "__main__":
    thread_parser = argparse.ArgumentParser(add_help=False)
    thread_parser.add_argument("--threads", type=int, default=2)
    thread_args, _ = thread_parser.parse_known_args()
    if thread_args.threads > 0:
        for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
            os.environ[variable] = str(thread_args.threads)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_preparation import prepare_data
from models import Adam, RecurrentRegressor

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "numpy"


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    mse = np.mean((actual - predicted) ** 2)
    return {"MSE": float(mse), "RMSE": float(np.sqrt(mse)),
            "R2": float(1 - np.sum((actual - predicted) ** 2)
                        / np.sum((actual - actual.mean()) ** 2))}


def predict(model, dataset, batch_size):
    return np.concatenate([model(x) for x, _ in dataset.batches(batch_size)])


def train(kind, datasets, args, out, info, artifact_name=None):
    train_set, val = datasets["train"], datasets["val"]
    rng = np.random.default_rng(args.seed)
    
    model = RecurrentRegressor(kind, input_size=train_set.features.shape[1], hidden_size=args.hidden_size, seed=args.seed)
    optimizer = Adam(model.weights, args.learning_rate)
    
    val_y = val.targets[val.start:val.stop]
    best_loss, stale, best_state = float("inf"), 0, None
    history = []
    started = time.perf_counter()
    
    for epoch in range(1, args.epochs + 1):
        # Прямой проход, BPTT и шаг Adam для каждого обучающего батча.
        total = 0.0
        for x, y in train_set.batches(args.batch_size, rng):
            loss, gradients = model.loss_and_gradients(x, y)
            optimizer.step(gradients)
            total += loss * len(y)
        # Валидация выбирает лучшую эпоху и управляет ранней остановкой.
        val_prediction = predict(model, val, args.batch_size)
        val_loss = float(np.mean((val_prediction - val_y) ** 2))
        train_loss = total / len(train_set)
        history.append({"epoch": epoch, "train_mse_scaled": train_loss, "val_mse_scaled": val_loss})
        if epoch == 1 or epoch % getattr(args, "log_every", 1) == 0:
            print(f"{kind} {epoch:03d}: train={train_loss:.5f}, val={val_loss:.5f}", flush=True)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = {name: weight.copy() for name, weight in model.weights.items()}
            best_epoch = epoch
        else:
            stale += 1
        if epoch >= getattr(args, "min_epochs", 0) and stale >= args.patience:
            break
    if best_state is None:
        raise FloatingPointError("Training did not produce a finite validation loss")
    model.weights = best_state
    elapsed = time.perf_counter() - started
    name = artifact_name or kind.lower()
    model.save(out / f"{name}.npz", info)
    history_name = "history.csv" if artifact_name else f"{name}_history.csv"
    pd.DataFrame(history).to_csv(out / history_name, index=False)
    return model, history, {"best_epoch": best_epoch, "epochs_run": len(history),
                            "best_val_mse_scaled": best_loss,
                            "training_seconds": elapsed,
                            "parameters": sum(weight.size for weight in model.weights.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/Steel_industry_data.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT)
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
    datasets, frame, info = prepare_data(args.data, args.window)
    args.output.mkdir(parents=True, exist_ok=True)
    test_start = datasets["test"].start
    actual = frame.Usage_kWh.to_numpy()[test_start:]
    predictions = pd.DataFrame({"date": frame.date.iloc[test_start:].to_numpy(), "actual": actual})
    persistence = frame.Usage_kWh.to_numpy()[test_start - 1:-1]
    predictions["Persistence"] = persistence
    scores = {"Persistence": metrics(actual, persistence)}
    histories, training = {}, {}

    for kind in ("RNN", "GRU", "LSTM"):
        model, histories[kind], training[kind] = train(kind, datasets, args, args.output, info)
        values = predict(model, datasets["test"], args.batch_size) * info["std"] + info["mean"]
        predictions[kind] = values
        scores[kind] = metrics(actual, values)

    table = pd.DataFrame(scores).T.rename_axis("model")
    table.to_csv(args.output / "metrics.csv")
    predictions.to_csv(args.output / "predictions.csv", index=False)
    metadata = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "data": info, "training": training, "metrics": scores,
                "versions": {"numpy": np.__version__, "pandas": pd.__version__}}
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
