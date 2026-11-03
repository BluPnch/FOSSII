"""Chronological one-step-ahead forecasting without future measurements."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Windows(Dataset):
    def __init__(self, features, targets, start, stop, window):
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)
        self.start = start
        self.stop = stop
        self.window = window

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        t = self.start + index
        return self.features[t - self.window:t], self.targets[t]


def prepare_data(path, window):
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], format="%d/%m/%Y %H:%M")
    frame = frame.sort_values("date").reset_index(drop=True)
    if not frame["date"].diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("Expected a continuous series sampled every 15 minutes")
    usage = frame["Usage_kWh"].to_numpy(dtype=np.float32)
    if not np.isfinite(usage).all():
        raise ValueError("Usage_kWh contains non-finite values")
    n = len(frame)
    train_end, val_end = int(n * 0.7), int(n * 0.85)
    if not 0 < window < train_end:
        raise ValueError("Window must be positive and shorter than training data")
    mean = float(usage[:train_end].mean())
    std = float(usage[:train_end].std())
    if std == 0:
        raise ValueError("Training target has zero variance")
    target = (usage - mean) / std
    time = frame["date"].dt
    day_phase = 2 * np.pi * (time.hour * 60 + time.minute) / 1440
    week_phase = 2 * np.pi * time.dayofweek / 7
    features = np.column_stack([
        target, np.sin(day_phase), np.cos(day_phase),
        np.sin(week_phase), np.cos(week_phase),
    ]).astype(np.float32)
    datasets = {
        "train": Windows(features, target, window, train_end, window),
        "val": Windows(features, target, train_end, val_end, window),
        "test": Windows(features, target, val_end, n, window),
    }
    info = {
        "rows": n, "window": window, "mean": mean, "std": std,
        "features": ["Usage_kWh", "time_sin", "time_cos", "weekday_sin", "weekday_cos"],
        "splits": {
            key: {"samples": len(ds), "first_target": str(frame.date.iloc[ds.start]),
                  "last_target": str(frame.date.iloc[ds.stop - 1])}
            for key, ds in datasets.items()
        },
    }
    return datasets, frame, info
