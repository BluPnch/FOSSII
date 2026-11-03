"""Six reproducible validation studies; test is used only for selected models."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from data_preparation import prepare_data
from main import train, predict, metrics
from models import RecurrentRegressor

BASE = dict(window=24, hidden_size=32, learning_rate=0.001, batch_size=256, seed=42)
FACTORS = {
    "window": [8, 24, 48],
    "hidden_size": [16, 32, 64],
    "learning_rate": [0.0003, 0.001, 0.003],
    "batch_size": [128, 256, 512],
    "seed": [7, 42, 123],
}
LABELS = {"window": "Window (15-minute steps)", "hidden_size": "Hidden units",
          "learning_rate": "Learning rate", "batch_size": "Batch size", "seed": "Seed"}


def configurations():
    configs = {"base": BASE.copy()}
    for factor, values in FACTORS.items():
        for value in values:
            if value != BASE[factor]:
                configs[f"{factor}_{value}"] = {**BASE, factor: value}
    return configs


def prepared(config):
    datasets, frame, info = prepare_data(ROOT / "data/Steel_industry_data.csv", config["window"])
    # Identical target dates for every window-length experiment.
    datasets["train"].start = 48
    info["splits"]["train"]["samples"] = len(datasets["train"])
    info["splits"]["train"]["first_target"] = str(frame.date.iloc[48])
    return datasets, frame, info


def run_one(name, kind, config, epochs, out):
    folder = out / "runs" / name / kind.lower()
    folder.mkdir(parents=True, exist_ok=True)
    settings = {**config, "epochs": epochs, "min_epochs": min(60, epochs),
                "patience": 25, "log_every": 10, "threads": 1}
    signature = {**settings, "kind": kind,
                 "sha256": hashlib.sha256((ROOT / "data/Steel_industry_data.csv").read_bytes()).hexdigest(),
                 "source_sha256": hashlib.sha256(b"".join(
                     (ROOT / "source" / f).read_bytes()
                     for f in ("models.py", "main.py", "data_preparation.py"))).hexdigest()}
    record_path = folder / "result.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["signature"] == signature:
            return record
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    datasets, frame, info = prepared(config)
    loaders = {key: DataLoader(ds, batch_size=config["batch_size"], shuffle=key == "train")
               for key, ds in datasets.items() if key != "test"}
    with (folder / "training.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        model, history, training = train(kind, loaders, SimpleNamespace(**settings), folder, info)
    ds = datasets["val"]
    actual = frame.Usage_kWh.to_numpy()[ds.start:ds.stop]
    values = predict(model, loaders["val"]) * info["std"] + info["mean"]
    pd.DataFrame({"date": frame.date.iloc[ds.start:ds.stop].to_numpy(),
                  "actual": actual, "predicted": values}).to_csv(folder / "validation_predictions.csv", index=False)
    record = {"name": name, "kind": kind, "config": config, "signature": signature,
              "training": training, "validation": metrics(actual, values),
              "torch_version": torch.__version__, "data": info}
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def plots_and_tables(records, out):
    rows = [{"run": r["name"], "model": r["kind"], **r["config"],
             **r["validation"], **r["training"],
             "optimizer_steps": r["training"]["epochs_run"] * math.ceil(
                 r["data"]["splits"]["train"]["samples"] / r["config"]["batch_size"])} for r in records]
    table = pd.DataFrame(rows).sort_values(["model", "run"])
    table.to_csv(out / "validation_summary.csv", index=False)
    overview = table.pivot(index="run", columns="model", values="RMSE")[["RNN", "GRU", "LSTM"]]
    fig, ax = plt.subplots(figsize=(8, 7))
    mesh = ax.imshow(overview.to_numpy(), cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(overview.columns)), overview.columns)
    ax.set_yticks(range(len(overview)), overview.index)
    for row in range(len(overview)):
        for col in range(len(overview.columns)):
            color = "black" if overview.iloc[row, col] < overview.to_numpy().mean() else "white"
            ax.text(col, row, f"{overview.iloc[row, col]:.3f}", ha="center", va="center", color=color)
    ax.set_title("Validation RMSE (kWh): all 33 runs")
    fig.colorbar(mesh, ax=ax, label="RMSE (kWh)")
    fig.tight_layout()
    fig.savefig(out / "validation_overview.png", dpi=150)
    plt.close(fig)
    configs = configurations()
    study_tables = {}
    for factor, values in FACTORS.items():
        names = ["base" if v == BASE[factor] else f"{factor}_{v}" for v in values]
        selected = table[table.run.isin(names)].copy()
        selected.to_csv(out / f"{factor}.csv", index=False)
        study_tables[factor] = selected
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, metric in zip(axes, ("MSE", "RMSE", "R2")):
            for kind in ("RNN", "GRU", "LSTM"):
                group = selected[selected.model == kind].sort_values(factor)
                ax.plot(group[factor].astype(str), group[metric], "o-", label=kind)
            ax.set(xlabel=LABELS[factor], ylabel=f"Validation {metric}")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{factor}.png", dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, kind in zip(axes, ("RNN", "GRU", "LSTM")):
            for value, name in zip(values, names):
                h = pd.read_csv(out / "runs" / name / kind.lower() / f"{kind.lower()}_history.csv")
                line, = ax.plot(h.epoch, h.val_mse_scaled, label=f"{value}: val")
                ax.plot(h.epoch, h.train_mse_scaled, linestyle="--", color=line.get_color(),
                        alpha=0.6, label=f"{value}: train")
            ax.set(title=kind, xlabel="Epoch", ylabel="MSE (standardized target)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"{factor}_loss.png", dpi=140)
        plt.close(fig)
    # Nested budgets use the same optimization trajectory, selected by validation.
    budgets = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, kind in zip(axes, ("RNN", "GRU", "LSTM")):
        h = pd.read_csv(out / "runs/base" / kind.lower() / f"{kind.lower()}_history.csv")
        base_record = next(r for r in records if r["kind"] == kind and r["name"] == "base")
        val_variance = base_record["validation"]["MSE"] / (1 - base_record["validation"]["R2"])
        ax.plot(h.epoch, h.train_mse_scaled, label="Train")
        ax.plot(h.epoch, h.val_mse_scaled, label="Validation")
        for budget in (30, 60, 100):
            prefix = h[h.epoch <= budget]
            best = prefix.loc[prefix.val_mse_scaled.idxmin()]
            budgets.append({"model": kind, "budget": budget, "epochs_available": len(prefix),
                            "best_epoch": int(best.epoch),
                            "validation_MSE": best.val_mse_scaled * records[0]["data"]["std"] ** 2,
                            "validation_RMSE": np.sqrt(best.val_mse_scaled) * records[0]["data"]["std"],
                            "validation_R2": 1 - best.val_mse_scaled * records[0]["data"]["std"] ** 2 / val_variance})
            ax.axvline(budget, color="gray", alpha=0.3)
        ax.set(title=kind, xlabel="Epoch", ylabel="MSE (standardized target)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "epochs.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(budgets).to_csv(out / "epochs.csv", index=False)
    return table, study_tables, budgets


def finalize(records, out):
    table, studies, budgets = plots_and_tables(records, out)
    selected = [min((r for r in records if r["kind"] == k),
                    key=lambda r: r["validation"]["MSE"]) for k in ("RNN", "GRU", "LSTM")]
    final = ROOT / "results"
    final.mkdir(exist_ok=True)
    predictions, scores, selections = None, {}, {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, record in zip(axes, selected):
        kind, config = record["kind"], record["config"]
        datasets, frame, info = prepared(config)
        folder = out / "runs" / record["name"] / kind.lower()
        checkpoint = torch.load(folder / f"{kind.lower()}.pt", weights_only=True)
        model = RecurrentRegressor(kind, hidden_size=config["hidden_size"])
        model.load_state_dict(checkpoint["state_dict"])
        ds = datasets["test"]
        if predictions is None:
            predictions = pd.DataFrame({"date": frame.date.iloc[ds.start:].to_numpy(),
                                        "actual": frame.Usage_kWh.iloc[ds.start:].to_numpy(),
                                        "Persistence": frame.Usage_kWh.iloc[ds.start-1:-1].to_numpy()})
            scores["Persistence"] = metrics(predictions.actual, predictions.Persistence)
        values = predict(model, DataLoader(ds, batch_size=config["batch_size"])) * info["std"] + info["mean"]
        predictions[kind] = values
        scores[kind] = metrics(predictions.actual, values)
        selections[kind] = record
        for suffix in (".pt", "_history.csv"):
            shutil.copy2(folder / f"{kind.lower()}{suffix}", final / f"{kind.lower()}{suffix}")
        h = pd.read_csv(folder / f"{kind.lower()}_history.csv")
        ax.plot(h.epoch, h.train_mse_scaled, label="Train")
        ax.plot(h.epoch, h.val_mse_scaled, label="Validation")
        ax.axvline(record["training"]["best_epoch"], color="gray", linestyle="--", label="Best epoch")
        ax.set(title=f"{kind}: {record['name']}", xlabel="Epoch", ylabel="MSE (standardized target)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(final / "loss.png", dpi=150)
    plt.close(fig)
    predictions.to_csv(final / "predictions.csv", index=False)
    pd.DataFrame(scores).T.rename_axis("model").to_csv(final / "metrics.csv")
    metadata = {"selection": "Minimum validation MSE across one-factor experiments; test excluded from selection",
                "selected": selections, "metrics": scores, "study_directory": str(out),
                "training": {k: r["training"] for k, r in selections.items()}}
    (final / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(13, 4))
    for column in ("actual", "RNN", "GRU", "LSTM"):
        ax.plot(predictions.date[:384], predictions[column][:384], label=column, alpha=0.8)
    ax.set(xlabel="Date", ylabel="Usage (kWh)", title="Selected models: rolling one-step test forecast")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(final / "predictions.png", dpi=150)
    plt.close(fig)
    write_report(table, studies, budgets, selections, scores, out)
    from update_readme import main as update_overview
    update_overview()


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(table, studies, budgets, selections, scores, out):
    lines = ["# Исследования RNN, GRU и LSTM", "",
             "Выполнены шесть исследований: бюджет эпох, длина окна, размер скрытого состояния, "
             "скорость обучения, размер батча и случайная инициализация. "
             "Всего 33 отдельных обучения: 11 конфигураций × 3 архитектуры.", "",
             "Каждый запуск: до 100 эпох, минимум 60, patience 25. На каждой эпохе считаются "
             "train MSE и validation MSE; лучшие веса выбираются по validation MSE. "
             "Все исследования используют полные данные, без сокращённой выборки.", "",
             "Базовая конфигурация: окно 24, hidden size 32, learning rate 0.001, batch size 256, seed 42. "
             "В каждом исследовании меняется один параметр. Сочетания лучших значений разных "
             "параметров отдельно не проверялись, поэтому выводы не описывают глобальный оптимум.", "",
             "Во всех запусках первые 48 строк служат контекстом: обучающие цели начинаются "
             "01.01.2018 12:00 (24 480 примеров). Даты валидации и теста прежние, по 5 256 точек. "
             "Нормализация только по обучающей части. В исследованиях тест не вычисляется: "
             "он используется после выбора одной конфигурации для каждой архитектуры.", "",
             "![Все 33 запуска](validation_overview.png)", "",
             "## 1. Бюджет эпох", "",
             "Сравниваются лучшие validation MSE в префиксах одной базовой траектории "
             "до 30, 60 и 100 эпох. Это не три независимых обучения. Если сработала ранняя "
             "остановка, фактическое число доступных эпох указано в таблице.", "",
             markdown_table(pd.DataFrame(budgets)), "", "![Эпохи](epochs.png)", ""]
    titles = {"window": "Длина входной последовательности", "hidden_size": "Размер скрытого состояния",
              "learning_rate": "Скорость обучения", "batch_size": "Размер батча", "seed": "Случайная инициализация"}
    for i, (factor, frame) in enumerate(studies.items(), start=2):
        lines += [f"## {i}. {titles[factor]}", "",
                  markdown_table(frame[["model", factor, "MSE", "RMSE", "R2", "best_epoch", "epochs_run"]].sort_values(["model", factor])),
                  "", f"![{titles[factor]}]({factor}.png)", "",
                  f"![Обучение и валидация: {titles[factor]}]({factor}_loss.png)", ""]
        for kind in ("RNN", "GRU", "LSTM"):
            part = frame[frame.model == kind]
            best = part.loc[part.MSE.idxmin()]
            base = part[part[factor] == BASE[factor]].iloc[0]
            change = (base.MSE - best.MSE) / base.MSE * 100
            lines += [f"- {kind}: лучшее значение {factor} = {best[factor]}, validation MSE = "
                      f"{best.MSE:.4f}; снижение относительно базовой конфигурации {change:.2f} %."]
        if factor == "seed":
            lines += ["", "Seed показывает чувствительность к инициализации и порядку батчей. "
                      "Выбор лучшего seed по валидации не заменяет оценку устойчивости:", "",
                      markdown_table(frame.groupby("model")["RMSE"].agg(["mean", "std", "min", "max"]).reset_index()), "",
                      "Различия между гиперпараметрами, сопоставимые с разбросом по seed, "
                      "не следует считать устойчивым преимуществом по одному запуску. "
                      "Здесь каждый вариант гиперпараметров проверен только при seed 42; "
                      "повторный подбор на нескольких seed не выполнялся."]
        if factor == "batch_size":
            lines += ["", "При фиксированном числе эпох меньший батч даёт больше обновлений весов. "
                      "Это сравнение бюджетов по эпохам, а не по одинаковому числу шагов Adam. "
                      "Число обновлений записано в `optimizer_steps` сводной таблицы. "
                      "Время обучения при параллельных запусках зависит от конкуренции за CPU, "
                      "поэтому оно не используется для вывода о скорости архитектур."]
        lines += [""]
    lines += ["## Итоговый тест после выбора по валидации", "",
              markdown_table(pd.DataFrame(scores).T.rename_axis("model").reset_index()), ""]
    for kind, r in selections.items():
        lines += [f"- {kind}: `{r['name']}`, параметры `{r['config']}`, лучшая эпоха {r['training']['best_epoch']}."]
    winner = min((k for k in scores if k != "Persistence"), key=lambda k: scores[k]["MSE"])
    lines += ["", f"На тесте лучший результат у {winner}: RMSE {scores[winner]['RMSE']:.4f} kWh, "
              f"R² {scores[winner]['R2']:.6f}. Результат относится к данному временному разбиению.", "",
              "Перебор нескольких конфигураций может переобучить выбор на валидацию; "
              "для более строгой оценки нужны дополнительные временные разбиения. "
              "Тестовый период уже использовался в первоначальной лабораторной, поэтому "
              "это повторная оценка на прежнем отрезке, а не совершенно новый независимый тест.", "",
              "## Воспроизведение", "", "```powershell", "python lab3/investigations/run.py --workers 3", "```", "",
              "Готовые запуски с совпадающими настройками, хешами кода и данных переиспользуются. "
              "`--force` повторяет обучение. Каждый запуск хранит веса, полный CSV истории, "
              "валидационные прогнозы и лог в `runs/`. Сводная таблица — `validation_summary.csv`."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    out = ROOT / "results/investigations"
    out.mkdir(parents=True, exist_ok=True)
    archive = ROOT / "results/baseline_30"
    if not archive.exists():
        archive.mkdir()
        for path in (ROOT / "results").iterdir():
            if path.is_file():
                shutil.copy2(path, archive / path.name)
    jobs = [(name, kind, config) for name, config in configurations().items()
            for kind in ("RNN", "GRU", "LSTM")]
    if args.force:
        for name, kind, _ in jobs:
            record = out / "runs" / name / kind.lower() / "result.json"
            record.unlink(missing_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, name, kind, config, 100, out): (name, kind)
                   for name, kind, config in jobs}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"[{len(records)}/{len(jobs)}] {record['name']} {record['kind']}: "
                  f"val RMSE={record['validation']['RMSE']:.4f}, "
                  f"epochs={record['training']['epochs_run']}, best={record['training']['best_epoch']}", flush=True)
    torch.set_num_threads(1)
    finalize(records, out)
    print(f"Completed: {out / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
