"""Продолжение первой и оптимальной NumPy CNN ровно до 60 эпох."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from common import (SequentialCNN, load_data, evaluate, confusion_matrix,
                    classification_metrics, roc_one_vs_rest, save)
from data_preparation import FILES, download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/investigations/compare_60_epochs"
NAMES = ("conv1", "conv2", "fc1", "fc2")
SPECS = {
    "first": {"label": "Первая", "filters": [8, 16], "pools": [True, True],
              "source": ROOT / "results/archive_baseline/weights.npz", "color": "#2a78d6"},
    "optimal": {"label": "Оптимальная", "filters": [16, 32],
                "pools": [(2, 2, "avg"), (2, 2, "avg")],
                "source": ROOT / "results/weights.npz", "color": "#eb6834"},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(key, threads, from_scratch):
    spec = SPECS[key]
    return {"model": key, "seed": 42, "learning_rate": 0.01, "lr_decay": 0.95,
            "train_part": 0.7, "threads": threads, "from_scratch": from_scratch,
            "source_weights_sha256": None if from_scratch else digest(spec["source"]),
            "source_code_sha256": hashlib.sha256(b"".join(
                (ROOT / "source" / name).read_bytes()
                for name in ("layers.py", "cnn.py", "data_preparation.py"))
                + (ROOT / "investigations/common.py").read_bytes()).hexdigest(),
            "dataset_sha256": {p.name: digest(p) for p in sorted((ROOT / "data").glob("*-ubyte.gz"))}}


def save_checkpoint(path, model, losses, rng, settings, prefix_epochs):
    state = rng.get_state()
    arrays = {"losses": np.asarray(losses, dtype=np.float64),
              "rng_keys": state[1], "rng_pos": state[2], "rng_has_gauss": state[3],
              "rng_cached_gaussian": state[4], "signature": json.dumps(settings, sort_keys=True),
              "prefix_epochs": prefix_epochs}
    for layer, name in zip(model.trainable(), NAMES):
        arrays[name + "_W"] = layer.W
        arrays[name + "_b"] = layer.b
    temporary = path.with_suffix(".next.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def assign_weights(model, arrays):
    for layer, name in zip(model.trainable(), NAMES):
        weight, bias = arrays[name + "_W"], arrays[name + "_b"]
        if weight.shape != layer.W.shape or bias.shape != layer.b.shape:
            raise ValueError(f"Checkpoint shape mismatch: {name}")
        layer.W = weight.copy()
        layer.b = bias.copy()


def start_or_resume(key, folder, settings, train_size):
    np.random.seed(42)
    spec = SPECS[key]
    model = SequentialCNN(spec["filters"], spec["pools"])
    # main.py initializes the model with this global RNG, then permutes each epoch.
    # Data splitting and TSNE(random_state=42) use their own random generators.
    rng = np.random.RandomState()
    rng.set_state(np.random.get_state())
    checkpoint = folder / "weights.npz"
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as arrays:
            if json.loads(str(arrays["signature"])) != settings:
                raise ValueError(f"Settings changed; use a separate --output directory: {folder}")
            assign_weights(model, arrays)
            rng.set_state(("MT19937", arrays["rng_keys"], int(arrays["rng_pos"]),
                           int(arrays["rng_has_gauss"]), float(arrays["rng_cached_gaussian"])))
            return model, arrays["losses"].tolist(), rng, int(arrays["prefix_epochs"])
    losses = []
    if not settings["from_scratch"]:
        with np.load(spec["source"], allow_pickle=False) as arrays:
            assign_weights(model, arrays)
            losses = arrays["losses"].tolist()
        # Recover the permutation stream after the archived completed epochs.
        for _ in losses:
            rng.permutation(train_size)
    save_checkpoint(checkpoint, model, losses, rng, settings, len(losses))
    return model, losses, rng, len(losses)


def write_history(folder, losses, prefix_epochs, timings):
    lines = ["epoch,train_loss,learning_rate,epoch_seconds,origin"]
    for index, loss in enumerate(losses):
        epoch = index + 1
        origin = "archive" if epoch <= prefix_epochs else "continued"
        seconds = timings.get(str(epoch), "")
        lines.append(f"{epoch},{loss:.17g},{0.01 * 0.95 ** index:.17g},{seconds},{origin}")
    (folder / "history.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_model(key, output, epochs, threads, from_scratch):
    folder = output / key
    folder.mkdir(parents=True, exist_ok=True)
    settings = signature(key, threads, from_scratch)
    result_path = folder / "result.json"
    if result_path.exists():
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if record["signature"] == settings and record["epochs"] == epochs:
            return record
    with threadpool_limits(limits=threads, user_api="blas"), \
            (folder / "training.log").open("a", encoding="utf-8", buffering=1) as log, \
            redirect_stdout(log):
        x_train, y_train, x_test, y_test = load_data(seed=42)
        model, losses, rng, prefix = start_or_resume(key, folder, settings, len(x_train))
        if len(losses) > epochs:
            raise ValueError("Requested epochs precede checkpoint; choose a new output and --from-scratch")
        timings_path = folder / "timings.json"
        timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
        print(f"{key}: resume after {len(losses)} epochs; train={len(x_train)}, test={len(x_test)}", flush=True)
        for epoch in range(len(losses) + 1, epochs + 1):
            learning_rate = 0.01 * 0.95 ** (epoch - 1)
            started = time.perf_counter()
            total_loss = 0.0
            for index in rng.permutation(len(x_train)):
                total_loss += model.train_one(x_train[index], y_train[index], learning_rate)
            mean_loss = float(total_loss / len(x_train))
            if not np.isfinite(mean_loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            losses.append(mean_loss)
            timings[str(epoch)] = time.perf_counter() - started
            save_checkpoint(folder / "weights.npz", model, losses, rng, settings, prefix)
            timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
            write_history(folder, losses, prefix, timings)
            print(f"{key}: epoch {epoch}/{epochs}, loss={mean_loss:.10f}, "
                  f"lr={learning_rate:.8f}, seconds={timings[str(epoch)]:.1f}", flush=True)
        # Evaluation does not control training; every requested epoch is executed.
        probabilities, ms_per_image = evaluate(model, x_test)
        predictions = probabilities.argmax(axis=1)
        matrix = confusion_matrix(y_test, predictions)
        scores = classification_metrics(matrix)
        curves = roc_one_vs_rest(y_test, probabilities)
        target_hits = np.flatnonzero(np.asarray(losses) <= 0.0001)
        record = {"key": key, "label": SPECS[key]["label"], "signature": settings,
                  "architecture": model.describe(), "parameters": int(model.num_parameters()),
                  "epochs": len(losses), "prefix_epochs": prefix,
                  "new_epochs": len(losses) - prefix, "train_size": len(x_train), "test_size": len(x_test),
                  "history": {"train_loss": losses}, "final_loss": losses[-1],
                  "first_target_epoch": int(target_hits[0] + 1) if len(target_hits) else None,
                  "new_training_seconds": sum(float(timings.get(str(e), 0)) for e in range(prefix+1,epochs+1)),
                  "accuracy": float(scores["accuracy"]),
                  "macro_precision": float(scores["macro_precision"]),
                  "macro_recall": float(scores["macro_recall"]),
                  "macro_f1": float(scores["macro_f1"]),
                  "macro_auc": float(np.mean([area for _, _, area in curves])),
                  "test_cross_entropy": float(-np.log(probabilities[np.arange(len(y_test)), y_test]+1e-12).mean()),
                  "errors": int(np.sum(predictions != y_test)), "ms_per_image": ms_per_image,
                  "confusion_matrix": matrix.tolist(), "numpy_version": np.__version__}
        result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(folder / "test_predictions.npz", actual=y_test, probabilities=probabilities)
        return record


def draw_curves(records, output, epochs, individual_only=False):
    def draw(selected, filename, title, start=1, logarithmic=True):
        fig, ax = plt.subplots(figsize=(11, 6))
        for record in selected:
            x = np.arange(1, record["epochs"] + 1)
            mask = x >= start
            ax.plot(x[mask], np.asarray(record["history"]["train_loss"])[mask],
                    label=f"{record['label']} ({record['parameters']:,} параметров)".replace(",", " "),
                    color=SPECS[record["key"]]["color"], linewidth=2, marker="o", markersize=3)
            if record["prefix_epochs"] >= start:
                ax.axvline(record["prefix_epochs"], color=SPECS[record["key"]]["color"],
                           linestyle=":", alpha=0.6)
        ax.axhline(0.0001, color="#898781", linestyle="--", label="Прежний порог 0.0001 (без остановки)")
        if logarithmic:
            ax.set_yscale("log")
        ax.set(xlabel="Эпоха", ylabel="Средняя train cross-entropy", title=title, xlim=(start, epochs))
        ax.legend()
        ax.set_xticks(sorted(set([start] + [e for e in range(10, epochs+1, 10) if e >= start] + [epochs])))
        save(fig, output / filename)
    if not individual_only:
        draw(records, "loss.png", f"Первая и оптимальная CNN: по {epochs} эпох")
        draw(records, "loss_linear.png", f"Train loss: по {epochs} эпох (линейная шкала)", logarithmic=False)
        draw(records, "loss_late.png", "Продолжение обучения после 20-й эпохи", start=min(20,epochs))
    for record in records:
        draw([record], f"{record['key']}_loss.png", f"{record['label']} CNN: {epochs} эпох")


def write_report(records, output, epochs):
    first, optimal = sorted(records, key=lambda r: r["key"])
    lines = [f"# Первая и оптимальная CNN: ровно по {epochs} эпох", "",
             "Сравниваются исходная модель (8/16 фильтров, max-пулинг 2×2) и "
             "оптимальная (16/32 фильтра, average-пулинг 2×2). Обе реализованы "
             "на NumPy с ручным обратным проходом и SGD по одному изображению.", "",
             f"Каждая модель имеет {epochs} реальных значений loss: без экстраполяции "
             "и без остановки при достижении 0.0001. Потеря — средняя cross-entropy "
             "на 49 000 обучающих изображениях, усреднённая во время обновлений весов.", "",
             "## Условия", "",
             "Разбиение MNIST прежнее: все 70 000 изображений перемешаны с seed 42, "
             "49 000 используются для обучения, 21 000 — для теста. "
             "Learning rate на эпохе e: `0.01 * 0.95**(e-1)`. "
             "Обновление после каждого изображения, без батчевого усреднения.", "",
             "| Модель | Параметров | Эпох из архива | Дообучено эпох | Всего |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for r in (first, optimal):
        lines.append(f"| {r['label']} | {r['parameters']} | {r['prefix_epochs']} | {r['new_epochs']} | {r['epochs']} |")
    continued = any(r["prefix_epochs"] for r in records)
    provenance = ("Архивные веса и начала историй сохранены без изменений. Для продолжения "
              "восстановлен поток перестановок: seed 42, инициализация соответствующей "
              "архитектуры, затем пропуск перестановок завершённых архивных эпох. "
              "Это соответствует схеме исходного `source/main.py`. Состояние SGD "
              "полностью задаётся весами, так как momentum не используется. "
              "Все новые checkpoint сохраняют также состояние генератора случайных чисел."
              if continued else "Обе модели обучены с первой эпохи с новой инициализацией "
              "seed 42, без загрузки архивных весов и историй.")
    limitation = ("Исходный архив не содержал состояния RNG; восстановление опирается "
              "на документированную инициализацию и порядок вызовов исходной программы. "
              "Побитовое совпадение с гипотетическим непрерывным запуском в старом "
              "окружении не утверждается. Для нового полного запуска доступен `--from-scratch`."
              if continued else "Checkpoint каждой эпохи содержит веса, историю и состояние RNG.")
    lines += ["", provenance, "", limitation, "",
              "## Графики", "", "Пунктир по горизонтали — прежняя цель 0.0001, "
              "теперь только ориентир. " +
              ("Вертикальные цветные линии отмечают конец архивного участка "
               "и начало нового обучения. " if continued else "Показаны полные истории с первой эпохи. ") +
              "Логарифмическая шкала помогает различить малые ошибки после достижения цели.", "",
              "![Общий loss](loss.png)", "", "![Первая модель](first_loss.png)", "",
              "![Оптимальная модель](optimal_loss.png)", "", "![Поздние эпохи](loss_late.png)", "",
              "![Линейная шкала](loss_linear.png)", "", "## Итоги", "",
              "| Показатель | Первая | Оптимальная |", "| --- | ---: | ---: |",
              f"| Train loss на последней эпохе | {first['final_loss']:.10f} | {optimal['final_loss']:.10f} |",
              f"| Первая эпоха с loss ≤ 0.0001 | {first['first_target_epoch']} | {optimal['first_target_epoch']} |",
              f"| Test cross-entropy на последней эпохе | {first['test_cross_entropy']:.6f} | {optimal['test_cross_entropy']:.6f} |",
              f"| Test accuracy на последней эпохе | {first['accuracy']:.6f} | {optimal['accuracy']:.6f} |",
              f"| Ошибок на 21 000 тестовых изображениях | {first['errors']} | {optimal['errors']} |", ""]
    for r in (first, optimal):
        before = r["history"]["train_loss"][r["prefix_epochs"]-1] if r["prefix_epochs"] else r["history"]["train_loss"][0]
        change = 100*(1-r["final_loss"]/before)
        basis = "конца архивного участка" if r["prefix_epochs"] else "первой эпохи"
        lines.append(f"- {r['label']}: train loss уменьшился относительно {basis} "
                     f"на {change:.2f} %. Итог: {r['final_loss']:.10f} после {epochs} эпох.")
    lower_loss = min(records, key=lambda r: r["final_loss"])
    best_accuracy = max(records, key=lambda r: r["accuracy"])
    lines += ["", f"При одинаковом бюджете минимальная обучающая ошибка у модели "
              f"«{lower_loss['label']}» ({lower_loss['final_loss']:.10f}). "
              f"Максимальная итоговая test accuracy у модели «{best_accuracy['label']}» "
              f"({best_accuracy['accuracy']:.6f}). Название «оптимальная» обозначает "
              "архитектуру, выбранную в предыдущих исследованиях, и не означает "
              "обязательного превосходства по любой метрике при любом бюджете."]
    lines += ["", "Сравнение при одинаковом числе эпох отличается от прежнего сравнения "
              "при одинаковом пороге ошибки. Меньший train loss не гарантирует лучшую "
              "точность на новых изображениях: тест посчитан отдельно после завершения "
              "всех эпох и не управляет остановкой. Время дообучения измерялось при "
              "параллельных запусках и не сопоставляется напрямую со старыми временами.", "",
              "## Файлы результатов", "",
              "Общая сводка сохранена в `results.json`. Для каждой модели есть история потерь "
              "и времени по эпохам, а также отдельный `result.json`."]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if min(args.epochs, args.workers, args.threads) < 1:
        parser.error("epochs, workers and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.replot:
        records = json.loads((args.output / "results.json").read_text(encoding="utf-8"))["results"]
        epochs = records[0]["epochs"]
    else:
        for filename in FILES.values():
            download(filename)
        with ProcessPoolExecutor(max_workers=min(args.workers, 2)) as pool:
            futures = {pool.submit(train_model, key, args.output, args.epochs, args.threads, args.from_scratch): key
                       for key in SPECS}
            records = []
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                draw_curves([record], args.output, args.epochs, individual_only=True)
                print(f"Completed {record['label']}: {record['epochs']} epochs, "
                      f"loss={record['final_loss']:.10f}", flush=True)
        records.sort(key=lambda r: r["key"])
        (args.output / "results.json").write_text(
            json.dumps({"results": records}, ensure_ascii=False, indent=2), encoding="utf-8")
        epochs = args.epochs
    draw_curves(records, args.output, epochs)
    write_report(records, args.output, epochs)
    print(f"Saved: {args.output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
