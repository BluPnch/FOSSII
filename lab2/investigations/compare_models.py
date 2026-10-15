"""
Итоговое сравнение: первая модель против модели с оптимальными параметрами.

Сравниваются три набора сохранённых весов:

    первая модель, 4 эпохи        results/archive_4_epochs/weights.npz
    первая модель, до цели        results/archive_baseline/weights.npz
    оптимальная модель, до цели   results/weights.npz

Архитектура первой модели:      conv(8) → max-пулинг 2×2 → conv(16) → max-пулинг 2×2
Архитектура оптимальной модели: conv(16) → avg-пулинг 2×2 → conv(32) → avg-пулинг 2×2

Ничего не обучает: загружает веса, классифицирует тестовую выборку
и строит графики.

Запуск (из этой папки):

    python compare_models.py

Результаты: results/investigations/compare_models/
"""

import json

import numpy as np

from common import (
    RESULTS_ROOT,
    SequentialCNN,
    confusion_matrix,
    classification_metrics,
    evaluate,
    load_data,
    plot_f1_per_class,
    plot_loss,
    print_table,
    roc_one_vs_rest,
    save,
    INK,
)

import matplotlib.pyplot as plt


RESULTS_DIR = RESULTS_ROOT.parent

MODELS = [
    {
        "key": "first_4_epochs",
        "label": "первая, 4 эпохи",
        "weights": RESULTS_DIR / "archive_4_epochs" / "weights.npz",
        "filters": [8, 16],
        "pools": [True, True],
        "color": "#898781",
    },
    {
        "key": "first_full",
        "label": "первая, до цели",
        "weights": RESULTS_DIR / "archive_baseline" / "weights.npz",
        "filters": [8, 16],
        "pools": [True, True],
        "color": "#2a78d6",
    },
    {
        "key": "optimal_full",
        "label": "оптимальная",
        "weights": RESULTS_DIR / "weights.npz",
        "filters": [16, 32],
        "pools": [(2, 2, "avg"), (2, 2, "avg")],
        "color": "#eb6834",
    },
]

# Порядок обучаемых слоёв в main.py: conv1, conv2, fc1, fc2
WEIGHT_NAMES = ["conv1", "conv2", "fc1", "fc2"]


def load_model(spec):

    model = SequentialCNN(spec["filters"], spec["pools"])

    arrays = np.load(spec["weights"])

    for layer, name in zip(model.trainable(), WEIGHT_NAMES):
        layer.W = arrays[name + "_W"]
        layer.b = arrays[name + "_b"]

    return model, list(arrays["losses"])


def evaluate_model(spec, x_test, y_test):

    model, losses = load_model(spec)

    probabilities, ms_per_image = evaluate(model, x_test)

    predictions = probabilities.argmax(axis=1)

    matrix = confusion_matrix(y_test, predictions)
    metrics = classification_metrics(matrix)
    curves = roc_one_vs_rest(y_test, probabilities)

    return {
        "key": spec["key"],
        "label": spec["label"],
        "color": spec["color"],
        "architecture": model.describe(),
        "parameters": int(model.num_parameters()),
        "dense_inputs": int(model.features),
        # В history только ошибка: accuracy по эпохам main.py не сохраняет.
        # epoch_seconds нужен print_table - времени эпох в весах нет
        "history": {"train_loss": losses, "epoch_seconds": [float("nan")]},
        "epochs": len(losses),
        "final_loss": float(losses[-1]),
        "ms_per_image": ms_per_image,
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "macro_auc": float(np.mean([area for _, _, area in curves])),
        "f1_per_class": metrics["f1"].tolist(),
        "precision_per_class": metrics["precision"].tolist(),
        "recall_per_class": metrics["recall"].tolist(),
        "auc_per_class": [float(area) for _, _, area in curves],
        "errors": int(np.sum(predictions != y_test)),
        "test_size": int(len(y_test)),
        "confusion_matrix": matrix.tolist(),
    }


def plot_errors_bar(results, path):
    """
    Ошибки на тесте и итоговая средняя ошибка обучения - две панели,
    потому что у величин разные единицы.
    """

    fig, (ax_err, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))

    labels = [r["label"] for r in results]
    colors = [r["color"] for r in results]

    errors = [r["errors"] for r in results]
    losses = [r["final_loss"] for r in results]

    for ax, values, title, fmt in [
        (ax_err, errors, f"Ошибок на тесте (из {results[0]['test_size']:,})"
         .replace(",", " "), "{:.0f}"),
        (ax_loss, losses, "Средняя ошибка в последней эпохе обучения",
         "{:.6f}"),
    ]:

        bars = ax.bar(
            labels, values, color=colors, width=0.6,
            edgecolor="#fcfcfb", linewidth=2,
        )

        for bar, number in zip(bars, values):
            ax.annotate(
                fmt.format(number),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                color=INK,
                fontsize=10,
            )

        ax.set_title(title)
        ax.grid(axis="x", visible=False)

    ax_err.set_ylim(0, max(errors) * 1.15)

    # Ошибка обучения отличается на порядки - логарифмическая шкала
    ax_loss.set_yscale("log")
    ax_loss.set_ylim(min(losses) / 3, max(losses) * 3)

    save(fig, path)


def main():

    out_dir = RESULTS_ROOT / "compare_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, x_test, y_test = load_data(seed=42)

    results = []

    for spec in MODELS:
        print("Оценка:", spec["label"])
        results.append(evaluate_model(spec, x_test, y_test))

    print_table(results)

    for r in results:
        print(
            f"{r['label']:16s} эпох: {r['epochs']:3d}, "
            f"последняя ошибка обучения: {r['final_loss']:.6f}"
        )

    with open(out_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump({"results": results}, file, ensure_ascii=False, indent=2)

    plot_loss(
        results, "Первая и оптимальная модель", out_dir / "loss.png",
        target=0.0001,
    )
    plot_errors_bar(results, out_dir / "summary.png")
    plot_f1_per_class(results, out_dir / "f1_per_class.png")


if __name__ == "__main__":
    main()
