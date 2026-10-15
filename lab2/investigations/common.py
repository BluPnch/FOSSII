"""
Общий код для исследований: сеть с настраиваемой свёрточной частью,
обучение в одинаковых условиях, метрики и графики сравнения.

Сами исследования лежат рядом:

    conv_layers_count.py - сколько нужно свёрточных слоёв (1, 2 или 3)
    double_filters.py    - что даёт удвоение числа фильтров (8/16 и 16/32)
    pooling_window.py    - размер окна пулинга (2×2, 3×3, 4×4, 5×5)
    kernel_size.py       - размер ядра свёртки (3×3, 5×5, 7×7)
    pooling_type.py      - тип пулинга (max или average)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Слои, функция ошибки, данные и метрики берутся из основной лабораторной
SOURCE_DIR = Path(__file__).resolve().parent.parent / "source"
sys.path.insert(0, str(SOURCE_DIR))

from cnn import SoftmaxCrossEntropy  # noqa: E402
from data_preparation import prepare_data  # noqa: E402
from layers import (  # noqa: E402
    Conv2D,
    ReLU,
    MaxPool2D,
    AvgPool2D,
    Flatten,
    Dense,
)
from metrics import (  # noqa: E402
    confusion_matrix,
    classification_metrics,
    roc_one_vs_rest,
)
from visualization import save, INK, MUTED  # noqa: E402


RESULTS_ROOT = (
    Path(__file__).resolve().parent.parent / "results" / "investigations"
)

SURFACE = "#fcfcfb"


class SequentialCNN:
    """
    Свёрточная сеть, собранная из списка слоёв.

    filters - число фильтров в каждом свёрточном слое, например [8, 16].
    pools   - пулинг после соответствующей свёртки:
                  None или False       - без пулинга,
                  True                 - max-пулинг 2×2 с шагом 2,
                  (size, stride)       - max-пулинг size×size с шагом stride,
                  (size, stride, kind) - kind = "max" или "avg".
    kernels - размер ядра каждой свёртки, по умолчанию везде 3.

    После свёрточной части у всех вариантов одинаково:

        Flatten → Dense(64) → ReLU → Dense(10)

    Вариант filters=[8, 16], pools=[True, True] совпадает с сетью из cnn.py.
    """

    def __init__(self, filters, pools, kernels=None, hidden=64):

        if kernels is None:
            kernels = [3] * len(filters)

        self.layers = []

        channels = 1
        size = 28

        for num_filters, pool, kernel in zip(filters, pools, kernels):

            if kernel > size:
                raise ValueError(
                    f"Ядро {kernel}×{kernel} больше карты {size}×{size}"
                )

            self.layers.append(
                Conv2D(channels, num_filters, kernel_size=kernel)
            )
            self.layers.append(ReLU())

            channels = num_filters
            size -= kernel - 1

            if pool:

                if pool is True:
                    pool = (2, 2)

                pool_size, stride = pool[0], pool[1]
                kind = pool[2] if len(pool) > 2 else "max"

                if pool_size > size:
                    raise ValueError(
                        f"Окно пулинга {pool_size}×{pool_size} больше "
                        f"карты признаков {size}×{size}"
                    )

                pool_class = MaxPool2D if kind == "max" else AvgPool2D

                self.layers.append(pool_class(pool_size, stride))
                size = (size - pool_size) // stride + 1

        self.features = channels * size * size

        self.layers += [
            Flatten(),
            Dense(self.features, hidden),
            ReLU(),
            Dense(hidden, 10),
        ]

        self.loss_function = SoftmaxCrossEntropy()

    def trainable(self):
        return [layer for layer in self.layers if hasattr(layer, "W")]

    def num_parameters(self):
        return sum(
            layer.W.size + layer.b.size for layer in self.trainable()
        )

    def describe(self):

        parts = []

        for layer in self.layers:
            if isinstance(layer, Conv2D):
                kernel = layer.kernel_size
                parts.append(
                    f"Conv({layer.num_filters})" if kernel == 3
                    else f"Conv({layer.num_filters}, {kernel}×{kernel})"
                )
            elif isinstance(layer, (MaxPool2D, AvgPool2D)):
                name = "Pool" if isinstance(layer, MaxPool2D) else "AvgPool"
                parts.append(
                    f"{name}({layer.pool_size}×{layer.pool_size}, "
                    f"шаг {layer.stride})"
                )
            elif isinstance(layer, Dense):
                parts.append(f"Dense({layer.W.shape[0]})")
            else:
                parts.append(type(layer).__name__)

        return " → ".join(parts)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train_one(self, x, target, learning_rate):

        logits = self.forward(x)

        loss = self.loss_function.forward(logits, target)

        grad = self.loss_function.backward()

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        for layer in self.trainable():
            layer.update(learning_rate)

        return loss

    def predict_proba(self, x):

        logits = self.forward(x)

        exp_values = np.exp(logits - np.max(logits))

        return exp_values / np.sum(exp_values)


def load_data(seed, train_size=None):

    x_train, y_train, x_test, y_test = prepare_data(
        train_part=0.7, seed=seed
    )

    if train_size is not None:
        x_train = x_train[:train_size]
        y_train = y_train[:train_size]

    return x_train, y_train, x_test, y_test


def evaluate(model, x):
    """
    Вероятности классов и время классификации одного изображения, мс.
    """

    start = time.time()

    probabilities = np.array([model.predict_proba(image) for image in x])

    ms_per_image = (time.time() - start) / len(x) * 1000

    return probabilities, ms_per_image


def run_variant(variant, data, epochs, learning_rate, seed):
    """
    Обучает один вариант сети и считает его показатели.

    Условия у всех вариантов одинаковые: те же данные, тот же порядок
    изображений в каждой эпохе, та же скорость обучения и число эпох.
    """

    x_train, y_train, x_test, y_test = data

    # Одинаковое зерно: начальные веса воспроизводимы
    np.random.seed(seed)

    model = SequentialCNN(
        variant["filters"], variant["pools"], variant.get("kernels")
    )

    # Отдельный генератор для порядка обхода: у всех вариантов
    # изображения идут в одном и том же порядке
    order_rng = np.random.default_rng(seed)

    print(
        f"\n=== {variant['label']}: "
        f"{model.num_parameters()} параметров, "
        f"{model.features} признаков на входе Dense ==="
    )
    print(model.describe())

    history = {
        "train_loss": [],
        "test_accuracy": [],
        "epoch_seconds": [],
    }

    for epoch in range(1, epochs + 1):

        start = time.time()

        total_loss = 0.0

        for index in order_rng.permutation(len(x_train)):
            total_loss += model.train_one(
                x_train[index], y_train[index], learning_rate
            )

        seconds = time.time() - start

        probabilities, _ = evaluate(model, x_test)

        accuracy = float(np.mean(probabilities.argmax(axis=1) == y_test))

        history["train_loss"].append(total_loss / len(x_train))
        history["test_accuracy"].append(accuracy)
        history["epoch_seconds"].append(seconds)

        print(
            f"Эпоха {epoch}: ошибка = {history['train_loss'][-1]:.5f}, "
            f"accuracy на тесте = {accuracy:.4f}, время = {seconds:.0f} с",
            flush=True,
        )

    probabilities, ms_per_image = evaluate(model, x_test)

    predictions = probabilities.argmax(axis=1)

    matrix = confusion_matrix(y_test, predictions)
    metrics = classification_metrics(matrix)
    curves = roc_one_vs_rest(y_test, probabilities)

    return {
        "key": variant["key"],
        "label": variant["label"],
        "color": variant["color"],
        "architecture": model.describe(),
        "parameters": int(model.num_parameters()),
        "dense_inputs": int(model.features),
        "history": history,
        "ms_per_image": ms_per_image,
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "macro_auc": float(np.mean([area for _, _, area in curves])),
        "f1_per_class": metrics["f1"].tolist(),
        "errors": int(np.sum(predictions != y_test)),
        "test_size": int(len(y_test)),
        "confusion_matrix": matrix.tolist(),
    }


def label_line_ends(ax, ends, min_gap_px=16):
    """
    Подписи у правых концов линий: вариант читается без легенды.

    ends - список (x, y, текст). Если концы линий близко,
    подписи раздвигаются по вертикали, чтобы не налезать друг на друга.
    """

    ax.autoscale_view()

    fig = ax.figure

    points = sorted(
        (ax.transData.transform((x, y))[1], x, y, text)
        for x, y, text in ends
    )

    # Сверху вниз: каждая подпись не ближе min_gap_px к предыдущей
    placed = []

    for pixel_y, x, y, text in reversed(points):

        target = pixel_y

        if placed and placed[-1] - target < min_gap_px:
            target = placed[-1] - min_gap_px

        placed.append(target)

        ax.annotate(
            text,
            xy=(x, y),
            xytext=(8, (target - pixel_y) * 72 / fig.dpi),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=10,
        )


def plot_history(
    results, key, title, ylabel, path,
    log=False, percent=False, target=None,
):
    """
    Линейный график показателя по эпохам, по линии на вариант.

    key    - поле из history: "train_loss" или "test_accuracy".
    target - если задано, рисуется пунктир целевого значения.

    Число эпох у вариантов может различаться.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    ends = []

    max_epochs = max(len(r["history"][key]) for r in results)

    for result in results:

        values = np.array(result["history"][key])

        if percent:
            values = 100 * values

        epochs = np.arange(1, len(values) + 1)

        ax.plot(
            epochs, values,
            color=result["color"],
            linewidth=2,
            marker="o",
            markersize=6,
            label=result["label"],
        )

        if percent:
            value_text = f"{values[-1]:.2f} %"
        elif values[-1] < 0.001:
            value_text = f"{values[-1]:.6f}"
        else:
            value_text = f"{values[-1]:.4f}"

        ends.append((epochs[-1], values[-1], f"{result['label']}: {value_text}"))

    if log:
        ax.set_yscale("log")

    if target is not None:
        ax.axhline(target, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(
            f"цель {target:g}",
            xy=(1, target),
            xytext=(0, 5),
            textcoords="offset points",
            color=MUTED,
        )

    ax.set_xlabel("Эпоха")

    if max_epochs <= 10:
        ax.set_xticks(np.arange(1, max_epochs + 1))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Справа запас под подписи концов линий
    ax.set_xlim(0.5, max_epochs + max(1.5, 0.3 * max_epochs))
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Ошибка убывает - правый верхний угол свободен;
    # accuracy растёт - свободен правый нижний
    ax.legend(loc="upper right" if log else "lower right")

    label_line_ends(ax, ends)

    save(fig, path)


def plot_loss(results, title, path, target=None):
    plot_history(
        results,
        "train_loss",
        f"{title}: ошибка на обучающей выборке",
        "Средняя ошибка (cross-entropy), логарифмическая шкала",
        path,
        log=True,
        target=target,
    )


def plot_accuracy(results, title, path):
    plot_history(
        results,
        "test_accuracy",
        f"{title}: accuracy на тестовой выборке",
        "Accuracy, %",
        path,
        percent=True,
    )


def plot_summary(results, path):
    """
    Три отдельные панели: у величин разные единицы,
    поэтому общей оси у них быть не может.
    """

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    test_size = results[0]["test_size"]

    panels = [
        ("Обучаемых параметров", lambda r: r["parameters"]),
        (
            "Время эпохи (медиана), с",
            lambda r: np.median(r["history"]["epoch_seconds"]),
        ),
        (
            f"Ошибок на тесте (из {test_size:,})".replace(",", " "),
            lambda r: r["errors"],
        ),
    ]

    labels = [r["label"] for r in results]
    colors = [r["color"] for r in results]

    for ax, (title, value) in zip(axes, panels):

        values = [value(r) for r in results]

        bars = ax.bar(
            labels, values,
            color=colors, width=0.6,
            edgecolor=SURFACE, linewidth=2,
        )

        for bar, number in zip(bars, values):

            text = (
                f"{number:.1f}" if number < 10
                else f"{number:,.0f}".replace(",", " ")
            )

            ax.annotate(
                text,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                color=INK,
                fontsize=10,
            )

        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(values) * 1.15)
        ax.tick_params(axis="y", colors=MUTED)

    save(fig, path)


def plot_f1_per_class(results, path):
    """
    F-мера по цифрам: где именно выигрывает или проигрывает каждый вариант.
    """

    fig, ax = plt.subplots(figsize=(12, 5))

    digits = np.arange(10)
    width = 0.8 / len(results)

    for index, result in enumerate(results):

        offset = (index - (len(results) - 1) / 2) * width

        ax.bar(
            digits + offset,
            result["f1_per_class"],
            width=width,
            color=result["color"],
            edgecolor=SURFACE,
            linewidth=2,
            label=result["label"],
        )

    lowest = min(min(r["f1_per_class"]) for r in results)

    ax.set_ylim(max(0.0, lowest - 0.02), 1.0)
    ax.set_xticks(digits)
    ax.set_xlabel("Цифра")
    ax.set_ylabel("F-мера")
    ax.set_title(
        "F-мера по цифрам (ось Y начинается не с нуля)", pad=36
    )
    ax.grid(axis="x", visible=False)

    # Легенда над областью графика, чтобы не закрывать столбцы
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.06),
        ncols=len(results),
        borderaxespad=0,
    )

    save(fig, path)


def print_table(results):

    print(
        "\nВариант          Параметры  Признаков  Время эпохи  мс/изобр.  "
        "Accuracy  F1 (макро)  AUC (макро)  Ошибок"
    )

    for r in results:
        print(
            f"{r['label']:16s} "
            f"{r['parameters']:9d}  "
            f"{r['dense_inputs']:9d}  "
            f"{np.median(r['history']['epoch_seconds']):9.1f} с  "
            f"{r['ms_per_image']:9.3f}  "
            f"{r['accuracy']:8.4f}  "
            f"{r['macro_f1']:10.4f}  "
            f"{r['macro_auc']:11.5f}  "
            f"{r['errors']:6d}"
        )


def plot_all(results, title, results_dir):

    plot_loss(results, title, results_dir / "loss.png")
    plot_accuracy(results, title, results_dir / "accuracy.png")
    plot_summary(results, results_dir / "summary.png")
    plot_f1_per_class(results, results_dir / "f1_per_class.png")


def run_investigation(name, title, variants, args):
    """
    Обучает все варианты, печатает таблицу,
    сохраняет результаты в JSON и строит графики.

    С флагом --replot ничего не обучает: берёт results.json
    от прошлого запуска и только перерисовывает графики.
    """

    results_dir = RESULTS_ROOT / name
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "results.json"

    if args.replot:

        with open(results_path, encoding="utf-8") as file:
            results = json.load(file)["results"]

    else:

        data = load_data(args.seed, args.train_size)

        results = [
            run_variant(variant, data, args.epochs, args.lr, args.seed)
            for variant in variants
        ]

        with open(results_path, "w", encoding="utf-8") as file:
            json.dump(
                {"settings": vars(args), "results": results},
                file, ensure_ascii=False, indent=2,
            )

    print_table(results)

    plot_all(results, title, results_dir)


def parse_args():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="обучаться только на части выборки (для быстрой проверки)",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="не обучать, а перерисовать графики по results.json",
    )

    return parser.parse_args()
