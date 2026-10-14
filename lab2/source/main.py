import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

from cnn import CNN
from data_preparation import prepare_data
from metrics import (
    confusion_matrix,
    classification_metrics,
    roc_one_vs_rest
)
from visualization import (
    plot_loss,
    plot_confusion_matrix,
    plot_roc,
    plot_tsne,
    plot_tsne_errors,
    plot_errors
)


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

WEIGHTS_PATH = RESULTS_DIR / "weights.npz"

TRAINABLE_LAYERS = ["conv1", "conv2", "fc1", "fc2"]


def save_weights(model, losses):

    arrays = {"losses": np.array(losses)}

    for name in TRAINABLE_LAYERS:
        layer = getattr(model, name)
        arrays[name + "_W"] = layer.W
        arrays[name + "_b"] = layer.b

    np.savez(WEIGHTS_PATH, **arrays)


def load_weights(model):

    arrays = np.load(WEIGHTS_PATH)

    for name in TRAINABLE_LAYERS:
        layer = getattr(model, name)
        layer.W = arrays[name + "_W"]
        layer.b = arrays[name + "_b"]

    return list(arrays["losses"])


def train(
    model,
    x_train,
    y_train,
    learning_rate,
    lr_decay,
    max_epochs,
    target_loss
):
    """
    Обучение обратным распространением ошибки
    (стохастический градиентный спуск, по одному изображению).

    Останавливаемся, когда средняя ошибка за эпоху
    стала не больше target_loss, либо когда закончились эпохи.
    """

    losses = []

    for epoch in range(1, max_epochs + 1):

        start = time.time()

        order = np.random.permutation(len(x_train))

        total_loss = 0.0

        for index in order:
            total_loss += model.train_one(
                x_train[index],
                y_train[index],
                learning_rate
            )

        mean_loss = total_loss / len(x_train)

        losses.append(mean_loss)

        print(
            f"Эпоха {epoch:3d}: "
            f"средняя ошибка = {mean_loss:.6f}, "
            f"learning rate = {learning_rate:.5f}, "
            f"время = {time.time() - start:.0f} с",
            flush=True
        )

        save_weights(model, losses)

        if mean_loss <= target_loss:
            print("Целевая средняя ошибка достигнута")
            break

        learning_rate *= lr_decay

    else:
        print(
            f"Целевая средняя ошибка {target_loss:g} не достигнута "
            f"за {max_epochs} эпох"
        )

    return losses


def predict_all(model, x):
    """
    Вероятности классов для всех изображений: (N, 10)
    """

    return np.array([
        model.predict_proba(image) for image in x
    ])


def print_metrics(matrix, metrics, curves):

    print("\nМатрица неточностей (строки - истина, столбцы - ответ сети):")
    print(matrix)

    print("\nКласс  Precision  Recall   F1       AUC")

    for digit in range(len(matrix)):
        print(
            f"{digit:5d}  "
            f"{metrics['precision'][digit]:.4f}     "
            f"{metrics['recall'][digit]:.4f}   "
            f"{metrics['f1'][digit]:.4f}   "
            f"{curves[digit][2]:.5f}"
        )

    print(f"\nAccuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision (макро): {metrics['macro_precision']:.4f}")
    print(f"Recall (макро):    {metrics['macro_recall']:.4f}")
    print(f"F-мера (макро):    {metrics['macro_f1']:.4f}")
    print(
        "AUC (макро):       "
        f"{np.mean([area for _, _, area in curves]):.5f}"
    )


def print_top_confusions(matrix, count=10):
    """
    Какие цифры сеть путает чаще всего.
    """

    errors = matrix.copy()

    np.fill_diagonal(errors, 0)

    print("\nСамые частые ошибки (истинная цифра → ответ сети):")

    for flat_index in np.argsort(-errors, axis=None)[:count]:

        true_digit, predicted_digit = np.unravel_index(
            flat_index,
            errors.shape
        )

        if errors[true_digit, predicted_digit] == 0:
            break

        print(
            f"  {true_digit} → {predicted_digit}: "
            f"{errors[true_digit, predicted_digit]} раз"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--target-loss", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--tsne-size",
        type=int,
        default=5000,
        help="сколько тестовых изображений брать для t-SNE"
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="обучаться только на части выборки (для быстрой проверки)"
    )

    parser.add_argument(
        "--load",
        action="store_true",
        help="не обучать, а загрузить сохранённые веса"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    RESULTS_DIR.mkdir(exist_ok=True)

    # ---------- Данные ----------

    x_train, y_train, x_test, y_test = prepare_data(
        train_part=0.7,
        seed=args.seed
    )

    print("Обучающая выборка:", x_train.shape)
    print("Тестовая выборка: ", x_test.shape)

    if args.train_size is not None:
        x_train = x_train[:args.train_size]
        y_train = y_train[:args.train_size]

    # ---------- t-SNE до обучения ----------

    tsne_size = min(args.tsne_size, len(x_test))

    print(f"\nПостроение t-SNE по {tsne_size} изображениям...")

    embedding = TSNE(
        n_components=2,
        init="pca",
        random_state=args.seed
    ).fit_transform(
        x_test[:tsne_size].reshape(tsne_size, -1)
    )

    plot_tsne(
        embedding,
        y_test[:tsne_size],
        RESULTS_DIR / "tsne.png"
    )

    # ---------- Обучение ----------

    model = CNN()

    if args.load:
        losses = load_weights(model)
        print("\nВеса загружены из", WEIGHTS_PATH)
    else:
        print("\nОбучение...")
        losses = train(
            model,
            x_train,
            y_train,
            learning_rate=args.lr,
            lr_decay=args.lr_decay,
            max_epochs=args.epochs,
            target_loss=args.target_loss
        )

    plot_loss(losses, args.target_loss, RESULTS_DIR / "loss.png")

    # ---------- Оценка на тестовой выборке ----------

    print("\nКлассификация тестовой выборки...")

    probabilities = predict_all(model, x_test)

    predictions = probabilities.argmax(axis=1)

    matrix = confusion_matrix(y_test, predictions)

    metrics = classification_metrics(matrix)

    curves = roc_one_vs_rest(y_test, probabilities)

    print_metrics(matrix, metrics, curves)

    print_top_confusions(matrix)

    plot_confusion_matrix(
        matrix,
        RESULTS_DIR / "confusion_matrix.png"
    )

    plot_roc(curves, RESULTS_DIR / "roc.png")

    # ---------- Анализ ошибок ----------

    plot_tsne_errors(
        embedding,
        y_test[:tsne_size],
        predictions[:tsne_size],
        RESULTS_DIR / "tsne_errors.png"
    )

    plot_errors(
        x_test,
        y_test,
        probabilities,
        RESULTS_DIR / "errors.png"
    )


if __name__ == "__main__":
    main()
