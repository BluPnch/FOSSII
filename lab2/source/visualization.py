import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm


# Цвета цифр 0-9. Десять оттенков на глаз различить трудно,
# поэтому на графиках цифры дополнительно подписаны
DIGIT_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
    "#008300", "#4a3aa7", "#e34948", "#898781", "#7a4b1c",
]

BLUE = "#2a78d6"
RED = "#d03b3b"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
POINT_GRAY = "#c3c2b7"

BLUES = LinearSegmentedColormap.from_list(
    "blues",
    ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"]
)

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
})


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён график:", path)


def plot_loss(losses, target_loss, path):
    """
    Средняя ошибка на обучающей выборке по эпохам.
    """

    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = np.arange(1, len(losses) + 1)

    ax.plot(epochs, losses, color=BLUE, linewidth=2, marker="o", markersize=4)

    ax.axhline(target_loss, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(
        f"целевая ошибка {target_loss:g}",
        xy=(epochs[-1], target_loss),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        color=MUTED,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Средняя ошибка (cross-entropy)")
    ax.set_title("Средняя ошибка на обучающей выборке")

    save(fig, path)


def plot_confusion_matrix(matrix, path):
    """
    Матрица неточностей: строки - истинный класс,
    столбцы - предсказанный.
    """

    fig, ax = plt.subplots(figsize=(8, 7))

    num_classes = len(matrix)

    # Логарифмическая шкала цвета: иначе на фоне диагонали
    # не видно редких ошибок
    norm = LogNorm(vmin=1, vmax=matrix.max())

    image = ax.imshow(np.maximum(matrix, 0.5), cmap=BLUES, norm=norm)

    for i in range(num_classes):
        for j in range(num_classes):

            if matrix[i, j] == 0:
                continue

            dark_cell = norm(matrix[i, j]) > 0.6

            ax.text(
                j, i, matrix[i, j],
                ha="center",
                va="center",
                fontsize=9,
                color="white" if dark_cell else INK,
            )

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Матрица неточностей (тестовая выборка)")
    ax.grid(False)

    fig.colorbar(image, ax=ax, label="Количество изображений", shrink=0.8)

    save(fig, path)


def plot_roc(curves, path):
    """
    ROC-кривые "один против остальных".

    curves: список (fpr, tpr, auc) по классам.

    Слева кривые целиком, справа - увеличенный левый верхний
    угол, потому что на общем графике кривые сливаются.
    """

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 6))

    for digit, (fpr, tpr, area) in enumerate(curves):
        for ax in (ax_full, ax_zoom):
            ax.plot(
                fpr, tpr,
                color=DIGIT_COLORS[digit],
                linewidth=1.5,
                label=f"цифра {digit}: AUC = {area:.5f}",
            )

    ax_full.plot(
        [0, 1], [0, 1],
        color=MUTED, linewidth=1, linestyle="--",
        label="случайный классификатор",
    )

    ax_full.set_title("ROC-кривые (один против остальных)")
    ax_full.set_xlabel("FPR")
    ax_full.set_ylabel("TPR")
    ax_full.legend(loc="lower right")

    ax_zoom.set_xlim(0, 0.05)
    ax_zoom.set_ylim(0.95, 1.0)
    ax_zoom.set_title("Увеличенный левый верхний угол")
    ax_zoom.set_xlabel("FPR")
    ax_zoom.set_ylabel("TPR")

    save(fig, path)


def label_clusters(ax, embedding, labels):
    """
    Подписывает каждую цифру в центре её облака точек.
    """

    for digit in range(10):

        center = np.median(embedding[labels == digit], axis=0)

        ax.text(
            center[0], center[1], str(digit),
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="center",
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": DIGIT_COLORS[digit],
                "linewidth": 2,
                "alpha": 0.9,
            },
        )


def plot_tsne(embedding, labels, path):
    """
    t-SNE исходных изображений, цвет - истинная цифра.
    """

    fig, ax = plt.subplots(figsize=(10, 9))

    for digit in range(10):

        points = embedding[labels == digit]

        ax.scatter(
            points[:, 0], points[:, 1],
            s=8,
            color=DIGIT_COLORS[digit],
            alpha=0.7,
            linewidths=0,
        )

    label_clusters(ax, embedding, labels)

    ax.set_title("t-SNE исходных изображений (пиксели)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    save(fig, path)


def plot_tsne_errors(embedding, labels, predictions, path):
    """
    То же t-SNE, но выделены изображения, на которых сеть ошиблась.

    Подпись "4→9" означает: на самом деле 4, сеть ответила 9.
    """

    fig, ax = plt.subplots(figsize=(10, 9))

    wrong = labels != predictions

    ax.scatter(
        embedding[~wrong, 0], embedding[~wrong, 1],
        s=8,
        color=POINT_GRAY,
        alpha=0.6,
        linewidths=0,
        label="классифицировано верно",
    )

    ax.scatter(
        embedding[wrong, 0], embedding[wrong, 1],
        s=40,
        color=RED,
        edgecolors="white",
        linewidths=1,
        label=f"ошибка сети ({wrong.sum()} из {len(labels)})",
    )

    label_clusters(ax, embedding, labels)

    for index in np.where(wrong)[0]:
        ax.annotate(
            f"{labels[index]}→{predictions[index]}",
            xy=embedding[index],
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )

    ax.set_title(
        "Ошибки сети на t-SNE (истинная цифра → ответ сети)"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.legend(loc="upper right")

    save(fig, path)


def plot_errors(images, labels, probabilities, path, max_images=24):
    """
    Примеры изображений, на которых сеть ошиблась.

    Показываем сначала самые "уверенные" ошибки.
    """

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    wrong = np.where(labels != predictions)[0]

    if len(wrong) == 0:
        print("Ошибок нет, график не строится")
        return

    wrong = wrong[np.argsort(-confidence[wrong])][:max_images]

    columns = 8
    rows = int(np.ceil(len(wrong) / columns))

    fig, axes = plt.subplots(
        rows, columns,
        figsize=(columns * 1.6, rows * 2.0),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.axis("off")

    for ax, index in zip(axes.ravel(), wrong):

        ax.imshow(images[index, 0], cmap="gray_r")

        ax.set_title(
            f"{labels[index]}→{predictions[index]}"
            f" ({confidence[index]:.0%})",
            fontsize=9,
        )

    fig.suptitle(
        "Самые уверенные ошибки: истинная цифра → ответ сети "
        "(уверенность сети)",
        color=INK,
    )

    save(fig, path)
