"""
Карты признаков одной цифры по слоям обученной сети.

Изображение прогоняется через сеть, и после каждого шага
свёрточной части сохраняются все карты:

    conv1 + ReLU   16 карт 26×26
    pool1          16 карт 13×13
    conv2 + ReLU   32 карты 11×11
    pool2          32 карты 5×5

Обучение не нужно: используются веса из results/weights.npz.

Запуск (из этой папки):

    python draw_feature_maps.py              восьмёрка
    python draw_feature_maps.py --digit 4    другая цифра

Результат: results/feature_maps_<цифра>.png
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from cnn import CNN
from data_preparation import prepare_data
from main import load_weights, RESULTS_DIR
from visualization import BLUES, INK, MUTED, save


# (атрибут CNN, после которого берутся карты, подпись)
STAGES = [
    ("relu1", "conv1 + ReLU"),
    ("pool1", "pool1 (average 2×2)"),
    ("relu2", "conv2 + ReLU"),
    ("pool2", "pool2 (average 2×2)"),
]

# Сколько карт в одном ряду
COLUMNS = 16

GRID_LINE = "#c3c2b7"


def maps_word(count):
    """
    «карта», «карты» или «карт» в зависимости от числа.
    """

    if count % 10 == 1 and count % 100 != 11:
        return "карта"

    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return "карты"

    return "карт"


def collect_maps(model, image):
    """
    Прямой проход по свёрточной части с сохранением выходов.
    """

    maps = {}

    x = image

    for name in ["conv1", "relu1", "pool1", "conv2", "relu2", "pool2"]:
        x = getattr(model, name).forward(x)
        maps[name] = x

    return maps


def pick_image(model, x_test, y_test, digit):
    """
    Первое тестовое изображение нужной цифры,
    которое сеть распознаёт правильно.
    """

    for index in np.where(y_test == digit)[0]:
        if model.predict(x_test[index]) == digit:
            return index

    raise ValueError(f"Нет правильно распознанных изображений цифры {digit}")


def draw_stage(subfig, stage_maps, title):
    """
    Все карты одного шага сеткой по COLUMNS в ряд.

    Каждая карта нормируется отдельно: у разных фильтров
    разный масштаб откликов, и при общей шкале слабые карты
    были бы неразличимо бледными.
    """

    count, height, width = stage_maps.shape

    rows = int(np.ceil(count / COLUMNS))

    axes = subfig.subplots(rows, COLUMNS, squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, feature_map in zip(axes.ravel(), stage_maps):

        # Тонкая рамка, чтобы были видны и почти пустые карты
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(GRID_LINE)
            spine.set_linewidth(0.8)

        top = feature_map.max()

        ax.imshow(
            feature_map,
            cmap=BLUES,
            vmin=0,
            vmax=top if top > 0 else 1,
            interpolation="nearest",
        )

    silent = int(np.sum(stage_maps.max(axis=(1, 2)) <= 0))

    note = f", без отклика: {silent}" if silent else ""

    subfig.suptitle(
        f"{title}: {count} {maps_word(count)} {height}×{width}{note}",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--digit", type=int, default=8)

    args = parser.parse_args()

    model = CNN()
    load_weights(model)

    _, _, x_test, y_test = prepare_data(train_part=0.7, seed=42)

    index = pick_image(model, x_test, y_test, args.digit)

    image = x_test[index]

    maps = collect_maps(model, image)

    probability = model.predict_proba(image)[args.digit]

    # Размер подобран так, чтобы квадратные карты заполняли ширину
    fig = plt.figure(figsize=(13, 8.6))

    left, right = fig.subfigures(1, 2, width_ratios=[1.4, 8], wspace=0.02)

    # Исходное изображение
    ax = left.subplots()
    ax.imshow(image[0], cmap="gray_r", interpolation="nearest")
    ax.set_title(
        f"Вход 28×28\nцифра {args.digit}\n"
        f"ответ сети: {args.digit} ({probability:.0%})",
        fontsize=11,
        color=INK,
    )
    ax.axis("off")

    # Высота блока пропорциональна числу рядов карт
    heights = [
        int(np.ceil(maps[name].shape[0] / COLUMNS)) for name, _ in STAGES
    ]

    stage_figs = right.subfigures(len(STAGES), 1, height_ratios=heights)

    for subfig, (name, title) in zip(stage_figs, STAGES):
        draw_stage(subfig, maps[name], title)

    # Над подфигурами, чтобы не наезжать на подпись первого ряда
    fig.suptitle(
        "Карты признаков по слоям: чем глубже, тем меньше разрешение "
        "и абстрактнее признаки (тёмное — сильный отклик)",
        y=1.04,
        fontsize=13,
        color=INK,
    )

    fig.text(
        0.99, 0.005,
        f"тестовое изображение №{index}; "
        "каждая карта нормирована отдельно",
        ha="right",
        fontsize=8,
        color=MUTED,
    )

    save(fig, RESULTS_DIR / f"feature_maps_{args.digit}.png")


if __name__ == "__main__":
    main()
