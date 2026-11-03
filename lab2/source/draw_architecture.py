"""
Схема архитектуры сети в виде блоков, как на классических
схемах VGG: каждый слой - вертикальный прямоугольник с подписью.

Запуск (из этой папки):

    python draw_architecture.py

Результат: results/architecture.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# (заливка, рамка и текст) для каждого типа слоя
STYLES = {
    "input": ("#e6e6e6", "#4d4d4d"),
    "conv": ("#fde3cc", "#f0962e"),
    "pool": ("#d5e3f7", "#4a86e8"),
    "fc": ("#d9ead3", "#38761d"),
    "softmax": ("#f6d0d0", "#e06666"),
}

# (тип, подпись в блоке, форма выхода под блоком)
LAYERS = [
    ("input", "Input 28×28", "1×28×28"),
    ("conv", "3x3 conv, 16", "16×26×26"),
    ("pool", "AvgPool 2×2", "16×13×13"),
    ("conv", "3x3 conv, 32", "32×11×11"),
    ("pool", "AvgPool 2×2", "32×5×5"),
    ("fc", "FC 64", "64"),
    ("fc", "FC 10", "10"),
    ("softmax", "Softmax", "10"),
]

BLOCK_WIDTH = 0.42
BLOCK_HEIGHT = 2.6
STEP = 0.62


def main():

    fig, ax = plt.subplots(figsize=(6.4, 2.6))

    for index, (kind, label, shape) in enumerate(LAYERS):

        fill, edge = STYLES[kind]

        x = index * STEP

        ax.add_patch(FancyBboxPatch(
            (x, 0),
            BLOCK_WIDTH,
            BLOCK_HEIGHT,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=fill,
            edgecolor=edge,
            linewidth=2.2,
        ))

        ax.text(
            x + BLOCK_WIDTH / 2,
            BLOCK_HEIGHT / 2,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
            color=edge,
        )

        # Форма данных на выходе слоя - мелко под блоком
        ax.text(
            x + BLOCK_WIDTH / 2,
            -0.12,
            shape,
            rotation=90,
            ha="center",
            va="top",
            fontsize=7,
            color="#898781",
        )

    ax.set_xlim(-0.1, (len(LAYERS) - 1) * STEP + BLOCK_WIDTH + 0.1)
    ax.set_ylim(-1.05, BLOCK_HEIGHT + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")

    path = RESULTS_DIR / "architecture.png"

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Сохранена схема:", path)


if __name__ == "__main__":
    main()
