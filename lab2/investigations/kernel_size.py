"""
Исследование 4: какой размер ядра свёртки (размерность свёртки) лучше.

Сравниваются три сети, у которых в обоих свёрточных слоях ядро
одного размера, а всё остальное как в основной сети
(8 и 16 фильтров, max-пулинг 2×2 с шагом 2):

    ядро 3×3:  28 → conv 26 → pool 13 → conv 11 → pool 5   → 16·5·5 = 400 признаков
    ядро 5×5:  28 → conv 24 → pool 12 → conv 8  → pool 4   → 16·4·4 = 256 признаков
    ядро 7×7:  28 → conv 22 → pool 11 → conv 5  → pool 2   → 16·2·2 = 64 признака

Свёртка идёт без дополнения нулями, поэтому чем больше ядро,
тем сильнее уменьшается карта признаков: ядро k×k отрезает k − 1
пикселей по каждой стороне.

Вариант 3×3 - основная сеть лабораторной.

Запуск (из этой папки):

    python kernel_size.py
    python kernel_size.py --train-size 1000 --epochs 2   быстрая проверка

Результаты: results/investigations/kernel_size/
"""

from common import parse_args, run_investigation


# Цвет закреплён за вариантом и одинаков на всех графиках;
# основная сеть - синяя, как и в других исследованиях
VARIANTS = [
    {
        "key": f"kernel_{kernel}",
        "label": f"ядро {kernel}×{kernel}",
        "filters": [8, 16],
        "pools": [True, True],
        "kernels": [kernel, kernel],
        "color": color,
    }
    for kernel, color in [
        (3, "#2a78d6"),
        (5, "#eb6834"),
        (7, "#1baf7a"),
    ]
]


if __name__ == "__main__":
    run_investigation(
        "kernel_size",
        "Размер ядра свёртки",
        VARIANTS,
        parse_args(),
    )
