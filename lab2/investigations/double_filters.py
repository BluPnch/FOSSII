"""
Исследование 2: что даёт удвоение числа фильтров.

Сравниваются две сети с одинаковым числом слоёв:

    8 / 16 фильтров:   conv(8)  → ReLU → pool → conv(16) → ReLU → pool → 16·5·5 = 400 признаков
    16 / 32 фильтра:   conv(16) → ReLU → pool → conv(32) → ReLU → pool → 32·5·5 = 800 признаков

Первый вариант - основная сеть лабораторной.

Удвоение затрагивает не только свёртки: признаков на выходе свёрточной
части тоже становится вдвое больше, поэтому вдвое растёт и первый
полносвязный слой.

Запуск (из этой папки):

    python double_filters.py
    python double_filters.py --train-size 1000 --epochs 2   быстрая проверка

Результаты: results/investigations/double_filters/
"""

from common import parse_args, run_investigation


# Цвет закреплён за вариантом и одинаков на всех графиках;
# основная сеть - синяя, как и в первом исследовании
VARIANTS = [
    {
        "key": "8_16",
        "label": "8 / 16 фильтров",
        "filters": [8, 16],
        "pools": [True, True],
        "color": "#2a78d6",
    },
    {
        "key": "16_32",
        "label": "16 / 32 фильтра",
        "filters": [16, 32],
        "pools": [True, True],
        "color": "#eb6834",
    },
]


if __name__ == "__main__":
    run_investigation(
        "double_filters",
        "Сравнение сетей с 8 / 16 и 16 / 32 фильтрами",
        VARIANTS,
        parse_args(),
    )
