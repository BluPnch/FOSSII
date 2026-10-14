"""
Исследование 1: сколько нужно свёрточных слоёв.

Сравниваются три сети, которые отличаются только свёрточной частью:

    1 слой:  conv(8) → ReLU → pool                                 → 8·13·13 = 1352 признака
    2 слоя:  conv(8) → ReLU → pool → conv(16) → ReLU → pool        → 16·5·5  = 400 признаков
    3 слоя:  conv(8) → ReLU → pool → conv(16) → ReLU → pool
                                   → conv(32) → ReLU               → 32·3·3  = 288 признаков

Вариант с двумя слоями - основная сеть лабораторной.

У третьего слоя нет пулинга: после conv(32) карты имеют размер 3×3,
и пулинг 2×2 оставил бы от них 1×1, отбросив последние строку и столбец.

Запуск (из этой папки):

    python conv_layers_count.py
    python conv_layers_count.py --train-size 1000 --epochs 2   быстрая проверка

Результаты: results/investigations/conv_layers_count/
"""

from common import parse_args, run_investigation


# Цвет закреплён за вариантом и одинаков на всех графиках
VARIANTS = [
    {
        "key": "1_layer",
        "label": "1 слой",
        "filters": [8],
        "pools": [True],
        "color": "#eb6834",
    },
    {
        "key": "2_layers",
        "label": "2 слоя",
        "filters": [8, 16],
        "pools": [True, True],
        "color": "#2a78d6",
    },
    {
        "key": "3_layers",
        "label": "3 слоя",
        "filters": [8, 16, 32],
        "pools": [True, True, False],
        "color": "#1baf7a",
    },
]


if __name__ == "__main__":
    run_investigation(
        "conv_layers_count",
        "Сравнение сетей с 1, 2 и 3 свёрточными слоями",
        VARIANTS,
        parse_args(),
    )
