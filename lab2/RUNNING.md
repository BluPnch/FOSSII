# Запуск лабораторной работы № 2

Понадобится Python 3.12. При первом запуске команд, которым нужны данные, MNIST автоматически скачивается в `lab2/data/`. Если автоматическое скачивание не удаётся, поместите туда четыре файла `*-ubyte.gz`, перечисленные в [data_preparation.py](source/data_preparation.py).

## Окружение

Из корня репозитория создайте виртуальное окружение, если его ещё нет.

```bash
python -m venv .venv
```

Активируйте его в PowerShell.

```powershell
.\.venv\Scripts\Activate.ps1
```

В Linux или macOS используйте команду ниже.

```bash
source .venv/bin/activate
```

Установите зависимости.

```bash
python -m pip install numpy matplotlib scikit-learn threadpoolctl
```

## Обучение и оценка CNN

Все команды ниже выполняются из корня репозитория.

```bash
python lab2/source/main.py --load
```

Команда загрузит опубликованный чекпойнт `lab2/results/weights.npz`, рассчитает метрики на всей тестовой выборке и сохранит графики в `lab2/results/`. Для обучения с нуля выполните следующую команду. Она обновит веса и графики итоговой модели в том же каталоге.

```bash
python lab2/source/main.py
```

По умолчанию обучение завершается при средней ошибке ≤ 0,0001 или после 60 эпох. Параметры можно посмотреть через `python lab2/source/main.py --help`. Флаг `--train-size` ограничивает обучающую выборку, а `--tsne-size` задаёт число тестовых изображений для t-SNE.

## Архитектура и карты признаков

```bash
python lab2/source/draw_architecture.py
python lab2/source/draw_feature_maps.py --digit 8
```

Схема и карты сохраняются в `lab2/results/`. Для карт признаков используются веса итоговой CNN.

## Исследования архитектуры

| Команда | Что сравнивается |
| --- | --- |
| `python lab2/investigations/conv_layers_count.py` | Число свёрточных слоёв |
| `python lab2/investigations/double_filters.py` | Число фильтров |
| `python lab2/investigations/kernel_size.py` | Размер ядра |
| `python lab2/investigations/pooling_window.py` | Размер окна пулинга |
| `python lab2/investigations/pooling_type.py` | Тип пулинга |

Каждый вариант по умолчанию обучается 5 эпох. Графики и `results.json` сохраняются в подпапках `lab2/results/investigations/`. Добавьте `--replot` к любой из пяти команд, чтобы построить графики по готовой сводке без обучения.

## Сравнение первой и итоговой моделей

```bash
python lab2/investigations/compare_models.py
python lab2/investigations/compare_60_epochs.py --replot
```

Первая команда оценивает модели по двум опубликованным чекпойнтам. Вторая строит графики 60 эпох по готовому `results.json` и обновляет `REPORT.md`. Обе сохраняют результаты в `lab2/results/investigations/`.

Для нового сравнения на 60 эпох можно продолжить обучение обеих моделей по исходным весам или запустить обучение с нуля. Для каждого варианта используйте свой выходной каталог.

```bash
python lab2/investigations/compare_60_epochs.py --output lab2/results/local/compare_60_epochs
python lab2/investigations/compare_60_epochs.py --from-scratch --output lab2/results/local/compare_60_epochs_fresh
```

Прерванное обучение можно продолжить той же командой. Чекпойнты, журналы и предсказания этого запуска сохраняются локально.
