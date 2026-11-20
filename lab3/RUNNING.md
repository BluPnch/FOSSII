# Запуск лабораторной работы № 3

Понадобится Python 3.12. Обучение выполняется на CPU. Датасет уже лежит в [data/Steel_industry_data.csv](data/Steel_industry_data.csv), доступ к интернету требуется только для установки зависимостей.

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
python -m pip install -r lab3/requirements.txt
```

## Обучение RNN, GRU и LSTM

Все команды ниже выполняются из корня репозитория.

```bash
python lab3/source/main.py --output lab3/results/numpy/main
```

Команда обучает все три модели с одинаковыми гиперпараметрами и сравнивает их с прогнозом по последнему известному значению. По умолчанию используются окно 24, скрытый размер 32, размер батча 256, скорость обучения 0,001, до 100 эпох и минимум 60 эпох. Скрипт сохраняет метрики, прогнозы, графики, истории обучения и чекпойнты `.npz`.

Для проверки запуска достаточно одной эпохи.

```bash
python lab3/source/main.py --epochs 1 --min-epochs 1 --output lab3/results/numpy/smoke
```

## Исследования

| Команда | Результат в `lab3/results/numpy/` |
| --- | --- |
| `python lab3/investigations/run.py --workers 3` | 33 однофакторных запуска, графики и отчёт `investigations/REPORT.md`, сводный `REPORT.md` и оценка выбранных моделей на тесте |
| `python lab3/investigations/ideal.py --workers 3` | Сочетания гиперпараметров и правила ранней остановки, чекпойнт и отчёт `ideal/REPORT.md` |
| `python lab3/investigations/features.py --workers 3` | Проверка дополнительных признаков, выбранная модель и отчёт `features/REPORT.md` |
| `python lab3/investigations/windows.py --workers 3 --factor window` | Повторная оценка длины окна на нескольких seed в `windows/` |
| `python lab3/investigations/windows.py --workers 3 --factor hidden_size` | Повторная оценка размера состояния в `hidden_sizes/` |
| `python lab3/investigations/windows.py --workers 3 --factor batch_size` | Повторная оценка размера батча в `batch_sizes/` |

Для полного цикла сначала запустите `run.py`, затем `ideal.py`. Скрипт `ideal.py` при необходимости сам выполнит недостающие запуски, а `features.py` — недостающие базовые запуски GRU. `--workers` управляет числом параллельных процессов. Завершённые запуски переиспользуются при совпадении параметров, кода и данных. Флаг `--force` у `run.py` повторяет обучение.

Текущий код обучает модели на NumPy. Сохранённые в Git отчёты и таблицы из [results/](results/) относятся к предыдущей реализации на PyTorch и используются как ориентир при сравнении. Новые значения метрик могут отличаться.

## Графики и проверки

```bash
python lab3/investigations/overview.py
python -m unittest discover -s lab3/investigations -p test_features.py -v
```

Первая команда обновляет сводный график по сохранённым таблицам в `lab3/results/`. Вторая проверяет, что подготовка дополнительных признаков использует только доступные прошлые данные.
