# Запуск лабораторной работы № 4

Понадобится Python 3.12. Основная реализация обучается на CPU с NumPy. Она использует файлы изображений в `lab4/data/MNIST/raw/` или распаковывает архивы из `lab2/data/`. Если локальной копии нет, скрипт скачивает изображения MNIST при первом запуске.

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
python -m pip install -r lab4/requirements.txt
```

## Обучение

Запускайте команды из корня репозитория.

```bash
python lab4/source/main.py
```

По умолчанию модель обучается три эпохи на 60 000 изображениях с размером батча 256 и двумя потоками BLAS, затем восстанавливает все 10 000 тестовых изображений. Число потоков можно задать через `--threads`.

Для короткой проверки на одной эпохе используйте отдельную папку результатов.

```bash
python lab4/source/main.py --epochs 1 --output lab4/results/numpy_smoke
```

## Файлы результатов

По умолчанию скрипт сохраняет результаты в `lab4/results/numpy/`.

| Файл | Содержимое |
| --- | --- |
| `history.csv` | Средние значения потерь за каждую эпоху |
| `checkpoint.npz` | Веса трёх компонентов и параметры зашумления |
| `metrics.json` | MSE и PSNR тестовых изображений до и после восстановления |
| `reconstructions.png` | Примеры оригиналов, зашумлённых входов и восстановленных изображений |

## Повторная оценка

Чтобы воспроизвести базовую оценку NumPy и построить примеры, укажите архивный чекпойнт.

```bash
python lab4/source/main.py --checkpoint lab4/baselines/numpy/checkpoint.npz --output lab4/results/numpy_evaluation
```

Параметры зашумления и seed скрипт прочитает из чекпойнта. Для оценки после собственного обучения укажите `lab4/results/numpy/checkpoint.npz` или путь из своего каталога результатов.

## Архив PyTorch

Исходная реализация и файлы первого прогона находятся в [baselines/pytorch/](baselines/pytorch/). Для запуска этой версии установите её отдельные зависимости и укажите путь к архивным весам.

```bash
python -m pip install -r lab4/baselines/pytorch/requirements.txt
python lab4/baselines/pytorch/source/main.py --device cpu --checkpoint lab4/baselines/pytorch/checkpoint.pt --output lab4/results/pytorch_evaluation
```

## Сравнение двух моделей

Установите зависимости обеих реализаций и запустите общую оценку из корня репозитория.

```bash
python -m pip install -r lab4/requirements.txt -r lab4/baselines/pytorch/requirements.txt
python lab4/baselines/compare.py
```

Скрипт загружает архивные чекпойнты и подаёт в обе модели одни и те же зашумлённые изображения. Он сохраняет `metrics.json` с MSE, PSNR и контрольной суммой общих входов и `reconstructions.png` с примерами в `lab4/results/comparison/`. [Сохранённый результат сравнения](baselines/comparison/)

## Вклад состязательной потери

Выполните команду из корня репозитория, чтобы запустить три парных обучения VAE и VAE-GAN на NumPy.

```bash
python lab4/source/adversarial_experiment.py --max-minutes 20
```

По умолчанию каждая модель обучается две эпохи на одной подвыборке из 8 192 изображений с батчем 32. Все модели оцениваются на одних и тех же 10 000 зашумлённых тестовых цифрах. При достижении лимита в сводку попадают завершённые пары.

Метрики, диаграмма, примеры, журналы обучения и веса записываются в `lab4/results/vae-vs-vae-gan/`. Для запуска подходят зависимости из [requirements.txt](requirements.txt). Сохранённые веса любой модели можно оценить через `source/main.py --checkpoint`. [Результаты исследования](investigations/vae-vs-vae-gan/)
