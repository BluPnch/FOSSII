# Запуск лабораторной работы № 1

Понадобятся Python 3.12 и доступ к UCI. Скрипты и ноутбуки получают датасет Mushroom через `ucimlrepo`.

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
python -m pip install numpy pandas matplotlib ucimlrepo
```

## Скрипты Python

Запускайте из корня репозитория.

```bash
python lab1/lab1_olp.py
python lab1/lab1_mlp.py
```

Первая команда запускает однослойный перцептрон, вторая — многослойный. Метрики выводятся в терминал, а графики открываются в окнах Matplotlib. Закройте открытый график, чтобы скрипт продолжил работу.

## Ноутбуки Jupyter

```bash
python -m pip install notebook
python -m notebook lab1
```

Откройте `lab1_olp.ipynb` или `lab1_mlp.ipynb`, выберите ядро Python из созданного окружения и запустите все ячейки через **Run → Run All Cells**. Метрики и графики появятся внутри ноутбука.
