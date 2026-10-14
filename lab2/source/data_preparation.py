import gzip
import urllib.request
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download(filename):
    """
    Скачивает файл MNIST в папку data,
    если его там ещё нет.
    """

    DATA_DIR.mkdir(exist_ok=True)

    path = DATA_DIR / filename

    if path.exists():
        return path

    for mirror in MIRRORS:
        try:
            print("Скачивание", mirror + filename)
            urllib.request.urlretrieve(mirror + filename, path)
            return path
        except OSError as error:
            print("  не удалось:", error)
            path.unlink(missing_ok=True)

    raise RuntimeError(
        f"Не удалось скачать {filename}. "
        f"Положите файл вручную в {DATA_DIR}"
    )


def read_images(path):
    """
    Формат IDX: 16 байт заголовка,
    дальше пиксели по одному байту.
    """

    with gzip.open(path, "rb") as file:
        data = np.frombuffer(file.read(), dtype=np.uint8, offset=16)

    return data.reshape(-1, 28, 28)


def read_labels(path):
    """
    Формат IDX: 8 байт заголовка,
    дальше метки по одному байту.
    """

    with gzip.open(path, "rb") as file:
        return np.frombuffer(file.read(), dtype=np.uint8, offset=8)


def load_mnist():
    """
    Возвращает все 70000 изображений MNIST:

        images: (70000, 28, 28), uint8
        labels: (70000,)
    """

    paths = {
        name: download(filename)
        for name, filename in FILES.items()
    }

    images = np.concatenate([
        read_images(paths["train_images"]),
        read_images(paths["test_images"]),
    ])

    labels = np.concatenate([
        read_labels(paths["train_labels"]),
        read_labels(paths["test_labels"]),
    ])

    return images, labels.astype(np.int64)


def prepare_data(train_part=0.7, seed=42):
    """
    Загружает MNIST, нормирует пиксели в [0, 1],
    перемешивает и делит на обучающую и тестовую выборки.

    Возвращает:

        x_train: (N_train, 1, 28, 28)
        y_train: (N_train,)
        x_test:  (N_test, 1, 28, 28)
        y_test:  (N_test,)
    """

    images, labels = load_mnist()

    x = images.astype(np.float64) / 255.0

    # Добавляем ось каналов: (N, 28, 28) -> (N, 1, 28, 28)
    x = x[:, None, :, :]

    order = np.random.default_rng(seed).permutation(len(x))

    train_size = int(len(x) * train_part)

    train_index = order[:train_size]
    test_index = order[train_size:]

    return (
        x[train_index],
        labels[train_index],
        x[test_index],
        labels[test_index],
    )
