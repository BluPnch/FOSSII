import gzip
from pathlib import Path
import struct
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "MNIST" / "raw"
MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)


def fetch_images(filename):
    archive = ROOT.parent / "lab2" / "data" / f"{filename}.gz"
    if archive.is_file():
        return gzip.decompress(archive.read_bytes())

    for mirror in MIRRORS:
        try:
            print(f"Скачивание {mirror}{filename}.gz", flush=True)
            with urlopen(f"{mirror}{filename}.gz", timeout=60) as response:
                return gzip.decompress(response.read())
        except (OSError, EOFError) as error:
            print(f"Загрузка не удалась: {error}", flush=True)
    raise RuntimeError(f"Поместите распакованный файл {filename} в {DATA_DIR}")


def load_images(*, train):
    filename = "train-images-idx3-ubyte" if train else "t10k-images-idx3-ubyte"
    expected = 60_000 if train else 10_000
    path = DATA_DIR / filename
    cached = path.exists()
    data = path.read_bytes() if cached else fetch_images(filename)
    if len(data) != 16 + expected * 28 * 28 or struct.unpack_from(">IIII", data) != (2051, expected, 28, 28):
        raise ValueError(f"Неверный формат изображений MNIST: {path}")

    if not cached:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=DATA_DIR, delete=False) as file:
            temporary = Path(file.name)
        try:
            temporary.write_bytes(data)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(expected, 28, 28, 1)
