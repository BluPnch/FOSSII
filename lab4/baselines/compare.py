import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from pytorch.source.models import Encoder as TorchEncoder, Generator as TorchGenerator


BASELINES = Path(__file__).resolve().parent
LAB = BASELINES.parent
BATCH_SIZE = 256
THREADS = 2
NUMPY_CHECKPOINT = BASELINES / "numpy" / "checkpoint.npz"
PYTORCH_CHECKPOINT = BASELINES / "pytorch" / "checkpoint.pt"
sys.path.insert(0, str(LAB / "source"))

from data import load_images
from main import add_noise, load_checkpoint, normalize, reconstruct, restoration_scores, save_examples
from models import Discriminator, Encoder, Generator


@torch.inference_mode()
def compare(images, numpy_models, torch_models, noise_std, noise_seed):
    numpy_encoder, numpy_generator = numpy_models
    torch_encoder, torch_generator = torch_models
    noise_rng = np.random.default_rng(noise_seed)
    noisy_hash = hashlib.sha256()
    errors = np.zeros(3, dtype=np.float64)
    pytorch_wins = numpy_wins = 0
    examples = None

    for start in range(0, len(images), BATCH_SIZE):
        clean = normalize(images[start:start + BATCH_SIZE])
        noisy = add_noise(clean, noise_std, noise_rng)
        noisy_hash.update(noisy.astype("<f4", copy=False).tobytes())
        numpy_restored = reconstruct(numpy_encoder, numpy_generator, noisy)

        # Меняем только порядок осей. Тензор содержит те же значения пикселей.
        torch_input = torch.from_numpy(noisy).permute(0, 3, 1, 2)
        mean, _ = torch_encoder(torch_input)
        pytorch_restored = torch_generator(mean).sigmoid().permute(0, 2, 3, 1).numpy()

        squared_errors = [np.square(image - clean) for image in (noisy, pytorch_restored, numpy_restored)]
        errors += [error.sum(dtype=np.float64) for error in squared_errors]
        pytorch_image_error, numpy_image_error = (
            error.sum(axis=(1, 2, 3), dtype=np.float64) for error in squared_errors[1:]
        )
        pytorch_wins += int(np.count_nonzero(pytorch_image_error < numpy_image_error))
        numpy_wins += int(np.count_nonzero(numpy_image_error < pytorch_image_error))

        if examples is None:
            examples = tuple(image[:10] for image in (clean, noisy, pytorch_restored, numpy_restored))

    metrics = {
        "test_images": len(images),
        "noise_std": noise_std,
        "noise_seed": noise_seed,
        "noisy_images_sha256": noisy_hash.hexdigest(),
    }
    metrics.update({
        name: restoration_scores(error, images.size)
        for name, error in zip(("noisy", "pytorch", "numpy"), errors)
    })
    metrics["images_with_lower_mse"] = {
        "pytorch": pytorch_wins,
        "numpy": numpy_wins,
        "equal": len(images) - pytorch_wins - numpy_wins,
    }
    return metrics, examples


def main():
    parser = argparse.ArgumentParser(description="Оценка двух архивных VAE-GAN на одном зашумлённом MNIST.")
    parser.add_argument("--output", type=Path, default=LAB / "results" / "comparison")
    args = parser.parse_args()

    torch.set_num_threads(THREADS)
    rng = np.random.default_rng(0)
    numpy_encoder, numpy_generator = Encoder(rng), Generator(rng)
    config = load_checkpoint(NUMPY_CHECKPOINT, (numpy_encoder, numpy_generator, Discriminator(rng)))
    torch_encoder, torch_generator = TorchEncoder().eval(), TorchGenerator().eval()
    checkpoint = torch.load(PYTORCH_CHECKPOINT, map_location="cpu", weights_only=True)
    torch_encoder.load_state_dict(checkpoint["encoder"])
    torch_generator.load_state_dict(checkpoint["generator"])

    images = load_images(train=False)
    with threadpool_limits(limits=THREADS, user_api="blas"):
        metrics, examples = compare(
            images,
            (numpy_encoder, numpy_generator),
            (torch_encoder, torch_generator),
            config["noise_std"],
            config["seed"] + 1,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    labels = ("Оригинал", "Зашумлённый вход", "PyTorch", "NumPy")
    save_examples(examples, args.output / "reconstructions.png", labels)
    report = json.dumps(metrics, indent=2)
    (args.output / "metrics.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
