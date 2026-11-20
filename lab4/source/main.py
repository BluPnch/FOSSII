import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from data import ROOT, load_images
from models import Adam, Discriminator, Encoder, Generator


NOISE_STD = 0.5
KL_WEIGHT = 0.001
ADVERSARIAL_WEIGHT = 0.02
LOSS_NAMES = ("reconstruction_bce", "kl", "generator_adversarial", "discriminator")


def sigmoid(logits):
    return np.exp(-np.logaddexp(0, -logits))


def binary_cross_entropy(logits, targets):
    return (np.logaddexp(0, logits) - targets * logits).mean(dtype=np.float64)


def normalize(images):
    return images.astype(np.float32) / 255


def add_noise(clean, noise_std, rng):
    noise = rng.standard_normal(clean.shape, dtype=np.float32)
    return np.clip(clean + noise_std * noise, 0, 1)


def train_step(clean, models, optimizers, rng):
    encoder, generator, discriminator = models
    vae_optimizer, discriminator_optimizer = optimizers
    batch_size = len(clean)

    noisy = add_noise(clean, NOISE_STD, rng)
    mean, log_variance = np.split(encoder.forward(noisy, retain=True), 2, axis=1)
    variance = np.exp(log_variance)
    offset = np.exp(0.5 * log_variance) * rng.standard_normal(mean.shape, dtype=np.float32)
    prior = rng.standard_normal(mean.shape, dtype=np.float32)
    logits = generator.forward(np.concatenate((mean + offset, prior)), retain=True)
    fake = sigmoid(logits)

    # Дискриминатор получает чистые изображения и оба вида синтетических.
    judged = discriminator.forward(np.concatenate((clean, fake)), retain=True)
    real_scores, fake_scores = judged[:batch_size], judged[batch_size:]
    discriminator_loss = (
        binary_cross_entropy(real_scores, 1) + binary_cross_entropy(fake_scores, 0)
    ) / 2
    judged_gradient = sigmoid(judged)
    judged_gradient[:batch_size] -= 1
    judged_gradient[:batch_size] *= 0.5 / len(real_scores)
    judged_gradient[batch_size:] *= 0.5 / len(fake_scores)
    discriminator.backward(judged_gradient)
    discriminator_optimizer.step()

    # Веса дискриминатора зафиксированы, градиент проходит к синтетическим пикселям.
    fake_scores = discriminator.forward(fake, retain=True)
    adversarial = binary_cross_entropy(fake_scores, 1)
    adversarial_gradient = (sigmoid(fake_scores) - 1) * (ADVERSARIAL_WEIGHT / len(fake_scores))
    fake_gradient = discriminator.backward(adversarial_gradient, parameter_grad=False, input_grad=True)

    reconstruction = binary_cross_entropy(logits[:batch_size], clean)
    logits_gradient = fake_gradient * fake * (1 - fake)
    logits_gradient[:batch_size] += (fake[:batch_size] - clean) / clean.size
    latent_gradient = generator.backward(logits_gradient, input_grad=True)[:batch_size]

    # Градиенты репараметризации и KL-дивергенции по среднему и логарифму дисперсии.
    kl = -0.5 * (1 + log_variance - mean**2 - variance).sum(axis=1).mean(dtype=np.float64)
    mean_gradient = latent_gradient + (KL_WEIGHT / batch_size) * mean
    log_variance_gradient = 0.5 * latent_gradient * offset + (0.5 * KL_WEIGHT / batch_size) * (variance - 1)
    encoder.backward(np.concatenate((mean_gradient, log_variance_gradient), axis=1))
    vae_optimizer.step()
    return np.array((reconstruction, kl, adversarial, discriminator_loss))


def train(images, models, args, rng):
    encoder, generator, discriminator = models
    vae_optimizer = Adam((*encoder.parameters(), *generator.parameters()), learning_rate=0.001)
    discriminator_optimizer = Adam(discriminator.parameters(), learning_rate=0.0002)

    with (args.output / "history.csv").open("w", encoding="utf-8", newline="") as file:
        history = csv.writer(file)
        history.writerow(("epoch", *LOSS_NAMES))
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            order = rng.permutation(len(images))
            totals = np.zeros(len(LOSS_NAMES))
            for start in range(0, len(images), args.batch_size):
                clean = normalize(images[order[start:start + args.batch_size]])
                losses = train_step(clean, models, (vae_optimizer, discriminator_optimizer), rng)
                totals += losses * len(clean)
            averages = totals / len(images)
            if not np.isfinite(averages).all():
                raise FloatingPointError(f"Неконечная функция потерь на эпохе {epoch}")
            history.writerow((epoch, *averages))
            file.flush()
            summary = ", ".join(f"{name}={value:.4f}" for name, value in zip(LOSS_NAMES, averages))
            seconds = time.perf_counter() - started
            print(f"Эпоха {epoch}/{args.epochs} ({seconds:.1f} с): {summary}", flush=True)


def model_weights(models):
    return {
        f"{network}.{name}": weight
        for network, model in zip(("encoder", "generator", "discriminator"), models)
        for name, weight in model.weights().items()
    }


def save_checkpoint(path, models, seed):
    metadata = json.dumps({"noise_std": NOISE_STD, "seed": seed})
    np.savez_compressed(path, metadata=metadata, **model_weights(models))


def load_checkpoint(path, models):
    with np.load(path, allow_pickle=False) as checkpoint:
        for name, weight in model_weights(models).items():
            value = checkpoint[name]
            if weight.shape != value.shape:
                raise ValueError(f"Неверный размер параметра {name} в {path}")
            weight[:] = value
        return json.loads(checkpoint["metadata"].item())


def reconstruct(encoder, generator, noisy):
    mean, _ = np.split(encoder.forward(noisy), 2, axis=1)
    return sigmoid(generator.forward(mean))


def restoration_scores(error, pixels):
    mse = float(error / pixels)
    return {"mse": mse, "psnr_db": -10 * math.log10(max(mse, 1e-12))}


def evaluate(images, models, batch_size, noise_std, seed):
    encoder, generator, _ = models
    noise_rng = np.random.default_rng(seed + 1)
    noisy_error = restored_error = 0
    examples = None

    for start in range(0, len(images), batch_size):
        clean = normalize(images[start:start + batch_size])
        noisy = add_noise(clean, noise_std, noise_rng)
        restored = reconstruct(encoder, generator, noisy)
        noisy_error += np.square(noisy - clean).sum(dtype=np.float64)
        restored_error += np.square(restored - clean).sum(dtype=np.float64)
        if examples is None:
            examples = tuple(batch[:10] for batch in (clean, noisy, restored))

    metrics = {
        "test_images": len(images),
        "noise_std": noise_std,
        "noisy": restoration_scores(noisy_error, images.size),
        "restored": restoration_scores(restored_error, images.size),
    }
    return metrics, examples


def save_examples(examples, path, labels=("Чистые", "Зашумлённые", "Восстановленные")):
    rows = len(examples)
    count = len(examples[0])
    fig, axes = plt.subplots(
        rows, count, figsize=(1.5 + count, 4 * rows / 3), squeeze=False, layout="constrained"
    )
    for row, (label, batch) in enumerate(zip(labels, examples)):
        axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center")
        for axis, image in zip(axes[row], batch):
            axis.imshow(image.squeeze(-1), cmap="gray", vmin=0, vmax=1)
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Восстановление зашумлённых изображений MNIST с VAE-GAN на NumPy.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "numpy")
    parser.add_argument("--checkpoint", type=Path, help="Оценить сохранённые веса без обучения")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.seed < 0:
        parser.error("Число эпох, размер батча и число потоков должны быть положительными, а seed — неотрицательным.")

    rng = np.random.default_rng(args.seed)
    models = Encoder(rng), Generator(rng), Discriminator(rng)
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads, user_api="blas"):
        if args.checkpoint is None:
            images = load_images(train=True)
            print(f"Обучение на CPU, {len(images)} изображений MNIST", flush=True)
            train(images, models, args, rng)
            save_checkpoint(args.output / "checkpoint.npz", models, args.seed)
            noise_std, seed = NOISE_STD, args.seed
        else:
            config = load_checkpoint(args.checkpoint, models)
            noise_std, seed = config["noise_std"], config["seed"]

        test = load_images(train=False)
        metrics, examples = evaluate(test, models, args.batch_size, noise_std, seed)

    save_examples(examples, args.output / "reconstructions.png")
    report = json.dumps(metrics, indent=2)
    (args.output / "metrics.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
