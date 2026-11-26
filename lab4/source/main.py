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


def train_step(
    clean, models, optimizers, rng, *, latent_normal=None, prior_normal=None,
    adversarial_weight=ADVERSARIAL_WEIGHT,
):
    encoder, generator, discriminator = models
    vae_optimizer, discriminator_optimizer = optimizers
    batch_size = len(clean)

    noisy = add_noise(clean, NOISE_STD, rng)
    
    mean, log_variance = np.split(encoder.forward(noisy, retain=True), 2, axis=1)
    variance = np.exp(log_variance)
    # z = μ + σ·ε. Шум ε из N(0, 1).
    draw_latent = rng.standard_normal if latent_normal is None else latent_normal
    offset = np.exp(0.5 * log_variance) * draw_latent(mean.shape, dtype=np.float32)
    latent = mean + offset
    if adversarial_weight:
        draw_prior = draw_latent if prior_normal is None else prior_normal
        prior = draw_prior(mean.shape, dtype=np.float32)
        latent = np.concatenate((latent, prior))
    logits = generator.forward(latent, retain=True)
    fake = sigmoid(logits)

    if adversarial_weight:
        logits_gradient, adversarial, discriminator_loss = adversarial_step(
            clean, fake, discriminator, discriminator_optimizer, adversarial_weight,
        )
    else:
        logits_gradient = np.zeros_like(logits)
        adversarial = discriminator_loss = 0.0

    # BCE с чистым оригиналом применяется к первым batch_size выходам, где лежат восстановления.
    # Добавляем её градиент к состязательному. У новых цифр есть только сигнал от дискриминатора.
    reconstruction = binary_cross_entropy(logits[:batch_size], clean)
    logits_gradient[:batch_size] += (fake[:batch_size] - clean) / clean.size
    latent_gradient = generator.backward(logits_gradient, input_grad=True)[:batch_size]

    kl = -0.5 * (1 + log_variance - mean**2 - variance).sum(axis=1).mean(dtype=np.float64)
    mean_gradient = latent_gradient + (KL_WEIGHT / batch_size) * mean
    log_variance_gradient = 0.5 * latent_gradient * offset + (0.5 * KL_WEIGHT / batch_size) * (variance - 1)
    encoder.backward(np.concatenate((mean_gradient, log_variance_gradient), axis=1))
    # Обновление весов энкодера и декодера.
    vae_optimizer.step()
    return np.array((reconstruction, kl, adversarial, discriminator_loss))


def adversarial_step(clean, fake, discriminator, optimizer, weight):
    batch_size = len(clean)
    # Фаза дискриминатора. При батче из двух цифр на вход - 2 оригинала, 2 восстановления и 2 новые цифры.
    judged = discriminator.forward(np.concatenate((clean, fake)), retain=True)
    real_scores, fake_scores = judged[:batch_size], judged[batch_size:]
    discriminator_loss = (binary_cross_entropy(real_scores, 1) + binary_cross_entropy(fake_scores, 0)) / 2
    judged_gradient = sigmoid(judged)
    judged_gradient[:batch_size] -= 1
    judged_gradient[:batch_size] *= 0.5 / len(real_scores)
    judged_gradient[batch_size:] *= 0.5 / len(fake_scores)
    discriminator.backward(judged_gradient)
    optimizer.step()

    # Фаза генератора. Дискриминатор с обновлёнными весами оценивает только синтетические изображения - 2 восстановления и 2 новые цифры.
    # Цель теперь 1. Веса дискриминатора зафиксированы
    fake_scores = discriminator.forward(fake, retain=True)
    adversarial = binary_cross_entropy(fake_scores, 1)
    adversarial_gradient = (sigmoid(fake_scores) - 1) * (weight / len(fake_scores))
    fake_gradient = discriminator.backward(adversarial_gradient, parameter_grad=False, input_grad=True)

    # Переходим от градиента по яркостям к градиенту по логитам 
    return fake_gradient * fake * (1 - fake), adversarial, discriminator_loss


def make_optimizers(models):
    encoder, generator, discriminator = models
    # Один экземпляр ручного Adam обновляет энкодер и декодер, другой обслуживает дискриминатор.
    vae_optimizer = Adam((*encoder.parameters(), *generator.parameters()), learning_rate=0.001)
    discriminator_optimizer = (Adam(discriminator.parameters(), learning_rate=0.0002) if discriminator is not None else None)
    return vae_optimizer, discriminator_optimizer


def training_batches(images, batch_size, rng):
    # На каждой эпохе перемешивается порядок цифр. батчи с яркостями в [0, 1].
    order = rng.permutation(len(images))
    for start in range(0, len(images), batch_size):
        yield normalize(images[order[start:start + batch_size]])


def train(
    images, models, args, rng, *, latent_normal=None, prior_normal=None,
    adversarial_weight=ADVERSARIAL_WEIGHT, deadline=None,
):
    optimizers = make_optimizers(models)
    with (args.output / "history.csv").open("w", encoding="utf-8", newline="") as file:
        history = csv.writer(file)
        history.writerow(("epoch", *LOSS_NAMES))
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            totals = np.zeros(len(LOSS_NAMES))
            for clean in training_batches(images, args.batch_size, rng):
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError("Достигнут лимит времени исследования.")
                losses = train_step(
                    clean, models, optimizers, rng, latent_normal=latent_normal,
                    prior_normal=prior_normal, adversarial_weight=adversarial_weight,
                )
                # Взвешиваем средние потери по числу изображений. Последний батч может быть меньше.
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
        if model is not None
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


def evaluate(images, models, batch_size, noise_std, seed, *, per_image_mse=None):
    encoder, generator, _ = models
    noise_rng = np.random.default_rng(seed + 1)
    noisy_error = restored_error = 0
    examples = None

    for start in range(0, len(images), batch_size):
        clean = normalize(images[start:start + batch_size])
        noisy = add_noise(clean, noise_std, noise_rng)
        restored = reconstruct(encoder, generator, noisy)

        noisy_error += np.square(noisy - clean).sum(dtype=np.float64)
        squared_error = np.square(restored - clean)
        restored_error += squared_error.sum(dtype=np.float64)
        if per_image_mse is not None:
            per_image_mse[start:start + len(clean)] = squared_error.mean(axis=(1, 2, 3), dtype=np.float64)

        if examples is None:
            examples = tuple(batch[:10] for batch in (clean, noisy, restored))

    metrics = {
        "test_images": len(images),
        "noise_std": noise_std,
        "noisy": restoration_scores(noisy_error, images.size),
        "restored": restoration_scores(restored_error, images.size),
    }
    return metrics, examples


def save_examples(examples, path, labels=("Чистые", "Зашумлённые", "Восстановленные"), *, error_indices=()):
    # Карта показывает |восстановление − оригинал|. Оригиналы лежат в первом ряду examples.
    errors = tuple(np.abs(examples[index] - examples[0]) for index in error_indices)
    rows = len(examples) + len(errors)
    count = len(examples[0])
    fig, axes = plt.subplots(
        rows, count, figsize=(count + (2.5 if errors else 1.5), 4 * rows / 3),
        squeeze=False, layout="constrained",
    )
    for row, (label, batch) in enumerate(zip(labels, examples)):
        axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center")
        for axis, image in zip(axes[row], batch):
            axis.imshow(image.squeeze(-1), cmap="gray", vmin=0, vmax=1)
            axis.set(xticks=[], yticks=[])
            axis.spines[:].set_visible(False)

    if errors:
        mse = np.mean(np.square(errors), axis=(2, 3, 4), dtype=np.float64)
        best_mse = mse.min(axis=0)
        for row, source_index, batch, scores in zip(axes[len(examples):], error_indices, errors, mse):
            row[0].set_ylabel(f"Ошибка {labels[source_index]}", rotation=0, ha="right", va="center")
            for axis, error, score, is_best in zip(row, batch, scores, scores == best_mse):
                colors = axis.imshow(error.squeeze(-1), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
                axis.set(xticks=[], yticks=[])
                axis.spines[:].set_visible(is_best)
                if is_best:
                    axis.spines[:].set_color("#176493")
                    axis.spines[:].set_linewidth(1.8)
                axis.set_xlabel(
                    f"{score:.5f}".replace(".", ","), fontsize=9, fontweight="bold" if is_best else "normal",
                )
        fig.colorbar(
            colors, ax=axes, label="Абсолютная разница яркостей",
            shrink=len(errors) / rows, anchor=(0, 0), pad=0.02,
        )
        fig.supxlabel("Под картами указана MSE. Рамкой выделен минимум в столбце.", fontsize=10)
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
            config = load_checkpoint(args.checkpoint, (*models[:2], None))
            noise_std, seed = config["noise_std"], config["seed"]

        test = load_images(train=False)
        metrics, examples = evaluate(test, models, args.batch_size, noise_std, seed)

    save_examples(examples, args.output / "reconstructions.png")
    report = json.dumps(metrics, indent=2)
    (args.output / "metrics.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
