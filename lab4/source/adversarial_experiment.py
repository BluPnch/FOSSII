import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import time
from types import SimpleNamespace

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from threadpoolctl import threadpool_limits

from data import ROOT, load_images
from main import (
    ADVERSARIAL_WEIGHT, KL_WEIGHT, NOISE_STD, add_noise, normalize, reconstruct,
    restoration_scores, save_checkpoint, save_examples, train,
)
from models import Discriminator, Encoder, Generator, LATENT_DIM
from padding_experiment import SUBSET_SEED, TEST_NOISE_SEED


VARIANTS = {"vae": 0.0, "vae_gan": ADVERSARIAL_WEIGHT}
NAMES = ("VAE", "VAE-GAN")
TEST_BATCH_SIZE = 256
coordinates = np.arange(-5, 6)
SSIM_WINDOW = np.exp(-0.5 * (coordinates / 1.5)**2)
SSIM_WINDOW /= SSIM_WINDOW.sum()


def gaussian_average(images):
    windows = len(SSIM_WINDOW)
    along_width = sliding_window_view(images, windows, axis=2) @ SSIM_WINDOW
    return sliding_window_view(along_width, windows, axis=1) @ SSIM_WINDOW


def per_image_ssim(reference, restored):
    """SSIM for [0, 1] images, averaged over complete 11×11 windows."""
    reference = reference[..., 0].astype(np.float64)
    restored = restored[..., 0].astype(np.float64)
    mean_x, mean_y = gaussian_average(reference), gaussian_average(restored)
    variance_x = gaussian_average(reference**2) - mean_x**2
    variance_y = gaussian_average(restored**2) - mean_y**2
    covariance = gaussian_average(reference * restored) - mean_x * mean_y
    c1, c2 = 0.01**2, 0.03**2
    local_ssim = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x**2 + mean_y**2 + c1) * (variance_x + variance_y + c2)
    )
    return local_ssim.mean(axis=(1, 2))


def evaluate_pair(test, models, deadline):
    noise_rng = np.random.default_rng(TEST_NOISE_SEED)
    squared_errors = np.zeros(3)
    ssim_sums = np.zeros(3)
    examples = None
    for start in range(0, len(test), TEST_BATCH_SIZE):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Достигнут лимит времени исследования.")
        clean = normalize(test[start:start + TEST_BATCH_SIZE])
        noisy = add_noise(clean, NOISE_STD, noise_rng)
        restored = tuple(reconstruct(encoder, decoder, noisy) for encoder, decoder, _ in models)
        for index, image in enumerate((noisy, *restored)):
            squared_errors[index] += np.square(image - clean).sum(dtype=np.float64)
            ssim_sums[index] += per_image_ssim(clean, image).sum()
        if examples is None:
            examples = tuple(image[:10].copy() for image in (clean, noisy, *restored))
    metrics = {
        variant: {
            **restoration_scores(error, test.size),
            "ssim": float(ssim / len(test)),
        }
        for variant, error, ssim in zip(("noisy", *VARIANTS), squared_errors, ssim_sums)
    }
    return metrics, examples


def train_variant(images, args, seed, variant, weight, deadline):
    encoder_rng, decoder_rng, disc_rng, train_rng, posterior_rng, prior_rng = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(6)
    )
    models = Encoder(encoder_rng), Generator(decoder_rng), Discriminator(disc_rng) if weight else None
    output = args.output / f"seed-{seed}" / variant
    output.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(output=output, batch_size=args.batch_size, epochs=args.epochs)
    print(f"seed={seed}, {variant}, λ_adv={weight}", flush=True)
    start = time.perf_counter()
    train(
        images, models, settings, train_rng, latent_normal=posterior_rng.standard_normal,
        prior_normal=prior_rng.standard_normal, adversarial_weight=weight, deadline=deadline,
    )
    seconds = time.perf_counter() - start
    save_checkpoint(output / "checkpoint.npz", models, seed)
    return models, seconds


def save_scores(runs, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), layout="constrained")
    colors = ("#346889", "#ad582f")
    for ax, (metric, title) in zip(axes, (("mse", "MSE ↓"), ("ssim", "SSIM ↑"))):
        for run in runs:
            values = [run[variant][metric] for variant in VARIANTS]
            ax.plot((0, 1), values, "o-", color="#7b8389", alpha=0.65, linewidth=1)
        averages = [mean(run[variant][metric] for run in runs) for variant in VARIANTS]
        ax.scatter((0, 1), averages, marker="D", s=50, c=colors, edgecolor="white", zorder=3)
        ax.set(xticks=(0, 1), xticklabels=NAMES, xlim=(-0.28, 1.28), title=title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.supxlabel("Линия — один seed, ромб — среднее по seed", fontsize=10)
    fig.savefig(path, dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Парное сравнение NumPy-моделей VAE и VAE-GAN на MNIST.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-minutes", type=float, default=25)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "vae-vs-vae-gan")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or not 1 <= args.train_size <= 60_000:
        parser.error("Нужны положительные размеры, train-size от 1 до 60000.")
    if not 0 < args.max_minutes <= 30:
        parser.error("Лимит времени должен быть больше нуля и не больше 30 минут.")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Значения seed должны быть различными и неотрицательными.")

    started = time.perf_counter()
    deadline = started + 60 * args.max_minutes
    full_train = load_images(train=True)
    indices = np.random.default_rng(SUBSET_SEED).choice(len(full_train), args.train_size, replace=False)
    images = full_train[indices]
    test = load_images(train=False)
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    panels = None
    with threadpool_limits(limits=args.threads, user_api="blas"):
        for seed in args.seeds:
            try:
                trained, times = zip(*(
                    train_variant(images, args, seed, variant, weight, deadline)
                    for variant, weight in VARIANTS.items()
                ))
                metrics, examples = evaluate_pair(test, trained, deadline)
            except TimeoutError as error:
                print(error, flush=True)
                break
            if panels is None:
                panels = examples
            baseline = metrics.pop("noisy")
            for variant, seconds in zip(VARIANTS, times):
                metrics[variant]["train_seconds"] = seconds
            runs.append({"seed": seed, **metrics})
            print(
                f"seed={seed}, MSE VAE={metrics['vae']['mse']:.6f}, "
                f"VAE-GAN={metrics['vae_gan']['mse']:.6f}, "
                f"SSIM VAE={metrics['vae']['ssim']:.6f}, VAE-GAN={metrics['vae_gan']['ssim']:.6f}",
                flush=True,
            )

    if not runs:
        parser.exit(1, "Лимит времени истёк до завершения первой пары.\n")
    summary = {}
    for variant in VARIANTS:
        averages = {}
        for metric in ("mse", "ssim", "psnr_db", "train_seconds"):
            values = [run[variant][metric] for run in runs]
            averages[metric] = {"mean": mean(values), "stdev": stdev(values) if len(values) > 1 else None}
        summary[variant] = averages

    save_examples(panels, args.output / "reconstructions.png", ("Оригинал", "Зашумлённый", *NAMES))
    save_scores(runs, args.output / "metrics.png")
    report = {
        "train_images": len(images),
        "test_images": len(test),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_batch_size": TEST_BATCH_SIZE,
        "threads": args.threads,
        "seeds_requested": args.seeds,
        "subset_seed": SUBSET_SEED,
        "test_noise_seed": TEST_NOISE_SEED,
        "noise_std": NOISE_STD,
        "latent_dim": LATENT_DIM,
        "kl_weight": KL_WEIGHT,
        "adversarial_weights": VARIANTS,
        "example_seed": runs[0]["seed"],
        "noisy": baseline,
        "runs": runs,
        "summary": summary,
        "time_limit_minutes": args.max_minutes,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Пар завершено: {len(runs)}/{len(args.seeds)}, секунд: {report['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
