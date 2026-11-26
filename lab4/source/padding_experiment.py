import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from data import ROOT, load_images
from layers import conv_output_size
from main import NOISE_STD, evaluate, save_examples, train
from models import Discriminator, Encoder, Generator


PADDINGS = (0, 1, 2)
SUBSET_SEED = 1729
TEST_NOISE_SEED = 43


def run(images, test, args, seed, padding):
    encoder_rng, generator_rng, discriminator_rng, training_rng = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)
    )
    encoder = Encoder(encoder_rng, second_padding=padding)
    models = encoder, Generator(generator_rng), Discriminator(discriminator_rng)
    output = args.output / f"seed-{seed}-p{padding}"
    output.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(output=output, batch_size=args.batch_size, epochs=args.epochs)
    print(f"seed={seed}, P={padding}", flush=True)
    train(images, models, settings, training_rng)
    image_mse = np.empty(len(test), dtype=np.float64)
    metrics, examples = evaluate(
        test, models, args.batch_size, NOISE_STD, TEST_NOISE_SEED - 1, per_image_mse=image_mse,
    )
    result = {
        "seed": seed,
        "padding": padding,
        "map_size": conv_output_size(14, padding),
        "encoder_parameters": sum(weight.size for weight, _ in encoder.parameters()),
        **metrics["restored"],
    }
    print(f"MSE={result['mse']:.6f}, PSNR={result['psnr_db']:.3f} dB", flush=True)
    return result, metrics["noisy"], examples, image_mse


def compare_images(mse, variants=PADDINGS):
    best = mse == mse.min(axis=0)
    single_winner = best.sum(axis=0) == 1
    return {
        "wins": {str(variant): int(np.count_nonzero(row & single_winner)) for variant, row in zip(variants, best)},
        "ties": int(np.count_nonzero(~single_winner)),
    }


def save_error_maps(clean, restored, mse, path, *, labels=None, title="Абсолютная ошибка восстановления"):
    import matplotlib.pyplot as plt

    count = len(clean)
    labels = [f"P={padding}" for padding in PADDINGS] if labels is None else labels
    fig, axes = plt.subplots(
        1 + len(labels), count, figsize=(max(5, 1.6 + count * 1.12), 5.8 + 1.12 * (len(labels) - 3)),
        squeeze=False, layout="constrained",
    )
    fig.suptitle(title)
    fig.supxlabel("Под картами — MSE. Рамкой отмечен минимум.", fontsize=10)
    axes[0, 0].set_ylabel("Оригинал", rotation=0, ha="right", va="center", labelpad=8)
    for axis, image in zip(axes[0], clean):
        axis.imshow(image[..., 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set(xticks=[], yticks=[])
        axis.spines[:].set_visible(False)

    best_mse = mse.min(axis=0)
    for index, (label, images) in enumerate(zip(labels, restored)):
        row = axes[index + 1]
        row[0].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=8)
        error = np.abs(images - clean)
        for column, axis in enumerate(row):
            colors = axis.imshow(error[column, ..., 0], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            axis.set(xticks=[], yticks=[])
            is_best = mse[index, column] == best_mse[column]
            axis.spines[:].set_visible(is_best)
            if is_best:
                axis.spines[:].set_color("#176493")
                axis.spines[:].set_linewidth(1.8)
            label = f"{mse[index, column]:.4f}".replace(".", ",")
            axis.set_xlabel(label, fontsize=9, labelpad=3, fontweight="bold" if is_best else "normal")
    fig.colorbar(colors, ax=axes, label="Абсолютная разница яркостей", shrink=.72, pad=.02)
    fig.savefig(path, dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Сравнение нулевого дополнения во второй свёртке энкодера.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "padding")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or not 1 <= args.train_size <= 60_000:
        parser.error("Нужны положительные размеры, train-size от 1 до 60000.")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Значения seed должны быть различными и неотрицательными.")

    full_train = load_images(train=True)
    indices = np.random.default_rng(SUBSET_SEED).choice(len(full_train), args.train_size, replace=False)
    images = full_train[indices]
    test = load_images(train=False)
    runs = []
    panels = []
    image_scores = []
    with threadpool_limits(limits=args.threads, user_api="blas"):
        for seed in args.seeds:
            for padding in PADDINGS:
                result, noisy_metrics, examples, image_mse = run(images, test, args, seed, padding)
                runs.append(result)
                image_scores.append(image_mse)
                if seed == args.seeds[0]:
                    if not panels:
                        panels.extend(examples[:2])
                    panels.append(examples[2])

    summaries = []
    for padding in PADDINGS:
        group = [result for result in runs if result["padding"] == padding]
        summaries.append({
            "padding": padding,
            "map_size": group[0]["map_size"],
            "encoder_parameters": group[0]["encoder_parameters"],
            "mean_mse": mean(result["mse"] for result in group),
            "std_mse": pstdev(result["mse"] for result in group),
            "mean_psnr_db": mean(result["psnr_db"] for result in group),
        })
    image_scores = np.stack(image_scores).reshape(len(args.seeds), len(PADDINGS), len(test))
    mean_image_mse = image_scores.mean(axis=0)
    example_mse = image_scores[0, :, :len(panels[0])]
    report = {
        "train_images": len(images),
        "test_images": len(test),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "subset_seed": SUBSET_SEED,
        "test_noise_seed": TEST_NOISE_SEED,
        "noise_std": NOISE_STD,
        "noisy": noisy_metrics,
        "runs": runs,
        "summary": summaries,
        "per_image_comparison": {
            "mean_over_seeds": compare_images(mean_image_mse),
            "by_seed": {str(seed): compare_images(scores) for seed, scores in zip(args.seeds, image_scores)},
            "example_seed": args.seeds[0],
            "example_mse_by_padding": {str(padding): row.tolist() for padding, row in zip(PADDINGS, example_mse)},
        },
    }
    save_examples(panels, args.output / "reconstructions.png", ("Оригинал", "Зашумлённый", "P=0", "P=1", "P=2"))
    save_error_maps(panels[0], panels[2:], example_mse, args.output / "error-maps.png")
    np.savetxt(
        args.output / "per-image-mse.csv", np.column_stack((np.arange(len(test)), mean_image_mse.T)),
        header="test_index,mse_p0,mse_p1,mse_p2", delimiter=",", fmt=("%d", "%.12f", "%.12f", "%.12f"), comments="",
    )
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
