import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from data import ROOT, load_images
from layers import KERNEL, PADDING, STRIDE, conv_output_size
from main import NOISE_STD, evaluate, train
from models import Discriminator, Encoder, Generator
from padding_experiment import SUBSET_SEED, TEST_NOISE_SEED


WINDOW_SIZES = tuple(range(1, 8))
FIXED_PADDING_WINDOW_SIZES = tuple(range(2, 7))
STRIDE_VALUES = (1, 2, 3, 4)
FIXED_PADDING = 1
EVALUATION_BATCH_SIZE = 256


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def trial_path(output, seed, value, *, vary_stride=False):
    prefix = "s" if vary_stride else "k"
    return output / f"seed-{seed}-{prefix}{value}" / "metrics.json"


def evaluate_trial(test, models, capture_path=None):
    image_mse = None if capture_path is None else np.empty(len(test), dtype=np.float64)
    metrics, examples = evaluate(
        test, models, EVALUATION_BATCH_SIZE, NOISE_STD, TEST_NOISE_SEED - 1, per_image_mse=image_mse,
    )
    if capture_path is not None:
        temporary = capture_path.with_suffix(".tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(file, mse=image_mse, restored=examples[2])
        temporary.replace(capture_path)
    return metrics["restored"]


def run_trial(args, seed, value, *, capture_image_errors=False):
    full_train = load_images(train=True)
    indices = np.random.default_rng(SUBSET_SEED).choice(len(full_train), args.train_size, replace=False)
    images = full_train[indices]
    test = load_images(train=False)
    encoder_seed, generator_seed, discriminator_seed, training_seed = np.random.SeedSequence(seed).spawn(4)
    if args.vary_stride:
        size, padding, stride = KERNEL, PADDING, value
    else:
        size, stride = value, STRIDE
        padding = FIXED_PADDING if args.fixed_padding else (size - 1) // 2
    map_size = conv_output_size(14, padding, size, stride)
    encoder = Encoder(
        np.random.default_rng(encoder_seed), second_padding=padding, second_kernel_size=size, second_stride=stride,
    )
    # Для карт одного размера берём одинаковую инициализацию линейного слоя.
    if size != KERNEL:
        reference_padding = map_size - conv_output_size(14, padding=0)
        reference = Encoder(np.random.default_rng(encoder_seed), second_padding=reference_padding).trainable["posterior"]
        posterior = encoder.trainable["posterior"]
        posterior.weight[:] = reference.weight
        posterior.bias[:] = reference.bias
    models = encoder, Generator(np.random.default_rng(generator_seed)), Discriminator(np.random.default_rng(discriminator_seed))
    metrics_path = trial_path(args.output, seed, value, vary_stride=args.vary_stride)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(output=metrics_path.parent, batch_size=args.batch_size, epochs=args.epochs)
    capture_path = metrics_path.with_name("image-errors.npz") if capture_image_errors else None
    with threadpool_limits(limits=1, user_api="blas"):
        train(images, models, settings, np.random.default_rng(training_seed))
        metrics = evaluate_trial(test, models, capture_path)

    conv2 = encoder.trainable["conv2"]
    result = {
        "seed": seed,
        "kernel_size": size,
        "padding": padding,
        "map_size": map_size,
        "convolution_parameters": conv2.weight.size + conv2.bias.size,
        **metrics,
    }
    if args.vary_stride:
        result["stride"] = stride
        result["encoder_parameters"] = sum(weight.size for weight, _ in encoder.parameters())
    write_json(metrics_path, result)
    return result


def summarize(runs, variants, *, field="kernel_size"):
    summary = []
    for value in variants:
        group = [run for run in runs if run[field] == value]
        mse = [run["mse"] for run in group]
        result = {
            "kernel_size": group[0]["kernel_size"],
            "padding": group[0]["padding"],
            "map_size": group[0]["map_size"],
            "convolution_parameters": group[0]["convolution_parameters"],
            "repeats": len(group),
            "mean_mse": mean(mse),
            "std_mse": stdev(mse) if len(mse) > 1 else 0.0,
            "mean_psnr_db": mean(run["psnr_db"] for run in group),
        }
        if field == "stride":
            result["stride"] = value
            result["encoder_parameters"] = group[0]["encoder_parameters"]
        summary.append(result)
    return summary


def save_metric_plot(path, runs, summary, *, field, labels, title, xlabel, y_decimals=3):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    positions = {item[field]: index for index, item in enumerate(summary)}
    fig, ax = plt.subplots(figsize=(8.4, 4.6), layout="constrained")
    ax.scatter(
        [positions[run[field]] for run in runs], [run["mse"] for run in runs],
        color="#6c7d92", s=23, alpha=.55, zorder=3, label="Отдельный повтор",
    )
    ax.errorbar(
        range(len(summary)), [item["mean_mse"] for item in summary],
        yerr=[item["std_mse"] for item in summary], color="#165d97",
        marker="o", markersize=5, capsize=4, linewidth=1.8, zorder=4,
        label="Среднее ± стандартное отклонение",
    )
    ax.set_title(title, loc="left", pad=12)
    ax.set(xlabel=xlabel, ylabel="MSE на тестовой выборке")
    ax.set_xticks(range(len(summary)), labels)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.{y_decimals}f}".replace(".", ",")))
    ax.grid(axis="y", color="#dce2e9", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_plot(path, runs, summary, *, fixed_padding=False):
    sizes = [item["kernel_size"] for item in summary]
    title = "Размер окна и ошибка восстановления MNIST"
    if fixed_padding:
        title += f" при P={FIXED_PADDING}"
    save_metric_plot(
        path, runs, summary, field="kernel_size", labels=[f"{size}×{size}" for size in sizes],
        title=title, xlabel="Размер окна второй свёртки",
    )


def main():
    parser = argparse.ArgumentParser(description="Сравнение размеров окна и шага второй свёртки энкодера.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--jobs", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fixed-padding", action="store_true", help="Пять окон при P=1, десять повторов по умолчанию.")
    mode.add_argument("--vary-stride", action="store_true", help="Шаги 1–4 при окне 4×4, десять повторов по умолчанию.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.vary_stride:
        variants, field, directory = STRIDE_VALUES, "stride", "stride-sizes"
    elif args.fixed_padding:
        variants, field, directory = FIXED_PADDING_WINDOW_SIZES, "kernel_size", "window-sizes-p1"
    else:
        variants, field, directory = WINDOW_SIZES, "kernel_size", "window-sizes"
    if args.seeds is None:
        args.seeds = list(range(42, 52 if args.fixed_padding or args.vary_stride else 49))
    if args.output is None:
        args.output = ROOT / "results" / directory
    if min(args.epochs, args.batch_size, args.jobs) < 1 or not 1 <= args.train_size <= 60_000:
        parser.error("Нужны положительные размеры, train-size от 1 до 60000.")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Значения seed должны быть различными и неотрицательными.")

    protocol = {
        "kernel_sizes": [KERNEL] if args.vary_stride else list(variants),
        "train_seeds": args.seeds,
        "train_images": args.train_size,
        "test_images": 10_000,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "subset_seed": SUBSET_SEED,
        "test_noise_seed": TEST_NOISE_SEED,
        "noise_std": NOISE_STD,
    }
    if args.fixed_padding:
        protocol["fixed_padding"] = FIXED_PADDING
    if args.vary_stride:
        protocol["strides"] = list(STRIDE_VALUES)
        protocol["fixed_padding"] = PADDING
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.is_file() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        parser.error("В выходном каталоге сохранён другой протокол. Укажите новый --output.")
    write_json(protocol_path, protocol)

    total = len(args.seeds) * len(variants)
    runs = []
    pending = []
    for seed in args.seeds:
        for value in variants:
            path = trial_path(args.output, seed, value, vary_stride=args.vary_stride)
            if path.is_file():
                runs.append(json.loads(path.read_text(encoding="utf-8")))
            else:
                pending.append((seed, value))
    print(f"Готовых прогонов {len(runs)}/{total}.", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_trial, args, seed, value) for seed, value in pending]
            label = "S" if args.vary_stride else "K"
            for future in as_completed(futures):
                result = future.result()
                runs.append(result)
                print(f"[{len(runs)}/{total}] {label}={result[field]}, seed={result['seed']}, "
                      f"MSE={result['mse']:.6f}", flush=True)

    runs.sort(key=lambda run: (run[field], run["seed"]))
    summary = summarize(runs, variants, field=field)
    if args.vary_stride:
        save_metric_plot(
            args.output / "mse-by-stride.png", runs, summary, field=field, labels=[str(stride) for stride in variants],
            title="Шаг свёртки и ошибка восстановления MNIST", xlabel="Шаг второй свёртки", y_decimals=4,
        )
    else:
        save_plot(args.output / "mse-by-window.png", runs, summary, fixed_padding=args.fixed_padding)
    write_json(args.output / "metrics.json", {**protocol, "runs": runs, "summary": summary})


if __name__ == "__main__":
    main()
