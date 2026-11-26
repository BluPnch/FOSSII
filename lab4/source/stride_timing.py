import argparse
from itertools import chain
import json
import os
import platform
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from data import ROOT, load_images
from main import add_noise, evaluate, make_optimizers, normalize, reconstruct, train_step, training_batches
from models import Discriminator, Encoder, Generator
from window_experiment import write_json


ROUNDS = 10
WARMUP_STEPS = 4
INFERENCE_IMAGES = 128


def make_training_run(images, config, stride, seed):
    encoder_seed, generator_seed, discriminator_seed, training_seed = np.random.SeedSequence(seed).spawn(4)
    models = (
        Encoder(np.random.default_rng(encoder_seed), second_stride=stride),
        Generator(np.random.default_rng(generator_seed)),
        Discriminator(np.random.default_rng(discriminator_seed)),
    )
    rng = np.random.default_rng(training_seed)
    batches = chain.from_iterable(
        training_batches(images, config["batch_size"], rng) for _ in range(config["epochs"])
    )
    return models, make_optimizers(models), rng, batches


def time_updates(run, steps):
    models, optimizers, rng, batches = run
    elapsed = 0.0
    for _ in range(steps):
        clean = next(batches)
        started = perf_counter()
        losses = train_step(clean, models, optimizers, rng)
        elapsed += perf_counter() - started
        if not np.isfinite(losses).all():
            raise FloatingPointError("Неконечная функция потерь в замере времени.")
    return elapsed


def time_reconstruction(models, batches):
    encoder, generator, _ = models
    started = perf_counter()
    for batch in batches:
        reconstruct(encoder, generator, batch)
    elapsed = perf_counter() - started
    return 1000 * elapsed / sum(len(batch) for batch in batches)


def summarize(values):
    return {"mean": mean(values), "std": stdev(values)}


def save_plot(path, summary):
    import matplotlib.pyplot as plt

    fig, (inference, training) = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    fig.suptitle("Время восстановления и обучения при разных шагах")
    strides = [row["stride"] for row in summary]
    for field, label, color in (
        ("single_image_ms", "По одной цифре", "#165d97"),
        ("batched_image_ms", "В батче из 32", "#af6020"),
    ):
        inference.errorbar(
            strides, [row[field]["mean"] for row in summary], yerr=[row[field]["std"] for row in summary],
            color=color, marker="o", markersize=4, capsize=4, label=label,
        )
    training.errorbar(
        strides, [row["training_batch_ms"]["mean"] for row in summary],
        yerr=[row["training_batch_ms"]["std"] for row in summary],
        color="#165d97", marker="o", markersize=4, capsize=4,
    )
    inference.set(title="Восстановление", ylabel="мс на цифру")
    training.set(title="Обучение", ylabel="мс на батч из 32")
    inference.legend(frameon=False, fontsize=9)
    for axis in (inference, training):
        axis.set(xlabel="Шаг второй свёртки", xticks=strides)
        axis.grid(axis="y", color="#dce2e9", linewidth=.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Замер времени восстановления и обучения для шагов 1–4.")
    parser.add_argument("--quality", type=Path, default=ROOT / "results" / "stride-sizes" / "metrics.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "stride-timing")
    args = parser.parse_args()
    config = json.loads(args.quality.read_text(encoding="utf-8"))
    strides = config["strides"]
    batch_size = config["batch_size"]
    if (strides, config["kernel_sizes"], config["fixed_padding"], batch_size) != ([1, 2, 3, 4], [4], 1, 32):
        parser.error("Нужны метрики исследования шагов 1–4 с K=4, P=1 и батчем 32.")
    seed = config["train_seeds"][0]
    full_train = load_images(train=True)
    indices = np.random.default_rng(config["subset_seed"]).choice(len(full_train), config["train_images"], replace=False)
    images = full_train[indices]
    test = load_images(train=False)
    noisy = add_noise(
        normalize(test[:INFERENCE_IMAGES]), config["noise_std"], np.random.default_rng(config["test_noise_seed"]),
    )
    singles = np.split(noisy, len(noisy))
    batches = np.split(noisy, len(noisy) // batch_size)
    steps = config["epochs"] * ((len(images) + batch_size - 1) // batch_size)
    block_size, extra = divmod(steps, ROUNDS)
    step_counts = [block_size + (round_index < extra) for round_index in range(ROUNDS)]
    order_rng = np.random.default_rng(316)
    series = {stride: {"train_seconds": [], "single_image_ms": [], "batched_image_ms": []} for stride in strides}
    summary = []

    with threadpool_limits(limits=1, user_api="blas"):
        blas = [{key: item[key] for key in ("internal_api", "version", "num_threads")}
                for item in threadpool_info() if item["user_api"] == "blas"]
        for stride in strides:
            time_updates(make_training_run(images, config, stride, seed), WARMUP_STEPS)

        runs = {stride: make_training_run(images, config, stride, seed) for stride in strides}
        for index, count in enumerate(step_counts, 1):
            for stride in order_rng.permutation(strides):
                series[stride]["train_seconds"].append(time_updates(runs[stride], count))
            print(f"Обучение: серия {index}/{ROUNDS}.", flush=True)

        for stride in strides:
            time_reconstruction(runs[stride][0], singles[:8])
            time_reconstruction(runs[stride][0], batches[:2])
        for _ in range(ROUNDS):
            for stride in order_rng.permutation(strides):
                models = runs[stride][0]
                series[stride]["single_image_ms"].append(time_reconstruction(models, singles))
                series[stride]["batched_image_ms"].append(time_reconstruction(models, batches))

        for stride, quality in zip(strides, config["summary"]):
            models = runs[stride][0]
            metrics, _ = evaluate(
                test, models, config["evaluation_batch_size"], config["noise_std"], config["test_noise_seed"] - 1,
            )
            times = series[stride]
            summary.append({
                "stride": stride,
                "map_size": quality["map_size"],
                "encoder_parameters": quality["encoder_parameters"],
                "mean_mse": quality["mean_mse"],
                "timed_model_mse": metrics["restored"]["mse"],
                "single_image_ms": summarize(times["single_image_ms"]),
                "batched_image_ms": summarize(times["batched_image_ms"]),
                "training_batch_ms": summarize([1000 * seconds / count for seconds, count in zip(times["train_seconds"], step_counts)]),
                "total_training_seconds": sum(times["train_seconds"]),
            })

    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "metrics.json", {
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "numpy_version": np.__version__,
        "blas": blas,
        "seed": seed,
        "batch_size": batch_size,
        "train_images": len(images),
        "epochs": config["epochs"],
        "inference_images_per_round": len(noisy),
        "warmup_training_steps": WARMUP_STEPS,
        "steps_per_round": step_counts,
        "series": series,
        "summary": summary,
    })
    save_plot(args.output / "stride-timing.png", summary)


if __name__ == "__main__":
    main()
