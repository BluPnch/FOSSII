import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import json
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from data import ROOT, load_images
from layers import KERNEL, PADDING, STRIDE
from main import ADVERSARIAL_WEIGHT, KL_WEIGHT, NOISE_STD, train
from models import LATENT_DIM, Discriminator, Encoder, Generator
from padding_experiment import SUBSET_SEED, TEST_NOISE_SEED
from window_experiment import EVALUATION_BATCH_SIZE, evaluate_trial, save_metric_plot, write_json


ENCODER_OUTPUTS = (16, 32, 64, 128, 256)


def draw_paired_latent(training_rng, extra_rng, shape, *, dtype):
    batch_size, latent_dim = shape
    # Основной поток делает столько же выборок, сколько и в базовой модели.
    common = training_rng.standard_normal((batch_size, LATENT_DIM), dtype=dtype)
    if latent_dim <= LATENT_DIM:
        return common[:, :latent_dim]
    extra = extra_rng.standard_normal((batch_size, latent_dim - LATENT_DIM), dtype=dtype)
    return np.concatenate((common, extra), axis=1)


def trial_path(output, seed, encoder_outputs):
    return output / f"seed-{seed}-outputs-{encoder_outputs}" / "metrics.json"


def run_trial(args, seed, encoder_outputs, *, capture_image_errors=False):
    latent_dim = encoder_outputs // 2
    full_train = load_images(train=True)
    indices = np.random.default_rng(SUBSET_SEED).choice(len(full_train), args.train_size, replace=False)
    images = full_train[indices]
    test = load_images(train=False)
    encoder_seed, generator_seed, discriminator_seed, training_seed, extra_seed = np.random.SeedSequence(seed).spawn(5)
    encoder = Encoder(np.random.default_rng(encoder_seed), latent_dim=latent_dim)
    generator = Generator(np.random.default_rng(generator_seed), latent_dim=latent_dim)
    if latent_dim != LATENT_DIM:
        # Сохраняем начальные веса свёрток при изменении размеров входа линейного слоя.
        reference = Generator(np.random.default_rng(generator_seed))
        for name in ("upsample1", "upsample2"):
            layer = generator.trainable[name]
            layer.weight[:] = reference.trainable[name].weight
            layer.bias[:] = reference.trainable[name].bias
    models = encoder, generator, Discriminator(np.random.default_rng(discriminator_seed))
    training_rng = np.random.default_rng(training_seed)
    latent_normal = partial(draw_paired_latent, training_rng, np.random.default_rng(extra_seed))
    path = trial_path(args.output, seed, encoder_outputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(output=path.parent, batch_size=args.batch_size, epochs=args.epochs)
    capture_path = path.with_name("image-errors.npz") if capture_image_errors else None
    with threadpool_limits(limits=1, user_api="blas"):
        train(images, models, settings, training_rng, latent_normal=latent_normal)
        metrics = evaluate_trial(test, models, capture_path)

    result = {
        "seed": seed,
        "encoder_outputs": encoder_outputs,
        "latent_dim": latent_dim,
        "encoder_parameters": sum(weight.size for weight, _ in encoder.parameters()),
        "decoder_parameters": sum(weight.size for weight, _ in generator.parameters()),
        **metrics,
    }
    write_json(path, result)
    return result


def summarize(runs):
    summary = []
    for outputs in ENCODER_OUTPUTS:
        group = [run for run in runs if run["encoder_outputs"] == outputs]
        mse = [run["mse"] for run in group]
        summary.append({
            "encoder_outputs": outputs,
            "latent_dim": group[0]["latent_dim"],
            "encoder_parameters": group[0]["encoder_parameters"],
            "decoder_parameters": group[0]["decoder_parameters"],
            "repeats": len(group),
            "mean_mse": mean(mse),
            "std_mse": stdev(mse) if len(mse) > 1 else 0.0,
            "mean_psnr_db": mean(run["psnr_db"] for run in group),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description="Сравнение пяти размеров латентного кода энкодера.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "latent-sizes")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.jobs) < 1 or not 1 <= args.train_size <= 60_000:
        parser.error("Нужны положительные размеры, train-size от 1 до 60000.")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Значения seed должны быть различными и неотрицательными.")

    protocol = {
        "encoder_outputs": list(ENCODER_OUTPUTS),
        "train_seeds": args.seeds,
        "train_images": args.train_size,
        "test_images": 10_000,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "subset_seed": SUBSET_SEED,
        "test_noise_seed": TEST_NOISE_SEED,
        "noise_std": NOISE_STD,
        "kernel_size": KERNEL,
        "padding": PADDING,
        "stride": STRIDE,
        "kl_weight": KL_WEIGHT,
        "adversarial_weight": ADVERSARIAL_WEIGHT,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.is_file() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        parser.error("В выходном каталоге сохранён другой протокол. Укажите новый --output.")
    write_json(protocol_path, protocol)

    total = len(args.seeds) * len(ENCODER_OUTPUTS)
    runs = []
    pending = []
    for seed in args.seeds:
        for outputs in ENCODER_OUTPUTS:
            path = trial_path(args.output, seed, outputs)
            if path.is_file():
                runs.append(json.loads(path.read_text(encoding="utf-8")))
            else:
                pending.append((seed, outputs))
    print(f"Готовых прогонов {len(runs)}/{total}.", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_trial, args, seed, outputs) for seed, outputs in pending]
            for future in as_completed(futures):
                result = future.result()
                runs.append(result)
                print(f"[{len(runs)}/{total}] Выходы={result['encoder_outputs']}, seed={result['seed']}, "
                      f"MSE={result['mse']:.6f}", flush=True)

    runs.sort(key=lambda run: (run["encoder_outputs"], run["seed"]))
    summary = summarize(runs)
    save_metric_plot(
        args.output / "mse-by-encoder-output.png", runs, summary, field="encoder_outputs",
        labels=[str(outputs) for outputs in ENCODER_OUTPUTS],
        title="Число выходов энкодера и ошибка восстановления MNIST",
        xlabel="Число выходов энкодера", y_decimals=4,
    )
    write_json(args.output / "metrics.json", {**protocol, "runs": runs, "summary": summary})


if __name__ == "__main__":
    main()
