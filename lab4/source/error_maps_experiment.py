import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import latent_experiment
import window_experiment
from data import ROOT, load_images
from layers import KERNEL, PADDING, STRIDE
from main import NOISE_STD, normalize
from models import LATENT_DIM
from padding_experiment import SUBSET_SEED, TEST_NOISE_SEED, compare_images, save_error_maps
from window_experiment import EVALUATION_BATCH_SIZE, write_json


STUDIES = {
    "window-sizes": ("kernel_size", "Окно", "Разные размеры окна"),
    "window-sizes-p1": ("kernel_size", "Окно", "Разные размеры окна при P=1"),
    "stride-sizes": ("stride", "Шаг", "Разные шаги второй свёртки"),
    "latent-sizes": ("encoder_outputs", "Выходы", "Разное число выходов энкодера"),
}
SHARED_SETTINGS = ("epochs", "batch_size", "train_images", "test_images", "evaluation_batch_size",
                   "subset_seed", "test_noise_seed", "noise_std")


def trial_key(run):
    return (run["seed"], run.get("kernel_size", KERNEL), run.get("padding", PADDING),
            run.get("stride", STRIDE), run.get("latent_dim", LATENT_DIM))


def trial_location(output, study, run):
    directory = output / "replay" / study
    if study == "latent-sizes":
        return latent_experiment.trial_path(directory, run["seed"], run["encoder_outputs"])
    vary_stride = study == "stride-sizes"
    value = run[STUDIES[study][0]]
    return window_experiment.trial_path(directory, run["seed"], value, vary_stride=vary_stride)


def verify_trial(run, expected):
    if not math.isclose(run["mse"], expected["mse"], rel_tol=1e-8, abs_tol=1e-12):
        raise ValueError(f"Повторный запуск {trial_key(expected)} дал MSE {run['mse']} вместо {expected['mse']}.")


def replay_trial(output, study, settings, expected):
    args = SimpleNamespace(
        output=output / "replay" / study, epochs=settings["epochs"],
        batch_size=settings["batch_size"], train_size=settings["train_images"],
        fixed_padding=study == "window-sizes-p1", vary_stride=study == "stride-sizes",
    )
    runner = latent_experiment.run_trial if study == "latent-sizes" else window_experiment.run_trial
    record = runner(args, expected["seed"], expected[STUDIES[study][0]], capture_image_errors=True)
    verify_trial(record, expected)
    return trial_key(expected)


def load_capture(path):
    with np.load(path.with_name("image-errors.npz"), allow_pickle=False) as data:
        return data["mse"], data["restored"]


def save_comparison(directory, values, measures, field, runs):
    seeds = sorted({run["seed"] for run in runs})
    averaged = measures.mean(axis=0)
    summary = compare_images(averaged, values)
    write_json(directory / "comparison.json", {
        "field": field,
        "variants": values,
        "seeds": seeds,
        "test_images": averaged.shape[1],
        "mean_over_seeds": summary,
        "by_seed": {str(seed): compare_images(scores, values) for seed, scores in zip(seeds, measures)},
    })
    header = ",".join(["test_index", *(f"mse_{field}_{value}" for value in values)])
    np.savetxt(
        directory / "per-image-mse.csv", np.column_stack((np.arange(averaged.shape[1]), averaged.T)),
        header=header, delimiter=",", fmt=("%d", *("%.12f",) * len(values)), comments="",
    )


def update_reports(output, reports, paths, ready, drawn, summarized, clean):
    for study, report in reports.items():
        runs = report["runs"]
        field, label, title = STUDIES[study]
        values = sorted({run[field] for run in runs})
        seed = min(report["train_seeds"])
        examples = sorted((run for run in runs if run["seed"] == seed), key=lambda run: run[field])
        directory = output / study
        if study not in drawn and all(trial_key(run) in ready for run in examples):
            directory.mkdir(exist_ok=True)
            captures = [load_capture(paths[trial_key(run)]) for run in examples]
            labels = [f"{value}×{value}" if field == "kernel_size" else f"{label} {value}" for value in values]
            save_error_maps(
                clean, [restored for _, restored in captures],
                np.stack([mse[:len(clean)] for mse, _ in captures]), directory / "error-maps.png",
                labels=labels, title=title,
            )
            drawn.add(study)
            print(f"Готовы карты ошибок: {study}.", flush=True)
        if study not in summarized and all(trial_key(run) in ready for run in runs):
            ordered = sorted(runs, key=lambda run: (run["seed"], run[field]))
            mse = np.stack([load_capture(paths[trial_key(run)])[0] for run in ordered])
            np.testing.assert_allclose(mse.mean(axis=1), [run["mse"] for run in ordered], rtol=1e-8, atol=1e-12)
            save_comparison(directory, values, mse.reshape(-1, len(values), mse.shape[1]), field, runs)
            summarized.add(study)
            print(f"Готово сравнение для всех цифр: {study}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Карты ошибок и сравнение отдельных цифр для четырёх исследований.")
    parser.add_argument("--source", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "error-maps")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("Число параллельных прогонов должно быть положительным.")

    reports = {study: json.loads((args.source / study / "metrics.json").read_text(encoding="utf-8"))
               for study in STUDIES}
    settings = {key: reports["window-sizes"][key] for key in SHARED_SETTINGS}
    if any({key: report[key] for key in SHARED_SETTINGS} != settings for report in reports.values()):
        parser.error("У исследований различаются настройки обучения или оценки.")
    if (settings["evaluation_batch_size"], settings["subset_seed"], settings["test_noise_seed"], settings["noise_std"]
            ) != (EVALUATION_BATCH_SIZE, SUBSET_SEED, TEST_NOISE_SEED, NOISE_STD):
        parser.error("Сохранённые настройки оценки отличаются от настроек воспроизведения.")

    trials = {}
    for study, report in reports.items():
        for record in report["runs"]:
            key = trial_key(record)
            if key in trials:
                verify_trial(record, trials[key][1])
            else:
                trials[key] = (study, record)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {key: trial_location(args.output, study, record) for key, (study, record) in trials.items()}
    ready = {key for key, path in paths.items() if path.is_file() and path.with_name("image-errors.npz").is_file()}
    for key in ready:
        verify_trial(json.loads(paths[key].read_text(encoding="utf-8")), trials[key][1])
    first_seed = min(key[0] for key in trials)
    pending = sorted(
        (key for key in trials if key not in ready),
        key=lambda key: (key[0] != first_seed, list(STUDIES).index(trials[key][0])),
    )
    clean = normalize(load_images(train=False)[:10])
    drawn, summarized = set(), set()
    update_reports(args.output, reports, paths, ready, drawn, summarized, clean)
    print(f"Готовых уникальных прогонов {len(ready)}/{len(trials)}.", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(replay_trial, args.output, trials[key][0], settings, trials[key][1]) for key in pending]
            for future in as_completed(futures):
                key = future.result()
                ready.add(key)
                print(f"[{len(ready)}/{len(trials)}] {trials[key][0]}, seed={key[0]}, "
                      f"K={key[1]}, P={key[2]}, S={key[3]}, L={key[4]}", flush=True)
                update_reports(args.output, reports, paths, ready, drawn, summarized, clean)


if __name__ == "__main__":
    main()
