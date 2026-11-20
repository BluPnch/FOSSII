"""Compare recurrent architectures across window, hidden, or batch sizes."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import run as study
import matplotlib.pyplot as plt
import pandas as pd


WINDOWS = (8, 16, 24, 32, 48)
SEEDS = (7, 21, 42, 84, 123)
KINDS = ("RNN", "GRU", "LSTM")
OUT = study.OUTPUT / "windows"
HIDDEN_SIZES = (16, 24, 32, 48, 64)
REPEAT_SEEDS = (7, 21, 42, 84)
HIDDEN_OUT = study.OUTPUT / "hidden_sizes"
BATCH_SIZES = (128, 192, 256, 384, 512)
BATCH_OUT = study.OUTPUT / "batch_sizes"


def summarize(records, factor, values, out, xlabel):
    runs = pd.DataFrame(
        {"model": r["kind"], factor: r["config"][factor], "seed": r["config"]["seed"],
         **r["validation"], "best_epoch": r["training"]["best_epoch"]}
        for r in records
    ).sort_values(["model", factor, "seed"])
    summary = runs.groupby(["model", factor], as_index=False).agg(
        repeats=("seed", "size"), mean_mse=("MSE", "mean"), sd_mse=("MSE", "std"),
        mean_rmse=("RMSE", "mean"), sd_rmse=("RMSE", "std"),
    )
    runs.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    for ax, kind, color in zip(axes, KINDS, ("tab:blue", "tab:orange", "tab:green")):
        points = runs[runs.model == kind]
        stats = summary[summary.model == kind]
        ax.scatter(points[factor], points.MSE, alpha=0.35, color=color, label="Individual seeds")
        ax.errorbar(stats[factor], stats.mean_mse, yerr=stats.sd_mse,
                    fmt="o-", capsize=5, color=color, label="Mean ± sample SD")
        ax.set(xticks=values, title=kind)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Validation MSE (kWh²)")
    axes[-1].legend()
    fig.supxlabel(xlabel)
    fig.savefig(out / f"{factor}_mse.png", dpi=160)
    plt.close(fig)
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--factor", choices=("window", "hidden_size", "batch_size"), default="window")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    factor = args.factor
    if factor == "window":
        values, seeds, out, xlabel = WINDOWS, SEEDS, OUT, "Window length (15-minute steps)"
    elif factor == "hidden_size":
        values, seeds, out, xlabel = HIDDEN_SIZES, REPEAT_SEEDS, HIDDEN_OUT, "Hidden size"
    else:
        values, seeds, out, xlabel = BATCH_SIZES, REPEAT_SEEDS, BATCH_OUT, "Batch size"

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {}
        for value in reversed(values):
            for seed in seeds:
                for kind in reversed(KINDS):
                    config = {**study.BASE, factor: value, "seed": seed}
                    name = f"{factor}_{value}_seed_{seed}"
                    run_out = out
                    if factor != "window" and value == study.BASE[factor]:
                        name = f"window_{study.BASE['window']}_seed_{seed}"
                        run_out = OUT
                    job = pool.submit(study.run_one, name, kind, config, 100, run_out)
                    jobs[job] = (kind, value, seed)
        for job in as_completed(jobs):
            records.append(job.result())
            kind, value, seed = jobs[job]
            print(f"{len(records)}/{len(jobs)} completed: {kind}, {factor}={value}, seed={seed}", flush=True)
    summarize(records, factor, values, out, xlabel)


if __name__ == "__main__":
    main()
