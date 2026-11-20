"""Plot the hyperparameter studies from the saved validation results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RESULTS = Path(__file__).resolve().parents[1] / "results"
KINDS = ("RNN", "GRU", "LSTM")


def plot_factor(ax, relative_path, factor, metric, title, deviation=None):
    table = pd.read_csv(RESULTS / relative_path)
    values = sorted(table[factor].unique())
    for kind in KINDS:
        rows = table[table.model == kind].sort_values(factor)
        spread = rows[deviation] if deviation else None
        ax.errorbar(rows[factor], rows[metric], yerr=spread,
                    fmt="o-", capsize=3, label=kind)
    ax.set(title=title, ylabel="MSE, (кВт·ч)²", xticks=values)
    ax.set_xticklabels([str(value).replace(".", ",") for value in values])
    ax.grid(axis="y", alpha=0.25)


def plot_stopping(ax):
    table = pd.read_csv(RESULTS / "ideal/stopping_summary.csv")
    grid = table.pivot(index=["budget", "min_epochs"], columns="patience",
                       values="mean_validation_MSE")
    ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set(title="Ранняя остановка · среднее по 33 траекториям",
           xlabel="Patience, эпохи", ylabel="Бюджет / минимум эпох",
           xticks=range(len(grid.columns)), yticks=range(len(grid.index)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticklabels([f"{budget} / {minimum}" for budget, minimum in grid.index])
    midpoint = (grid.min().min() + grid.max().max()) / 2
    for row, values in enumerate(grid.to_numpy()):
        for col, value in enumerate(values):
            ax.text(col, row, f"{value:.2f}".replace(".", ","), ha="center", va="center",
                    color="white" if value > midpoint else "black")


def main():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    panels = (
        ("hidden_sizes/summary.csv", "hidden_size", "mean_mse", "Размер состояния · 4 seed", "sd_mse"),
        ("batch_sizes/summary.csv", "batch_size", "mean_mse", "Размер батча · 4 seed", "sd_mse"),
        ("investigations/learning_rate.csv", "learning_rate", "MSE", "Скорость обучения · seed 42"),
        ("investigations/seed.csv", "seed", "MSE", "Случайные seed базового запуска"),
        ("investigations/epochs.csv", "budget", "validation_MSE", "Бюджет эпох · базовые траектории"),
    )
    for ax, panel in zip(axes.flat, panels):
        plot_factor(ax, *panel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncol=3)
    plot_stopping(axes[1, 2])
    fig.savefig(RESULTS / "investigations/parameter_overview.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
