"""Combine one-factor winners, replicate across seeds and export a validation winner."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil

import run as study
import numpy as np
import pandas as pd

ROOT = study.ROOT
SEEDS = (7, 42, 123)
KINDS = ("RNN", "GRU", "LSTM")


def existing_records():
    records = []
    for name, config in study.configurations().items():
        for kind in KINDS:
            # Validate cache signatures through the original runner.
            records.append(study.run_one(name, kind, config, 100, study.STUDY_OUT))
    return records


def stopping_study(records, out):
    """Replay stopping rules on recorded trajectories, explicitly marking truncation."""
    rows = []
    for r in records:
        h = pd.read_csv(study.STUDY_OUT / "runs" / r["name"] /
                        r["kind"].lower() / f"{r['kind'].lower()}_history.csv")
        for budget in (30, 60, 100):
            for minimum in (0, 30, 60):
                if minimum > budget:
                    continue
                for patience in (7, 15, 25):
                    best, stale, stopped, best_epoch = float("inf"), 0, False, 0
                    prefix = h[h.epoch <= budget]
                    for row in prefix.itertuples():
                        if row.val_mse_scaled < best:
                            best, stale, best_epoch = row.val_mse_scaled, 0, row.epoch
                        else:
                            stale += 1
                        if row.epoch >= minimum and stale >= patience:
                            stopped = True
                            break
                    complete = stopped or row.epoch == budget
                    rows.append(dict(model=r["kind"], run=r["name"], budget=budget,
                                     min_epochs=minimum, patience=patience, epochs_used=row.epoch,
                                     best_epoch=best_epoch, complete=complete,
                                     MSE=best * r["data"]["std"] ** 2))
    table = pd.DataFrame(rows)
    table.to_csv(out / "stopping_rules.csv", index=False)
    # Only paired complete observations enter comparisons of stopping policies.
    complete_runs = table.groupby(["model", "run"]).complete.all()
    paired = table.set_index(["model", "run"]).loc[complete_runs[complete_runs].index].reset_index()
    summary = paired.groupby(["budget", "min_epochs", "patience"]).agg(
        mean_validation_MSE=("MSE", "mean"), mean_epochs=("epochs_used", "mean"),
        runs=("run", "size")).reset_index()
    summary.to_csv(out / "stopping_summary.csv", index=False)
    return summary


def finish(records, new_records, combinations, out):
    candidates = []
    for kind in KINDS:
        for family in ("base", "combined"):
            group = ([r for r in records if r["kind"] == kind and
                      r["name"] in ("base", "seed_7", "seed_123")] if family == "base"
                     else [r for r in new_records if r["kind"] == kind])
            for r in group:
                candidates.append(dict(model=kind, family=family, seed=r["config"]["seed"],
                                       MSE=r["validation"]["MSE"], RMSE=r["validation"]["RMSE"],
                                       run=r["name"]))
    trials = pd.DataFrame(candidates)
    trials.to_csv(out / "replications.csv", index=False)
    summary = trials.groupby(["model", "family"]).agg(
        mean_MSE=("MSE", "mean"), std_MSE=("MSE", "std"),
        mean_RMSE=("RMSE", "mean"), std_RMSE=("RMSE", "std"), seeds=("seed", "size")).reset_index()
    assert summary.seeds.eq(3).all()
    summary.to_csv(out / "selection.csv", index=False)
    winner = summary.sort_values(["mean_MSE", "model", "family"]).iloc[0]
    # Seed is fixed in advance; do not pick a lucky initialization.
    chosen = next(r for r in (records if winner.family == "base" else new_records)
                  if r["kind"] == winner.model and r["config"]["seed"] == 42
                  and (winner.family != "base" or r["name"] == "base"))
    source = (study.STUDY_OUT if winner.family == "base" else out)
    folder = source / "runs" / chosen["name"] / chosen["kind"].lower()
    checkpoint_path = folder / f"{chosen['kind'].lower()}.npz"
    model, _ = study.RecurrentRegressor.load(checkpoint_path)
    datasets, frame, info = study.prepared(chosen["config"])
    ds = datasets["test"]
    prediction = study.predict(model, ds, chosen["config"]["batch_size"])
    prediction = prediction * info["std"] + info["mean"]
    actual = frame.Usage_kWh.iloc[ds.start:ds.stop].to_numpy()
    scores = study.metrics(actual, prediction)
    original = pd.read_csv(ROOT / "results/baseline_30/predictions.csv")
    assert np.allclose(original.actual, actual)
    assert pd.to_datetime(original.date).equals(pd.Series(frame.date.iloc[ds.start:ds.stop].to_numpy()))
    comparison = []
    for kind in KINDS:
        comparison.append(dict(model=f"Original {kind}", **study.metrics(actual, original[kind])))
    comparison.append(dict(model=f"Ideal {chosen['kind']}", **scores))
    table = pd.DataFrame(comparison)
    table.to_csv(out / "comparison.csv", index=False)
    previous = pd.read_csv(ROOT / "results/metrics.csv")
    previous.to_csv(out / "previous_selection_metrics.csv", index=False)
    pd.DataFrame(dict(date=frame.date.iloc[ds.start:ds.stop].to_numpy(), actual=actual,
                      original=original[chosen["kind"]].to_numpy(), ideal=prediction)).to_csv(
                          out / "predictions.csv", index=False)
    shutil.copy2(checkpoint_path, out / "ideal_model.npz")
    shutil.copy2(folder / f"{chosen['kind'].lower()}_history.csv", out / "ideal_history.csv")
    metadata = dict(selection="Minimum mean validation MSE over seeds 7,42,123; final seed fixed to 42",
                    selected=chosen, family=winner.family, test=scores, combinations=combinations,
                    checkpoint_sha256=hashlib.sha256((out / "ideal_model.npz").read_bytes()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "ideal_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    stopping = stopping_study(records, out)
    fig, axes = study.plt.subplots(1, 2, figsize=(12, 4))
    for family in ("base", "combined"):
        part = summary[summary.family == family].set_index("model").loc[list(KINDS)]
        axes[0].errorbar(KINDS, part.mean_RMSE, yerr=part.std_RMSE, marker="o", capsize=5, label=family)
    axes[0].set(ylabel="Validation RMSE (kWh)", title="Mean and sample SD across 3 seeds")
    axes[0].legend()
    axes[1].bar(table.model, np.sqrt(table.MSE))
    axes[1].set(ylabel="Test RMSE (kWh)")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    study.plt.close(fig)
    old = comparison[list(KINDS).index(chosen["kind"])]["MSE"]
    gain = (old - scores["MSE"]) / old * 100
    conclusions = []
    for kind in KINDS:
        part = summary[summary.model == kind].set_index("family")
        change = (part.loc["base", "mean_MSE"] - part.loc["combined", "mean_MSE"]) / part.loc["base", "mean_MSE"] * 100
        conclusions.append(f"- {kind}: изменение среднего validation MSE относительно базы "
                           f"{change:+.2f}% (положительное число означает снижение ошибки).")
    lines = ["# Итоговая модель после исследования гиперпараметров", "",
             "«Идеальная» означает лучшую среди проверенных вариантов по заданному критерию; глобальный оптимум не доказан.", "",
             "## План и границы исследования", "",
             "Переиспользованы 33 обучения с проверкой хешей кода, данных и настроек. "
             "Исследованы window = 8/24/48, hidden_size = 16/32/64, learning_rate = 0.0003/0.001/0.003, "
             "batch_size = 128/256/512, seed = 7/42/123, бюджеты = 30/60/100 эпох. "
             "План однофакторных экспериментов описан в [лабораторной работе](../../../README.md#3-план-исследования).", "",
             "Для каждой архитектуры лучшие значения четырёх параметров при seed 42 объединены. "
             "Каждое сочетание обучено на трёх seed; сравнение с базой выполнено по среднему validation MSE. "
             "Складывать отдельные выигрыши нельзя: результат сочетания измеряется отдельно. "
             "Лимит 100 эпох, min_epochs 60, patience 25 одинаковы для базы и сочетаний. "
             "Финальный seed заранее фиксирован: 42, а не самый удачный seed.", "",
             "Сочетания: " + " ; ".join(f"{k}: `{v}`" for k, v in combinations.items()), "",
             "## Проверка сочетаний на трёх seed", "", study.markdown_table(summary), "",
             *conclusions, "",
             "Стандартное отклонение выборочное; три повтора не доказывают статистическую значимость. "
             "Окончательный выбор делается между шестью кандидатами: 3 архитектуры × база/сочетание. "
             "Отдельные однофакторные победители вне этих семейств не объявляются устойчивыми победителями.", "",
             "Базовая GRU имеет минимальное среднее MSE; уменьшение batch_size до 128 "
             "не подтвердило выигрыш на трёх seed и увеличило разброс. У RNN сочетание "
             "улучшило средний результат, но не достигло качества GRU. У LSTM объединение "
             "индивидуально лучших значений ухудшило средний результат. Поэтому итоговые "
             "параметры GRU совпали с базовыми: исследование может подтвердить исходные значения.", "",
             "![Сравнение](comparison.png)", "", "## Ранняя остановка и эпохи", "",
             "Дополнительно проверены patience = 7/15/25 и min_epochs = 0/30/60 "
             "для бюджетов 30/60/100 (min_epochs ≤ budget). Правила воспроизведены по фактическим "
             "историям обучения: они не меняют градиенты до остановки. "
             "Если исходная история закончилась раньше требуемого правила, complete=False; "
             "продолжение не выдумывается. Сводка использует только запуски, полные для всех правил, "
             "чтобы сравнивать одинаковые траектории. Эти расчёты описательные; "
             "для итогового обучения сохранён общий протокол 100/60/25.", "",
             study.markdown_table(stopping), "",
             "В этой сводке минимальное среднее validation MSE достигается при бюджете 100 "
             "и patience 25. min_epochs = 0/30/60 дают одинаковый минимум; сохранено 60 "
             "для соответствия протоколу повторных обучений. Малый patience 7 завершает "
             "обучение быстрее, но даёт более высокую ошибку. Это обосновывает протокол "
             "100/60/25 в пределах проверенной сетки.", "",
             "Все строки: [stopping_rules.csv](stopping_rules.csv). "
             "threads/log_every — технические настройки, не параметры качества. "
             "Архитектура ячеек, один слой, Adam, MSE, clip_grad_norm=1 и признаки сохранены как в исходной реализации. "
             "Новые семейства оптимизаторов, число слоёв, dropout и регуляризация в этот поиск не входят.", "",
             "## Выбранная модель и сравнение с исходной", "",
             f"Выбрана **{chosen['kind']}**, семейство **{winner.family}**, параметры `{chosen['config']}`. "
             f"Лучшая эпоха: {chosen['training']['best_epoch']}; выполнено эпох: {chosen['training']['epochs_run']}.", "",
             study.markdown_table(table), "",
             "Результаты ранее выбранных однофакторных моделей (для полноты сравнения):", "",
             study.markdown_table(previous), "",
             "Прежние модели выбирались по лучшему отдельному запуску, текущая — по среднему "
             "трёх seed. Поэтому новая процедура не гарантирует минимальную тестовую ошибку "
             "среди всех ранее просмотренных моделей; тест не используется для смены победителя.", "",
             f"Снижение тестового MSE относительно исходной {chosen['kind']}: **{gain:.2f}%**. "
             "Исходные модели взяты из сохранённого запуска baseline_30; сравниваются те же 5256 целевых дат. "
             "Даты и значения проверяются программно. Исходное обучение начинало цели с индекса 24, "
             "исследования — с 48: это сравнение всего обновлённого процесса, а не чистый эффект одного параметра. "
             "Чистое сравнение сочетаний с базой выше использует одинаковые обучающие даты.", "",
             "Разбиение хронологическое 70/15/15, нормализация только по train. "
             "Тест не участвует в выборе конфигурации; прежний тестовый период уже использовался в лабораторной. "
             "Вывод ограничен этим временным разбиением и исследованной сеткой.", "",
             "## Файлы результатов", "",
             "`ideal_model.npz` — обученные веса и preprocessing; `ideal_config.json` — конфигурация, "
             "критерий выбора и хеши; `ideal_history.csv` — обучение; `replications.csv`, `selection.csv`, "
             "`comparison.csv`, `predictions.csv` — численные результаты. Новые обучения хранятся в `runs/`. "
             "[Команды запуска](../../../RUNNING.md#исследования)"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(f"Selected {chosen['kind']} {winner.family}: {chosen['config']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    out = study.OUTPUT / "ideal"
    out.mkdir(parents=True, exist_ok=True)
    records = existing_records()
    combinations = {}
    for kind in KINDS:
        config = study.BASE.copy()
        for factor in ("window", "hidden_size", "learning_rate", "batch_size"):
            eligible = [r for r in records if r["kind"] == kind and
                        (r["name"] == "base" or r["name"].startswith(factor + "_"))]
            config[factor] = min(eligible, key=lambda r: r["validation"]["MSE"])["config"][factor]
        combinations[kind] = config
    print(f"Combinations: {combinations}", flush=True)
    new_records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(study.run_one, f"combined_seed_{seed}", kind,
                               {**config, "seed": seed}, 100, out)
                   for kind, config in combinations.items() for seed in SEEDS]
        for future in as_completed(futures):
            r = future.result()
            new_records.append(r)
            print(f"[{len(new_records)}/9] {r['kind']} seed={r['config']['seed']} "
                  f"validation RMSE={r['validation']['RMSE']:.4f}", flush=True)
    finish(records, new_records, combinations, out)


if __name__ == "__main__":
    main()
