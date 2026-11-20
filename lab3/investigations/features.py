"""Controlled GRU ablations using only historical additional CSV measurements."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from contextlib import redirect_stdout

import run as study
import numpy as np
import pandas as pd

ROOT = study.ROOT
OUT = study.OUTPUT / "features"
REACTIVE = ["Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh"]
FACTORS = ["Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"]
CO2 = ["CO2(tCO2)"]
CALENDAR = ["NSM", "WeekStatus", "Day_of_week"]
GROUPS = dict(reactive=REACTIVE, power_factor=FACTORS, co2=CO2,
              load_type=["Load_Type"], calendar=CALENDAR,
              numeric=REACTIVE + FACTORS + CO2,
              all=REACTIVE + FACTORS + CO2 + CALENDAR + ["Load_Type"])
SEEDS = (7, 42, 123)


def prepared(group):
    datasets, frame, info = study.prepared(study.BASE)
    split = datasets["train"].stop
    columns = GROUPS[group]
    features = [datasets["train"].features]
    names = list(info["features"])
    transform = {}
    for column in columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            values = series.to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite values: {column}")
            mean, std = float(values[:split].mean()), float(values[:split].std())
            std = std or 1.0
            features.append(((values - mean) / std)[:, None])
            names.append(column)
            transform[column] = dict(type="standardize", mean=mean, std=std)
        else:
            categories = sorted(series.iloc[:split].unique().tolist())
            if not series.isin(categories).all():
                raise ValueError(f"Unseen validation/test category: {column}")
            features.append(np.column_stack([series.eq(c).to_numpy(dtype=np.float32) for c in categories]))
            names.extend(f"{column}={c}" for c in categories)
            transform[column] = dict(type="one_hot", categories=categories)
    matrix = np.concatenate(features, axis=1).astype(np.float32)
    for ds in datasets.values():
        ds.features = matrix
    info.update(features=names, additional_transform=transform, input_size=len(names),
                availability="All input rows are strictly before the target timestamp")
    return datasets, frame, info


def signature(group, seed):
    files = [Path(__file__), ROOT / "investigations/run.py"] + [ROOT / "source" / f for f in
             ("models.py", "main.py", "data_preparation.py")]
    return dict(group=group, seed=seed, config=study.BASE, epochs=100, min_epochs=60, patience=25,
                code_sha256=hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest(),
                data_sha256=hashlib.sha256((ROOT / "data/Steel_industry_data.csv").read_bytes()).hexdigest())


def train_one(group, seed):
    folder = OUT / "runs" / group / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    sig = signature(group, seed)
    if (folder / "result.json").exists():
        r = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if r["signature"] == sig:
            return r
    datasets, frame, info = prepared(group)
    args = SimpleNamespace(**{**study.BASE, "seed": seed}, epochs=100, min_epochs=60,
                           patience=25, log_every=10)
    with (folder / "training.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        model, _, training = study.train("GRU", datasets, args, folder, info, artifact_name="model")
    val = datasets["val"]
    predicted = study.predict(model, val, args.batch_size) * info["std"] + info["mean"]
    actual = frame.Usage_kWh.iloc[val.start:val.stop].to_numpy()
    pd.DataFrame(dict(date=frame.date.iloc[val.start:val.stop].to_numpy(), actual=actual, predicted=predicted)).to_csv(folder / "validation_predictions.csv", index=False)
    r = dict(group=group, seed=seed, signature=sig, validation=study.metrics(actual, predicted),
             training=training, preprocessing=info)
    (folder / "result.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    return r


def baseline_records():
    result = []
    for seed in SEEDS:
        name = "base" if seed == 42 else f"seed_{seed}"
        r = study.run_one(name, "GRU", {**study.BASE, "seed": seed}, 100, study.STUDY_OUT)
        result.append(dict(group="baseline", seed=seed, validation=r["validation"], training=r["training"]))
    return result


def report(records):
    rows = pd.DataFrame([dict(group=r["group"], seed=r["seed"], **r["validation"],
                              input_size=5 if r["group"] == "baseline" else r["preprocessing"]["input_size"],
                              best_epoch=r["training"]["best_epoch"]) for r in records])
    rows.to_csv(OUT / "validation_runs.csv", index=False)
    summary = rows.groupby("group").agg(mean_MSE=("MSE", "mean"), sd_MSE=("MSE", "std"),
                mean_RMSE=("RMSE", "mean"), sd_RMSE=("RMSE", "std"), input_size=("input_size", "first"), seeds=("seed", "size")).reset_index()
    base = summary.set_index("group").loc["baseline", "mean_MSE"]
    summary["MSE_reduction_percent"] = (base-summary.mean_MSE)/base*100
    summary = summary.sort_values(["mean_MSE", "group"])
    summary.to_csv(OUT / "validation_summary.csv", index=False)
    winner = summary.iloc[0].group
    # Selection is complete before opening test targets; seed 42 is fixed.
    selected_folder = (study.STUDY_OUT / "runs/base/gru" if winner == "baseline" else OUT / "runs" / winner / "42")
    selected_path = selected_folder / ("gru.npz" if winner == "baseline" else "model.npz")
    ds, frame, info = (study.prepared(study.BASE) if winner == "baseline" else prepared(winner))
    model, checkpoint = study.RecurrentRegressor.load(selected_path)
    test = ds["test"]
    predicted = study.predict(model, test, batch_size=256) * info["std"] + info["mean"]
    reference = pd.read_csv(ROOT / "results/ideal/predictions.csv")
    actual = frame.Usage_kWh.iloc[test.start:test.stop].to_numpy()
    np.testing.assert_allclose(actual, reference.actual)
    assert pd.to_datetime(reference.date).equals(pd.Series(frame.date.iloc[test.start:test.stop].to_numpy()))
    scores = study.metrics(actual, predicted)
    compare = pd.DataFrame([dict(model="Previous GRU (5 features)", **study.metrics(actual, reference.ideal)),
                            dict(model=f"Selected GRU ({winner})", **scores)])
    compare.to_csv(OUT / "test_comparison.csv", index=False)
    pd.DataFrame(dict(date=reference.date, actual=actual, baseline=reference.ideal, selected=predicted)).to_csv(OUT / "test_predictions.csv", index=False)
    shutil.copy2(selected_path, OUT / "selected_model.npz")
    metadata = dict(winner=winner, seed=42, criterion="Minimum mean validation MSE over seeds 7,42,123",
                    config=study.BASE, test=scores, preprocessing=checkpoint["preprocessing"],
                    sha256=hashlib.sha256((OUT / "selected_model.npz").read_bytes()).hexdigest())
    (OUT / "selected_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fig, axes = study.plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(summary.group, summary.mean_RMSE, yerr=summary.sd_RMSE, capsize=4)
    axes[0].set(ylabel="Validation RMSE (kWh)", title="Mean and sample SD over 3 seeds")
    axes[0].tick_params(axis="x", rotation=50)
    for group in GROUPS:
        h = pd.read_csv(OUT / "runs" / group / "42/history.csv")
        axes[1].plot(h.epoch, h.val_mse_scaled, label=group)
    axes[1].set(xlabel="Epoch", ylabel="Validation MSE (scaled)", title="Seed 42 learning curves")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "features.png", dpi=150)
    study.plt.close(fig)
    change = (1-scores["MSE"]/compare.iloc[0].MSE)*100
    lines = ["# Отдельное исследование дополнительных прошлых признаков", "",
             "Проверено влияние дополнительных столбцов CSV при фиксированных настройках итоговой GRU. "
             "Результаты прежних исследований не перезаписываются.", "", "## Методика", "",
             "База: 5 признаков — история Usage_kWh и циклический календарь. Окно 24, hidden_size 32, "
             "Adam lr=0.001, batch_size 256, до 100 эпох, min_epochs 60, patience 25, clipping 1.0. "
             "Для каждого набора выполнены 3 обучения с seed 7/42/123; 21 новое обучение, "
             "3 базовых запуска переиспользованы с проверкой сигнатур. Изменяются входные признаки, "
             "поэтому число весов входной проекции также меняется. Другие гиперпараметры не подбираются заново.", "",
             "Все дополнительные значения берутся из [t−24, …, t−1], никогда из строки цели t. "
             "Предполагается, что измерения этих прошлых отсчётов уже доступны в момент прогноза. "
             "Это условие особенно существенно для CO₂ и категории нагрузки, которые могут "
             "появляться с задержкой в реальной системе.", "",
             "Нормализация численных столбцов и словарь one-hot категорий настраиваются только "
             "на первых 70% строк. Значения test не участвуют в расчёте среднего и стандартного отклонения. "
             "Для категории вне обучающего словаря скрипт завершится ошибкой, а не расширит словарь по тесту. "
             "Обучающие цели начинаются с индекса 48; train/val/test и даты полностью совпадают с прежним исследованием.", "",
             "## Наборы признаков", "", "Каждый набор добавляется к базовым пяти признакам.", ""]
    for group, cols in GROUPS.items():
        lines.append(f"- **{group}**: " + ", ".join(f"`{c}`" for c in cols) + ".")
    lines += ["", "NSM, WeekStatus и Day_of_week проверяются как альтернативные представления "
              "календаря; они в основном дублируют информацию даты. Категориальные столбцы "
              "кодируются one-hot, численные — стандартизуются.", "", "## Валидация: выбор набора", "",
              study.markdown_table(summary), "", "Положительный MSE_reduction_percent означает снижение ошибки относительно базы. "
              "SD — выборочное стандартное отклонение. Три seed не доказывают статистическую значимость. "
              "Победитель определяется по среднему validation MSE, итоговый seed заранее фиксирован: 42.", "",
              "![Сравнение наборов и обучение](features.png)", "", "## Итоговое сравнение на тесте", "",
              f"По валидации выбран набор **{winner}**.", "", study.markdown_table(compare), "",
              f"Снижение тестового MSE относительно прежней GRU: **{change:.2f}%** "
              "(отрицательное значение означает ухудшение). Тест вычисляется только после выбора "
              "одного набора, а не для перебора всех кандидатов.", "", "## Выводы", ""]
    for row in summary.itertuples():
        if row.group != "baseline":
            lines.append(f"- **{row.group}**: средний validation MSE {row.mean_MSE:.4f}; "
                         f"снижение относительно базы {row.MSE_reduction_percent:+.2f}%.")
    lines += ["", "Это исследование оценивает добавление признаков при фиксированной GRU. "
              "Выводы не распространяются автоматически на другие архитектуры, горизонты и периоды. "
              "Тестовый период уже использовался ранее; для независимой оценки необходим новый период. "
              "Добавление признаков не заменяет проверку их доступности при практическом прогнозировании.", "",
              "## Воспроизведение", "", "Из папки lab3:", "", "```powershell",
              "python investigations/features.py --workers 3", "```", "",
              "`validation_runs.csv` и `validation_summary.csv` — результаты всех повторов и сводка; "
              "`runs/` — веса, преобразования, истории и validation-прогнозы; "
              "`selected_model.npz` и `selected_config.json` — выбранная модель; "
              "`test_comparison.csv` и `test_predictions.csv` — итоговый тест."]
    (OUT / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(compare.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    records = baseline_records()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(train_one, group, seed) for group in GROUPS for seed in SEEDS]
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            records.append(r)
            print(f"[{i}/21] {r['group']} seed={r['seed']} validation RMSE={r['validation']['RMSE']:.4f}", flush=True)
    report(records)


if __name__ == "__main__":
    main()
