"""Re-evaluate immutable legacy signals; never tune models on evaluation outcomes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fx_signal.metrics.customer_value_and_regret import add_customer_outcomes, empirical_cvar
from fx_signal.metrics.frequency import cluster_rate, weekly_push_counts

METRICS = ['lift', 'moment_advantage_bps', 'customer_regret_cvar_95_bps',
           'signals_per_week', 'lift_at_risk', 'cluster_rate']
KEYS = ['experiment', 'training_horizon', 'evaluation_horizon', 'currency', 'fold']
EXPERIMENT_META = {
    'random': ('random', ''),
    'momentum': ('momentum', ''),
    'level': ('level', ''),
    'reversal': ('reversal', ''),
    'seasonality': ('seasonality', ''),
    'logreg_a': ('logreg', 'A'),
    'logreg_ab': ('logreg', 'A,B'),
    'logreg_abc': ('logreg', 'A,B,C'),
    'logreg_abcd': ('logreg', 'A,B,C,D'),
    'catboost_ab': ('catboost', 'A,B'),
    'catboost_abcd': ('catboost', 'A,B,C,D'),
    'catboost_a': ('catboost', 'A'),
    'catboost_abc': ('catboost', 'A,B,C'),
}
# Rungs the legacy matrix skipped, so LogReg and CatBoost can be compared on every group set.
EXTRA_JOBS = [('catboost', ['A']), ('catboost', ['A', 'B', 'C'])]
SUMMARY_KEYS = ['experiment', 'method', 'feature_groups', 'corridor', 'currency',
                'training_horizon', 'evaluation_horizon', 'fold']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_seed(seed, key):
    return int.from_bytes(hashlib.sha256(f'{seed}:{key}'.encode()).digest()[:8], 'little')


def outcomes(rates, h):
    result = add_customer_outcomes(rates, horizon=h)
    price = result.rub_per_unit
    future = pd.concat([result.groupby('currency').rub_per_unit.shift(-j)
                        for j in range(1, h + 1)], axis=1)
    complete = future.notna().all(axis=1)
    result['y'] = future.min(axis=1).ge(price).astype(float).where(complete)
    result['v'] = (future.mean(axis=1) / price - 1).mul(10000).where(complete)
    result['r'] = result[f'customer_regret_bps_h{h}']
    result['m'] = result[f'moment_advantage_bps_h{h}']
    return result[['effective_date', 'currency', 'y', 'v', 'r', 'm']]


def matched_masks(dates, signal, draws, rng):
    weeks = pd.DatetimeIndex(dates).to_period('W-SUN')
    masks = np.zeros((draws, len(signal)), dtype=bool)
    for week in weeks.unique():
        idx = np.flatnonzero(weeks == week)
        count = int(signal[idx].sum())
        if count:
            rank = np.argsort(rng.random((draws, len(idx))), axis=1)[:, :count]
            masks[np.arange(draws)[:, None], idx[rank]] = True
    return masks


def mean_or_nan(x):
    return float(np.mean(x)) if len(x) else np.nan


def lar_value(hit, value, risk, baseline, rho, scale):
    hr, vr, tr = baseline
    if not np.isfinite([hit, value, risk, hr, vr, tr]).all() or hr <= 0:
        return np.nan
    return float(hit / hr * np.exp(np.clip(((value-vr)-rho*(risk-tr))/scale, -745, 709)))


def summarize(part, cfg, rng):
    signal = part.eval_signal.to_numpy(bool)
    valid = part.y.notna().to_numpy()
    valid_m = part.m.notna().to_numpy()
    chosen = signal & valid
    y, v, r, m = (part[c].to_numpy(float) for c in ('y', 'v', 'r', 'm'))
    hit = mean_or_nan(y[chosen])
    base_hit = mean_or_nan(y[valid])
    risk = empirical_cvar(r[chosen]) if chosen.any() else np.nan
    counts = weekly_push_counts(part.effective_date, part.eval_signal)
    masks = matched_masks(part.effective_date[valid], signal[valid],
                          cfg['baseline_draws'], rng)
    if chosen.any():
        n = int(chosen.sum())
        hr = float((masks @ y[valid] / n).mean())
        vr = float((masks @ v[valid] / n).mean())
        tr = float(np.mean([empirical_cvar(r[valid][row]) for row in masks]))
    else:
        hr = vr = tr = np.nan
    row = dict(
        lift=hit/base_hit if base_hit > 0 else np.nan,
        moment_advantage_bps=mean_or_nan(m[signal & valid_m]),
        customer_regret_cvar_95_bps=risk,
        signals_per_week=float(counts.mean()),
        lift_at_risk=lar_value(hit, mean_or_nan(v[chosen]), risk, (hr, vr, tr),
                              cfg['rho'], cfg['scale_bps']),
        cluster_rate=cluster_rate(part.effective_date, part.eval_signal),
        hit_rate=hit, base_hit_rate=base_hit, eligible_days=int(valid.sum()),
        target_positives=int(np.nansum(y)), positive_target_per_week=float(np.nansum(y)/len(counts)),
        all_signals=int(signal.sum()), outcome_signals=int(chosen.sum()),
        excluded_outcome_signals=int((signal & ~valid).sum()),
        moment_signals=int((signal & valid_m).sum()),
        excluded_moment_signals=int((signal & ~valid_m).sum()),
        empty_week_share=float(counts.eq(0).mean()), over_two_week_share=float(counts.gt(2).mean()),
        weeks=len(counts), period_start=str(part.effective_date.min().date()),
        period_end=str(part.effective_date.max().date()), small_sample=int(chosen.sum()) < 100,
        worst_five_mean_bps=mean_or_nan(np.sort(r[chosen])[-min(5, int(chosen.sum())):])
        if chosen.any() and chosen.sum() < 100 else np.nan,
        random_hit_rate_lar=hr, random_forward_bps_lar=vr, random_cvar_lar=tr,
        forward_bps_lar=mean_or_nan(v[chosen]),
    )
    return row


def evaluate(output, rates, cfg, suffix):
    """Every slice is scored on its own days, so signal sets can be evaluated in separate passes."""
    predictions = pd.read_csv(output/f'predictions{suffix}.csv', parse_dates=['effective_date'])
    fingerprint = hashlib.sha256((json.dumps(cfg, sort_keys=True)+digest(output/f'predictions{suffix}.csv')+
                                 digest(__file__)).encode()).hexdigest()[:12]
    rows = []
    # Evaluate one horizon at a time; checkpoints keep a long run resumable.
    for h in cfg['evaluation_horizons']:
        checkpoint = output/f'.h{h}{suffix}-{fingerprint}.csv'
        if checkpoint.exists():
            rows.extend(pd.read_csv(checkpoint).to_dict('records'))
            continue
        merged = predictions.merge(outcomes(rates, h), on=['effective_date','currency'], validate='many_to_one')
        hrows = []
        groups = list(merged.groupby(['experiment','currency','fold'], sort=True))
        # Compatibility aggregation is additional, never a replacement for yearly folds.
        groups += [( (e,c,'wf_oos'), p) for (e,c),p in merged[merged.fold.isin(
            ['wf_2023','wf_2024','wf_2025'])].groupby(['experiment','currency'], sort=True)]
        for i, ((exp, currency, fold), part) in enumerate(groups):
            part = part.sort_values('effective_date').reset_index(drop=True)
            key = f'{exp}:{currency}:{fold}:{h}'
            result = summarize(part, cfg, np.random.default_rng(stable_seed(cfg['seed'],key)))
            hrows.append(dict(experiment=exp, currency=currency, fold=fold,
                              training_horizon=5 if exp.startswith(('logreg','catboost')) else np.nan,
                              evaluation_horizon=h, **result))
            if i % 30 == 0:
                print(f'h={h}{suffix}: {i+1}/{len(groups)} slices', flush=True)
        pd.DataFrame(hrows).to_csv(checkpoint,index=False)
        rows.extend(hrows)
    return rows


def recover(source, output, cfg):
    snapshots = output / 'provenance'
    snapshots.mkdir(parents=True, exist_ok=True)
    manifest = {'legacy_commit': cfg['legacy_commit'], 'data': {}, 'runtime': {},
                'source_head': subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()}
    for package in ('numpy','pandas','scikit-learn','catboost','holidays'):
        manifest['runtime'][package] = importlib.metadata.version(package)
    manifest['runtime']['python'] = sys.version
    for name in ('rates.csv','context_rates.csv','manifest.json'):
        file = source / 'data/raw/cbr/baseline-2019-2026' / name
        manifest['data'][name] = digest(file)
    for path in ('configs/model.yaml','configs/data.yaml','uv.lock','pyproject.toml'):
        content = subprocess.check_output(['git','-C',str(source),'show',f"{cfg['legacy_commit']}:{path}"])
        (snapshots / Path(path).name).write_bytes(content)
    manifest['available_prediction_artifact'] = {}
    artifact = source / 'artifacts/experiments/logreg_ab'
    if artifact.exists():
        manifest['available_prediction_artifact'] = {
            'sha256': digest(artifact / 'predictions.csv'),
            'journal': json.loads((artifact / 'journal.json').read_text()),
            'decision': 'Not reused: journal identifies a different parent commit and does not prove matrix-run provenance.'}
    train_with_legacy_code(source, output, cfg)
    manifest['predictions_sha256'] = digest(output/'predictions.csv')
    (snapshots/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def train_with_legacy_code(source, output, cfg, jobs=None, suffix=''):
    """Fit inside an extracted checkout of the legacy commit, never against current src."""
    with tempfile.TemporaryDirectory(prefix='ml-legacy-') as tmp:
        legacy = Path(tmp)
        archive = legacy / 'source.tar'
        with archive.open('wb') as stream:
            subprocess.run(['git','-C',str(source),'archive',cfg['legacy_commit']], stdout=stream, check=True)
        with tarfile.open(archive) as tar:
            tar.extractall(legacy, filter='data')
        dest = legacy / 'data/raw/cbr/baseline-2019-2026'
        dest.mkdir(parents=True)
        for name in ('rates.csv','context_rates.csv'):
            shutil.copy2(source / 'data/raw/cbr/baseline-2019-2026' / name, dest / name)
        env = {**os.environ, 'PYTHONPATH': str(legacy / 'src'), 'PYTHONHASHSEED':'0'}
        command = [sys.executable, str(ROOT/'scripts/recover_ml_legacy.py'), str(legacy), str(output)]
        if jobs is not None:
            command += [json.dumps(jobs), suffix]
        subprocess.run(command, env=env, check=True)


def extend(source, output, cfg):
    """Train the skipped rungs; frozen legacy predictions stay byte-identical."""
    path = output / 'provenance' / 'manifest.json'
    manifest = json.loads(path.read_text())
    train_with_legacy_code(source, output, cfg, jobs=EXTRA_JOBS, suffix='_extra')
    manifest['extra_jobs'] = {
        'configurations': [f"{method} {','.join(groups)}" for method, groups in EXTRA_JOBS],
        'predictions_sha256': digest(output/'predictions_extra.csv'),
        'reason': 'Legacy matrix fitted CatBoost only on A,B and A,B,C,D, leaving two rungs unpaired.'}
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-repo', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ml_recomputed.yaml')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/ml_recomputed')
    parser.add_argument('--reuse-predictions', action='store_true')
    parser.add_argument('--publish-only', action='store_true')
    parser.add_argument('--train-extra', action='store_true')
    args = parser.parse_args()
    source, output = args.source_repo.resolve(), args.output.resolve()
    cfg = yaml.safe_load(args.config.read_text())
    # Formula, rather than a rounded YAML constant, is the source of truth.
    cfg['scale_bps'] = 100/math.log(1.3)
    output.mkdir(parents=True, exist_ok=True)
    if args.publish_only:
        publish(output)
        print(f'Published canonical tables from {output}', flush=True)
        return
    if args.train_extra:
        rates = pd.read_csv(source/'data/raw/cbr/baseline-2019-2026/rates.csv', parse_dates=['effective_date'])
        rates = rates.sort_values(['currency','effective_date']).reset_index(drop=True)
        extend(source, output, cfg)
        append_rows(output, evaluate(output, rates, cfg, '_extra'))
        publish(output)
        print(f'Added {len(EXTRA_JOBS)} configurations: {output}', flush=True)
        return
    if not args.reuse_predictions:
        recover(source, output, cfg)
    else:
        manifest = json.loads((output/'provenance/manifest.json').read_text())
        assert digest(output/'predictions.csv') == manifest['predictions_sha256']
        for name in ('rates.csv','context_rates.csv','manifest.json'):
            assert digest(source/'data/raw/cbr/baseline-2019-2026'/name) == manifest['data'][name]
    rates = pd.read_csv(source/'data/raw/cbr/baseline-2019-2026/rates.csv', parse_dates=['effective_date'])
    rates = rates.sort_values(['currency','effective_date']).reset_index(drop=True)
    rows = evaluate(output, rates, cfg, '')
    if (output/'predictions_extra.csv').exists():
        rows += evaluate(output, rates, cfg, '_extra')
    table = pd.DataFrame(rows)
    main = table[table.fold.ne('wf_oos')]
    main[KEYS+METRICS].to_csv(output/'metrics.csv',index=False)
    table[table.fold.eq('wf_oos')][KEYS+METRICS].to_csv(output/'wf_oos_metrics.csv',index=False)
    table.drop(columns=METRICS).to_csv(output/'diagnostics.csv',index=False)
    (output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    write_report(output, table)
    publish_canonical(output)
    print(f'Completed: {output}',flush=True)


def append_rows(output, rows):
    """Replace only the added experiments, so re-running stays idempotent."""
    table = pd.DataFrame(rows)
    parts = {'metrics.csv': table[table.fold.ne('wf_oos')][KEYS+METRICS],
             'wf_oos_metrics.csv': table[table.fold.eq('wf_oos')][KEYS+METRICS],
             'diagnostics.csv': table.drop(columns=METRICS)}
    for name, part in parts.items():
        previous = pd.read_csv(output/name)
        kept = previous[~previous.experiment.isin(table.experiment.unique())]
        pd.concat([kept, part], ignore_index=True).to_csv(output/name, index=False)


def publish(output):
    table = pd.concat([pd.read_csv(output/'metrics.csv'), pd.read_csv(output/'wf_oos_metrics.csv')]).merge(
        pd.read_csv(output/'diagnostics.csv'), on=['experiment', 'currency', 'fold', 'evaluation_horizon'],
        how='left', suffixes=('', '_diag'))
    write_report(output, table)
    publish_canonical(output)


def labeled(frame):
    out = frame.copy()
    out['method'] = out.experiment.map(lambda name: EXPERIMENT_META[name][0])
    out['feature_groups'] = out.experiment.map(lambda name: EXPERIMENT_META[name][1])
    out['corridor'] = 'RUB->' + out.currency.astype(str)
    return out


def publish_canonical(output):
    tables = ROOT / 'reports' / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    metrics = labeled(pd.concat([
        pd.read_csv(output/'metrics.csv'),
        pd.read_csv(output/'wf_oos_metrics.csv'),
    ], ignore_index=True))
    diagnostics = labeled(pd.read_csv(output/'diagnostics.csv'))
    metrics[SUMMARY_KEYS + METRICS].to_csv(tables/'ml_summary.csv', index=False)
    diagnostics.to_csv(tables/'ml_diagnostics.csv', index=False)


def write_report(output, table):
    main = table[table.fold.ne('wf_oos')]
    report = f'''# Пересчёт старого ML-отчёта

Обучающий горизонт ML: 5 наблюдений. Неизменные сигналы оценены на 1/3/5/10/20.
Сформировано {len(main)} основных строк: {main.experiment.nunique()} конфигураций × 5 коридоров × 5 горизонтов × 5 фолдов.
Легаси-матрица обучала CatBoost только на `A,B` и `A,B,C,D`. Ступени `A` и `A,B,C` доучены тем же
архивным кодом на тех же фолдах; сигналы легаси-матрицы не пересчитывались, их прогнозы лежат отдельно
в [predictions.csv](predictions.csv), новые — в [predictions_extra.csv](predictions_extra.csv).

## Восстановление

Архив исходного кода: `4c12350`. Прогнозы взяты оттуда без пересчёта.
Метрики старой схемы удалены: отчёт описывает только определения из `metrics-ground.md`,
поэтому построчная сверка со старой таблицей больше не ведётся.
Найденный отдельный артефакт LogReg не использован: его journal не удостоверяет принадлежность запуску матрицы.
Версии окружения, SHA256 данных и прогнозов: [manifest](provenance/manifest.json).
Старые конфиги и lock-файл сохранены в `provenance/`; модели запускались в текущем окружении,
поэтому исторический lock-файл сам по себе не является свидетельством совпадения runtime.

## Что изменилось в оценке

Основная таблица перезаписывает [reports/tables/ml_summary.csv](../tables/ml_summary.csv).
Технический след прогона: [metrics.csv](metrics.csv).
Все шесть определений соответствуют [metrics-ground.md](metrics-ground.md).
`moment_advantage_bps` использует ±h и не является переименованием `bps_forward`.
Частота считается по календарным неделям на всём периоде отправки, независимо от зрелости метки.
`cluster_rate` использует 3 календарных дня, а не соседние строки старого `cluster_share`.
Добавлены хвостовой regret и фиксированный LAR; разные горизонты здесь оценивают одну модель h=5,
а не пять отдельно обученных моделей. Сигналы проходят ровно старый фильтр `has_fact` для ML;
нового cooldown нет. Частота и кучность характеризуют этот исторический поток без политики пилота.

Диагностика: [../tables/ml_diagnostics.csv](../tables/ml_diagnostics.csv) и [diagnostics.csv](diagnostics.csv). Приведены оба hit rate, число
оценённых и исключённых сигналов, доля пустых недель и превышений лимита, границы периода.
Доверительных интервалов нет: все значения — точные описательные величины наблюдавшегося потока.
Случайная база для LAR остаётся частью самой метрики: 200 потоков с тем же числом сигналов
в каждой календарной неделе, их hit rate, выгода и CVaR сохранены в диагностике отдельными колонками.

## Периоды и ограничения

2022, 2023, 2024, январь–август 2025 и сентябрь 2025 — конец данных показаны отдельно.
[wf_oos_metrics.csv](wf_oos_metrics.csv) объединяет 2023 — август 2025 только для сопоставления.
Будущие исходы могут пересекать границу отчётного периода, но не используются для изменения сигналов.

**Исторический дефект:** код `4c12350` удалял хвост train, но не удалял незавершённые метки
validation перед тестом. Изотоника, порог и ранняя остановка могли использовать исходы начала теста.
Это сохранено ради восстановления; результаты не являются доказательством полностью корректного
walk-forward. Исправление требует отдельного эксперимента с другими сигналами.
OOT уже просматривался и не является новым нетронутым holdout. Победитель по тестовому LAR не выбирается.
Курс ЦБ — proxy курса исполнения; нулевой лаг доступности не доказывает фактическое время публикации.

## Воспроизведение

Из корня этого checkout, в окружении с зависимостями проекта:

```bash
python scripts/recompute_ml_report.py --source-repo /path/to/original/fx-signals
```

Для повторной оценки зафиксированных прогнозов добавьте `--reuse-predictions`.
SHA256 исходных данных проверяется. Промежуточные `.h*.csv` кешируются по хешу кода, конфига и прогнозов.
Сохранённые пороги и границы: [folds.json](folds.json); сигналы и оценки: [predictions.csv](predictions.csv).
'''
    (output/'REPORT.md').write_text(report)


if __name__ == '__main__':
    run()
