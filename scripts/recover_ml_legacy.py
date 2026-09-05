"""Run only with the archived legacy src directory on PYTHONPATH."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fx_signal.data import load_yaml
from fx_signal.features import columns_for_groups
from fx_signal.splits import make_walk_forward_folds, mask_test
from fx_signal.train import (
    _random_signal,
    _walk_forward_model,
    build_frame,
)

LEGACY_JOBS = [('random', []), *[(x, []) for x in ('momentum','level','reversal','seasonality')],
        ('logreg',['A']), ('logreg',['A','B']), ('logreg',['A','B','C']),
        ('catboost',['A','B']), ('logreg',['A','B','C','D']),
        ('catboost',['A','B','C','D'])]

root, out = map(Path, sys.argv[1:3])
jobs = [tuple(job) for job in json.loads(sys.argv[3])] if len(sys.argv) > 3 else LEGACY_JOBS
suffix = sys.argv[4] if len(sys.argv) > 4 else ''
names = (f'predictions{suffix}.csv', f'folds{suffix}.json')
config = load_yaml(root / 'configs/model.yaml')
frame = build_frame(config, root)
folds = make_walk_forward_folds(frame.effective_date.min(), frame.effective_date.max())
predictions, audits = [], []
for method, groups in jobs:
    name = method + ('_' + ''.join(groups).lower() if groups else '')
    print('Recovering ' + name, flush=True)
    work = frame.copy()
    scores = pd.Series(np.nan, index=frame.index)
    thresholds = {}
    if method in ('logreg','catboost'):
        scores, signal, thresholds = _walk_forward_model(
            frame, kind=method, feature_cols=columns_for_groups(groups, frame),
            target_col='target_stay_not_worse_h5', horizon=5, folds=folds, config=config)
    elif method == 'random':
        signal = _random_signal(frame, np.random.default_rng(0))
    else:
        signal = frame['signal_' + method].eq(True)
    work['eval_signal'] = signal.fillna(False).astype(bool)
    for fold in folds:
        part = work.loc[mask_test(work, fold), ['effective_date','currency','eval_signal']].copy()
        part['score'] = scores.loc[part.index]
        part['threshold'] = thresholds.get(fold.name, np.nan)
        part['fold'] = fold.name
        part['experiment'] = name
        part['training_horizon'] = 5 if groups else np.nan
        predictions.append(part)
        audits.append(dict(experiment=name, fold=fold.name, threshold=thresholds.get(fold.name),
                           train_end=str(fold.train_end), val_start=str(fold.val_start),
                           val_end=str(fold.val_end), test_start=str(fold.test_start),
                           test_end=str(fold.test_end)))
pd.concat(predictions).to_csv(out / names[0], index=False)
(out / names[1]).write_text(json.dumps(audits, indent=2))
