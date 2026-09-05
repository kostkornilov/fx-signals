"""Contract tests for re-evaluation of immutable historic signals."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    'recomputed', Path(__file__).parents[1]/'scripts/recompute_ml_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

CFG = dict(baseline_draws=20, rho=2, scale_bps=100/np.log(1.3))


def make(prices, dates=None):
    return pd.DataFrame(dict(currency='AMD', rub_per_unit=prices,
                             effective_date=pd.to_datetime(dates) if dates else
                             pd.date_range('2024-01-01', periods=len(prices))))


def test_complete_windows_and_equal_prices():
    out = report.outcomes(make([1., 1., 1., 1.]), 1)
    assert out.y.tolist()[:3] == [1., 1., 1.]
    assert np.isnan(out.y.iloc[-1])
    assert out.r.iloc[:3].eq(0).all()
    assert out.m.iloc[1:3].eq(0).all()
    assert out.m.iloc[[0,3]].isna().all()


def test_future_loss_and_symmetric_advantage_are_different():
    out = report.outcomes(make([12., 10., 8., 11.]), 1)
    assert out.y.iloc[1] == 0
    assert np.isclose(out.r.iloc[1],2500)
    assert np.isclose(out.v.iloc[1],-2000)
    assert np.isclose(out.m.iloc[1],0)


def test_windows_cross_report_boundary_without_splicing_prices():
    out = report.outcomes(make([10., 9., 8., 10.]), 2)
    first_period = out.iloc[:2]
    assert first_period.y.notna().all()
    assert np.isclose(first_period.r.iloc[1],1250)


def test_week_matching_and_seed():
    dates = pd.to_datetime(['2024-01-07','2024-01-08','2024-01-10','2024-01-21'])
    s = np.array([True,True,False,False])
    a = report.matched_masks(dates,s,20,np.random.default_rng(4))
    b = report.matched_masks(dates,s,20,np.random.default_rng(4))
    np.testing.assert_array_equal(a,b)
    assert a[:,0].all()
    assert (a[:,1:3].sum(axis=1)==1).all()
    assert not a[:,3].any()


def test_zero_signals_and_calendar_weeks():
    p = report.outcomes(make([1.,1.,1.],['2024-01-07','2024-01-08','2024-01-22']),1)
    p['eval_signal'] = False
    row = report.summarize(p,CFG,np.random.default_rng(0))
    assert row['weeks'] == 4
    assert row['signals_per_week'] == 0
    assert row['empty_week_share'] == 1
    for c in ('lift','lift_at_risk','cluster_rate','customer_regret_cvar_95_bps'):
        assert np.isnan(row[c])


def test_cluster_denominator_and_horizon_invariance():
    rates = make([1.]*30)
    rows=[]
    for h in (1,3,5,10,20):
        part = report.outcomes(rates,h)
        part['eval_signal'] = part.index.isin([0,2,10,29])
        rows.append(report.summarize(part,CFG,np.random.default_rng(0)))
    assert len({r['signals_per_week'] for r in rows})==1
    assert all(r['cluster_rate']==.25 for r in rows)
    assert rows[-1]['excluded_outcome_signals']==2


def test_discrete_cvar_and_random_neutrality():
    assert report.empirical_cvar(np.arange(1.,101.))==98
    assert report.lar_value(.5,10,20,(.5,10,20),2,100/np.log(1.3))==1
    assert np.isnan(report.lar_value(.5,10,20,(0,10,20),2,100))


def test_summary_is_reproducible():
    p=report.outcomes(make([1.,.9,1.1,1.,1.2,1.,1.3]),1)
    p['eval_signal']=[True,False,True,False,True,False,True]
    a=report.summarize(p,CFG,np.random.default_rng(123))
    b=report.summarize(p,CFG,np.random.default_rng(123))
    for key in a:
        if isinstance(a[key],float) and np.isnan(a[key]):
            assert np.isnan(b[key])
        else:
            assert a[key]==b[key]
