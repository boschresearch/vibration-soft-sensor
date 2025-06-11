# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:57:32 2022

@author: WET2RNG
"""


import pytest
from softsensor.data_gen import Zero, sine, sine_2, white_noise, sweep
from softsensor.data_gen import get_academic_data
import statsmodels.stats.diagnostic as diagn
import numpy as np
import pandas as pd
import numpy.testing as npt


@pytest.fixture()
def create_input_DF():
    fs = 2048
    t = np.arange(0, 2, 1/fs)
    ts_df = pd.DataFrame({"sine": np.sin(10 * 2 * np.pi * t),
                          "cos": 10 * np.cos(5 * 2 * np.pi * t),
                          "wn": np.random.randn(len(t))},
                         index=t)
    ts_df.index.name = "t"
    return ts_df


def test_Zero():
    time = np.linspace(0, 100, 1000)
    F = Zero(time)

    np.testing.assert_array_equal(F.F, np.zeros((1000)))
    np.testing.assert_array_equal(F.dF, np.zeros((1000)))

    assert F.comp(t=5) == 0
    assert F.comp_dt(t=5) == 0


def test_Sine():
    time = np.linspace(1, 100, 1000)

    F = sine(time, gamma=1, w0=1)

    npt.assert_array_almost_equal(F.F, 1 * np.sin(time))
    npt.assert_array_almost_equal(F.dF, 1 * np.cos(time))

    npt.assert_almost_equal(F.comp(t=2*np.pi), 0)
    npt.assert_almost_equal(F.comp_dt(t=2*np.pi), 1)

    F = sine(time, gamma=0, w0=1)

    npt.assert_almost_equal(F.F, 0, decimal=1)
    npt.assert_almost_equal(F.dF, 0, decimal=1)


def test_sine_2():
    time = np.linspace(1, 100, 1000)

    F = sine_2(time, gamma1=1, w01=1, gamma2=0.5, w02=2)

    npt.assert_array_almost_equal(F.F, 1 * np.sin(time) + 0.5 * np.sin(2*time))
    npt.assert_array_almost_equal(F.dF, 1 * np.cos(time) + 1 * np.cos(2*time))

    npt.assert_almost_equal(F.comp(t=2*np.pi), 0)
    npt.assert_almost_equal(F.comp(t=0.5*np.pi), 1)

    npt.assert_almost_equal(F.comp_dt(t=2*np.pi), 2)
    npt.assert_almost_equal(F.comp_dt(t=0.5*np.pi), -1)

    F = sine_2(time, gamma1=0, w01=1, gamma2=0, w02=2)

    npt.assert_almost_equal(F.F, 0, decimal=1)
    npt.assert_almost_equal(F.dF, 0, decimal=1)


def test_white_noise():
    time = np.linspace(0, 1000, 10000)

    F = white_noise(time)

    npt.assert_almost_equal(np.mean(F.F), 0, decimal=1)
    npt.assert_almost_equal(np.std(F.F), 1, decimal=1)
    npt.assert_almost_equal(np.mean(F.dF), 0, decimal=1)

    test_statistic = diagn.acorr_ljungbox(F.F, lags=[20], return_df=True)
    assert float(test_statistic['lb_pvalue']) > 0.01

    F = white_noise(time, f=0)

    npt.assert_almost_equal(F.F, 0, decimal=1)
    npt.assert_almost_equal(F.dF, 0, decimal=1)

    F1 = F.comp(0)
    F2 = F.comp(1/10)

    npt.assert_almost_equal(F.comp(1/20), (F2 + F1)/2)


def test_chirp():
    time = np.linspace(0, 100, 1000)

    F = sweep(time, f0=0, f1=1, t1=100, method='linear', direction='up', f=.1)

    npt.assert_almost_equal(F.comp(0), .1)
    npt.assert_almost_equal(F.comp(100), .1)
    npt.assert_almost_equal(F.comp_dt(0), 0, decimal=3)

    F = sweep(time, f0=0, f1=1, t1=100, method='linear', direction='down')

    npt.assert_almost_equal(F.comp(0), 1)
    npt.assert_almost_equal(F.comp(100), 1)

    try:
        F = sweep(time, f0=0, f1=1, t1=100, method='log', direction='up')
        assert False
    except ValueError:
        assert True

    F = sweep(time, f0=.1, f1=1, t1=100, method='log', direction='up')


def test_get_academic_data():
    time = np.linspace(0, 100, 1000)

    params = {'D': 0.05,
              'c_lin': 1,
              'c_nlin': 0.1,
              'epsilon': 3,
              'mue': .1,
              'kappa': 1,
              'delta': .2}
    F = Zero(time)

    df = get_academic_data(time, Model='Duffing', F=F, params=params)
    npt.assert_almost_equal(df['F(t)'], np.zeros(1000))
    npt.assert_almost_equal(df['x'], np.zeros(1000))
    npt.assert_almost_equal(df['v'], np.zeros(1000))

    df = get_academic_data(time, Model='Duffing_fp', F=F, params=params)
    npt.assert_almost_equal(df['z(t)'], np.zeros(1000))
    npt.assert_almost_equal(df['x'], np.zeros(1000))
    npt.assert_almost_equal(df['v'], np.zeros(1000))

    df = get_academic_data(time, Model='vd_Pol', F=F, params=params)
    npt.assert_almost_equal(df['F(t)'], np.zeros(1000))
    npt.assert_almost_equal(df['x'], np.zeros(1000))
    npt.assert_almost_equal(df['v'], np.zeros(1000))
    
    df = get_academic_data(time, Model='Pendulum', F=F, params=params)
    npt.assert_almost_equal(df['F(t)'], np.zeros(1000))
    npt.assert_almost_equal(df['x'], np.zeros(1000))
    npt.assert_almost_equal(df['v'], np.zeros(1000))
    
    
    df = get_academic_data(time, Model='Two_Mass_System', F=F, params=params)
    npt.assert_almost_equal(df['F(t)'], np.zeros(1000))
    npt.assert_almost_equal(df['x1'], np.zeros(1000))
    npt.assert_almost_equal(df['v1'], np.zeros(1000))
    npt.assert_almost_equal(df['x2'], np.zeros(1000))
    npt.assert_almost_equal(df['v2'], np.zeros(1000))
    
    try:
        df = get_academic_data(time, Model='some dumb shit', F=F,
                               params=params)
        assert False
    except ValueError:
        assert True

    df = get_academic_data(time, Model='vd_Pol', F=F, params=params,
                           x0=[1e-6, 0])

    check_vdPol = np.allclose(df['x'], np.zeros(1000))

    npt.assert_almost_equal(check_vdPol, False)

    F = white_noise(time)

    df = get_academic_data(time, Model='Duffing', F=F, params=params)
    df = get_academic_data(time, Model='Duffing_fp', F=F, params=params)
    df = get_academic_data(time, Model='vd_Pol', F=F, params=params)
    df = get_academic_data(time, Model='Pendulum', F=F, params=params)
