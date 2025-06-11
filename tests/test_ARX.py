# -*- coding: utf-8 -*-
"""
Created on Wes Aug 26 16:08:00 2022

@author: SJT2RT
"""

import numpy as np
import pandas as pd
from softsensor.data_gen import white_noise, sine, get_academic_data
from sklearn.metrics import mean_squared_error as mse
from softsensor.arx import ARX


def test_arx_sine():
    fs = 10
    end_t = 100
    steps = (end_t * fs) + 1
    time = np.linspace(0, 10 * np.pi, steps)

    x = np.sin(time) * 2
    y = np.cos(time)
    df = pd.DataFrame(data={'x': x, 'y': y}, index=time)

    inp_sens = ['x']
    out_sens = ['y']

    arx = ARX(order=(2, 2))
    arx.fit(data_train=[df], input_sensors=inp_sens, output_sensors=out_sens, windows=[(0, 800)])
    assert len(arx.parameters) == 4

    # start at zero
    test_df = df.iloc[50:, :]

    prediction = arx.prediction(test_df)
    assert test_df[out_sens].shape == prediction[out_sens].shape

    error = mse(test_df[out_sens], prediction[out_sens])
    assert error < 5e-4


def test_arx_duffing():
    ems_params = {'D': .5, 'c_lin': 1, 'c_nlin': 0}
    model = 'Duffing'

    fs = 10
    end_t = 100
    steps = end_t * fs
    time = np.linspace(0, end_t, steps)

    force = white_noise(time)
    train_df = [get_academic_data(time, model, ems_params, force, x0=[0, 0])]

    end_t = 100
    steps = end_t * fs
    time = np.linspace(0, end_t, steps)
    force = sine(time)
    test_df = get_academic_data(time, model, ems_params, force, x0=[0, 0])

    inp_sens = ['F(t)']
    out_sens = ['x', 'v']

    order = (2, 6)
    arx = ARX(order=order)
    arx.fit(train_df, inp_sens, out_sens, verbose=True)
    assert len(arx.parameters) == order[0] + order[1] * len(inp_sens)

    prediction = arx.prediction(test_df)
    assert test_df[out_sens].shape == prediction[out_sens].shape

    error = mse(test_df[out_sens], prediction[out_sens])
    assert error < 3e-2
