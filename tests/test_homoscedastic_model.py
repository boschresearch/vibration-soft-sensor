from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from softsensor.meas_handling import Meas_handling
from softsensor.eval_tools import comp_pred
from softsensor.homoscedastic_model import fit_homoscedastic_var, HomoscedasticModel, _fit_var
from softsensor.datasets import SlidingWindow
from softsensor.autoreg_models import ARNN

def dataframe(le):
    d = {'in_col1': np.linspace(0, 100, le),
         'in_col2': np.sin(np.linspace(0, 100, le)),
         'out_col1': np.cos(np.linspace(0, 100, le)),
         'out_col2': np.linspace(0, 20, le)}

    return pd.DataFrame(d)

def test_compare_models():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    handler = Meas_handling([dataframe(le=1000)], train_names=['train'],
                            input_sensors=input_col, output_sensors=output_col, fs=100)
    
    torch.random.manual_seed(0)
    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=windowsize)



    result_df = comp_pred([model], handler, 'train', names=['ARNN'], batch_size=16)
    
    homoscedastic_var = fit_homoscedastic_var([result_df], ['out_col1_ARNN', 'out_col2_ARNN'], 
                                              ['out_col1', 'out_col2'])
    assert homoscedastic_var.shape[0] == 2 
    homoscedastic_model = HomoscedasticModel(model, homoscedastic_var)
    
    loader = handler.give_list(windowsize, keyword='training', rnn_window=windowsize, batch_size=1)
    predicted_ts = homoscedastic_model.prediction(loader[0])
    mean = predicted_ts[0]
    var = predicted_ts[1]

    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    assert torch.equal(var[0], torch.full(([1000]), homoscedastic_var[0]))
    assert torch.equal(var[1], torch.full(([1000]), homoscedastic_var[1]))


def test_different_models():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=20, forecast=3)
    loader = [DataLoader(data, shuffle=False, batch_size=1)]

    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20, forecast=3)

    var = torch.tensor([1, 1])
    homoscedastic_model = HomoscedasticModel(model, var)

def test_forward_function():
    m = ARNN(2, 1, 10, 10, [16, 8])
    vars = torch.tensor([1])
    homosc_m = HomoscedasticModel(m, vars)
    input = torch.randn(32, 2, 10)
    rec_input = torch.randn(32, 1, 10)
    output = homosc_m(input, rec_input)
    assert output[0].shape == torch.Size([32, 1, 1])
    assert output[1].shape == torch.Size([32, 1, 1])

def test_fit_var():
    mean = torch.zeros([3, 100])
    targets = torch.rand([3, 100])
    
    var = _fit_var(mean, targets)
    assert var.shape[0] == 3