import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from softsensor.calibration import TemperatureScaling, optimize_temperature

from softsensor.meas_handling import Meas_handling
from softsensor.datasets import SlidingWindow
from softsensor.autoreg_models import SeparateMVEARNN, ARNN

def dataframe(le):
    d = {'in_col1': np.linspace(0, 100, le),
         'in_col2': np.sin(np.linspace(0, 100, le)),
         'out_col1': np.cos(np.linspace(0, 100, le)),
         'out_col2': np.linspace(0, 20, le)}

    return pd.DataFrame(d)

def test_temperature_scaling():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=20)
    loader = [DataLoader(data, shuffle=True, batch_size=1)]

    mean_model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20)
    model = SeparateMVEARNN(input_channels=2, pred_size=2, window_size=windowsize,
                mean_model=mean_model, var_hidden_size=[4, 4], activation='leaky_relu',
                rnn_window=20)

    torch.random.manual_seed(0)
    
    mean, var = model.prediction(loader[0])
    
    handler = Meas_handling([dataframe(le=1000)], ['train'], input_col,
                            output_col, 100)
    temperature = optimize_temperature(model, handler, ['train'])
    scaled_model = TemperatureScaling(model, temperature)

    inp, out = next(iter(loader[0]))
    
    out_model = model(inp[0], inp[1])
    out_model[1][:, 0, :] *= temperature.square()[0]
    out_model[1][:, 1, :] *= temperature.square()[1]
    out_model_scaled = scaled_model(inp[0], inp[1])

    assert torch.equal(out_model[1].detach(), out_model_scaled[1].detach())

    torch.random.manual_seed(0)
    (mean_2, var_2) = scaled_model.prediction(loader[0])
    assert mean_2.shape == (2, 1000)
    assert var_2.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean_2)))
    assert not any(torch.isnan(torch.flatten(var_2)))
    assert torch.equal(mean, mean_2)
    assert torch.equal(var_2, var*temperature.square()[:, None])
    

    
