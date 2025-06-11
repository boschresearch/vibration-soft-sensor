# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:22:35 2023

@author: WET2RNG
"""

import pandas as pd
import numpy as np
import pytest
from torch.utils.data import DataLoader

from softsensor.meas_handling import SlidingWindow

from softsensor.temporal_fusion_transformer import TFT

@pytest.fixture()
def dataframe():
    d = {'in_col1': np.linspace(0, 100, 1000),
         'in_col2': np.sin(np.linspace(0, 100, 1000)),
         'out_col1': np.cos(np.linspace(0, 100, 1000)),
         'out_col2': np.linspace(0, 20, 1000)}

    return pd.DataFrame(d)


def test_transformer(dataframe):

    windowsize = 1
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    model = TFT(input_channels=2, pred_size=2, blocks=16, hidden_window=10,
                num_layers=1, blocktype='LSTM', n_heads=4, bias=True, dropout=None,
                forecast=1)

    data = SlidingWindow(dataframe, windowsize, output_col, input_col,
                         Add_zeros=True, full_ds=False, forecast=1)
    loader = DataLoader(data, shuffle=False, batch_size=32)

    x_, y_ = next(iter(loader))
    out = model(x_)
    
    model = TFT(input_channels=2, pred_size=2, blocks=16, hidden_window=10,
                num_layers=1, blocktype='LSTM', n_heads=4, bias=True, dropout=None,
                forecast=1, GRN=False)

    x_, y_ = next(iter(loader))
    out = model(x_)
    
    