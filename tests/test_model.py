# -*- coding: utf-8 -*-
"""
Created on Feb  8 16:57 2022

@author: WET2RNG
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import pytest

from softsensor.meas_handling import SlidingWindow
from softsensor.model import (Feed_ForwardNN, CNN, _conv_block, _RNN, _GRU, _LSTM,
                              _Parallel_Conv, Freq_Att_CNN, _activation_function,
                              Sine)
from softsensor.model import _filter_parameters
from softsensor.layers import ConcreteDropout

@pytest.fixture()
def dataframe():
    d = {'in_col1': np.linspace(0, 5, 100),
         'in_col2': np.sin(np.linspace(0, 5, 100)),
         'out_col1': np.cos(np.linspace(0, 5, 100)),
         'out_col2': np.linspace(0, 20, 100)}

    return pd.DataFrame(d)


def test_Feed_ForwardNN(dataframe):
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe, windowsize, output_col, input_col)

    loader = DataLoader(data, shuffle=True, batch_size=32)

    model = Feed_ForwardNN(input_size=20*2, output_size=2,
                           hidden_size=[12, 4], activation='relu',
                           dropout=0.5, bias=True)

    inp, _ = next(iter(loader))
    inp = torch.flatten(inp, start_dim=1)
    out = model(inp)
    assert out.shape == (32, 2)


def test_build_dnn():
    model = Feed_ForwardNN(20*2, 2, [12, 4], dropout=0.5, concrete_dropout=False).DNN

    weights = [[12, 40], [12], [4, 12], [4], [2, 4], [2]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    model = Feed_ForwardNN(20*2, 2, [12, 4], dropout=0.5, concrete_dropout=True).DNN

    weights = [[12, 40], [12], [1], [4, 12], [4], [2, 4], [2]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    model = Feed_ForwardNN(20*2, 2, [12, 4], dropout=None).DNN

    weights = [[12, 40], [12], [4, 12], [4], [2, 4], [2]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    model = Feed_ForwardNN(20*2, 2, hidden_size=None).DNN

    weights = [[2, 40], [2]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

def test_concrete_dropout(dataframe):
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe, windowsize, output_col, input_col)

    loader = DataLoader(data, shuffle=True, batch_size=32)

    linear_layer = nn.Linear(20*2, 2)
    very_low_dropout_layer = ConcreteDropout(linear_layer, init_min=0.00000001, init_max=0.00000001)
    dropout_layer = ConcreteDropout(linear_layer, init_min=0.1, init_max=0.1)

    inp, _ = next(iter(loader))
    inp = torch.flatten(inp, start_dim=1)
    out_dropout = dropout_layer(inp)
    out_very_low_dropout = very_low_dropout_layer(inp)
    out_linear = linear_layer(inp)
    assert out_very_low_dropout.shape == out_linear.shape
    assert torch.allclose(out_very_low_dropout, out_linear)
    assert out_dropout.shape == out_linear.shape
    assert not torch.allclose(out_dropout, out_linear)

def test_conv_block(dataframe):

    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe, windowsize, output_col, input_col)

    loader = DataLoader(data, shuffle=True, batch_size=32)

    model = _conv_block(in_channels=2, filters=6, kernel_size=12,
                       window_size=20, activation='relu', pooling='average',
                       pooling_size=2)

    shapes = [[6, 2, 12], [6]]
    for (name, W), shape in zip(model.named_parameters(), shapes):
        assert W.shape == torch.Size(shape)

    model = _conv_block(in_channels=2, filters=6, kernel_size=12,
                       window_size=20, activation='relu', pooling='max',
                       pooling_size=2)

    shapes = [[6, 2, 12], [6]]
    for (name, W), shape in zip(model.named_parameters(), shapes):
        assert W.shape == torch.Size(shape)

    inp, _ = next(iter(loader))
    out = model(inp)

    assert out.shape == (32, 6, 5)


def test_CNN(dataframe):

    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe, windowsize, output_col, input_col)

    loader = DataLoader(data, shuffle=True, batch_size=32)

    model = CNN(input_channels=2, window_size=20, filters=6,
                kernel_size=5, depth=2, activation='relu',
                pooling='max', pooling_size=2, dropout=0.5)

    inp, _ = next(iter(loader))
    out = model(inp)

    shapes = [[6, 2, 5], [6], [6, 6, 5], [6]]
    for (name, W), shape in zip(model.named_parameters(), shapes):
        assert W.shape == torch.Size(shape)

    assert out.shape == (32, 6, 2)

    model = CNN(input_channels=2, window_size=20, filters=[12, 6],
                kernel_size=[9, 3], depth=2, activation='relu',
                pooling=None, pooling_size=None, dropout=None)

    shapes = [[12, 2, 9], [12], [6, 12, 3], [6]]
    for (name, W), shape in zip(model.named_parameters(), shapes):
        assert W.shape == torch.Size(shape)


def test_parrallel_conv(dataframe):
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    model = _Parallel_Conv(in_channels=2, kernel_size=5, max_dilation=3,
                          filters=5, oscillations=3)
    assert model.Lout == 11
    assert model.window_size == 43

    shapes = [[5, 2, 5], [5], [5, 2, 5], [5], [5, 2, 5], [5]]
    for (name, W), shape in zip(model.named_parameters(), shapes):
        assert W.shape == torch.Size(shape)

    data = SlidingWindow(dataframe, model.window_size, output_col, input_col)
    loader = DataLoader(data, shuffle=True, batch_size=32)

    inp, _ = next(iter(loader))
    out = model(inp)

    assert out.shape == (32, 15, 11)


def test_Freq_Att_CNN(dataframe):
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    model = Freq_Att_CNN(input_channels=2, kernel_size=5, max_dilation=3,
                         filters=5, oscillations=3, depth=1,
                         activation='relu', bypass=True)

    shapes = [[5, 2, 5], [5], [5, 2, 5], [5], [5, 2, 5], [5],
              [5, 15, 5], [5]]
    for (name, W), shape in zip(model.named_parameters(), shapes):
        assert W.shape == torch.Size(shape)

    data = SlidingWindow(dataframe, model.window_size, output_col, input_col)
    loader = DataLoader(data, shuffle=True, batch_size=32)

    inp, _ = next(iter(loader))
    out = model(inp)

    assert out.shape == (32, 5+2, model.ts_length)

    model = Freq_Att_CNN(input_channels=2, kernel_size=5, max_dilation=3,
                         filters=5, oscillations=3, depth=2,
                         activation='relu', bypass=False)

    data = SlidingWindow(dataframe, model.window_size, output_col, input_col)
    loader = DataLoader(data, shuffle=True, batch_size=32)

    shapes = [[5, 2, 5], [5], [5, 2, 5], [5], [5, 2, 5], [5],
              [5, 15, 5], [5], [5, 5, 5], [5]]
    for (name, W), shape in zip(model.named_parameters(), shapes):
        assert W.shape == torch.Size(shape)

    inp, _ = next(iter(loader))
    out = model(inp)
    assert out.shape == (32, 5, model.ts_length)


def test_RNN(dataframe):

    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe, windowsize, output_col, input_col,
                         Add_zeros=True)
    loader = DataLoader(data, shuffle=False, batch_size=32)

    model = _RNN(input_size=windowsize*2, hidden_size=16, num_layers=2)

    shapes = [[16, 40], [16, 16], [16], [16], [16, 16], [16, 16], [16], [16]]
    for (name, W), shape in zip(model.named_parameters(), shapes):
        assert W.shape == torch.Size(shape)

    rec_weights = model.get_recurrent_weights()
    shapes = [[16, 16], [16, 16]]
    for W, shape in zip(rec_weights, shapes):
        assert W.shape == torch.Size(shape)

    inp, _ = next(iter(loader))
    out = model(inp)
    assert out.shape == (32, 16)

    model = _GRU(input_size=windowsize*2, hidden_size=16, num_layers=2)

    shapes = [[3*16, 40], [3*16, 16], [3*16], [3*16],
              [3*16, 16], [3*16, 16], [3*16], [3*16]]
    for (name, W), shape in zip(model.named_parameters(), shapes):
        assert W.shape == torch.Size(shape)

    rec_weights = model.get_recurrent_weights()
    shapes = [[3*16, 16], [3*16, 16]]
    for W, shape in zip(rec_weights, shapes):
        assert W.shape == torch.Size(shape)

    inp, _ = next(iter(loader))
    out = model(inp)
    assert out.shape == (32, 16)

    model = _LSTM(input_size=windowsize*2, hidden_size=16, num_layers=2)

    shapes = [[4*16, 40], [4*16, 16], [4*16], [4*16],
              [4*16, 16], [4*16, 16], [4*16], [4*16]]
    for (name, W), shape in zip(model.named_parameters(), shapes):
        assert W.shape == torch.Size(shape)

    rec_weights = model.get_recurrent_weights()
    shapes = [[4*16, 16], [4*16, 16]]
    for W, shape in zip(rec_weights, shapes):
        assert W.shape == torch.Size(shape)

    inp, _ = next(iter(loader))
    out = model(inp)
    assert out.shape == (32, 16)


def test_act_functions():
    act = _activation_function('relu')
    assert isinstance(act, nn.ReLU)
    act = _activation_function('sigmoid')
    assert isinstance(act, nn.Sigmoid)
    act = _activation_function('tanh')
    assert isinstance(act, nn.Tanh)
    act = _activation_function('leaky_relu')
    assert isinstance(act, nn.LeakyReLU)
    act = _activation_function('sine')
    assert isinstance(act, Sine)

    try:
        act = _activation_function('not_valid_inp')
        assert False
    except ValueError:
        assert True


def test_filter_params():
    params = {'self': None,
              '__class__': Sine,
              'input1': 12}

    params = _filter_parameters(params)

    assert params == {'input1': 12}
