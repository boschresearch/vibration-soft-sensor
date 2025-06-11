# -*- coding: utf-8 -*-
"""
Created on Wed May 18 20:03:39 2022

@author: WET2RNG
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import pandas as pd
import numpy as np
import pytest

from softsensor.meas_handling import SlidingWindow
from softsensor.recurrent_models import AR_RNN, RNN_DNN, _parallel_RNN, parr_RNN_DNN

@pytest.fixture()
def dataframe():
    d = {'in_col1': np.linspace(0, 100, 1000),
         'in_col2': np.sin(np.linspace(0, 100, 1000)),
         'out_col1': np.cos(np.linspace(0, 100, 1000)),
         'out_col2': np.linspace(0, 20, 1000)}
    return pd.DataFrame(d)


def test_RNN_DNN(dataframe):
    windowsize = 17
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    model = RNN_DNN(input_channels=2, window_size=17, pred_size=2, blocktype='RNN',
                    blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                    dropout=0.2)

    weights = [[16, 34], [16, 16], [16], [16], [16, 16], [16, 16], [16], [16],
               [8, 16], [8], [2, 8], [2]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    rec_weights = model.get_recurrent_weights()
    weights = [[16, 16], [16, 16]]
    for W, shape in zip(rec_weights, weights):
        assert W.shape == torch.Size(shape)

    data = SlidingWindow(dataframe, windowsize, output_col, input_col,
                         Add_zeros=True, full_ds=False, forecast=1)
    loader = DataLoader(data, shuffle=False, batch_size=32)

    x_, y_ = next(iter(loader))
    out = model(x_)
    assert out.shape == y_.shape

    x_ = Variable(x_, requires_grad=True)
    out = model(x_)
    assert out.shape == y_.shape
    assert out.requires_grad is True


    model = RNN_DNN(input_channels=2, window_size=17, pred_size=2, blocktype='GRU',
                    blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                    dropout=0.2, forecast=3)

    weights = [[48, 34], [48, 16], [48], [48], [48, 16], [48, 16], [48], [48],
               [8, 16], [8], [6, 8], [6]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    rec_weights = model.get_recurrent_weights()
    weights = [[48, 16], [48, 16]]
    for W, shape in zip(rec_weights, weights):
        assert W.shape == torch.Size(shape)

    data = SlidingWindow(dataframe, windowsize, output_col, input_col,
                         Add_zeros=True, full_ds=False, forecast=3)
    loader = DataLoader(data, shuffle=False, batch_size=32)

    x_, y_ = next(iter(loader))
    out = model(x_)
    assert out.shape == y_.shape

    x_ = Variable(x_, requires_grad=True)
    out = model(x_)
    assert out.shape == y_.shape
    assert out.requires_grad is True


    model = RNN_DNN(input_channels=2, window_size=17, pred_size=2, blocktype='LSTM',
                    blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                    dropout=0.2, forecast=3)

    weights = [[64, 34], [64, 16], [64], [64], [64, 16], [64, 16], [64], [64],
               [8, 16], [8], [6, 8], [6]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    rec_weights = model.get_recurrent_weights()
    weights = [[64, 16], [64, 16]]
    for W, shape in zip(rec_weights, weights):
        assert W.shape == torch.Size(shape)

    x_, y_ = next(iter(loader))
    out = model(x_)
    assert out.shape == y_.shape

    x_ = Variable(x_, requires_grad=True)
    out = model(x_)
    assert out.shape == y_.shape
    assert out.requires_grad is True

    predicted_ts = model.prediction(loader)
    assert predicted_ts.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(predicted_ts)))

    predicted_ts, sensitivities = model.prediction(loader, sens_params={'method':'gradient', 'comp':True})
    assert predicted_ts.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(predicted_ts)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Point'}
    assert sensitivities['Point'].shape == (1000, 2, 2*windowsize)

    predicted_ts, sensitivities = model.prediction(loader, sens_params={'method':'smooth_grad', 'comp':True, 'sens_length':96})
    assert predicted_ts.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(predicted_ts)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Point'}
    assert sensitivities['Point'].shape == (96, 2, 2*windowsize)

    predicted_ts, sensitivities = model.prediction(loader, sens_params={'method':'perturbation', 'comp':True, 'sens_length':100})
    assert predicted_ts.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(predicted_ts)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Point'}
    assert sensitivities['Point'].shape == (96, 2, 2*windowsize)

    predicted_ts, sensitivities = model.prediction(loader, sens_params={'method':'gradient', 'comp':True, 'sens_length':150})
    assert predicted_ts.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(predicted_ts)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Point'}
    assert sensitivities['Point'].shape == (192, 2, 2*windowsize)


def test_AR_RNN(dataframe):
    windowsize = 17
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    # Test different AR_RNN types of point forward pass
    model = AR_RNN(input_channels=2, window_size=17, pred_size=2, blocktype='RNN',
                    blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                    dropout=0.2, rnn_window=17)

    weights = [[16, 68], [16, 16], [16], [16], [16, 16], [16, 16], [16], [16],
               [8, 16], [8], [2, 8], [2]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    rec_weights = model.get_recurrent_weights()
    weights = [[16, 16], [16, 16], [16, 34]]
    for W, shape in zip(rec_weights, weights):
        assert W.shape == torch.Size(shape)

    data = SlidingWindow(dataframe, windowsize, output_col, input_col,
                         rnn_window=17, full_ds=True, forecast=1)
    loader = DataLoader(data, shuffle=False, batch_size=32)

    x_, y_ = next(iter(loader))

    out = model(x_[0], x_[1])
    assert out.shape == y_.shape

    # assert forward pass for sensitivity analysis
    x_[0] = Variable(x_[0], requires_grad=True)
    x_[1] = Variable(x_[1], requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    out = model.forward_sens(inp)
    assert inp.shape == (32, 4, windowsize)
    assert out.shape == y_.shape
    assert out.requires_grad is True


    model = AR_RNN(input_channels=2, window_size=17, pred_size=2, blocktype='GRU',
                   blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                   dropout=0.2, rnn_window=17, forecast=3)

    weights = [[48, 68], [48, 16], [48], [48], [48, 16], [48, 16], [48], [48],
               [8, 16], [8], [6, 8], [6]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    rec_weights = model.get_recurrent_weights()
    weights = [[48, 16], [48, 16], [48, 34]]
    for W, shape in zip(rec_weights, weights):
        assert W.shape == torch.Size(shape)

    data = SlidingWindow(dataframe, windowsize, output_col, input_col,
                         rnn_window=17, full_ds=False, forecast=3)
    loader = DataLoader(data, shuffle=False, batch_size=32)

    x_, y_ = next(iter(loader))

    out = model(x_[0], x_[1])
    assert out.shape == y_.shape

    # assert forward pass for sensitivity analysis
    x_[0] = Variable(x_[0], requires_grad=True)
    x_[1] = Variable(x_[1], requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    out = model.forward_sens(inp)
    assert inp.shape == (32, 4, windowsize)
    assert out.shape == y_.shape
    assert out.requires_grad is True

    model = AR_RNN(input_channels=2, window_size=17, pred_size=2, blocktype='LSTM',
                   blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                   dropout=0.2, rnn_window=17, forecast=3)

    weights = [[64, 68], [64, 16], [64], [64], [64, 16], [64, 16], [64], [64],
               [8, 16], [8], [6, 8], [6]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    rec_weights = model.get_recurrent_weights()
    weights = [[64, 16], [64, 16], [64, 34]]
    for W, shape in zip(rec_weights, weights):
        assert W.shape == torch.Size(shape)

    data = SlidingWindow(dataframe, windowsize, output_col, input_col,
                         rnn_window=17, full_ds=False, forecast=3)
    loader = DataLoader(data, shuffle=False, batch_size=32)

    x_, y_ = next(iter(loader))

    out = model(x_[0], x_[1])
    assert out.shape == y_.shape

    # assert forward pass for sensitivity analysis
    x_[0] = Variable(x_[0], requires_grad=True)
    x_[1] = Variable(x_[1], requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    out = model.forward_sens(inp)
    assert inp.shape == (32, 4, windowsize)
    assert out.shape == y_.shape
    assert out.requires_grad is True


    # Test Point prediction
    loader = DataLoader(data, shuffle=False, batch_size=1)

    predicted_ts = model.prediction(loader)
    assert predicted_ts.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(predicted_ts)))

    predicted_ts, sensitivities = model.prediction(loader, sens_params={'method':'gradient', 'comp':True})
    assert predicted_ts.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(predicted_ts)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Point'}
    assert sensitivities['Point'].shape == (1000, 2, 4*windowsize)

    # Test Mean_Var forward pass
    model = AR_RNN(input_channels=2, window_size=17, pred_size=2, blocktype='LSTM',
                   blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                   dropout=0.2, rnn_window=17, forecast=3, Pred_Type='Mean_Var')

    weights = [[64, 68], [64, 16], [64], [64], [64, 16], [64, 16], [64], [64],
               [8, 16], [8], [12, 8], [12]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    rec_weights = model.get_recurrent_weights()
    weights = [[64, 16], [64, 16], [64, 34]]
    for W, shape in zip(rec_weights, weights):
        assert W.shape == torch.Size(shape)

    data = SlidingWindow(dataframe, windowsize, output_col, input_col,
                         rnn_window=17, full_ds=False, forecast=3)
    loader = DataLoader(data, shuffle=False, batch_size=32)

    x_, y_ = next(iter(loader))
    out = model(x_[0], x_[1])

    assert out[0].shape == y_.shape
    assert out[1].shape == y_.shape

    # assert forward pass for sensitivity analysis
    x_[0] = Variable(x_[0], requires_grad=True)
    x_[1] = Variable(x_[1], requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    out = model.forward_sens(inp)
    assert inp.shape == (32, 4, windowsize)
    assert out[0].shape == y_.shape
    assert out[1].shape == y_.shape
    assert out[0].requires_grad is True
    assert out[1].requires_grad is True

    # Test Mean_Var prediction
    loader = DataLoader(data, shuffle=False, batch_size=1)

    predicted_ts = model.prediction(loader)
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))

    predicted_ts, sensitivities = model.prediction(loader, sens_params={'method':'gradient', 'comp':True})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Aleatoric_UQ'}
    assert sensitivities['Mean'].shape == (1000, 2, 4*windowsize)
    assert sensitivities['Aleatoric_UQ'].shape == (1000, 2, 4*windowsize)

    predicted_ts, sensitivities = model.prediction(loader, sens_params={'method':'smooth_grad', 'comp':True, 'sens_length':99})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Aleatoric_UQ'}
    assert sensitivities['Mean'].shape == (99, 2, 4*windowsize)
    assert sensitivities['Aleatoric_UQ'].shape == (99, 2, 4*windowsize)

    predicted_ts, sensitivities = model.prediction(loader, sens_params={'method':'perturbation', 'comp':True, 'sens_length':100})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Aleatoric_UQ'}
    assert sensitivities['Mean'].shape == (102, 2, 4*windowsize)
    assert sensitivities['Aleatoric_UQ'].shape == (102, 2, 4*windowsize)


def test_parallel_RNN(dataframe):
    windowsize = 1
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']
    
    model = _parallel_RNN(input_channels=2, hidden_window=5, pred_size=2,
                         blocktype='RNN', blocks=16, num_layers=2)
    
    
    data = SlidingWindow(dataframe, windowsize, output_col, input_col)
    loader = DataLoader(data, shuffle=False, batch_size=32)
    
    x_, y_ = next(iter(loader))
    out = model(x_)


def test_parr_RNN_DNN(dataframe):
    windowsize = 1
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']
    
    data = SlidingWindow(dataframe, windowsize, output_col, input_col)
    loader = DataLoader(data, shuffle=False, batch_size=32)
    
    model = parr_RNN_DNN(input_channels=2, hidden_window=5, pred_size=2,
                         blocktype='RNN', blocks=16, num_layers=2)
    
    predicted_ts = model.prediction(loader)
    
    assert predicted_ts.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(predicted_ts)))
    
    model = parr_RNN_DNN(input_channels=2, hidden_window=5, pred_size=2,
                         blocktype='RNN', blocks=16, num_layers=2,
                         Pred_Type='Mean_Var')
    
    predicted_ts = model.prediction(loader)
    
    assert predicted_ts[0].shape == (2, 1000)
    assert predicted_ts[1].shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(predicted_ts)))
