# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:29:35 2022

@author: WET2RNG
"""
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from softsensor.datasets import SlidingWindow, batch_rec_SW
from softsensor.autoreg_models import (_pred_ARNN_batch, _reshape_array, _comp_grad_sens, _comp_smooth_grad_sens,
                                       _comp_integrated_grad_sens, _comp_perturb_sens, _comp_sensitivity,
                                       _postprocess_sens, _random_subset_sens_indices, _check_sens_params_pred)
from softsensor.autoreg_models import *
from softsensor.autoreg_models import _predict_arnn_uncertainty, _predict_arnn_uncertainty_both
from softsensor.ensemble_wrappers import AsyncMCDropoutMVE
from softsensor.metrics import quantile_ece
from softsensor.recurrent_models import AR_RNN, RNN_DNN


def dataframe(le):
    d = {'in_col1': np.linspace(0, 100, le),
         'in_col2': np.sin(np.linspace(0, 100, le)),
         'out_col1': np.cos(np.linspace(0, 100, le)),
         'out_col2': np.linspace(0, 20, le)}

    return pd.DataFrame(d)


def test_model_narx():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=20)
    loader = DataLoader(data, shuffle=True, batch_size=32)

    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20, dropout=None)

    weights = [[8, 80], [8], [4, 8], [4], [2, 4], [2]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    x_, y_ = next(iter(loader))
    out = model(x_[0], x_[1])

    assert out.shape == y_.shape

    rec_weights = model.get_recurrent_weights()

    weights = [[8, 40], [4, 8], [2, 4]]
    for W, shape in zip(rec_weights, weights):
        assert W.shape == torch.Size(shape)

    loader = DataLoader(data, shuffle=False, batch_size=1)
    predicted_ts = model.prediction(loader)

    assert predicted_ts.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(predicted_ts)))

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=20, forecast=3, full_ds=False)
    loader = DataLoader(data, shuffle=True, batch_size=32)

    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20, dropout=None, forecast=3)

    weights = [[8, 80], [8], [4, 8], [4], [6, 4], [6]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    x_, y_ = next(iter(loader))
    out = model(x_[0], x_[1])
    assert out.shape == y_.shape

    loader = DataLoader(data, shuffle=False, batch_size=1)
    predicted_ts = model.prediction(loader)

    assert predicted_ts.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(predicted_ts)))

    model_params = model.params
    state_dict = model.state_dict()

    model2 = ARNN(**model_params)
    model2.load_state_dict(state_dict)

    res = _compare_models(model, model2)
    assert res is True


def test_quantile_narx():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']
    
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=20)
    loader = DataLoader(data, shuffle=True, batch_size=32)
    
    model = QuantileARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=20, n_quantiles=3)
    weights = [[8, 80], [8], [4, 8], [4], [6, 4], [6]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        print(W.shape, torch.Size(shape))
        assert W.shape == torch.Size(shape)
    
    x_, y_ = next(iter(loader))
    pred = model(x_[0], x_[1])
    for i in range(model.n_quantiles):
        assert pred[...,0].shape == y_.shape
    
    model = QuantileARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)#, n_quantiles=4)
    weights = [[8, 80], [8], [4, 8], [4], [78, 4], [78]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)
    
    pred = model(x_[0], x_[1])
    for i in range(model.n_quantiles):
        assert pred[...,0].shape == y_.shape
    
    loader = DataLoader(data, shuffle=False, batch_size=1)
    prediction = model.prediction(loader)
    
    assert len(prediction) == model.n_quantiles
    for quantile in prediction:
        assert quantile.shape == (2, 1000)
        assert not any(torch.isnan(torch.flatten(quantile)))


def test_comp_grad_sens():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20)
    loader = DataLoader(data, shuffle=False, batch_size=1)

    # Test Point prediction with different configurations
    # test ARNN with batch_size=1, forecast=1
    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)

    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp = _comp_grad_sens(inp, predicted_ts, model.Pred_Type)
    assert isinstance(sens_temp, torch.Tensor)
    assert sens_temp.shape == (1, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # test ARNN with batch_size=1, forecast=3
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20, forecast=3)
    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20, forecast=3)
    
    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp = _comp_grad_sens(inp, predicted_ts, model.Pred_Type)
    assert sens_temp.shape == (3, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # test AR_RNN with batch_size=32, forecast=1
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=windowsize, full_ds=False)
    loader = DataLoader(data, shuffle=False, batch_size=32)
    model = AR_RNN(input_channels=2, window_size=windowsize, pred_size=2, blocktype='LSTM',
                   blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                   dropout=0.2, rnn_window=windowsize)
    
    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp = _comp_grad_sens(inp, predicted_ts, model.Pred_Type)
    assert sens_temp.shape == (32, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # test RNN_DNN with batch_size=32, forecast=3
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         Add_zeros=True, full_ds=False, forecast=3)
    loader = DataLoader(data, shuffle=False, batch_size=32)
    model = RNN_DNN(input_channels=2, window_size=20, pred_size=2, blocktype='GRU',
                    blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                    dropout=0.2, forecast=3)
    
    x_, y_ = next(iter(loader))
    x_ = Variable(torch.flatten(x_, start_dim=1), requires_grad=True)
    predicted_ts = model.forward_sens(x_)
    sens_temp = _comp_grad_sens(x_, predicted_ts, model.Pred_Type)
    assert sens_temp.shape == (32*3, 2, 2*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # Test Mean_Var prediction
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20)
    loader = DataLoader(data, shuffle=False, batch_size=1)
    model = DensityEstimationARNN(input_channels=2, pred_size=2, window_size=windowsize,
                    hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)

    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp_mean, sens_temp_var, gradients, samples = _comp_grad_sens(inp, predicted_ts, model.Pred_Type, random_samples=5)
    assert isinstance(sens_temp_mean, torch.Tensor)
    assert isinstance(sens_temp_var, torch.Tensor)
    assert isinstance(gradients, torch.Tensor)
    assert isinstance(samples, torch.Tensor)
    assert sens_temp_mean.shape == (1, 2, 4*windowsize)
    assert sens_temp_var.shape == (1, 2, 4*windowsize)
    assert gradients.shape == (5, 2, 4*windowsize)
    assert samples.shape == (5, 2)
    assert not any(torch.isnan(torch.flatten(sens_temp_mean)))
    assert not any(torch.isnan(torch.flatten(sens_temp_var)))
    assert not any(torch.isnan(torch.flatten(gradients)))
    assert not any(torch.isnan(torch.flatten(samples)))


def test_comp_smooth_grad_sens():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20)
    loader = DataLoader(data, shuffle=False, batch_size=1)

    # Test Point prediction with different configurations
    # test ARNN with batch_size=1, forecast=1
    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)

    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp = _comp_smooth_grad_sens(model, inp, predicted_ts, model.Pred_Type)
    assert isinstance(sens_temp, torch.Tensor)
    assert sens_temp.shape == (1, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # test ARNN with batch_size=1, forecast=3
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20, forecast=3)
    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20, forecast=3)
    
    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp = _comp_smooth_grad_sens(model, inp, predicted_ts, model.Pred_Type)
    assert sens_temp.shape == (3, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # test AR_RNN with batch_size=32, forecast=1
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=windowsize, full_ds=False)
    loader = DataLoader(data, shuffle=False, batch_size=32)
    model = AR_RNN(input_channels=2, window_size=windowsize, pred_size=2, blocktype='LSTM',
                   blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                   dropout=0.2, rnn_window=windowsize)
    
    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp = _comp_smooth_grad_sens(model, inp, predicted_ts, model.Pred_Type)
    assert sens_temp.shape == (32, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # test RNN_DNN with batch_size=32, forecast=3
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         Add_zeros=True, full_ds=False, forecast=3)
    loader = DataLoader(data, shuffle=False, batch_size=32)
    model = RNN_DNN(input_channels=2, window_size=20, pred_size=2, blocktype='GRU',
                    blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                    dropout=0.2, forecast=3)
    
    x_, y_ = next(iter(loader))
    x_ = Variable(torch.flatten(x_, start_dim=1), requires_grad=True)
    predicted_ts = model.forward_sens(x_)
    sens_temp = _comp_smooth_grad_sens(model, x_, predicted_ts, model.Pred_Type, num_samples=5)
    assert sens_temp.shape == (32*3, 2, 2*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # Test Mean_Var prediction
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20)
    loader = DataLoader(data, shuffle=False, batch_size=1)
    model = DensityEstimationARNN(input_channels=2, pred_size=2, window_size=windowsize,
                    hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)

    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp_mean, sens_temp_var = _comp_smooth_grad_sens(model, inp, predicted_ts, model.Pred_Type, std_dev=0.1)
    assert isinstance(sens_temp_mean, torch.Tensor)
    assert isinstance(sens_temp_var, torch.Tensor)
    assert sens_temp_mean.shape == (1, 2, 4*windowsize)
    assert sens_temp_var.shape == (1, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp_mean)))
    assert not any(torch.isnan(torch.flatten(sens_temp_var)))

    # catch Value Error if num_samples is <= 0
    try:
        mean, var = _comp_smooth_grad_sens(model, inp, predicted_ts, model.Pred_Type, num_samples=0)
    except AssertionError as e:
        assert str(e) == 'Number of samples must be greater than 0 for SmoothGrad!'
    
    # catch Value Error if std_dev is <= 0
    try:
        mean, var = _comp_smooth_grad_sens(model, inp, predicted_ts, model.Pred_Type, std_dev=0)
    except AssertionError as e:
        assert str(e) == 'Standard deviation for Gaussian noise must be greater than 0 for SmoothGrad!'


def test_comp_integrated_grad_sens():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20)
    loader = DataLoader(data, shuffle=False, batch_size=1)

    # Test Point prediction with different configurations
    # test ARNN with batch_size=1, forecast=1
    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)

    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp = _comp_integrated_grad_sens(model, inp, predicted_ts, model.Pred_Type)
    assert isinstance(sens_temp, torch.Tensor)
    assert sens_temp.shape == (1, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # test ARNN with batch_size=1, forecast=3
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20, forecast=3)
    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20, forecast=3)
    
    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp = _comp_integrated_grad_sens(model, inp, predicted_ts, model.Pred_Type)
    assert sens_temp.shape == (3, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # test AR_RNN with batch_size=32, forecast=1
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=windowsize, full_ds=False)
    loader = DataLoader(data, shuffle=False, batch_size=32)
    model = AR_RNN(input_channels=2, window_size=windowsize, pred_size=2, blocktype='LSTM',
                   blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                   dropout=0.2, rnn_window=windowsize)
    
    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp = _comp_integrated_grad_sens(model, inp, predicted_ts, model.Pred_Type)
    assert sens_temp.shape == (32, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # test RNN_DNN with batch_size=32, forecast=3
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         Add_zeros=True, full_ds=False, forecast=3)
    loader = DataLoader(data, shuffle=False, batch_size=32)
    model = RNN_DNN(input_channels=2, window_size=20, pred_size=2, blocktype='GRU',
                    blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                    dropout=0.2, forecast=3)
    
    x_, y_ = next(iter(loader))
    x_ = Variable(torch.flatten(x_, start_dim=1), requires_grad=True)
    predicted_ts = model.forward_sens(x_)
    sens_temp = _comp_integrated_grad_sens(model, x_, predicted_ts, model.Pred_Type, num_steps=5)
    assert sens_temp.shape == (32*3, 2, 2*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # Test Mean_Var prediction
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20)
    loader = DataLoader(data, shuffle=False, batch_size=1)
    model = DensityEstimationARNN(input_channels=2, pred_size=2, window_size=windowsize,
                    hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)

    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp_mean, sens_temp_var = _comp_integrated_grad_sens(model, inp, predicted_ts, model.Pred_Type, num_steps=2)
    assert isinstance(sens_temp_mean, torch.Tensor)
    assert isinstance(sens_temp_var, torch.Tensor)
    assert sens_temp_mean.shape == (1, 2, 4*windowsize)
    assert sens_temp_var.shape == (1, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp_mean)))
    assert not any(torch.isnan(torch.flatten(sens_temp_var)))

    # catch Value Error if num_steps is <= 0
    try:
        mean, var = _comp_integrated_grad_sens(model, inp, predicted_ts, model.Pred_Type, num_steps=0)
    except AssertionError as e:
        assert str(e) == 'Number of integration steps must be greater than 1 for IG!'


def test_comp_perturb_sens():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20)
    loader = DataLoader(data, shuffle=False, batch_size=1)

    # Test Point prediction with different configurations
    # test ARNN with batch_size=1, forecast=1
    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)

    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    
    sens_temp = _comp_perturb_sens(model, inp, predicted_ts)
    assert isinstance(sens_temp, torch.Tensor)
    assert sens_temp.shape == (1, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # test ARNN with batch_size=1, forecast=3
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20, forecast=3)
    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20, forecast=3)
    
    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp = _comp_perturb_sens(model, inp, predicted_ts, perturb_size=5)
    assert sens_temp.shape == (3, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # test AR_RNN with batch_size=32, forecast=1
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=windowsize, full_ds=False)
    loader = DataLoader(data, shuffle=False, batch_size=32)
    model = AR_RNN(input_channels=2, window_size=windowsize, pred_size=2, blocktype='LSTM',
                   blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                   dropout=0.2, rnn_window=windowsize)
    
    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp = _comp_perturb_sens(model, inp, predicted_ts, std_dev=0.3)
    assert sens_temp.shape == (32, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # test RNN_DNN with batch_size=32, forecast=3
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         Add_zeros=True, full_ds=False, forecast=3)
    loader = DataLoader(data, shuffle=False, batch_size=32)
    model = RNN_DNN(input_channels=2, window_size=20, pred_size=2, blocktype='GRU',
                    blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                    dropout=0.2, forecast=3)
    
    x_, y_ = next(iter(loader))
    inp = Variable(torch.flatten(x_, start_dim=1), requires_grad=True)
    predicted_ts = model.forward_sens(inp)
    sens_temp = _comp_perturb_sens(model, inp, predicted_ts, correlated=False)
    assert sens_temp.shape == (32*3, 2, 2*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    # Test Mean_Var prediction
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20)
    loader = DataLoader(data, shuffle=False, batch_size=1)
    model = DensityEstimationARNN(input_channels=2, pred_size=2, window_size=windowsize,
                    hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)

    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)
    sens_temp_mean, sens_temp_var, gradients, samples = _comp_perturb_sens(model, inp, predicted_ts, perturb_size=3, random_samples=5)
    assert isinstance(sens_temp_mean, torch.Tensor)
    assert isinstance(sens_temp_var, torch.Tensor)
    assert isinstance(gradients, torch.Tensor)
    assert isinstance(samples, torch.Tensor)
    assert sens_temp_mean.shape == (1, 2, 4*windowsize)
    assert sens_temp_var.shape == (1, 2, 4*windowsize)
    assert gradients.shape == (5, 2, 4*windowsize)
    assert samples.shape == (5, 2)
    assert not any(torch.isnan(torch.flatten(sens_temp_mean)))
    assert not any(torch.isnan(torch.flatten(sens_temp_var)))
    assert not any(torch.isnan(torch.flatten(gradients)))
    assert not any(torch.isnan(torch.flatten(samples)))

    # catch Value Error if perturb_size is <= 0
    try:
        mean, var = _comp_perturb_sens(model, inp, predicted_ts, perturb_size=0)
    except AssertionError as e:
        assert str(e) == 'Permutation size must me greater than 0 for PFI!'
    
    # catch Value Error if std_dev is <= 0
    try:
        mean, var = _comp_perturb_sens(model, inp, predicted_ts, std_dev=0)
    except AssertionError as e:
        assert str(e) == 'Standard deviation for Gaussian noise must me greater than 0 for PFI!'


def test_comp_sensitivity():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20)
    loader = DataLoader(data, shuffle=True, batch_size=1)
    # test for same input/output channels and same window_sizes
    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=windowsize)
    
    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)

    method = 'gradient'
    sens_temp = _comp_sensitivity(method, model, inp, predicted_ts)
    assert isinstance(sens_temp, torch.Tensor)
    assert sens_temp.shape == (1, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))
    
    method = 'smooth_grad'
    sens_temp = _comp_sensitivity(method, model, inp, predicted_ts)
    assert isinstance(sens_temp, torch.Tensor)
    assert sens_temp.shape == (1, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))
    
    method = 'integrated_gradient'
    sens_temp = _comp_sensitivity(method, model, inp, predicted_ts)
    assert isinstance(sens_temp, torch.Tensor)
    assert sens_temp.shape == (1, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))
    
    method = 'perturbation'
    sens_temp = _comp_sensitivity(method, model, inp, predicted_ts)
    assert isinstance(sens_temp, torch.Tensor)
    assert sens_temp.shape == (1, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp)))

    model = DensityEstimationARNN(input_channels=2, pred_size=2, window_size=windowsize,
                    hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)
    x_, y_ = next(iter(loader))
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    predicted_ts = model.forward_sens(inp)

    method = 'gradient'
    sens_temp_mean, sens_temp_var = _comp_sensitivity(method, model, inp, predicted_ts)
    assert isinstance(sens_temp_mean, torch.Tensor)
    assert isinstance(sens_temp_var, torch.Tensor)
    assert sens_temp_mean.shape == (1, 2, 4*windowsize)
    assert sens_temp_var.shape == (1, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp_mean)))
    assert not any(torch.isnan(torch.flatten(sens_temp_var)))

    method = 'integrated_gradient'
    sens_temp_mean, sens_temp_var = _comp_sensitivity(method, model, inp, predicted_ts)
    assert isinstance(sens_temp_mean, torch.Tensor)
    assert isinstance(sens_temp_var, torch.Tensor)
    assert sens_temp_mean.shape == (1, 2, 4*windowsize)
    assert sens_temp_var.shape == (1, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp_mean)))
    assert not any(torch.isnan(torch.flatten(sens_temp_var)))

    method = 'smooth_grad'
    sens_temp_mean, sens_temp_var = _comp_sensitivity(method, model, inp, predicted_ts, num_samples=3, std_dev=0.1)
    assert isinstance(sens_temp_mean, torch.Tensor)
    assert isinstance(sens_temp_var, torch.Tensor)
    assert sens_temp_mean.shape == (1, 2, 4*windowsize)
    assert sens_temp_var.shape == (1, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp_mean)))
    assert not any(torch.isnan(torch.flatten(sens_temp_var)))

    method = 'perturbation'
    sens_temp_mean, sens_temp_var = _comp_sensitivity(method, model, inp, predicted_ts, num_samples=6, std_dev=0.3)
    assert isinstance(sens_temp_mean, torch.Tensor)
    assert isinstance(sens_temp_var, torch.Tensor)
    assert sens_temp_mean.shape == (1, 2, 4*windowsize)
    assert sens_temp_var.shape == (1, 2, 4*windowsize)
    assert not any(torch.isnan(torch.flatten(sens_temp_mean)))
    assert not any(torch.isnan(torch.flatten(sens_temp_var)))

    # catch Assertion Error if wrong / invalid method is provided
    try:
        method = 'foo'
        sens_temp_mean, sens_temp_var = _comp_sensitivity(method, model, inp, predicted_ts)
    except ValueError as e:
        assert str(e) == ("Given method 'foo' is not implemented! Choose from: "
                          "['gradient', 'smooth_grad', 'integrated_gradient', 'perturbation']")


def test_reshape_array():
    ## Test with 1D array
    arr = np.random.rand(2*20 + 2*20)
    # test for same input/output channels and same window_sizes
    model = ARNN(input_channels=2, pred_size=2, window_size=20,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)

    reshaped_arr = _reshape_array(model, arr)
    assert reshaped_arr.shape == (2+2, 21)
    # assert that first column is only nans
    assert all(np.isnan(reshaped_arr[:,0]))

    reshaped_arr = _reshape_array(model, arr, remove_nans=True)
    assert not any(np.isnan(np.concatenate(reshaped_arr)))
    assert all(x.shape == (20,) for x in reshaped_arr)

    reshaped_arr = _reshape_array(model, arr, aggregation='mean')
    assert reshaped_arr.shape == (2+2,)

    reshaped_arr = _reshape_array(model, arr, aggregation='sum')
    assert reshaped_arr.shape == (2+2,)

    reshaped_arr = _reshape_array(model, arr, aggregation='rms')
    assert reshaped_arr.shape == (2+2,)

    reshaped_arr = _reshape_array(model, arr, aggregation='rms', repeat=True)
    assert reshaped_arr.shape == (1+4*20,)

    reshaped_arr = _reshape_array(model, arr, aggregation='rms', repeat=True, repeat_size=30)
    assert reshaped_arr.shape == (1+4*30,)

    # catch Value Error if wrong agg method is given
    try:
        reshaped_arr = _reshape_array(model, arr, aggregation='max')
    except ValueError as e:
        assert str(e) == 'Invalid aggregation method "max" given! Choose from: mean, sum, rms.'

    # test for different input/output channels and different window_sizes
    model = ARNN(input_channels=2, pred_size=3, window_size=10,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=40)
    arr = np.random.rand(2*10 + 3*40)

    reshaped_arr = _reshape_array(model, arr)
    assert reshaped_arr.shape == (2+3, 41)
    # assert that first column is only nans
    assert all(np.isnan(reshaped_arr[:,0]))
    assert all(np.isnan(reshaped_arr[:2,:31].flatten()))

    reshaped_arr = _reshape_array(model, arr, remove_nans=True)
    assert not any(np.isnan(np.concatenate(reshaped_arr)))
    assert all(x.shape == (10,) for x in reshaped_arr[:2])
    assert all(x.shape == (40,) for x in reshaped_arr[2:])

    reshaped_arr = _reshape_array(model, arr, aggregation='mean')
    assert reshaped_arr.shape == (2+3,)

    reshaped_arr = _reshape_array(model, arr, aggregation='sum')
    assert reshaped_arr.shape == (2+3,)

    reshaped_arr = _reshape_array(model, arr, aggregation='rms')
    assert reshaped_arr.shape == (2+3,)

    reshaped_arr = _reshape_array(model, arr, aggregation='rms', repeat=True)
    assert reshaped_arr.shape == (1+2*10+3*40,)

    reshaped_arr = _reshape_array(model, arr, aggregation='rms', repeat=True, repeat_size=30)
    assert reshaped_arr.shape == (1+5*30,)


    ## Test with 2D array
    arr = np.random.rand(2, 2*20)
    # test for RNN_DNN with only input channels being used
    model = RNN_DNN(input_channels=2, window_size=20, pred_size=2, blocktype='GRU',
                    blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                    dropout=0.2)

    reshaped_arr = _reshape_array(model, arr)
    assert reshaped_arr.shape == (2, 2, 21)
    # assert that first columns are only nans
    assert all(np.isnan(reshaped_arr[:,:,0].flatten()))

    reshaped_arr = _reshape_array(model, arr, remove_nans=True)
    assert not any(np.isnan(np.concatenate(reshaped_arr).flatten()))
    assert all([x.shape == (20,) for x in y] for y in reshaped_arr)

    reshaped_arr = _reshape_array(model, arr, aggregation='mean')
    assert reshaped_arr.shape == (2, 2)

    reshaped_arr = _reshape_array(model, arr, aggregation='sum')
    assert reshaped_arr.shape == (2, 2)

    reshaped_arr = _reshape_array(model, arr, aggregation='rms')
    assert reshaped_arr.shape == (2, 2)

    # test for different input/output channels and different window_sizes
    model = ARNN(input_channels=2, pred_size=3, window_size=30,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)
    arr = np.random.rand(3, 2*30 + 3*20)

    reshaped_arr = _reshape_array(model, arr)
    assert reshaped_arr.shape == (3, 2+3, 31)
    # assert that first column is only nans
    assert all(np.isnan(reshaped_arr[:,:,0].flatten()))
    assert all(np.isnan(reshaped_arr[:,2:,:11].flatten()))

    reshaped_arr = _reshape_array(model, arr, remove_nans=True)
    assert not any(any(np.isnan(np.concatenate(x))) for x in reshaped_arr)
    assert all([x.shape == (30,) for x in y] for y in reshaped_arr[:2])
    assert all([x.shape == (20,) for x in y] for y in reshaped_arr[2:])

    reshaped_arr = _reshape_array(model, arr, aggregation='mean')
    assert reshaped_arr.shape == (3, 2+3)

    reshaped_arr = _reshape_array(model, arr, aggregation='sum')
    assert reshaped_arr.shape == (3, 2+3)

    reshaped_arr = _reshape_array(model, arr, aggregation='rms')
    assert reshaped_arr.shape == (3, 2+3)


def test_postprocess_sens():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20)
    loader = DataLoader(data, shuffle=True, batch_size=1)
    # test for same input/output channels and same window_sizes
    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=windowsize)
    
    _, sensitivity = model.prediction(loader, sens_params={'method':'gradient', 'comp':True, 'plot':False})
    sum_mean_std_feature, sum_mean_std_inp_chs, out_ch_sens = _postprocess_sens(model, sensitivity['Point'])

    # assert that sum_mean_std_feature is a touple of np.arrays
    assert isinstance(sum_mean_std_feature, tuple)
    assert isinstance(sum_mean_std_feature[0], np.ndarray)
    assert isinstance(sum_mean_std_feature[1], np.ndarray)
    assert isinstance(sum_mean_std_inp_chs, tuple)
    assert isinstance(sum_mean_std_inp_chs[0], np.ndarray)
    assert isinstance(sum_mean_std_inp_chs[1], np.ndarray)
    assert isinstance(out_ch_sens, tuple)
    assert isinstance(out_ch_sens[0], np.ndarray)
    assert isinstance(out_ch_sens[1], np.ndarray)
    assert sum_mean_std_feature[0].shape == (4*windowsize,)
    assert sum_mean_std_feature[1].shape == (4*windowsize,)
    assert sum_mean_std_inp_chs[0].shape == (2, 4)
    assert sum_mean_std_inp_chs[1].shape == (2, 4)
    assert out_ch_sens[0].shape == (1000, 4*windowsize)
    assert out_ch_sens[1].shape == (1000, 4*windowsize)


def test_random_subset_sens_indices():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    # test ARNN with batch_size=1, forecast=1
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20)
    loader = DataLoader(data, shuffle=True, batch_size=1)
    # test for same input/output channels and same window_sizes
    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=windowsize)
    num_timesteps, sens_indices = _random_subset_sens_indices(100, model.forecast, model.Type, loader)
    assert num_timesteps == 100
    assert len(sens_indices) == 100

    # test ARNN with batch_size=1, forecast=3
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20, forecast=3)
    loader = DataLoader(data, shuffle=True, batch_size=1)
    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20, forecast=3)
    num_timesteps, sens_indices = _random_subset_sens_indices(100, model.forecast, model.Type, loader)
    assert num_timesteps == 102
    assert len(sens_indices) == 34

    # test RNN_DNN with batch_size=32, forecast=3
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         Add_zeros=True, full_ds=False, forecast=3)
    loader = DataLoader(data, shuffle=False, batch_size=32)
    model = RNN_DNN(input_channels=2, window_size=20, pred_size=2, blocktype='GRU',
                    blocks=16, num_layers=2, hidden_size=[8], activation='relu',
                    dropout=0.2, forecast=3)
    num_timesteps, sens_indices = _random_subset_sens_indices(100, model.forecast, model.Type, loader)
    assert num_timesteps == 96
    assert len(sens_indices) == 1

    # catch Assertion Error if sens_length is too small
    try:
        num_timesteps, sens_indices = _random_subset_sens_indices(10, model.forecast, model.Type, loader)
    except AssertionError as e:
        assert str(e) == 'Given sensitivity length of 10 must be at least of size 96!'

    # catch Assertion Error if sens_length exceeds dataloader length
    try:
        num_timesteps, sens_indices = _random_subset_sens_indices(1005, model.forecast, model.Type, loader)
    except AssertionError as e:
        assert str(e) == 'Given sensitivity length of 1005 exceeds maximum dataloader length of 1000!'


def test_check_sens_params_pred():
    sens_params = {
        'method': 'gradient',
        'comp': True,
        'foo': False, # invalid key
    }
    checks = _check_sens_params_pred(sens_params)
    gt_tuple = (
        'gradient',
        True,
        False,
        None,
        10,
        0.2,
        True,
        0,
        1,
    )
    assert checks == gt_tuple

    # assert empty dict
    sens_params = {}
    checks = _check_sens_params_pred(sens_params)
    assert checks == (False, False)


def test_density_estimation():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=20)
    loader = DataLoader(data, shuffle=True, batch_size=32)

    model = DensityEstimationARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)

    weights = [[8, 80], [8], [4, 8], [4], [4, 4], [4]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    x_, y_ = next(iter(loader))
    mean, std = model(x_[0], x_[1])
    assert mean.shape == y_.shape
    assert std.shape == y_.shape

    mean, std = model.estimate_uncertainty_mean_std(x_[0], x_[1])
    assert mean.shape == y_.shape
    assert std.shape == y_.shape

    # assert forward pass for sensitivity analysis
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    mean, std = model.forward_sens(inp)

    assert x_[0].shape == (32, 2*windowsize)
    assert x_[1].shape == (32, 2*windowsize)
    assert inp.shape == (32, 4*windowsize)
    assert mean.shape == y_.shape
    assert std.shape == y_.shape
    assert mean.requires_grad is True
    assert std.requires_grad is True


    model = DensityEstimationARNN(input_channels=2, pred_size=2, window_size=windowsize,
                hidden_size=[8, 4], activation='leaky_relu',
                rnn_window=20, dropout=0.1)

    x_, y_ = next(iter(loader))
    mean, std = model.estimate_uncertainty_mean_std(x_[0], x_[1])
    assert mean.shape == y_.shape
    assert std.shape == y_.shape

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
    assert sensitivities['Mean'].shape == (len(loader), 2, 4*windowsize)
    assert sensitivities['Aleatoric_UQ'].shape == (len(loader), 2, 4*windowsize)

    predicted_ts, sensitivities = model.prediction(loader, sens_params={'method':'integrated_gradient', 'comp':True})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Aleatoric_UQ'}
    assert sensitivities['Mean'].shape == (len(loader), 2, 4*windowsize)
    assert sensitivities['Aleatoric_UQ'].shape == (len(loader), 2, 4*windowsize)

    predicted_ts, sensitivities = model.prediction(loader, sens_params={'method':'smooth_grad', 'comp':True})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Aleatoric_UQ'}
    assert sensitivities['Mean'].shape == (len(loader), 2, 4*windowsize)
    assert sensitivities['Aleatoric_UQ'].shape == (len(loader), 2, 4*windowsize)

    predicted_ts, sensitivities = model.prediction(loader, sens_params={'method':'gradient', 'comp':True, 'sens_length':100})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Aleatoric_UQ'}
    assert sensitivities['Mean'].shape == (100, 2, 4*windowsize)
    assert sensitivities['Aleatoric_UQ'].shape == (100, 2, 4*windowsize)

    predicted_ts, sensitivities = model.prediction(loader, sens_params={'method':'perturbation', 'comp':True, 'sens_length':100})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Aleatoric_UQ'}
    assert sensitivities['Mean'].shape == (100, 2, 4*windowsize)
    assert sensitivities['Aleatoric_UQ'].shape == (100, 2, 4*windowsize)


    model_params = model.params
    state_dict = model.state_dict()

    model2 = DensityEstimationARNN(**model_params)
    model2.load_state_dict(state_dict)

    res = _compare_models(model, model2)
    assert res is True


def test_separate_mve():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=20)
    loader = DataLoader(data, shuffle=True, batch_size=32)

    mean_model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20, dropout=None)
    model = SeparateMVEARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 mean_model=mean_model, var_hidden_size=[4, 4], activation='leaky_relu', rnn_window=20)

    weights = [[8, 80], [8], [4, 8], [4], [2, 4], [2]]
    for (name, W), shape in zip(model.mean_model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    weights = [[4, 80], [4], [4, 4], [4], [2, 4], [2]]
    for (name, W), shape in zip(model.DNN.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

    x_, y_ = next(iter(loader))
    mean, std = model(x_[0], x_[1])
    assert mean.shape == y_.shape
    assert std.shape == y_.shape

    mean, std = model.estimate_uncertainty_mean_std(x_[0], x_[1])
    assert mean.shape == y_.shape
    assert std.shape == y_.shape

    mean_2, std_2 = model.estimate_uncertainty_mean_std(x_[0], x_[1])
    assert mean.shape == y_.shape
    assert std.shape == y_.shape
    assert torch.allclose(mean, mean_2, atol=1e-06)
    assert torch.allclose(std, std_2, atol=1e-06)

    # assert forward pass for sensitivity analysis
    x_[0] = Variable(torch.flatten(x_[0], start_dim=1), requires_grad=True)
    x_[1] = Variable(torch.flatten(x_[1], start_dim=1), requires_grad=True)
    inp = torch.cat((x_[0], x_[1]), dim=1)
    assert x_[0].shape == (32, 2*windowsize)
    assert x_[1].shape == (32, 2*windowsize)
    assert inp.shape == (32, 4*windowsize)

    mean, std = model.forward_sens(inp)
    assert mean.shape == y_.shape
    assert std.shape == y_.shape
    assert mean.requires_grad is True
    assert std.requires_grad is True


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
    assert sensitivities['Mean'].shape == (len(loader), 2, 4*windowsize)
    assert sensitivities['Aleatoric_UQ'].shape == (len(loader), 2, 4*windowsize)

    predicted_ts, sensitivities = model.prediction(loader, sens_params={'method':'gradient', 'comp':True, 'sens_length':len(loader)//10})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Aleatoric_UQ'}
    assert sensitivities['Mean'].shape == (100, 2, 4*windowsize)
    assert sensitivities['Aleatoric_UQ'].shape == (100, 2, 4*windowsize)

    predicted_ts, sensitivities = model.prediction(loader, sens_params={'method':'perturbation', 'comp':True,
                                                                        'sens_length':50, 'perturb_size':5, 'std_dev':0.1})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Aleatoric_UQ'}
    assert sensitivities['Mean'].shape == (50, 2, 4*windowsize)
    assert sensitivities['Aleatoric_UQ'].shape == (50, 2, 4*windowsize)

    model_params = model.params
    state_dict = model.state_dict()

    model2 = SeparateMVEARNN(**model_params)
    model2.load_state_dict(state_dict)

    res = _compare_models(model, model2)
    assert res is True


def test_batch_pred():
    length = [12, 12, 6, 16]
    list_SW = []
    loader_list = []
    for le in length:
        data = SlidingWindow(dataframe(le), 10, ['out_col1', 'out_col2'],
                             ['in_col1', 'in_col2'],
                             rnn_window=10)
        loader = DataLoader(data, shuffle=False, batch_size=1)
        loader_list.append(loader)
        list_SW.append(data)

    bsw = batch_rec_SW(list_SW)

    model = ARNN(input_channels=2, pred_size=2, window_size=10,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=10, dropout=None)
    pred = _pred_ARNN_batch(model, bsw, device='cpu')

    assert pred.shape == torch.Size([4, 2, 16])
    pred = _pred_ARNN_batch(model, bsw, device='cpu')
    assert pred.shape == torch.Size([4, 2, 16])


def test_batch_pred_forecast():
    length = [12, 12, 6, 16]
    list_SW = []
    for le in length:
        data = SlidingWindow(dataframe(le), 10, ['out_col1', 'out_col2'],
                             ['in_col1', 'in_col2'],
                             rnn_window=10, forecast=3, full_ds=False)
        list_SW.append(data)

    bsw = batch_rec_SW(list_SW)
    model = ARNN(input_channels=2, pred_size=2, window_size=10,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=10, dropout=None, forecast=3)
    pred = _pred_ARNN_batch(model, bsw, device='cpu')

    assert pred.shape == torch.Size([4, 2, 18])
    pred = _pred_ARNN_batch(model, bsw, device='cpu')

    assert pred.shape == torch.Size([4, 2, 18])


def test_predict_arnn_uncertainty():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=20)
    # loader = DataLoader(data, shuffle=True, batch_size=32)

    model = DensityEstimationARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', dropout=0.1,
                 rnn_window=20)

    loader = DataLoader(data, shuffle=False, batch_size=1)
    
    predicted_ts = _predict_arnn_uncertainty(model, loader)
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))

    predicted_ts = _predict_arnn_uncertainty(model, loader)
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))

    predicted_ts, sensitivities = _predict_arnn_uncertainty(model, loader, sens_params={'method':'gradient', 'comp':True})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Aleatoric_UQ'}
    assert sensitivities['Mean'].shape == (len(loader), 2, 4*windowsize)
    assert sensitivities['Aleatoric_UQ'].shape == (len(loader), 2, 4*windowsize)

    predicted_ts, sensitivities = _predict_arnn_uncertainty(model, loader, sens_params={'method':'perturbation', 'comp':True, 'sens_length':50})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Aleatoric_UQ'}
    assert sensitivities['Mean'].shape == (50, 2, 4*windowsize)
    assert sensitivities['Aleatoric_UQ'].shape == (50, 2, 4*windowsize)


def _compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(),
                                      model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        return True
    else:
        return False
