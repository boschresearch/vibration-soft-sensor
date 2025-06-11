import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from softsensor.datasets import SlidingWindow
from softsensor.autoreg_models import ARNN, SeparateMVEARNN, QuantileARNN
from softsensor.ensemble_wrappers import *

def dataframe(le):
    d = {'in_col1': np.linspace(0, 100, le),
         'in_col2': np.sin(np.linspace(0, 100, le)),
         'out_col1': np.cos(np.linspace(0, 100, le)),
         'out_col2': np.linspace(0, 20, le),
         'out_col3': np.linspace(0, 20, le)}
    return pd.DataFrame(d)


def test_narx_ensemble():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2', 'out_col3']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=20)
    loader = DataLoader(data, shuffle=True, batch_size=1)

    model = ARNN(input_channels=2, pred_size=3, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)
    model_2 = ARNN(input_channels=2, pred_size=3, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=20)
    
    ensemble_weights = [model.state_dict(), model_2.state_dict()]

    sync_ensemble = SyncEnsemble(model, ensemble_weights)
    predicted_ts = sync_ensemble.prediction(loader)
    mean, var = predicted_ts
    assert mean.shape == (3, 1000)
    assert var.shape == (3, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))

    async_ensemble = AsyncEnsemble(model, ensemble_weights)
    predicted_ts = async_ensemble.prediction(loader)
    mean, var = predicted_ts
    assert mean.shape == (3, 1000)
    assert var.shape == (3, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))


def test_point_ensemble():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']
    
    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=20)
    loader = DataLoader(data, shuffle=False, batch_size=1)
    
    model_1 = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20)
    model_2 = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20)
    model_3 = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20)
    
    ensemble_weights = [model_1.state_dict(), model_2.state_dict(), model_3.state_dict()]
    async_ensemble = AsyncEnsemble(model_1, ensemble_weights)
    
    predicted_ts = async_ensemble.prediction(loader)
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    
    predicted_ts, sensitivities = async_ensemble.prediction(loader, sens_params={'method':'gradient', 'comp':True})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Var_UQ'}
    assert sensitivities['Mean'].shape == (len(loader), 2, 4*windowsize)
    assert sensitivities['Var_UQ'].shape == (len(loader), 2, 4*windowsize)
    
    predicted_ts, sensitivities = async_ensemble.prediction(loader, sens_params={'method':'integrated_gradient', 'comp':True})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Var_UQ'}
    assert sensitivities['Mean'].shape == (len(loader), 2, 4*windowsize)
    assert sensitivities['Var_UQ'].shape == (len(loader), 2, 4*windowsize)
    
    predicted_ts, sensitivities = async_ensemble.prediction(loader, sens_params={'method':'perturbation', 'comp':True, 'sens_length':50})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Var_UQ'}
    assert sensitivities['Mean'].shape == (50, 2, 4*windowsize)
    assert sensitivities['Var_UQ'].shape == (50, 2, 4*windowsize)


def test_mve_ensemble():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=20)
    loader = DataLoader(data, shuffle=False, batch_size=1)

    mean_model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20)
    model_1 = SeparateMVEARNN(input_channels=2, pred_size=2, window_size=windowsize,
                mean_model=mean_model, var_hidden_size=[4, 4], activation='leaky_relu',
                rnn_window=20)
    model_2 = SeparateMVEARNN(input_channels=2, pred_size=2, window_size=windowsize,
                mean_model=mean_model, var_hidden_size=[4, 4], activation='leaky_relu',
                rnn_window=20)
    model_3 = SeparateMVEARNN(input_channels=2, pred_size=2, window_size=windowsize,
                mean_model=mean_model, var_hidden_size=[4, 4], activation='leaky_relu',
                rnn_window=20)

    ensemble_weights = [model_1.state_dict(), model_2.state_dict(), model_3.state_dict()]
    async_ensemble = AsyncMVEEnsemble(model_1, ensemble_weights)

    predicted_ts = async_ensemble.prediction(loader, reduce=True)
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))

    predicted_ts, sensitivities = async_ensemble.prediction(loader, reduce=True, sens_params={'method':'gradient', 'comp':True})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Var_UQ'}
    assert sensitivities['Mean'].shape == (len(loader), 2, 4*windowsize)
    assert sensitivities['Var_UQ'].shape == (len(loader), 2, 4*windowsize)

    predicted_ts, sensitivities = async_ensemble.prediction(loader, reduce=True, sens_params={'method':'perturbation', 'comp':True, 'sens_length':75})
    mean, var = predicted_ts
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Var_UQ'}
    assert sensitivities['Mean'].shape == (75, 2, 4*windowsize)
    assert sensitivities['Var_UQ'].shape == (75, 2, 4*windowsize)

    mean, epistemic_var, aleatoric_var = async_ensemble.prediction(loader, reduce=False)
    assert mean.shape == (2, 1000)
    assert epistemic_var.shape == (2, 1000)
    assert aleatoric_var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(epistemic_var)))
    assert not any(torch.isnan(torch.flatten(aleatoric_var)))

    predicted_ts, sensitivities = async_ensemble.prediction(loader, reduce=False, sens_params={'method':'gradient', 'comp':True})
    mean, epistemic_var, aleatoric_var = predicted_ts
    assert mean.shape == (2, 1000)
    assert epistemic_var.shape == (2, 1000)
    assert aleatoric_var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(epistemic_var)))
    assert not any(torch.isnan(torch.flatten(aleatoric_var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Var_UQ'}
    assert sensitivities['Mean'].shape == (len(loader), 2, 4*windowsize)
    assert sensitivities['Var_UQ'].shape == (len(loader), 2, 4*windowsize)

    predicted_ts, sensitivities = async_ensemble.prediction(loader, reduce=False, sens_params={'method':'integrated_gradient', 'comp':True})
    mean, epistemic_var, aleatoric_var = predicted_ts
    assert mean.shape == (2, 1000)
    assert epistemic_var.shape == (2, 1000)
    assert aleatoric_var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(epistemic_var)))
    assert not any(torch.isnan(torch.flatten(aleatoric_var)))
    # assert sensitivities dict
    assert sensitivities.keys() == {'Mean', 'Var_UQ'}
    assert sensitivities['Mean'].shape == (len(loader), 2, 4*windowsize)
    assert sensitivities['Var_UQ'].shape == (len(loader), 2, 4*windowsize)


def test_async_mcdo():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=20)
    loader = DataLoader(data, shuffle=True, batch_size=1)

    model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=20, dropout=0.1)

    async_mcdo = AsyncMCDropout(model, 5)
    predicted_ts = async_mcdo.prediction(loader)
    mean = predicted_ts[0]
    var = predicted_ts[1]
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))


def test_async_mve_mcdo():
    windowsize = 20
    input_col = ['in_col1', 'in_col2']
    output_col = ['out_col1', 'out_col2']

    data = SlidingWindow(dataframe(le=1000), windowsize, output_col, input_col,
                         rnn_window=20)
    loader = DataLoader(data, shuffle=True, batch_size=1)

    mean_model = ARNN(input_channels=2, pred_size=2, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=20, dropout=0.1)
    model = SeparateMVEARNN(input_channels=2, pred_size=2, window_size=windowsize,
                mean_model=mean_model, var_hidden_size=[4, 4], activation='leaky_relu',
                rnn_window=20, dropout=0.1)

    async_mcdo = AsyncMCDropoutMVE(model, 5)
    mean, var = async_mcdo.prediction(loader, reduce=True)    
    assert mean.shape == (2, 1000)
    assert var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(var)))

    prediction = async_mcdo.prediction(loader, reduce=False)
    mean, epistemic_var, aleatoric_var = prediction
    assert mean.shape == (2, 1000)
    assert epistemic_var.shape == (2, 1000)
    assert aleatoric_var.shape == (2, 1000)
    assert not any(torch.isnan(torch.flatten(mean)))
    assert not any(torch.isnan(torch.flatten(epistemic_var)))
    assert not any(torch.isnan(torch.flatten(aleatoric_var)))
