# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:22:07 2022

@author: WET2RNG
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pytest
import numpy as np
import pandas as pd
import copy
from softsensor.meas_handling import Meas_handling
from softsensor.train_model import train_model, early_stopping, _print_results
from softsensor.train_model import _relative_permutation
from softsensor.autoreg_models import ARNN, DensityEstimationARNN, SeparateMVEARNN
from softsensor.recurrent_models import AR_RNN, RNN_DNN, parr_RNN_DNN
from softsensor.temporal_fusion_transformer import TFT
from softsensor.losses import GaussianNLLLoss

# Model to use for testing
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(40, 1)

    def forward(self, inp):
        inp = inp.view(-1, 40)
        return self.lin1(inp)


@pytest.fixture()
def handler():
    d = {'in_col1': np.random.rand(1000),
         'in_col2': np.random.rand(1000),
         'out_col1': np.random.rand(1000)}

    handler = Meas_handling([pd.DataFrame(d)], ['input_df'],
                            ['in_col1', 'in_col2'], ['out_col1'], 1000,
                            [pd.DataFrame(d)], ['test'])

    return handler

def test_ar_model(handler):
    windowsize = 50
    data = handler.give_torch_loader(windowsize, rnn_window=windowsize,
                                     Add_zeros=False, forecast=3)
    model = DensityEstimationARNN(input_channels=2, pred_size=1, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu', rnn_window=windowsize, forecast=3)

    original = copy.deepcopy(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=.1)
    criterion = GaussianNLLLoss()
    train_model(model, data[0], max_epochs=3, optimizer=optimizer,
                device='cpu', criterion=criterion, val_loader=data[1],
                patience=None, print_results=False, give_results=False,
                rel_perm=0, stabelizer=1e-3, local_wd=1e-3)

    for key_item_1, key_item_2 in zip(original.items(),
                                      model.state_dict().items()):
        assert not any(torch.flatten(torch.eq(key_item_1[1], key_item_2[1])))

    mean_model = ARNN(input_channels=2, pred_size=1, window_size=windowsize,
                 hidden_size=[8, 4], activation='leaky_relu',
                 rnn_window=windowsize, forecast=3)
    model = SeparateMVEARNN(input_channels=2, pred_size=1, window_size=windowsize,
                mean_model=mean_model, var_hidden_size=[8, 4], activation='leaky_relu',
                rnn_window=windowsize, forecast=3)

    original = copy.deepcopy(model.DNN.state_dict())

    optimizer = optim.Adam(model.DNN.parameters(), lr=0.001, weight_decay=.1)
    criterion = GaussianNLLLoss()
    train_model(model, data[0], max_epochs=3, optimizer=optimizer,
                device='cpu', criterion=criterion, val_loader=data[1],
                patience=None, print_results=False, give_results=False,
                rel_perm=0, stabelizer=1e-3, local_wd=1e-3)

    for key_item_1, key_item_2 in zip(original.items(),
                                      model.DNN.state_dict().items()):
        assert not any(torch.flatten(torch.eq(key_item_1[1], key_item_2[1])))

def test_rnn_model(handler):
    windowsize = 50
    data = handler.give_list(windowsize, keyword='training', Add_zeros=False,
                             forecast=3, full_ds=False)

    model = RNN_DNN(input_channels=2, window_size=50, pred_size=1, blocktype='GRU',
                    blocks=16, num_layers=1, hidden_size=None, activation='relu',
                    dropout=None, forecast=3)
    original = copy.deepcopy(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=.1)
    train_model(model, data, max_epochs=1, optimizer=optimizer,
                device='cpu', criterion=nn.MSELoss(), val_loader=data,
                patience=3, print_results=False, give_results=False,
                rel_perm=0, stabelizer=None, local_wd=None)

    for key_item_1, key_item_2 in zip(original.items(),
                                      model.state_dict().items()):
        assert not any(torch.flatten(torch.eq(key_item_1[1], key_item_2[1])))

    data = handler.give_list(windowsize, keyword='training',
                             rnn_window=windowsize)

    model = AR_RNN(input_channels=2, window_size=50, pred_size=1, blocktype='LSTM',
                    blocks=16, num_layers=1, hidden_size=None, activation='sine',
                    dropout=None, rnn_window=50)
    original = copy.deepcopy(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=.1)
    train_model(model, data, max_epochs=1, optimizer=optimizer,
                device='cpu', criterion=nn.MSELoss(), val_loader=data,
                patience=3, print_results=False, give_results=False,
                rel_perm=0, stabelizer=1e-3, local_wd=1e-3)

    for key_item_1, key_item_2 in zip(original.items(),
                                      model.state_dict().items()):
        assert not any(torch.flatten(torch.eq(key_item_1[1], key_item_2[1])))


    data = handler.give_list(window_size=1, keyword='training', Add_zeros=False,
                             forecast=1, full_ds=False)
    
    model = parr_RNN_DNN(input_channels=2, hidden_window=5, pred_size=1,
                         blocktype='RNN', blocks=16, num_layers=2)
    original = copy.deepcopy(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=.1)
    train_model(model, data, max_epochs=1, optimizer=optimizer,
                device='cpu', criterion=nn.MSELoss(), val_loader=data,
                patience=3, print_results=False, give_results=False,
                rel_perm=0, stabelizer=None, local_wd=None)
    
    for key_item_1, key_item_2 in zip(original.items(),
                                      model.state_dict().items()):
        assert not any(torch.flatten(torch.eq(key_item_1[1], key_item_2[1])))


def test_train_rnn_prob(handler):
    windowsize = 50
    data = handler.give_list(windowsize, keyword='training', Add_zeros=False,
                             forecast=3, full_ds=False)

    model = RNN_DNN(input_channels=2, window_size=50, pred_size=1, blocktype='GRU',
                    blocks=16, num_layers=1, hidden_size=None, activation='relu',
                    dropout=None, forecast=3, Pred_Type='Mean_Var')
    original = copy.deepcopy(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=.1)
    train_model(model, data, max_epochs=1, optimizer=optimizer,
                device='cpu', criterion=GaussianNLLLoss(), val_loader=data,
                patience=3, print_results=False, give_results=False,
                rel_perm=0, stabelizer=None, local_wd=None)

    for key_item_1, key_item_2 in zip(original.items(),
                                      model.state_dict().items()):
            assert not any(torch.flatten(torch.eq(key_item_1[1], key_item_2[1])))

    data = handler.give_list(windowsize, keyword='training',
                             rnn_window=windowsize)

    model = AR_RNN(input_channels=2, window_size=50, pred_size=1, blocktype='LSTM',
                    blocks=16, num_layers=1, hidden_size=None, activation='sine',
                    dropout=None, rnn_window=50, Pred_Type='Mean_Var')
    original = copy.deepcopy(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=.1)
    train_model(model, data, max_epochs=1, optimizer=optimizer,
                device='cpu', criterion=GaussianNLLLoss(), val_loader=data,
                patience=3, print_results=False, give_results=False,
                rel_perm=0, stabelizer=1e-3, local_wd=1e-3)

    for key_item_1, key_item_2 in zip(original.items(),
                                      model.state_dict().items()):
        assert not any(torch.flatten(torch.eq(key_item_1[1], key_item_2[1])))
        
    data = handler.give_list(window_size=1, keyword='training', Add_zeros=False,
                             forecast=1, full_ds=False)
    
    model = parr_RNN_DNN(input_channels=2, hidden_window=5, pred_size=1,
                         blocktype='RNN', blocks=16, num_layers=2,
                         Pred_Type='Mean_Var')
    original = copy.deepcopy(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=.1)
    train_model(model, data, max_epochs=1, optimizer=optimizer,
                device='cpu', criterion=GaussianNLLLoss(), val_loader=data,
                patience=3, print_results=False, give_results=False,
                rel_perm=0, stabelizer=None, local_wd=None)
    
    for key_item_1, key_item_2 in zip(original.items(),
                                      model.state_dict().items()):
        assert not any(torch.flatten(torch.eq(key_item_1[1], key_item_2[1])))

def test_tft(handler):
    windowsize = 1
    data = handler.give_list(windowsize, keyword='training', Add_zeros=False,
                             forecast=1, full_ds=False)

    model = TFT(input_channels=2, pred_size=2, blocks=16, hidden_window=10,
                num_layers=1, blocktype='LSTM', n_heads=4, bias=True, dropout=None,
                forecast=1)
    
    original = copy.deepcopy(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=.1)
    train_model(model, data, max_epochs=1, optimizer=optimizer,
                device='cpu', criterion=nn.MSELoss(), val_loader=data,
                patience=3, print_results=False, give_results=False,
                rel_perm=0, stabelizer=None, local_wd=None)

    for key_item_1, key_item_2 in zip(original.items(),
                                      model.state_dict().items()):
        assert not any(torch.flatten(torch.eq(key_item_1[1], key_item_2[1])))

    model = TFT(input_channels=2, pred_size=2, blocks=16, hidden_window=10,
                num_layers=1, blocktype='GRU', n_heads=1, bias=True, dropout=None,
                forecast=1, Pred_Type='Mean_Var', GRN=False)
    
    original = copy.deepcopy(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=.1)
    train_model(model, data, max_epochs=1, optimizer=optimizer,
                device='cpu', criterion=GaussianNLLLoss(), val_loader=data,
                patience=3, print_results=False, give_results=False,
                rel_perm=0, stabelizer=None, local_wd=None)

    for key_item_1, key_item_2 in zip(original.items(),
                                      model.state_dict().items()):
        assert not any(torch.flatten(torch.eq(key_item_1[1], key_item_2[1])))

def test_early_stopping_stop():
    net = model()
    es = early_stopping(patience=3)
    val_loss = [1]
    out = es.call(val_loss, net)
    assert out is False

    val_loss = [1, 2, 3, 4]
    out = es.call(val_loss, net)
    assert out is True


def test_print_results():
    max_epochs = 50
    results = {'train_loss': [5, 1, 2],
               'val_loss': [],
               'stability_score': []}
    _print_results(max_epochs, results)

    results = {'train_loss': [5, 1, 2],
               'val_loss': [3, 2, 1],
               'stability_score': [2, 1, 1]}
    _print_results(max_epochs, results)


def test_rel_perm():
    t = torch.arange(100000)
    x = torch.sin(t)

    x_new = _relative_permutation(x, sigma=0.05)
    noise = (x - x_new) / x
    noise = noise[~noise.isnan()]
    assert np.isclose(torch.mean(noise).item(), 0, atol=5e-3)
    assert np.isclose(torch.std(noise).item(), 0.05, atol=5e-3)

    xs = torch.linspace(-5, 5, steps=100)
    ys = torch.linspace(-5, 5, steps=100)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    z = torch.sin(torch.sqrt(x * x + y * y))

    z_new = _relative_permutation(z, sigma=0.05)
    noise = (z - z_new) / z
    noise = noise[~noise.isnan()]
    assert np.isclose(torch.mean(noise).item(), 0, atol=5e-3)
    assert np.isclose(torch.std(noise).item(), 0.05, atol=5e-3)
