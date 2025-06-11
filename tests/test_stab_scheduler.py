# -*- coding: utf-8 -*-
"""
Created on Sat May 13 11:27:13 2023

@author: WET2RNG
"""
import pytest
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import softsensor.stab_scheduler as ss
from softsensor.train_model import train_model
from softsensor.autoreg_models import ARNN
from softsensor.meas_handling import Meas_handling

@pytest.fixture()
def handler():
    d = {'in_col1': np.random.rand(1000),
         'in_col2': np.random.rand(1000),
         'out_col1': np.random.rand(1000)}

    handler = Meas_handling([pd.DataFrame(d)], ['input_df'],
                            ['in_col1', 'in_col2'], ['out_col1'], 1000,
                            [pd.DataFrame(d)], ['test'])

    return handler


def test_stability_loss():
    net = ARNN(input_channels=2, pred_size=1, window_size=37, rnn_window=37,
               hidden_size=None, activation='sigmoid')

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            base_stability = ss.Stability_Criterion(net)
            assert False
        except Warning:
            assert True

    net = ARNN(input_channels=2, pred_size=1, window_size=37, rnn_window=37,
               hidden_size=None)

    set_to_one(net)
    norm = torch.ones((1, 37)).norm(p='fro')
    crit = norm - 1 / (1 * np.sqrt(net.rnn_window))
    base_stability = ss.Stability_Criterion(net)
    assert crit == base_stability

    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    optimizer.zero_grad()
    criterion = ss.Stability_Loss(nn.MSELoss(), net, stabelizer=0, local_wd=0)
    target = torch.tensor((32, 2, 50), dtype=torch.float)
    out = torch.tensor((32, 2, 50), dtype=torch.float, requires_grad=True)
    loss = criterion(out, target)
    
    loss.backward()
    optimizer.step()

    stability = ss.Stability_Criterion(net)
    assert stability == base_stability
    
    optimizer.zero_grad()
    criterion = ss.Stability_Loss(nn.MSELoss(), net, stabelizer=0.1, local_wd=0)
    target = torch.tensor((32, 2, 50), dtype=torch.float)
    out = torch.tensor((32, 2, 50), dtype=torch.float, requires_grad=True)
    loss = criterion(out, target)

    loss.backward()
    optimizer.step()

    stability = ss.Stability_Criterion(net)
    assert stability < base_stability
    base_stability = stability
    
    optimizer.zero_grad()
    criterion = ss.Stability_Loss(nn.MSELoss(), net, stabelizer=0, local_wd=0.1)
    target = torch.tensor((32, 2, 50), dtype=torch.float)
    out = torch.tensor((32, 2, 50), dtype=torch.float, requires_grad=True)
    loss = criterion(out, target)

    loss.backward()
    optimizer.step()

    stability = ss.Stability_Criterion(net)
    assert stability < base_stability
    base_stability = stability
    
    optimizer.zero_grad()
    criterion = ss.Stability_Loss(nn.MSELoss(), net, stabelizer=0.1, local_wd=0.1)
    target = torch.tensor((32, 2, 50), dtype=torch.float)
    out = torch.tensor((32, 2, 50), dtype=torch.float, requires_grad=True)
    loss = criterion(out, target)

    loss.backward()
    optimizer.step()
    
    stability = ss.Stability_Criterion(net)
    assert stability < base_stability
    

def None_stab():
    net = ARNN(input_channels=2, pred_size=1, window_size=37, rnn_window=37,
               hidden_size=None, activation='sigmoid')

    base_stability = ss.Stability_Criterion(net)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = ss.Stability_Loss(nn.MSELoss(), net, stabelizer=None, local_wd=None)
    target = torch.tensor((32, 2, 50), dtype=torch.float)
    out = torch.tensor((32, 2, 50), dtype=torch.float, requires_grad=True)
    loss = criterion(out, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    stability = ss.Stability_Criterion(net)
    assert stability == base_stability

def test_scheduler(handler):
    scheduler = ss.const_stab(10**(-4))
    windowsize = 10
    data = handler.give_torch_loader(windowsize, rnn_window=windowsize,
                                     forecast=1, batch_size=100)
    model = ARNN(input_channels=2, pred_size=1, window_size=windowsize,
                 rnn_window=windowsize, hidden_size=[32, 16])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    scheduler = ss.const_stab(1e-4, track=False)
    results = train_model(model, data[0], max_epochs=10, optimizer=optimizer,
                          device='cpu', criterion=nn.MSELoss(), val_loader=data[1],
                          print_results=False, give_results=True,
                          rel_perm=0, stabelizer=scheduler, local_wd=None)
    
    assert len(results['stability_score']) == 11
    
    scheduler = ss.none_stab(track=False)
    results = train_model(model, data[0], max_epochs=10, optimizer=optimizer,
                          device='cpu', criterion=nn.MSELoss(), val_loader=data[1],
                          print_results=False, give_results=True,
                          rel_perm=0, stabelizer=scheduler, local_wd=None)
    
    assert len(results['stability_score']) == 11
    
    
    scheduler = ss.linear_stab(model, s1=10^-2, track=False)
    results = train_model(model, data[0], max_epochs=10, optimizer=optimizer,
                          device='cpu', criterion=nn.MSELoss(), val_loader=data[1],
                          print_results=False, give_results=True,
                          rel_perm=0, stabelizer=scheduler, local_wd=None)
    
    assert len(results['stability_score']) == 11
    
    scheduler = ss.heaviside_stab(10**(-1))    
    results = train_model(model, data[0], max_epochs=10, optimizer=optimizer,
                          device='cpu', criterion=nn.MSELoss(), val_loader=data[1],
                          print_results=False, give_results=True,
                          rel_perm=0, stabelizer=scheduler, local_wd=None)
    
    assert len(results['stability_score']) == 11
    assert len(results['stabelizer']['eta']) == 80
    assert len(results['stabelizer']['sc']) == 80

    scheduler = ss.log_lin_stab(model, s0=10**(-8), s1=10**(-1), m=.1)
    results = train_model(model, data[0], max_epochs=10, optimizer=optimizer,
                          device='cpu', criterion=nn.MSELoss(), val_loader=data[1],
                          print_results=False, give_results=True,
                          rel_perm=0, stabelizer=scheduler, local_wd=None)
    
    assert len(results['stability_score']) == 11
    assert len(results['stabelizer']['eta']) == 80
    assert len(results['stabelizer']['sc']) == 80


    scheduler = ss.linear_stab(model, s1=10^-2, track_n=10)
    results = train_model(model, data[0], max_epochs=5, optimizer=optimizer,
                          device='cpu', criterion=nn.MSELoss(), val_loader=data[1],
                          print_results=False, give_results=True,
                          rel_perm=0, stabelizer=scheduler, local_wd=None)
    
    assert len(results['stability_score']) == 6
    assert len(results['stabelizer']['eta']) == 4
    assert len(results['stabelizer']['sc']) == 4

    scheduler = ss.log_lin_stab(model, s0=10**(-8), s1=10**(-1), m=.1, track_n=10)
    results = train_model(model, data[0], max_epochs=5, optimizer=optimizer,
                          device='cpu', criterion=nn.MSELoss(), val_loader=data[1],
                          print_results=False, give_results=True,
                          rel_perm=0, stabelizer=scheduler, local_wd=None)
    
    assert len(results['stability_score']) == 6
    assert len(results['stabelizer']['eta']) == 4
    assert len(results['stabelizer']['sc']) == 4

def test_get_scheduler():
    model = ARNN(input_channels=2, pred_size=1, window_size=10,
                 rnn_window=10, hidden_size=[32, 16])
    scheduler = ss.get_scheduler(stab_method=1e-4, model=None)
    assert isinstance(scheduler, ss.const_stab)
    
    scheduler = ss.get_scheduler(stab_method=None, model=None)
    assert isinstance(scheduler, ss.none_stab)
    
    scheduler = ss.get_scheduler(stab_method='lin', s1=1e-2, model=model)
    assert isinstance(scheduler, ss.linear_stab)
    
    scheduler = ss.get_scheduler(stab_method='log_lin', s1=1e-2, s0=1e-6, model=model)
    assert isinstance(scheduler, ss.log_lin_stab)
    
    scheduler = ss.get_scheduler(stab_method='heaviside', model=model)
    assert isinstance(scheduler, ss.heaviside_stab)
    
    try:
        scheduler = ss.get_scheduler(stab_method='nonsense', model=model)
        assert False
    except Warning:
        assert True
    

def set_to_one(model):
    for name, param in model.named_parameters():
        values = torch.ones(param.shape)
        param.data = values