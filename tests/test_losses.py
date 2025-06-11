import torch
import torch.nn as nn
import numpy as np

from softsensor.losses import *

def example():
    mean = torch.tensor([0,7,4])
    std = torch.tensor([0.1,0.2,0.01])
    outputs = (mean, std)
    targets = torch.tensor([0.5,7.3,3.7])
    return mean, std, outputs, targets

def test_dist_mse_loss():
    mean, _, outputs, targets = example()
    loss = DistributionalMSELoss()
    assert loss.distributional_loss
    assert loss(outputs, targets) == nn.MSELoss()(mean, targets)

def test_nll_loss():
    mean, std, outputs, targets = example()

    loss = GaussianNLLLoss()
    assert loss.distributional_loss
    assert loss(outputs, targets) == nn.GaussianNLLLoss()(mean, targets, std)

def test_betanll_loss():
    mean, std, outputs, targets = example()

    beta = 0.17
    loss = BetaNLL(beta)
    assert loss.distributional_loss

    expected = (nn.GaussianNLLLoss(reduction='none')(mean, targets, std**2) * (std**2)**beta).mean()
    assert loss(outputs, targets) == expected

def test_pinball_loss():
    mean, std, _, targets = example()
    lb = mean - 1.96 * std
    ub = mean + 1.96 * std

    outputs = torch.vstack([mean, lb, ub]).reshape((1,1,3,3))

    loss = PinballLoss([0.5, 0.05, 0.95])
    assert not loss.distributional_loss

    score = loss(outputs, targets)
    assert not np.isnan(score)
    
def example_ts(n):
    mean = np.random.rand(2, n)
    std = np.random.rand(2, n)
    return mean, std

def test_psd():
    mean, std = example_ts(1000)
    loss = PSDLoss(fs=100)
    
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    outputs = (mean, std)
    targets = torch.tensor(np.random.rand(2, 1000))
    loss(mean, targets)
    
    loss = PSDLoss(fs=100, window=128, type='log_area')
    loss(mean, targets)
    