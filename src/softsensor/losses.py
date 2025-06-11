# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
import sklearn.metrics as metr
import scipy.signal as signal
import softsensor.metrics as ssmetr

class DistributionalMSELoss(nn.Module):
    """
    Wrapper that computes MSE loss on the mean of the distribution prediction

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    def __init__(self):
        self.distributional_loss = True

    def __call__(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Computes the loss function

        Parameters
        ----------
        outputs (torch.Tensor): [mean, std]
        targets (torch.Tensor): targets

        Returns
        -------
        torch.Tensor: loss
        """
        
        #loss = nn.MSELoss()
        #return loss(outputs[0], targets)
        return F.mse_loss(outputs[0], targets)

class GaussianNLLLoss(nn.Module):
    """
    Compute the Gaussian negative log likelihood loss

    Heteroscedastic NLL loss for aleatoric uncertainty (equation 5) from the paper
    "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
    [Kendall & Gal 2017 https://arxiv.org/pdf/1703.04977.pdf]

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    def __init__(self):
        self.distributional_loss = True

    def __call__(self, outputs, targets):
        """Computes the loss function

        Parameters
        ----------
        outputs (torch.Tensor): [mean, std]
        targets (torch.Tensor): ground truth

        Returns
        -------
        torch.Tensor: loss
        """
        mu = outputs[0]
        var = outputs[1]
        # log_var = torch.log(var)
        # return (1/2 * (log_var + (targets - mu)**2 / var)).mean()
        # print(mu.shape, targets.shape, var.shape)
        return F.gaussian_nll_loss(mu, targets, var)

class BetaNLL(nn.Module):
    """
    Compute the beta negative log likelihood loss

    "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks"
    [Seitzer et al. 2022 https://openreview.net/forum?id=aPOpXlnV1T]

    Parameters
    ----------
    beta: float in range (0, 1)
        beta parameter that defines the degree to which the gradients are weighted by predicted variance
    
    Returns
    -------
    None
    """
    def __init__(self, beta=0.15):
        self.distributional_loss = True
        self.beta = beta

    def __call__(self, outputs, targets):
        """Computes the loss function

        Parameters
        ----------
        outputs (torch.Tensor): [mean, std]
        targets (torch.Tensor): targets

        Returns
        -------
        torch.Tensor: loss
        """
        mu = outputs[0]
        var = outputs[1]**2
        log_var = torch.log(var)
        return (1/2 * (log_var + (targets - mu)**2 / var) * (var ** self.beta)).mean()

class PinballLoss(nn.Module):
    """
    Compute the Pinball loss (quantile loss)
    
    Based on the qr loss as defined in
    "Estimating conditional quantiles with the help of the pinball loss" [Steinwart & Christmann 2011]
    https://arxiv.org/pdf/1102.2101.pdf

    Parameters
    ----------
    quantiles: list[x] with x float in range (0, 1)
        quantiles to compute the loss on

    Returns
    -------
    None
    """
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
        self.distributional_loss = False
        
    def forward(self, pred, target):
        """Computes the loss

        Parameters
        ----------
        pred (torch.Tensor): predicted quantiles
        target (torch.Tensor): ground truth for median

        Returns
        -------
        torch.Tensor: loss
        """
        # Compute residual for each quantile
        error = target[:, None] - pred
        
        upper =  self.quantiles * error # loss if error >= 0
        lower = (self.quantiles - 1) * error # loss if error < 0

        # The indicator function can be replaced by max since the value of choice is positive and the other one is negative
        losses = torch.max(lower, upper)
        loss = torch.mean(torch.sum(losses, dim=1))
        return loss
    
class PSDLoss():
    """
    Compute the PSD loss

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    def __init__(self, fs, freq_range=None, window=128, type='msle'):
        self.fs = fs
        self.freq_range = freq_range
        self.window = window
        self.distributional_loss = False
        self.type = type

    def __call__(self, outputs, targets, type='msle'):
        """Computes the loss function

        Parameters
        ----------
        outputs (torch.Tensor): outputs
        targets (torch.Tensor): targets

        Returns
        -------
        torch.Tensor: loss
        """

        # Compute the power spectral density

        f, psd_original = signal.welch(outputs, fs=self.fs, nperseg=self.window)
        f, psd_targets = signal.welch(targets, fs=self.fs, nperseg=self.window)

        # Optionally, you can specify a frequency range to focus on
        if self.freq_range is not None:
            psd_original = psd_original[:, 
                (self.freq_range[0] < f) & (f < self.freq_range[1])
            ]
            psd_targets = psd_targets[:, 
                (self.freq_range[0] < f) & (f < self.freq_range[1])
            ]

        # remove zero frequency component
        if f[0] == 0:
            f = f[1:]
            psd_original = psd_original[:, 1:]
            psd_targets = psd_targets[:, 1:]

        # Compute the mean squared log error
        if self.type=='msle':
            try:
                loss = metr.mean_squared_log_error(psd_original, psd_targets)
                return torch.tensor(loss)
            except ValueError:
                return torch.tensor(float('nan'))
        elif self.type=='log_area':
            try:
                return ssmetr.log_area_error(psd_original, psd_targets, f)
            except ValueError:
                return torch.tensor(float('nan'))
            

    
    