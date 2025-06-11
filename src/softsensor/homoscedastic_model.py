# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from softsensor.autoreg_models import _prediction


def fit_homoscedastic_var(dataframes, model_out, target_out):
    """
    Determines a homoscedastic uncertainty as the average sample variance on the training set
    where each point prediction is regarded as a single sample from a distribution with the ground truth as mean.

    Parameters
    ----------
    model : torch.Module
        Point prediction model
    train_list : list[Dataloader]
        dataset to fit the homoscedastic variance

    Returns
    -------
    homoscedastic_var: tensor[float], shape=[model.pred_size]
        Homoscedastic variance
    """

    var_list = []
 
    for track in dataframes:
        mean = torch.tensor(track[model_out].values.transpose())
        targets = torch.tensor(track[target_out].values.transpose())
        homoscedastic_var = _fit_var(mean, targets)
        var_list.append(homoscedastic_var)
        
    homoscedastic_var = torch.vstack(var_list).mean(dim=0)
    return homoscedastic_var


def _fit_var(mean, targets):
    """
    Fit homoscedastic variance to tensors of shape [channels, time series length]

    Parameters
    ----------
    mean : tensor[float] shape: [channels, ts length]
        Model prediction.
    targets : tensor[float] shape: [channels, ts length]
        Mground truth data

    Returns
    -------
    homoscedastic_var : tensor[float], shape=[channels]

    """
    homoscedastic_var = 1/(mean.shape[1] - 1) * torch.sum((mean - targets)**2, dim=1)
    return homoscedastic_var.to(torch.float)


class HomoscedasticModel(nn.Module):
    """
    Wrapper class for point prediction models that uses a constant variance for all inputs

    Parameters
    ----------
    model : uncertainty model 
        model that is used for prediction
    homoscedastic_var : tensor[float], shape=[model.pred_size]
        Homoscedastic variance

    Returns
    -------
    None.
    
    Examples
    --------
    
    Define Model

    >>> import softsensor.autoreg_models
    >>> import softsensor.homoscedastic_model as hm
    >>> import torch
    >>> m = softsensor.autoreg_models.ARNN(2, 1, 10, 10, [16, 8])
    >>> vars = torch.tensor([1])
    >>> homosc_m = hm.HomoscedasticModel(m, vars)
    >>> input = torch.randn(32, 2, 10)
    >>> rec_input = torch.randn(32, 1, 10)
    >>> output = homosc_m(input, rec_input)
    >>> print(output[0].shape)
    torch.Size([32, 1, 1])
    >>> print(output[1].shape)
    torch.Size([32, 1, 1])
    
    Define Data

    >>> import softsensor.meas_handling as ms
    >>> import numpy as np
    >>> import pandas as pd
    >>> t = np.linspace(0, 1.0, 101)
    >>> d = {'inp1': np.random.randn(101),
             'inp2': np.random.randn(101),
             'out': np.random.randn(101)}
    >>> handler = ms.Meas_handling([pd.DataFrame(d, index=t)], ['train'],
                                   ['inp1', 'inp2'], ['out'], fs=100)

    Define Model with Uncertainty

    >>> loader = handler.give_list(window_size=10, keyword='training',
                                   rnn_window=10, batch_size=1)
    >>> vars = torch.tensor([1])
    >>> homosc_m = hm.HomoscedasticModel(m, vars)
    >>> mean, var = homosc_m.prediction(loader[0])
    >>> print(mean.shape)
    torch.Size([1, 101])
    >>> print(var.shape)
    torch.Size([1, 101])

    """

    def __init__(self, model, homoscedastic_var):
        super().__init__()
        
        self.model = model
        self.homoscedastic_var = homoscedastic_var

        self.Type = model.Type
        self.window_size = model.window_size
        self.rnn_window = model.rnn_window
        self.forecast = model.forecast
        self.pred_size = model.pred_size
        
        self.Type = model.Type
        self.Pred_Type = 'Mean_Var'
        self.Ensemble = False
    
    def forward(self, *args):
        """
        Forward function to propagate through the network

        Parameters
        ----------
        *args : input arguments 
            (need to be the same as the Point model in __init__() needs)
        Returns
        -------
        mean: torch.tensor dtype=torch.float()
            shape=[batch size, pred_size]
        std: torch.tensor dtype=torch.float() in [0,1]
            shape=[batch size, pred_size]

        """
        mean = self.model(*args)
        #var = [torch.ones([mean.shape[0], 1, mean.shape[2]]) * v for v in self.homoscedastic_var]
        var = torch.ones(mean.shape)
        for i, v in enumerate(self.homoscedastic_var):
            var[:, i, :] = var[:, i, :]*v
        return mean, var
    

    def estimate_uncertainty_mean_std(self, *args):
        return self(*args)

    def prediction(self, dataloader, device="cpu"):
        """
        Prediction of a whole Time Series

        Parameters
        ----------
        dataloader : Dataloader
            Dataloader to predict output
        device : str, optional
            device to compute on. The default is 'cpu'.

        Returns
        -------
        if loss_ft=None:
            (torch.Tensor, list[torch.Tensor])
                tuple of Torch Tensor of same length as input and var
        if loss_ft=torch loss funciton:
            (torch.Tensor, list[torch.Tensor], loss)
                tuple of Torch Tensor of same length as input, var and loss

        """
        return _prediction(self, dataloader, device)
