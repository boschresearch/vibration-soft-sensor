# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:26:16 2022

@author: WET2RNG
"""

import math
import inspect
from enum import Enum
import numpy as np
#from import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import (grad, Variable)
from torchinfo import summary
# from captum.attr import LRP
from scipy.stats.qmc import LatinHypercube

from softsensor.losses import GaussianNLLLoss
from softsensor.model import (CNN, Feed_ForwardNN, Freq_Att_CNN,
                              _filter_parameters)
from softsensor.train_model import train_model



class SensitivityMethods(Enum):
    """
    Enumeration class, representing all currently available sensitivity methods.
    Applicable methods are: 'gradient' and 'perturbation'.
    """
    GRADIENT = 'gradient'
    SMOOTH_GRAD = 'smooth_grad'
    INTEGRATED_GRADIENT = 'integrated_gradient'
    PERTURBATION = 'perturbation'


class _Autoregressive_Model(nn.Module):
    """
    Parent class for all Autoregressive models

    Parameters
    ----------
    input_channels : int
        Number of input channels
    pred_size : int
        Number of predicted values
    window_size : int
        window size of the input. Number of Datapoints in the windowed
        external excitation signal
    rnn_window : int, optional
        Window Size of the Recurrent Connection

    Returns
    -------
    None.
    """
    def __init__(self, input_channels, pred_size, window_size, rnn_window,
                 forecast):

        super().__init__()

        self.input_channels = input_channels
        self.pred_size = pred_size
        self.window_size = window_size
        self.rnn_window = rnn_window
        self.forecast = forecast

        self.Type = 'AR'
        self.Pred_Type = 'Point'
        self.Ensemble = False

    def prediction(self, dataloader, device='cpu', sens_params=None):
        """
        Prediction of a whole Time Series

        Parameters
        ----------
        dataloader : Dataloader
            Dataloader to predict output
        device : str, optional
            device to compute on. The default is 'cpu'.
        sens_params : dict, optional
            Dictionary that contains the parameters for the sensitivity analysis.
            Key 'method' defines the method for sensitivity analysis: 'gradient' or 'perturbation'.
            Key 'comp' defines whether gradients are computed for sensitivity analysis.
            Key 'plot' defines whether the results of the sensitivity analysis are visualized.
            Key 'verbose' defines whether the information about the sensitivity analysis is printed.
            Key 'sens_length' defines the number of randomly sampled subset of timesteps for the analysis.
            The default is None, i.e. no sensitivity analysis is computed.

        Returns
        -------
        if loss_ft=None:
            torch.Tensor
                Torch Tensor of same langth as input
        if loss_ft=torch loss function:
            (torch.Tensor, loss)
                tuple of Torch Tensor of same langth as input and loss
        """
        return _predict_ARNN(self, dataloader, device, sens_params)


class ARNN(_Autoregressive_Model):
    """
    Autoregressive Neural Network with linear layers
        
    .. math:: window_size = rnn_window = tau
    .. math:: forecast = 1
    
    Parameters
    ----------
    input_channels : int
        Number of input channels
    pred_size : int
        Number of predicted values
    window_size : int
        Size of the sliding window applied to the time series
    rnn_window : int
        Window Size of the Recurrent Connection before the DNN.
    hidden_size : list of int or None, optional
        List gives the size of hidden units. The default is None.
    activation : str, optional
        Activation function to activate the feature space.
        The default is 'relu'.
    bias : bool, optional
        If True, bias weights are used. The default is True.
    dropout : float [0,1], optional
        Adds dropout Layers after each Linear Layer. The default is None.
    forecast : int, optional
        Size of the forecast. The default is 1
    concrete_dropout : bool, optional
        Whether to use normal or concrete dropout Layers if dropout is not None. The default is False

    Returns
    -------
    None.
    
    Examples
    --------
    >>> import softsensor.autoreg_models
    >>> import torch
    >>> m = softsensor.autoreg_models.ARNN(2, 1, 10, 10, [16, 8])
    >>> input = torch.randn(32, 2, 10)
    >>> rec_input = torch.randn(32, 1, 10)
    >>> output = m(input, rec_input)
    >>> print(output.shape)
    torch.Size([32, 1, 1])
    
    
    
    >>> import softsensor.meas_handling as ms
    >>> import numpy as np
    >>> import pandas as pd
    >>> t = np.linspace(0, 1.0, 101)
    >>> d = {'inp1': np.random.randn(101),
             'inp2': np.random.randn(101),
             'out': np.random.randn(101)}
    >>> handler = ms.Meas_handling([pd.DataFrame(d, index=t)], ['train'],
                                   ['inp1', 'inp2'], ['out'], fs=100)
    >>> loader = handler.give_list(window_size=10, keyword='training',
                                   rnn_window=10, batch_size=1)
    >>> pred = m.prediction(loader[0])
    >>> print(pred.shape)
    torch.Size([1, 101])
    """
    def __init__(self, input_channels, pred_size, window_size, rnn_window,
                 hidden_size=None, activation='relu', bias=True, dropout=None,
                 forecast=1, concrete_dropout=False, bn=False):

        _Autoregressive_Model.__init__(self, input_channels, pred_size,
                                      window_size, rnn_window, forecast)

        self.params = _filter_parameters(locals().copy())
        self.activation = activation

        flatten_size = window_size*input_channels+rnn_window*pred_size

        # Define Linear Network
        self.DNN = Feed_ForwardNN(flatten_size, pred_size*forecast,
                                hidden_size, activation=activation,
                                bias=bias, dropout=dropout, concrete_dropout=concrete_dropout,
                                bn=bn)

    def forward(self, inp, x_rec):
        """
        Forward function to propagate through the network

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation,
            shape=[batch size, external channels, window_size]
        x_rec : torch.tensor, dtype=torch.float
            Recurrent Input for forward Propagation.
            shape=[batch size, pred_size, rnn_window]

        Returns
        -------
        output: torch.tensor dtype=torch.float()
            shape=[batch size, pred_size, forecast]
        """
        inp = torch.flatten(inp, start_dim=1)
        x_rec = torch.flatten(x_rec, start_dim=1)

        pred = self.DNN(torch.cat((inp, x_rec), dim=1))
        pred = pred.reshape(-1, self.pred_size, self.forecast)
        return pred

    def forward_sens(self, inp):
        """
        Forward function to propagate through the network, but only with one input tensor
        that is already concatenated to allow for gradient-based sensitivity analysis

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation that is already concatenated,
            shape=[batch size, external channels*window_size + pred_size*rnn_window]

        Returns
        -------
        output: torch.tensor dtype=torch.float()
            shape=[batch size, pred_size, forecast]
        """
        pred = self.DNN(inp)
        pred = pred.reshape(-1, self.pred_size, self.forecast)
        return pred

    def get_recurrent_weights(self):
        """
        Function that returns the weight that effect the Recurrent input of the
        Network

        Returns
        -------
        recurrent_weights : list of weight Tensors
            List of the Weights that effect the Recurren input of the Network.
            
        Example
        -------
        Based on the example in the introduction
        
        >>> rec_w = m.get_recurrent_weights()
        >>> print(rec_w[0].shape)
        torch.Size([16, 10])
        >>> print(rec_w[1].shape)
        torch.Size([8, 16])
        >>> print(rec_w[2].shape)
        torch.Size([1, 8])
        """
        input_pred_slice = slice(-self.rnn_window*self.pred_size, None)
        return _get_recurrent_weights(self.DNN.named_parameters(), input_pred_slice)


class DensityEstimationARNN(ARNN):
    """
    ARNN with two outputs to predict mean and variance (aleatoric uncertainty)

    Parameters
    ----------
    input_channels : int
        Number of input channels
    pred_size : int
        Number of predicted values
    window_size : int
        Size of the sliding window applied to the time series
    rnn_window : int
        Window Size of the Recurrent Connection before the DNN.
    hidden_size : list of int or None, optional
        List gives the size of hidden units. The default is None.
    activation : str, optional
        Activation function to activate the feature space.
        The default is 'relu'.
    bias : bool, optional
        If True, bias weights are used. The default is True.
    dropout : float [0,1], optional
        Adds dropout Layers after each Linear Layer. The default is None
    forecast : int, optional
        Size of the forecast. The default is 1
    concrete_dropout : bool, optional
        Whether to use normal or concrete dropout Layers if dropout is not None. The default is False

    Returns
    -------
    None.
    
    Examples
    --------
    >>> import softsensor.autoreg_models
    >>> import torch
    >>> params = {'input_channels': 2,
                  'pred_size': 1,
                  'window_size': 10,
                  'rnn_window': 10}
    >>> m = softsensor.autoreg_models.DensityEstimationARNN(**params, hidden_size=[16, 8])
    >>> input = torch.randn(32, 2, 10)
    >>> rec_input = torch.randn(32, 1, 10)
    >>> output = m(input, rec_input)
    >>> print(output[0].shape) #Mean Prediction
    torch.Size([32, 1, 1]) 
    >>> print(output[1].shape) #Var Prediction
    torch.Size([32, 1, 1]) 
    """
    def __init__(self, input_channels, pred_size, window_size, rnn_window,
                 hidden_size=None, activation='relu', bias=True, dropout=None, forecast=1, concrete_dropout=False,
                 bn=False):
        # Define Linear Network with twice the forecast (mean and var of Gaussian)
        ARNN.__init__(self, input_channels, pred_size, window_size, rnn_window,
                 hidden_size, activation, bias, dropout, 2*forecast, concrete_dropout, bn)
        self.params = _filter_parameters(locals().copy())
        self.forecast = forecast
        self.Pred_Type = 'Mean_Var'

    def forward(self, inp, x_rec):
        """
        Forward function to propagate through the network

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation,
            shape=[batch size, external channels, window_size]
        x_rec : torch.tensor, dtype=torch.float
            Recurrent Input for forward Propagation.
            shape=[batch size, pred_size, rnn_window]

        Returns
        -------
        mean: torch.tensor dtype=torch.float()
            shape=[batch size, pred_size, forecast]
        var torch.tensor dtype=torch.float() in [0,1]
            shape=[batch size, pred_size, forecast]
        """
        inp = torch.flatten(inp, start_dim=1)
        x_rec = torch.flatten(x_rec, start_dim=1)
        pred = self.DNN(torch.cat((inp, x_rec), dim=1))
        pred = pred.reshape(-1, self.pred_size, self.forecast, 2)
        
        mean, hidden_std = pred[:,:,:,0], pred[:,:,:,1]
        var = F.softplus(hidden_std)
        return mean, var

    def forward_sens(self, inp):
        """
        Forward function to propagate through the network, but only with one input tensor
        that is already concatenated to allow for gradient-based sensitivity analysis

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation,
            shape=[batch size, external channels*window_size + pred_size*rnn_window]

        Returns
        -------
        mean: torch.tensor dtype=torch.float()
            shape=[batch size, pred_size, forecast]
        var torch.tensor dtype=torch.float() in [0,1]
            shape=[batch size, pred_size, forecast]
        """
        pred = self.DNN(inp)
        pred = pred.reshape(-1, self.pred_size, self.forecast, 2)
        
        mean, hidden_std = pred[:,:,:,0], pred[:,:,:,1]
        var = F.softplus(hidden_std)
        return mean, var
    
    def estimate_uncertainty(self, inp, x_rec):
        """
        Wrapper of forward pass

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation,
            shape=[batch size, external channels, window_size]
        x_rec : torch.tensor, dtype=torch.float
            Recurrent Input for forward Propagation.
            shape=[batch size, pred_size, rnn_window]

        Returns
        -------
        (mean, var)
            mean: torch.tensor dtype=torch.float()
                shape=[batch size, pred_size]

            var: torch.tensor dtype=torch.float() in [0,1]
                shape=[batch size, pred_size]
        """
        return self(inp, x_rec)

    def estimate_uncertainty_mean_std(self, inp, x_rec):
        return self(inp, x_rec)
    
    def prediction(self, dataloader, device='cpu', sens_params=None):
        """
        Prediction of a whole Time Series

        Parameters
        ----------
        dataloader : Dataloader
            Dataloader to predict output
        device : str, optional
            device to compute on. The default is 'cpu'.
        sens_params : dict, optional
            Dictionary that contains the parameters for the sensitivity analysis.
            Key 'method' defines the method for sensitivity analysis: 'gradient' or 'perturbation'.
            Key 'comp' defines whether gradients are computed for sensitivity analysis.
            Key 'plot' defines whether the results of the sensitivity analysis are visualized.
            Key 'verbose' defines whether the information about the sensitivity analysis is printed.
            Key 'sens_length' defines the number of randomly sampled subset of timesteps for the analysis.
            The default is None, i.e. no sensitivity analysis is computed.

        Returns
        -------
        if loss_ft=None:
            (torch.Tensor, list[torch.Tensor])
                tuple of Torch Tensor of same length as input and var
        if loss_ft=torch loss funciton:
            (torch.Tensor, list[torch.Tensor], loss)
                tuple of Torch Tensor of same length as input, var and loss
        """
        return _predict_arnn_uncertainty(self, dataloader, device, sens_params)

    def get_recurrent_weights(self):
        """
        Function that returns the weight that effect the Recurrent input of the
        Network (mean network)

        Returns
        -------
        recurrent_weights : list of weight Tensors
            List of the Weights that effect the Recurrent input of the Network.
        """
        input_pred_slice = slice(-self.rnn_window*self.pred_size, None)
        return _get_recurrent_weights(self.DNN.named_parameters(), input_pred_slice, True,
                                      self.pred_size, self.forecast)

class SeparateMVEARNN(ARNN):
    """
    ARNN with two independent subnetworks to predict mean and variance (aleatoric uncertainty)
    
    .. image:: C:/Users/wet2rng/Desktop/Coding/SoftSensor/doc/img/Separate_MVE.png

    Parameters
    ----------
    input_channels : int
        Number of input channels
    pred_size : int
        Number of predicted values
    window_size : int
        Size of the sliding window applied to the time series
    rnn_window : int
        Window Size of the Recurrent Connection before the DNN.
    mean_model : torch.Module
        Model for point prediction
    var_hidden_size : list[int] or None, optional
        List gives the size of hidden variance network units. The default is None.
    activation : str, optional
        Activation function to activate the feature space.
        The default is 'relu'.
    bias : bool, optional
        If True, bias weights are used. The default is True.
    dropout : float [0,1], optional
        Adds dropout Layers after each Linear Layer. The default is None
    forecast : int, optional
        Size of the forecast. The default is 1
    concrete_dropout : bool, optional
        Whether to use normal or concrete dropout Layers if dropout is not None. The default is False

    Returns
    -------
    None.
    
    Note
    -------
    
    See "Optimal Training of Mean Variance Estimation Neural Networks"
    [Sluijterman et al. 2023 https://arxiv.org/abs/2302.08875]

    Examples
    --------
    >>> import softsensor.autoreg_models
    >>> import torch
    >>> params = {'input_channels': 2,
                  'pred_size': 1,
                  'window_size': 10,
                  'rnn_window': 10}
    >>> mean_model = softsensor.autoreg_models.ARNN(**params, hidden_size=[16, 8])
    >>> m = softsensor.autoreg_models.SeparateMVEARNN(**params,mean_model=mean_model,
                                                      var_hidden_size=[16, 8])
    >>> input = torch.randn(32, 2, 10)
    >>> rec_input = torch.randn(32, 1, 10)
    >>> output = m(input, rec_input)
    >>> print(output[0].shape) #Mean Prediction
    torch.Size([32, 1, 1]) 
    >>> print(output[1].shape) #VarPrediction
    torch.Size([32, 1, 1]) 
    """
    def __init__(self, input_channels, pred_size, window_size, rnn_window, mean_model, var_hidden_size=None,
                 activation='relu', bias=True, dropout=None, forecast=1, concrete_dropout=False, bn=False):
        # Network for variance prediction
        ARNN.__init__(self, input_channels, pred_size, window_size, rnn_window,
                 var_hidden_size, activation, bias, dropout, forecast, concrete_dropout, bn)

        self.params = _filter_parameters(locals().copy())

        # Network for mean prediction
        self.mean_model = mean_model
        self.Pred_Type = 'Mean_Var'

    def forward(self, inp, x_rec):
        """
        Forward function to propagate through the MVE network

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation,
            shape=[batch size, external channels, window_size]
        x_rec : torch.tensor, dtype=torch.float
            Recurrent Input for forward Propagation.
            shape=[batch size, pred_size, rnn_window]
            
        Returns
        -------
        mean: torch.tensor dtype=torch.float()
            shape=[batch size, pred_size, forecast]
        var: torch.tensor dtype=torch.float() in [0,1]
            shape=[batch size, pred_size, forecast]
        """
        inp = torch.flatten(inp, start_dim=1)
        x_rec = torch.flatten(x_rec, start_dim=1)

        mean = self.mean_model(inp, x_rec)

        hidden_std = self.DNN(torch.cat((inp, x_rec), dim=1))
        hidden_std = hidden_std.reshape(-1, self.pred_size, self.forecast)
        var = F.softplus(hidden_std)

        return mean, var
    
    def forward_sens(self, inp):
        """
        Forward function to propagate through the network, but only with one input tensor
        that is already concatenated to allow for gradient-based sensitivity analysis

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation,
            shape=[batch size, external channels*window_size + pred_size*rnn_window]

        Returns
        -------
        mean: torch.tensor dtype=torch.float()
            shape=[batch size, pred_size, forecast]
        var torch.tensor dtype=torch.float() in [0,1]
            shape=[batch size, pred_size, forecast]
        """
        mean = self.mean_model.forward_sens(inp)
        
        hidden_std = self.DNN(inp)
        hidden_std = hidden_std.reshape(-1, self.pred_size, self.forecast)
        var = F.softplus(hidden_std)
        
        return mean, var
    
    def estimate_uncertainty(self, inp, x_rec):
        """
        Wrapper of forward pass

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation,
            shape=[batch size, external channels, window_size]
        x_rec : torch.tensor, dtype=torch.float
            Recurrent Input for forward Propagation.
            shape=[batch size, pred_size, rnn_window]

        Returns
        -------
        (mean, var)
            mean: torch.tensor dtype=torch.float()
                shape=[batch size, pred_size]

            var: torch.tensor dtype=torch.float() in [0,1]
                shape=[batch size, pred_size]
        """
        return self(inp, x_rec)

    def estimate_uncertainty_mean_std(self, inp, x_rec):
        """
        Wrapper of forward pass

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation,
            shape=[batch size, external channels, window_size]
        x_rec : torch.tensor, dtype=torch.float
            Recurrent Input for forward Propagation.
            shape=[batch size, pred_size, rnn_window]

        Returns
        -------
        (mean, var)
            mean: torch.tensor dtype=torch.float()
                shape=[batch size, pred_size]

            var: torch.tensor dtype=torch.float() in [0,1]
                shape=[batch size, pred_size]
        """
        return self(inp, x_rec)

    def prediction(self, dataloader, device='cpu', sens_params=None):
        """
        Prediction of a whole Time Series

        Parameters
        ----------
        dataloader : Dataloader
            Dataloader to predict output
        device : str, optional
            device to compute on. The default is 'cpu'.
        sens_params : dict, optional
            Dictionary that contains the parameters for the sensitivity analysis.
            Key 'method' defines the method for sensitivity analysis: 'gradient' or 'perturbation'.
            Key 'comp' defines whether gradients are computed for sensitivity analysis.
            Key 'plot' defines whether the results of the sensitivity analysis are visualized.
            Key 'verbose' defines whether the information about the sensitivity analysis is printed.
            Key 'sens_length' defines the number of randomly sampled subset of timesteps for the analysis.
            The default is None, i.e. no sensitivity analysis is computed.

        Returns
        -------
        if loss_ft=None:
            (torch.Tensor, list[torch.Tensor])
                tuple of Torch Tensor of same length as input and var
        if loss_ft=torch loss function:
            (torch.Tensor, list[torch.Tensor], loss)
                tuple of Torch Tensor of same length as input, var and loss
        """
        return _predict_arnn_uncertainty(self, dataloader, device, sens_params)


    def get_recurrent_weights(self):
        """
        Function that returns the weight that effect the Recurrent input of the
        Network (mean network)

        Returns
        -------
        recurrent_weights : list of weight Tensors
            List of the Weights that effect the Recurrent input of the Network.
        """
        return self.mean_model.get_recurrent_weights()


class QuantileARNN(ARNN):
    """
    ARNN with multiple outputs to predict quantiles

    Parameters
    ----------
    input_channels : int
        Number of input channels
    pred_size : int
        Number of predicted values
    window_size : int
        Size of the sliding window applied to the time series
    rnn_window : int
        Window Size of the Recurent Connection before the DNN.
    hidden_sizes : list of three lists of int or None, optional
        [hidden_mean_size, hidden_var_size, hidden_shared_size]
        List gives the size of hidden mean, variance and shared network units. The default is None.
    activation : str, optional
        Activation function to activate the feature space.
        The default is 'relu'.
    bias : bool, optional
        If True, bias weights are used. The default is True.
    dropout : float [0,1], optional
        Adds dropout Layers after each Linear Layer. The default is None
    forecast : int, optional
        Size of the forecast. The default is 1
    concrete_dropout : bool, optional
        Whether to use normal or concrete dropout Layers if dropout is not None. The default is False
    n_quantiles: int, optional
        Number of quantiles to predict. The default is 39 (median and 19 PIs between 0 and 1)
    mean_model : torch.Module, optional
        Model for point prediction. The default is None

    Returns
    -------
    None.
    """
    def __init__(self, input_channels, pred_size, window_size, rnn_window,
                 hidden_size=None, activation='relu', bias=True, dropout=None, forecast=1, concrete_dropout=False,  n_quantiles=39, mean_model=None, bn=False):
        ARNN.__init__(self, input_channels, pred_size, window_size, rnn_window,
                 hidden_size, activation, bias, dropout, n_quantiles*forecast, concrete_dropout, bn)
        self.params = _filter_parameters(locals().copy())
        self.n_quantiles = n_quantiles
        self.forecast = forecast
        self.n_layers = len(hidden_size) if hidden_size else 0
        self.mean_model = mean_model
        self.Pred_Type = 'Quantile'

    def forward(self, inp, x_rec):
        """
        Forward function to propagate through the quantile network
        
        If mean_model is not None but a point prediction model, the mean_model is used for point prediction
        This is useful to keep the point prediction frozen during training without teacher forcing

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation,
            shape=[batch size, external channels, window_size]
        x_rec : torch.tensor, dtype=torch.float
            Recurrent Input for forward Propagation.
            shape=[batch size, pred_size, rnn_window]

        Returns
        -------
        pred: torch.tensor dtype=torch.float()
            shape=[batch size, pred_size, n_quantiles]
        """
        inp = torch.flatten(inp, start_dim=1)
        x_rec = torch.flatten(x_rec, start_dim=1)
        
        pred = self.DNN(torch.cat((inp, x_rec), dim=1))
        pred = pred.reshape(-1, self.pred_size, self.forecast, self.n_quantiles)

        if self.mean_model:
            pred[...,:,0] = self.mean_model(inp, x_rec)

        return pred
    
    def prediction(self, dataloader, device='cpu'):
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
            quantiles: list[torch.Tensor]
                list of n_quantile tensors of same length as input
        if loss_ft=torch loss funciton:
            (list[torch.Tensor], loss)
                list of n_quantile tensors of same length as input and loss
        """
        return _prediction(self, dataloader, device)
    
    def get_recurrent_weights(self):
        """
        Function that returns the weight that effect the Recurrent input of the
        Network (mean network)

        Returns
        -------
        recurrent_weights : list of weight Tensors
            List of the Weights that effect the Recurrent input of the Network.
        """
        input_pred_slice = slice(-self.rnn_window*self.pred_size, None)
        return _get_recurrent_weights(self.DNN.named_parameters(), input_pred_slice,
                                      distribution_layer=True)

'''
helpers
'''
def _get_recurrent_weights(parameters, input_pred_slice=None, distribution_layer=False, pred_size=1, forecast=1):
    Layer = 0
    recurrent_weights = []

    for name, W in parameters:
        if 'weight' in name:
            if input_pred_slice and Layer == 0:
                temp_weights = W[:, input_pred_slice]
                recurrent_weights.append(temp_weights)
            else:
                recurrent_weights.append(W)
            Layer += 1

    if distribution_layer:
        #recurrent_weights[-1] = recurrent_weights[-1][0, :]
        recurrent_weights[-1] = recurrent_weights[-1][:pred_size*forecast, :]

    return recurrent_weights


def _forward_AR(model, inp, x_rec):
    inp = torch.cat([inp, x_rec], dim=1)

    inp = model.ConvNet(inp)

    if model.bn:
        inp = model.BNLayer(inp)
    inp = torch.flatten(inp, start_dim=1)
    inp = model.DNN(inp)
    inp = inp.reshape(-1, model.pred_size, model.forecast)
    return inp



def _predict_ARNN(model, dataloader, device='cpu', sens_params=None):
    """
    Predict function for forward ARNN models

    Parameters
    ----------
    model : Model consisting of nn.Modules
    dataloader : Dataloader
        Dataloader to predict output
    device : str, optional
        device to compute on. The default is 'cpu'.
    sens_params : dict, optional
        Dictionary that contains the parameters for the sensitivity analysis.
        Key 'method' defines the method for sensitivity analysis: 'gradient' or 'perturbation'.
        Key 'comp' defines whether gradients are computed for sensitivity analysis.
        Key 'plot' defines whether the results of the sensitivity analysis are visualized.
        Key 'verbose' defines whether the information about the sensitivity analysis is printed.
        Key 'sens_length' defines the number of randomly sampled subset of timesteps for the analysis.
        (If not a multiple of the model's forecast, the number will be rounded up to the next multiple.)
        The default is None, i.e. no sensitivity analysis is computed.

    Returns
    -------
    if comp_sens is False:
        torch.Tensor : Tensor of same langth as input, containing the predictions.
    if comp_sens is True:
        (torch.Tensor, dict) : Tuple of Tensor of same length as input and sensitivity dict.
        Key is the prediction type, value is the sensitivity tensor.
    """
    fc = model.forecast
    # The Model starts with zeros as recurrent System state
    size = max(model.window_size, model.rnn_window)
    prediction = torch.zeros((model.pred_size, size))
    original_out = torch.zeros((model.pred_size, size))
    x_rec = torch.zeros(1, model.pred_size, model.rnn_window)

    # Tensors to device
    x_rec = x_rec.to(device)
    prediction = prediction.to(device)
    original_out = original_out.to(device)
    model.to(device)

    checks = _check_sens_params_pred(sens_params)
    if sens_params:
        method, comp_sens, verbose, sens_length, num_samples, std_dev, correlated = checks[:-2]
    else:
        comp_sens, verbose = checks # False

    # Initialise 3D tensor for sensitivity analysis
    if comp_sens:
        flatten_size = model.window_size*model.input_channels + model.rnn_window*model.pred_size
        num_timesteps = len(dataloader)*fc
        loader_length = num_timesteps - dataloader.dataset.subclass.add_zero
        sens_indices = np.arange(len(dataloader))

        if sens_length:
            num_timesteps, sens_indices = _random_subset_sens_indices(sens_length, fc, model.Type, dataloader)
        
        sensitivity = torch.zeros((num_timesteps, model.pred_size, flatten_size))
        if verbose:
            print(f'Shape of sensitivity tensor: {sensitivity.shape}')
            print(f'Start {method.upper()}-based Sensitivity Analysis...\n')

    # Iterate over dataloader
    idx = 0
    for i, data in enumerate(tqdm(dataloader) if verbose else dataloader):
        inputs, output = data
        inputs, output = inputs[0].to(device), output.to(device)

        # Prepare input for model to allow autograd computing gradients of outputs w.r.t. inputs
        inputs = Variable(torch.flatten(inputs, start_dim=1), requires_grad=True)
        x_rec = Variable(torch.flatten(x_rec, start_dim=1), requires_grad=True)
        inp = torch.cat((inputs, x_rec), dim=1)
        pred = model.forward_sens(inp) if comp_sens else model(inputs, x_rec)

        if comp_sens and i in sens_indices:
            sensitivity[idx:idx+fc] = _comp_sensitivity(method, model, inp, pred, num_samples, std_dev, correlated)
            idx += fc
        
        prediction = torch.cat((prediction,
                                pred.detach().reshape(model.pred_size, -1)), dim=1)
        # Recurrent Input that is used for the next prediction -> autoregressive feedback!
        x_rec = torch.unsqueeze(prediction[:, -model.rnn_window:], dim=0)
        original_out = torch.cat((original_out,
                                  output.reshape(model.pred_size, -1)), dim=1)
    
    # cut zeros from initialisation
    prediction = prediction[:, size:] # shape = [pred_size, len(dataloader)]
    original_out = original_out[:, size:]

    cut_zeros = dataloader.dataset.subclass.add_zero
    if cut_zeros != 0:
        prediction = prediction[:, :-cut_zeros]
        original_out = original_out[:, :-cut_zeros]

    prediction.cpu()
    model.to('cpu')

    if comp_sens:
        sensitivity = sensitivity[:loader_length]
        sensitivity_dict = {f'{model.Pred_Type}': sensitivity.cpu()}
        if verbose:
            print(f'{method.upper()}-based Sensitivity Analysis completed!\n')
        return prediction, sensitivity_dict
    else:
        return prediction


def _comp_grad_sens(inputs, pred, pred_type, ensemble=False, random_samples=0, amplification=1):
    """
    Compute the gradient-based sensitivity of the output w.r.t. the inputs in each timestep.

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor with already concatenated external excitation and recurrent state signals,
        shape=[batch_size, input_channels*window_size + pred_size*rnn_window]
    pred : torch.Tensor
        Output tensor that contains the predictions,
        shape=[batch_size, pred_size, forecast]
    pred_type : str
        The model's prediction type out of ('Point, 'Mean_Var'), which defines
        the number of outputs that the sensitivity analysis is performed on.
    ensemble : bool, optional
        If True, sensitivity is computed for an ensemble of models. The default is False.
    random_samples : int, optional
        Number of random samples, drawn from a standard normal distribution, to approximate the
        expected sensitivity range/distribution across the aleatoric uncertainty of MVE models.
        The default is 0 samples, i.e. no sampling is performed.
    amplification : float, optional
        Amplification factor for the uncertainty quantification of the sensitivity analysis.
        Only applicable for MVE models and only used if random_samples > 0. The default is 1.
    
    Returns
    -------
    sens_temp : torch.Tensor
        Sensitivity tensor as Jacobian that contains the gradients of the output w.r.t. the inputs,
        shape=[batch_size*forecast, pred_size, input_channels*window_size + pred_size*rnn_window]
    sens_temp_mean : torch.Tensor, optional
        Sensitivity tensor that contains the gradients of the mean output w.r.t. the inputs when using MVE models.
    sens_temp_var : torch.Tensor, optional
        Sensitivity tensor that contains the gradients of the aleatoric variance output w.r.t. the inputs when using MVE models.
    """
    batch_size, pred_size, forecast = pred[0].shape if (pred_type == 'Mean_Var' or ensemble) else pred.shape
    flatten_size = inputs.shape[1]

    def jacobian(inputs, pred):
        """
        Compute the Jacobian of the output w.r.t. the inputs for one timestep.
        """
        # initialize Jacobian tensor with NaNs
        jac = torch.full((batch_size*forecast, pred_size, flatten_size), float('nan'))
        for k in range(forecast): # loop over forecasting horizon
            for j in range(pred_size): # loop over output channels
                # shape [batch_size, input_channels*window_size + pred_size*rnn_window]
                grad_temp = grad(pred[:,j,k], inputs, grad_outputs=
                                torch.ones_like(pred[:,j,k]), create_graph=True)[0].detach()
                if j == 0:
                    jac_temp = grad_temp.unsqueeze(1)
                else:
                    jac_temp = torch.cat((jac_temp, grad_temp.unsqueeze(1)), dim=1)
            jac[k::forecast] = jac_temp
        return jac # shape [batch_size*forecast, pred_size, flatten_size]
        # return jac * inputs.unsqueeze(1).detach() # gradient * input -> between pure gradient & IG!
    
    # Compute and return the gradients
    if pred_type == 'Point':
        return jacobian(inputs, pred)
    elif pred_type == 'Mean_Var' or ensemble:
        mean_pred, var_pred = pred
        sens_temp_mean = jacobian(inputs, mean_pred)
        sens_temp_var = torch.zeros_like(sens_temp_mean)

        if random_samples:
            grads = torch.full((random_samples, batch_size*forecast, pred_size, flatten_size), float('nan'))
            samples = torch.full((random_samples, pred_size), float('nan'))
            for i in range(random_samples): # not direct sampling, but reparametrization trick: mean + eps*std, with eps ~ N(0,1)
                eps = torch.randn(1, pred_size, 1) * amplification # used for later up-scaling
                sampled_pred = mean_pred + eps * torch.sqrt(var_pred)
                grads[i] = jacobian(inputs, sampled_pred)
                samples[i] = eps[0,:,0].detach()
            sens_temp_var = grads.mean(dim=0)
            return sens_temp_mean, sens_temp_var, grads.mean(dim=1), samples
        return sens_temp_mean, sens_temp_var


def _comp_smooth_grad_sens(model, inputs, pred, pred_type, ensemble=False, num_samples=20, std_dev=0.2):
    """
    Compute the SmoothGrad-based sensitivity of the output w.r.t. the inputs in each timestep.
    
    Parameters
    ----------
    model : Model consisting of nn.Modules
    inputs : torch.Tensor
        Input tensor with already concatenated external excitation and recurrent state signals,
        shape=[batch_size, input_channels*window_size + pred_size*rnn_window]
    pred : torch.Tensor
        Output tensor that contains the predictions,
        shape=[batch_size, pred_size, forecast]
    pred_type : str
        The model's prediction type out of ('Point', 'Mean_Var'), which defines
        the number of outputs that the sensitivity analysis is performed on.
    ensemble : bool, optional
        If True, sensitivity is computed for an ensemble of models. The default is False.
    num_samples : int, optional
        Number of noisy samples to generate. The default is 10.
    std_dev : float, optional
        Standard deviation used for sampling the noisy variations of the input. The default is 0.1.

    Returns
    -------
    sens_temp : torch.Tensor
        Sensitivity tensor as Jacobian that contains the gradients of the output w.r.t. the inputs,
        shape=[batch_size*forecast, pred_size, flatten_size]
    sens_temp_mean : torch.Tensor, optional
        Sensitivity tensor that contains the gradients of the mean output w.r.t. the inputs when using MVE models.
    sens_temp_var : torch.Tensor, optional
        Sensitivity tensor that contains the gradients of the aleatoric variance output w.r.t.
        the inputs when using MVE models.

    Raises
    ------
    AssertionError
        If the number of samples is less than 1 or the standard deviation is less than 0.
    """
    assert num_samples > 0, 'Number of samples must be greater than 0 for SmoothGrad!'
    assert std_dev > 0, 'Standard deviation for Gaussian noise must be greater than 0 for SmoothGrad!'

    batch_size, pred_size, forecast = pred[0].shape if (pred_type == 'Mean_Var' or ensemble) else pred.shape
    flatten_size = inputs.shape[1]

    def smooth_grad():
        """
        Compute the SmoothGrad sensitivity of the output w.r.t. the inputs for one timestep.
        """
        if pred_type == 'Point':
            gradients = torch.full((num_samples, batch_size*forecast, pred_size, flatten_size), float('nan'))
        else:
            gradients_mean = torch.full((num_samples, batch_size*forecast, pred_size, flatten_size), float('nan'))
            gradients_var = torch.full((num_samples, batch_size*forecast, pred_size, flatten_size), float('nan'))

        for i in range(num_samples):
            noise = torch.randn_like(inputs, requires_grad=True) * std_dev
            inputs_noisy = inputs + noise
            pred_noisy = model.forward_sens(inputs_noisy)
            if i < num_samples-1:
                grads = _comp_grad_sens(inputs_noisy, pred_noisy, pred_type, ensemble)
            else: # compute gradients for the original input
                grads = _comp_grad_sens(inputs, pred, pred_type, ensemble)

            if pred_type == 'Point':
                gradients[i] = grads
            else:
                gradients_mean[i], gradients_var[i] = grads
            
        if pred_type == 'Point':
            return gradients.mean(dim=0)
        elif pred_type == 'Mean_Var' or ensemble:
            return gradients_mean.mean(dim=0), gradients_var.mean(dim=0)
    return smooth_grad()


def _comp_integrated_grad_sens(model, inputs, pred, pred_type, ensemble=False, num_steps=10):
    """
    Compute the integrated gradient-based sensitivity of the output w.r.t. the inputs in each timestep.

    Parameters
    ----------
    model : Model consisting of nn.Modules
    inputs : torch.Tensor
        Input tensor with already concatenated external excitation and recurrent state signals,
        shape=[batch_size, input_channels*window_size + pred_size*rnn_window]
    pred : torch.Tensor
        Output tensor that contains the predictions,
        shape=[batch_size, pred_size, forecast]
    pred_type : str
        The model's prediction type out of ('Point, 'Mean_Var'), which defines
        the number of outputs that the sensitivity analysis is performed on.
    ensemble : bool, optional
        If True, sensitivity is computed for an ensemble of models. The default is False.
    num_steps : int, optional
        Number of steps along the linearly interpolated path from the baseline to the input.
        The default is 4 steps.
    
    Returns
    -------
    sens_temp : torch.Tensor
        Sensitivity tensor as Jacobian that contains the gradients of the output w.r.t. the inputs,
        shape=[batch_size*forecast, pred_size, input_channels*window_size + pred_size*rnn_window]
    sens_temp_mean : torch.Tensor, optional
        Sensitivity tensor that contains the gradients of the mean output w.r.t. the inputs when using MVE models.
    sens_temp_var : torch.Tensor, optional
        Sensitivity tensor that contains the gradients of the aleatoric variance output w.r.t. the inputs when using MVE models.
    
    Raises
    ------
    AssertionError
        If the number of steps is less than 2.
    """
    assert num_steps > 1, 'Number of integration steps must be greater than 1 for IG!'

    batch_size, pred_size, forecast = pred[0].shape if (pred_type == 'Mean_Var' or ensemble) else pred.shape
    flatten_size = inputs.shape[1]
    alphas = torch.linspace(0, 1, num_steps+1)[1:-1] # exclude the baseline bc it's zero
    baseline_inp = torch.zeros_like(inputs) # zero baseline
    # baseline_inp = torch.mean(inputs, dim=1, keepdim=True).repeat(1, flatten_size) # mean baseline as alternative option
    difference_inp = (inputs - baseline_inp).repeat_interleave(forecast, dim=0).unsqueeze(1).detach()

    def integrated_grads(alphas, baseline_inp, difference_inp):
        """
        Compute the integrated gradients of the output w.r.t. the inputs for one timestep.
        """
        gradients_mean = torch.full((num_steps, batch_size*forecast, pred_size, flatten_size), float('nan'))
        if pred_type == 'Mean_Var' or ensemble:
            gradients_var = torch.full((num_steps, batch_size*forecast, pred_size, flatten_size), float('nan'))
        
        for i, alpha in enumerate(alphas):
            inputs_interpol = baseline_inp + alpha * (inputs - baseline_inp) # interpolate between baseline and input
            pred_interpol = model.forward_sens(inputs_interpol)
            grads = _comp_grad_sens(inputs_interpol, pred_interpol, pred_type, ensemble)
            gradients_mean[i] = grads[0] if pred_type == 'Mean_Var' or ensemble else grads
            if pred_type == 'Mean_Var' or ensemble:
                gradients_var[i] = grads[1]
        
        grads = _comp_grad_sens(inputs, pred, pred_type, ensemble)
        gradients_mean[-1] = grads[0] if pred_type == 'Mean_Var' or ensemble else grads
        ig_mean = (gradients_mean[:-1] + gradients_mean[1:]) / 2.0 # trapezoidal integration rule
        ig_mean = ig_mean.mean(dim=0) * difference_inp
        # print(gradients_mean[:,0,0,50]) # check how gradients evolve along the path

        if pred_type == 'Mean_Var' or ensemble:
            gradients_var[-1] = grads[1]
            ig_var = (gradients_var[:-1] + gradients_var[1:]) / 2.0
            ig_var = ig_var.mean(dim=0) * difference_inp
            return ig_mean, ig_var
        return ig_mean # shape [batch_size*forecast, pred_size, flatten_size]
    return integrated_grads(alphas, baseline_inp, difference_inp)


def _reshape_array(model, array, aggregation=None, remove_nans=False, repeat=False, repeat_size=None):
    """
    Reshape a post-processed array of the sensitivity tensor for
        further analysis, while keeping the information of possibly
        different window sizes between input and recurrent signals

    Parameters
    ----------
    model : Model consisting of nn.Modules
    array : np.ndarray
        Post-processed array of the sensitivity tensor
    aggregation : str, optional
        Specifies the aggregation method for the reshaped array, performed
        on its last axis. Choose from: mean, sum, rms.
        The default is None, i.e. only reshaping and hstacking is performed.
    remove_nans : bool, optional
        If True, NaN values are removed from the reshaped array. The default is False.
    repeat : bool, optional
        If True, the array is repeated after the aggregation (only if 1D!)
        for each input and recurrent channel. The default is False.
    repeat_size : int, optional
        Size of the repetition, which is the same for every channel. The default is None.

    Returns
    -------
    np.ndarray
        Reshaped and hstacked array of the sensitivity tensor.
    
    Raises
    ------
    ValueError
        If an invalid aggregation method is given.
    """
    m_type = model.Type
    win_size = max(model.window_size, model.rnn_window) if m_type in ['AR', 'AR_RNN'] else model.window_size
    ch_size = model.input_channels + model.pred_size if m_type in ['AR', 'AR_RNN'] else model.input_channels
    rec_start_idx = model.input_channels*model.window_size

    # Reshape the array such that the input and recurrent signals with their corresponding window sizes are separated
    if array.ndim == 1:
        temp1 = array[:rec_start_idx].reshape(model.input_channels, model.window_size)
        if m_type in ['AR', 'AR_RNN']:
            temp2 = array[rec_start_idx:].reshape(model.pred_size, model.rnn_window)
    elif array.ndim == 2:
        temp1 = array[:, :rec_start_idx].reshape(array.shape[0], model.input_channels, model.window_size)
        if m_type in ['AR', 'AR_RNN']:
            temp2 = array[:, rec_start_idx:].reshape(array.shape[0], model.pred_size, model.rnn_window)
    elif array.ndim == 3:
        temp1 = array[..., :rec_start_idx].reshape(*array.shape[:2], model.input_channels, model.window_size)
        if m_type in ['AR', 'AR_RNN']:
            temp2 = array[..., rec_start_idx:].reshape(*array.shape[:2], model.pred_size, model.rnn_window)

    # Apply aggregation method if specified, otherwise only stack arrays together
    if aggregation is None:
        if temp1.ndim == 2:
            temp = np.full((ch_size, win_size+1), np.nan)
            temp[:model.input_channels, -model.window_size:] = temp1 # right-align all input channels
            if m_type in ['AR', 'AR_RNN']:
                temp[model.input_channels:, -model.rnn_window:] = temp2 # right-align all recurrent channels
        elif temp1.ndim == 3:
            temp = np.full((array.shape[0], ch_size, win_size+1), np.nan)
            temp[:, :model.input_channels, -model.window_size:] = temp1
            if m_type in ['AR', 'AR_RNN']:
                temp[:, model.input_channels:, -model.rnn_window:] = temp2
        if remove_nans:
            temp = [x[~np.isnan(x)] for x in temp]
            if temp1.ndim == 3:
                inp_ch_lst = [i*model.window_size for i in range(1,model.input_channels)]
                if m_type in ['AR', 'AR_RNN']:
                    rec_ch_lst = [rec_start_idx + i*model.rnn_window for i in range(model.pred_size)]
                    temp = [np.split(x, inp_ch_lst + rec_ch_lst) for x in temp]
                else:
                    temp = [np.split(x, inp_ch_lst) for x in temp]
        return temp
    else:
        if aggregation == 'mean':
            agg1 = np.mean(temp1, axis=-1)
            if m_type in ['AR', 'AR_RNN']:
                agg2 = np.mean(temp2, axis=-1)
        elif aggregation == 'median':
            agg1 = np.median(temp1, axis=-1)
            if m_type in ['AR', 'AR_RNN']:
                agg2 = np.median(temp2, axis=-1)
        elif aggregation == 'sum':
            agg1 = np.sum(temp1, axis=-1)
            if m_type in ['AR', 'AR_RNN']:
                agg2 = np.sum(temp2, axis=-1)
        elif aggregation == 'rms':
            agg1 = np.sqrt(np.mean(np.square(temp1), axis=-1))
            if m_type in ['AR', 'AR_RNN']:
                agg2 = np.sqrt(np.mean(np.square(temp2), axis=-1))
        else:
            raise ValueError(f'Invalid aggregation method "{aggregation}" given! Choose from: mean, sum, rms.')
        
        if agg1.ndim == 1 and repeat:
            if repeat_size is None:
                agg1 = agg1.repeat(model.window_size)
                if m_type in ['AR', 'AR_RNN']:
                    agg2 = agg2.repeat(model.rnn_window)
            else:
                agg1 = agg1.repeat(repeat_size)
                if m_type in ['AR', 'AR_RNN']:
                    agg2 = agg2.repeat(repeat_size)
            
            if m_type in ['AR', 'AR_RNN']:
                return np.concatenate((np.append(np.nan, agg1), agg2), axis=-1)
            else:
                return np.append(np.nan, agg1)
        
        if m_type in ['AR', 'AR_RNN']:
            return np.concatenate((agg1, agg2), axis=-1)
        else:
            return agg1


def _postprocess_sens(model, sensitivity):
    """
    Postprocess the sensitivity tensor to get information about mean and std of the gradients,
    aggregated over the timesteps, output channels and window sizes.

    Parameters
    ----------
    model : Model consisting of nn.Modules
    sensitivity : torch.Tensor
        Sensitivity tensor that contains the gradients of the output with respect to the inputs,
        with shape=[len(data_loader), pred_size, input_channels*window_size + pred_size*rnn_window]
    
    Returns
    -------
    sum_mean_feature : np.ndarray
        RMS of the time-averaged sensitivities over all output channels for each input feature
    sum_std_feature : np.ndarray
        Sum of the std of the sensitivities over all output channels for each input feature
    sum_inp_channels : np.ndarray
        RMS of the time-avg. sensitivities for each input-output channel combination
    std_inp_channels : np.ndarray
        Sum of the std of the sensitivities for each input-output channel combination
    rms_out_ch_sens : np.ndarray
        RMS of the sensitivities over all output channels for each timestep and input feature
    mean_out_ch_sens : np.ndarray
        Mean of the sensitivities over all output channels for each timestep and input feature
    """
    sensitivity = sensitivity.numpy()

    # Compute the RMS across/over all output channels for each timestep
    rms_out_ch_sens = np.sqrt(np.mean(np.square(sensitivity), axis=1)) # shape [len(dataloader), input_channels*window_size + pred_size*rnn_window]
    mean_out_ch_sens = np.mean(sensitivity, axis=1)

    # Compute the mean and var of the sensitivity tensor along the time axis
    temp_mean_sens = np.mean(sensitivity, axis=0) # shape [pred_size, input_channels*window_size + pred_size*rnn_window]
    temp_std_sens = np.std(sensitivity, axis=0)

    # Compute mean-squared sensitivity (RMS) and mean-squared std.-dev. across all output channels
    sum_mean_feature = np.sqrt(np.mean(temp_mean_sens**2, axis=0)) # shape [input_channels*window_size + pred_size*rnn_window]
    sum_std_feature = np.sqrt(np.mean(temp_std_sens**2 + (temp_mean_sens - temp_mean_sens.mean(axis=0, keepdims=True))**2, axis=0))

    # Compute the RMS across entire window size for each input/recurrent-output channel combination
    sum_inp_channels = _reshape_array(model, temp_mean_sens, aggregation='rms') # shape [pred_size, input_channels+pred_size]
    std_inp_channels = _reshape_array(model, temp_std_sens, aggregation='rms')

    # Delete sensitivity tensor for less memory usage
    del sensitivity
    return (sum_mean_feature, sum_std_feature), (sum_inp_channels, std_inp_channels), (rms_out_ch_sens, mean_out_ch_sens)


def _pred_ARNN_batch(model, batch_sw, device='cpu', sens_params=None):
    """
    Predict function for forward ARNN models with batched dataset for faster
    computation.

    Parameters
    ----------
    model : Model consisting of nn.Modules
    batch_sw : Batched Sliding Window
        Dataset to compute output for
    device : str, optional
        device to compute on. The default is 'cpu'.
    sens_params : dict, optional
        Dictionary containing the parameters for sensitivity analysis.
        Key 'method' defines the method for sensitivity analysis: 'gradient' or 'perturbation'.
        Key 'comp' defines whether gradients are computed for sensitivity analysis.
        Key 'plot' defines whether the results of the sensitivity analysis are visualized.
        Key 'sens_length' defines the number of randomly sampled subset of timesteps for the analysis.
        (If not a multiple of the model's forecast, the number will be rounded up to the next multiple.)
        The default is None, i.e. no sensitivity analysis is performed.

    Returns
    -------
    if comp_sens is False:
        torch.Tensor : Tensor of same langth as input, containing the predictions.
    if comp_sens is True:
        (torch.Tensor, dict) : Tuple of Tensor of same length as input and sensitivity dict.
        Key is the prediction type, value is the sensitivity tensor.
    if comp_sens is True and random_samples > 0:
        (torch.Tensor, dict, torch.Tensor, torch.Tensor) : Tuple of Tensor of same length as input,
        sensitivity dict, uncertainty quantification tensor and random samples tensor.
    """
    size = max(model.window_size, model.rnn_window)
    forecast = model.forecast
    rnn_size = model.rnn_window
    w = batch_sw.__widths__()[0]

    pred_warmup = torch.zeros((w, model.pred_size, size))
    prediction = torch.full((w, model.pred_size, len(batch_sw)*forecast), float('nan'))
    prediction = torch.cat((pred_warmup, prediction), dim=2)
    if model.Pred_Type == 'Mean_Var':
        var_prediction = torch.zeros(prediction.shape)

    original_out = torch.full((w, model.pred_size, len(batch_sw)*forecast), float('nan'))

    # Tensors to device
    prediction = prediction.to(device)
    original_out = original_out.to(device)
    model.to(device)

    checks = _check_sens_params_pred(sens_params)
    if sens_params:
        method, comp_sens, verbose, sens_length, num_samples, std_dev, correlated, random_samples, amplification = checks
    else:
        comp_sens, verbose = checks # False

    # Initialise 3D tensor for sensitivity analysis
    if comp_sens:
        flatten_size = model.window_size*model.input_channels + model.rnn_window*model.pred_size
        num_timesteps = len(batch_sw)*forecast
        sens_indices = np.arange(len(batch_sw))
        sens_uq, eps = None, None

        if sens_length:
            num_timesteps, sens_indices = _random_subset_sens_indices(sens_length, forecast,
                                                            model.Type, batch_sw, batched=True)

        sens_mean = torch.full((w, num_timesteps, model.pred_size, flatten_size), float('nan'))
        if model.Pred_Type == "Mean_Var":
            sens_var = sens_mean.clone()
            if random_samples:
                sens_uq = torch.full((num_timesteps, random_samples, model.pred_size, flatten_size), float('nan'))
                eps = torch.full((num_timesteps, random_samples, model.pred_size), float('nan'))
        if verbose:
            print(f'Start {method.upper()}-based Sensitivity Analysis...\n')
            print(f'Shape of sensitivity tensor: {sens_mean.shape}')

    # Iterate over dataloader
    idx = 0
    for i in tqdm(range(len(batch_sw))) if verbose else range(len(batch_sw)):
        offset = i*forecast + size
        inputs, output = batch_sw[i]
        inputs, output = inputs[0].to(device), output.to(device)
        original_out[:batch_sw.valid_sws[i], :,
                     i*forecast:(i+1)*forecast] = output
        x_rec = prediction[:batch_sw.valid_sws[i], :,
                           (offset - rnn_size):offset]
        
        # Prepare input for model to allow autograd computing gradients of outputs w.r.t. inputs
        inputs = Variable(torch.flatten(inputs, start_dim=1), requires_grad=True)
        x_rec = Variable(torch.flatten(x_rec, start_dim=1), requires_grad=True)
        inp = torch.cat((inputs, x_rec), dim=1)
        pred = model.forward_sens(inp) if comp_sens else model(inputs, x_rec)

        if comp_sens and i in sens_indices:
            sens_temp = _comp_sensitivity(method, model, inp, pred, num_samples, std_dev, correlated, random_samples, amplification)
            if model.Pred_Type == "Mean_Var":
                sens_mean[:batch_sw.valid_sws[i], idx:idx+forecast, :, :] = sens_temp[0].reshape(batch_sw.valid_sws[i], forecast, model.pred_size, flatten_size)
                sens_var[:batch_sw.valid_sws[i], idx:idx+forecast, :, :] = sens_temp[1].reshape(batch_sw.valid_sws[i], forecast, model.pred_size, flatten_size)
                if random_samples:
                    sens_uq[idx], eps[idx] = sens_temp[2], sens_temp[3]
            else:
                sens_mean[:batch_sw.valid_sws[i], idx:idx+forecast, :, :] = sens_temp.reshape(batch_sw.valid_sws[i], forecast, model.pred_size, flatten_size)
            idx += forecast
        
        if model.Pred_Type == 'Mean_Var':
            prediction[:batch_sw.valid_sws[i], :,
                       i*forecast+size:(i+1)*forecast+size] = pred[0].detach()
            var_prediction[:batch_sw.valid_sws[i], :,
                           i*forecast+size:(i+1)*forecast+size] = pred[1].detach()
        else:
            prediction[:batch_sw.valid_sws[i], :,
                       i*forecast+size:(i+1)*forecast+size] = pred.detach()
    
    # cut zeros from initialisation
    prediction = prediction[:, :, size:]

    if model.Pred_Type == 'Mean_Var':
        var_prediction = var_prediction[:, :, size:]
        prediction = (prediction.cpu(), var_prediction.cpu())
    
    if comp_sens:
        if model.Pred_Type == 'Mean_Var':
            sensitivities = (sens_mean.cpu(), sens_var.cpu())
        else:
            sensitivities = sens_mean.cpu()
        if verbose:
            print(f'\n{method.upper()}-based Sensitivity Analysis completed!\n')

        if random_samples: # cut all lines that contain NaNs and flatten first two dimensions
            sens_uq, eps = sens_uq.cpu(), eps.cpu()
            sens_uq = sens_uq[~torch.isnan(sens_uq).any(dim=(1,2,3))].flatten(start_dim=0, end_dim=1)
            eps = eps[~torch.isnan(eps).any(dim=(1,2))].flatten(start_dim=0, end_dim=1)
        return prediction, sensitivities, sens_uq, eps
        # return prediction, sensitivities
    else:
        return prediction


def _predict_arnn_uncertainty(model, dataloader, device='cpu', sens_params=None):
    """
    Predict function for ARNN models that support uncertainty estimation

    Parameters
    ----------
    model : Model consisting of nn.Modules
    dataloader : Dataloader
        Dataloader to predict output
    device : str, optional
        device to compute on. The default is 'cpu'.
    sens_params : dict, optional
        Dictionary that contains the parameters for the sensitivity analysis.
        Key 'method' defines the method for sensitivity analysis: 'gradient' or 'perturbation'.
        Key 'comp' defines whether gradients are computed for sensitivity analysis.
        Key 'plot' defines whether the results of the sensitivity analysis are visualized.
        Key 'sens_length' defines the number of randomly sampled subset of timesteps for the analysis.
        The default is None, i.e. no sensitivity analysis is computed.

    Returns
    -------
    if comp_sens is False:
        torch.Tensor : Tensor of same langth as input, containing the predictions.
        if comp_sens is True:
            (torch.Tensor, dict) : Tuple of Tensor of same length as input and sensitivity dict.
            Key is the prediction type, value is the sensitivity tensor.
    """
    return _prediction(model, dataloader, device, sens_params)


def _predict_arnn_uncertainty_both(model, dataloader, device='cpu'):
    """
    Predict function for ARNN models that support uncertainty estimation
    and capture heteroscedastic and aleatoric uncertainty

    Parameters
    ----------
    model : Model consisting of nn.Modules
    dataloader : Dataloader
        Dataloader to predict output
    device : str, optional
        device to compute on. The default is 'cpu'.

    Returns
    -------
    if comp_sens is False:
        torch.Tensor : Tensor of same langth as input, containing the predictions.
        if comp_sens is True:
            (torch.Tensor, dict) : Tuple of Tensor of same length as input and sensitivity dict.
            Key is the prediction type, value is the sensitivity tensor.
    """
    return _prediction(model, dataloader, device)


def _async_prediction(model, dataloader, device='cpu', n_samples=1, reduce=True, ensemble_weights=None, sens_params=None):
    """
    Prediction of a whole Time Series with a model wrapper
    in case of MVE ensemble, the weighting is done due to
    https://arxiv.org/pdf/1612.01474.pdf

    Parameters
    ----------
    dataloader : Dataloader
        Dataloader to predict output
    device : str, optional
        device to compute on. The default is 'cpu'.
    n_samples : int
        Numbers of samples to take for Monte Carlo estimation. The default is 1.
    reduce: bool, optional
        Whether the combined uncertainty (True) or both uncertainties should be returned. The default is True.
    ensemble_weights: list[dict]
        List of torch state dicts containing weights. The default is None
    sens_params : dict, optional
        Dictionary that contains the parameters for the sensitivity analysis.
        Key 'method' defines the method for sensitivity analysis: 'gradient' or 'perturbation'.
        Key 'comp' defines whether gradients are computed for sensitivity analysis.
        Key 'plot' defines whether the results of the sensitivity analysis are visualized.
        Key 'verbose' defines whether the information about the sensitivity analysis is printed.
        Key 'sens_length' defines the number of randomly sampled subset of timesteps for the analysis.
        The default is None, i.e. no sensitivity analysis is computed.
    
    Returns
    -------
    if comp_sens is False:
        torch.Tensor : Tensor of same langth as input, containing the predictions.
    if comp_sens is True:
        (torch.Tensor, dict) : Tuple of Tensor of same length as input and sensitivity dict.
        Key is the prediction type, value is the sensitivity tensor.
    """
    model.eval()
    prediction_type = model.Pred_Type

    if n_samples > 1:
        # Enable dropout Layers during test time
        for m in model.modules():
            if m.__class__.__name__ == 'Dropout':
                m.train()
    
    checks = _check_sens_params_pred(sens_params)
    comp_sens = checks[1] if sens_params else False # True if sens_params are given, False otherwise
    verbose = checks[2] if sens_params else False

    if ensemble_weights is None:
        if comp_sens:
            predictions = [model.prediction(dataloader, device, sens_params=sens_params) for _ in range(n_samples)]
            predictions, sens_dicts = zip(*predictions)
            predictions, sens_dicts = list(predictions), list(sens_dicts)
        else:
            predictions = [model.prediction(dataloader, device) for _ in range(n_samples)]
    else:
        predictions = []
        sens_dicts = []
        for w in ensemble_weights:
            for _ in range(n_samples):
                model.load_state_dict(w)
                if comp_sens:
                    pred, sensitivity = model.prediction(dataloader, device, sens_params=sens_params)
                    predictions.append(pred)
                    sens_dicts.append(sensitivity)
                else:
                    predictions.append(model.prediction(dataloader, device))

    if verbose:
        print(f'Shape of single prediction: {predictions[0].shape}')
    predictions = torch.stack(predictions)
    if verbose:
        print(f'Shape of ensemble predictions: {predictions.shape}\n')

    if comp_sens:
        mean_str = 'Mean' if prediction_type == "Mean_Var" else 'Point'
        means_senses = torch.stack([sens_dict[mean_str] for sens_dict in sens_dicts])
        mean_sens = torch.mean(means_senses, dim=0)
        if prediction_type == "Mean_Var":
            var_sens = torch.stack([sens_dict['Aleatoric_UQ'] for sens_dict in sens_dicts]).mean(dim=0)
        else:
            var_sens = torch.var(means_senses, dim=0)
        sens_avg_dict = {'Mean': mean_sens, 'Var_UQ': var_sens}

    # epistemic_var = torch.var(predictions[0], dim=0) if ensemble_weights is not None or n_samples > 1 else torch.zeros(mean.shape)

    if prediction_type == "Mean_Var":
        mean = torch.mean(predictions[:, 0, :, :], dim=0)
        total_var = torch.mean(predictions[:, 1, :, :] + predictions[:, 0, :, :].square(), dim=0) - mean.square()
        aleatoric_var = torch.mean(predictions[:, 1, :, :], dim=0)
        epistemic_var = total_var - aleatoric_var
        predictions = (mean, total_var) if reduce else (mean, epistemic_var, aleatoric_var)
    else:
        mean = torch.mean(predictions, dim=0)
        epistemic_var = torch.var(predictions, dim=0) if ensemble_weights is not None or n_samples > 1 else torch.zeros(mean.shape)
        predictions = (mean, epistemic_var)

    if comp_sens:
        return predictions, sens_avg_dict
    else:
        return predictions


def _prediction(model, dataloader, device='cpu', sens_params=None):
    """
    Prediction of a whole Time Series

    Parameters
    ----------
    dataloader : Dataloader
        Dataloader to predict output
    device : str, optional
        device to compute on. The default is 'cpu'.
    sens_params : dict, optional
        Dictionary that contains the parameters for the sensitivity analysis.
        Key 'method' defines the method for sensitivity analysis: 'gradient' or 'perturbation'.
        Key 'comp' defines whether gradients are computed for sensitivity analysis.
        Key 'plot' defines whether the results of the sensitivity analysis are visualized.
        Key 'verbose' defines whether the information about the sensitivity analysis is printed.
        Key 'sens_length' defines the number of randomly sampled subset of timesteps for the analysis.
        (If not a multiple of the model's forecast, the number will be rounded up to the next multiple.)
        The default is None, i.e. no sensitivity analysis is computed.

    Returns
    -------
    if comp_sens is False:
        torch.Tensor : Tensor of same langth as input, containing the predictions.
    if comp_sens is True:
        (torch.Tensor, dict) : Tuple of Tensor of same length as input and sensitivity dict.
        Key is the prediction type, value is the sensitivity tensor.
    """
    fc = model.forecast
    prediction_type = model.Pred_Type
    if prediction_type == "Quantile":
        num_outputs = model.n_quantiles
    else:
        num_outputs = {
            "Point": 1,
            "Mean_Var": 2,
        }
        num_outputs = num_outputs[prediction_type]
        if model.Ensemble:
            num_outputs = 2

    # The Model starts with zeros as recurrent System state
    size = max(model.window_size, model.rnn_window)
    prediction = torch.zeros((num_outputs, model.pred_size, size))
    original_out = torch.zeros((model.pred_size, size))
    x_rec = torch.zeros(1, model.pred_size, model.rnn_window)

    # Tensors to device
    x_rec = x_rec.to(device)
    prediction = prediction.to(device)
    original_out = original_out.to(device)
    model.to(device)

    checks = _check_sens_params_pred(sens_params)
    if sens_params:
        method, comp_sens, verbose, sens_length, num_samples, std_dev, correlated, random_samples, amplification = checks
    else:
        comp_sens, verbose = checks # False

    # Initialise 3D tensor for sensitivity analysis
    if comp_sens:
        flatten_size = model.window_size*model.input_channels + model.rnn_window*model.pred_size
        num_timesteps = len(dataloader)*fc
        loader_length = num_timesteps - dataloader.dataset.subclass.add_zero
        sens_indices = np.arange(len(dataloader))

        if sens_length:
            num_timesteps, sens_indices = _random_subset_sens_indices(sens_length, fc, model.Type, dataloader)
        
        sens_mean = torch.full((num_timesteps, model.pred_size, flatten_size), float('nan'))
        if prediction_type == "Mean_Var" or model.Ensemble:
            sens_var = sens_mean.clone()
            if random_samples:
                sens_uq = torch.full((num_timesteps, random_samples, model.pred_size, flatten_size), float('nan'))
                eps = torch.full((num_timesteps, random_samples, model.pred_size), float('nan'))
        if verbose:
            print(f'Start {method.upper()}-based Sensitivity Analysis...\n')
            print(f'Shape of sensitivity tensor: {sens_mean.shape}')

    # Iterate over dataloader
    idx = 0
    for i, data in enumerate(tqdm(dataloader) if verbose else dataloader):
        inputs, output = data
        inputs, output = inputs[0].to(device), output.to(device)

        # Prepare input for model to allow autograd computing gradients of outputs w.r.t. inputs
        inputs = Variable(torch.flatten(inputs, start_dim=1), requires_grad=True)
        x_rec = Variable(torch.flatten(x_rec, start_dim=1), requires_grad=True)
        inp = torch.cat((inputs, x_rec), dim=1)
        if comp_sens:
            pred = model.forward_sens(inp)
        elif prediction_type == 'Mean_Var' or model.Ensemble:
            pred = model.estimate_uncertainty_mean_std(inputs, x_rec)
        else:
            pred = model(inputs, x_rec)
        
        if comp_sens and i in sens_indices:
            sens_temp = _comp_sensitivity(method, model, inp, pred, num_samples, std_dev, correlated, random_samples, amplification)
            if prediction_type == "Mean_Var" or model.Ensemble:
                sens_mean[idx:idx+fc], sens_var[idx:idx+fc] = sens_temp[0], sens_temp[1]
                if random_samples:
                    sens_uq[idx], eps[idx] = sens_temp[2], sens_temp[3]
            else:
                sens_mean[idx:idx+fc] = sens_temp
            idx += fc
        
        if type(pred) == tuple:
            pred = torch.vstack(pred)

        if prediction_type == "Quantile":
            # QR uses a different output shape [1,pred_size,forecast,num_outputs] than other models [num_outputs,pred_size,forecast]
            # We can adjust for this by squeezing the first and swapping the remaining axes
            pred = pred.squeeze(0)
            pred = torch.transpose(pred, 0, 2)
            pred = torch.transpose(pred, 1, 2)
        
        prediction = torch.cat((prediction,
                                pred.detach().reshape(num_outputs, model.pred_size, -1)), -1)
        x_rec = torch.unsqueeze(prediction[0, :, -model.rnn_window:], dim=0) # only feed back the mean values!
        original_out = torch.cat((original_out,
                                    output.reshape(model.pred_size, -1)), 1)
    
    # cut zeros from initialisation
    # shape = [num_outputs, pred_size, len(dataset)] = [num_outputs, pred_size, forecast*len(dataloader)]
    prediction = prediction[..., size:]
    original_out = original_out[:, size:]

    cut_zeros = dataloader.dataset.subclass.add_zero
    if cut_zeros != 0:
        prediction = prediction[..., :-cut_zeros]
        original_out = original_out[:, :-cut_zeros]

    prediction.cpu()
    model.to('cpu')

    if comp_sens:
        # cut to the same length as the prediction
        sens_mean = sens_mean[:loader_length]
        if prediction_type == "Mean_Var" or model.Ensemble:
            sens_var = sens_var[:loader_length]
            sensitivity_dict = {'Mean': sens_mean.cpu(), 'Aleatoric_UQ': sens_var.cpu()}
        else:
            sensitivity_dict = {'Point': sens_mean.cpu()}
        if verbose:
            print(f'{method.upper()}-based Sensitivity Analysis completed!\n')

        if random_samples: # cut all lines that contain NaNs and flatten first two dimensions
            if verbose:
                mean, std = prediction[0].mean(dim=-1), torch.sqrt(prediction[1].mean(dim=-1))
                print(f'Mean of predictions: {torch.round(mean, decimals=3)}, Std of predictions: {torch.round(std, decimals=4)}')
            sens_uq, eps = sens_uq.cpu(), eps.cpu()
            sens_uq = sens_uq[~torch.isnan(sens_uq).any(dim=(1,2,3))].flatten(start_dim=0, end_dim=1)
            eps = eps[~torch.isnan(eps).any(dim=(1,2))].flatten(start_dim=0, end_dim=1)
            return prediction, sensitivity_dict, sens_uq, eps
        return prediction, sensitivity_dict
    else:
        return prediction


def _comp_perturb_sens(model, inputs, pred, perturb_size=10, std_dev=0.2, correlated=True, random_samples=0, amplification=1):
    """
    Compute the perturbation-based sensitivity of the output w.r.t. the inputs, also known as
    Permutation Feature Importance (PFI). The method is based on the paper by Altmann et al. (2010).

    Parameters
    ----------
    model : Model consisting of nn.Modules
    inputs : torch.Tensor
        Input tensor with already concatenated external excitation and recurrent state signals,
        shape=[batch_size, input_channels*window_size + pred_size*rnn_window]
    pred : torch.Tensor
        Output tensor that contains the predictions,
        shape=[batch_size, pred_size, forecast]
    output : torch.Tensor
        Output tensor that contains the true values,
        shape=[batch_size, pred_size, forecast]
    perturb_size : int, optional
        Number of permutations per input feature. The default is 4.
    std_dev : float, optional
        Standard deviation of the Gaussian noise as form of permutation.
        The default is 0.2.
    correlated : bool, optional
        If True, the relative perturbations are decayed in a local area of the current feature
        to be perturbed, based on the strongest region of the signal's auto-correlation.
        The default is True, i.e. the perturbations are decayed.
    random_samples : int, optional
        Number of random samples to take for "neural" Monte Carlo uncertainty estimation for MVE models.
        The default is 0, i.e. no random samples are taken.

    Returns
    -------
    sens_temp : torch.Tensor
        Sensitivity tensor as Jacobian that contains the gradients of the output w.r.t. the inputs,
        shape=[batch_size*forecast, pred_size, input_channels*window_size + pred_size*rnn_window]
    
    Raises
    ------
    AssertionError
        If the perturb_size is less than 1 or the std_dev is less than 0.
    """
    assert perturb_size > 0, 'Permutation size must me greater than 0 for PFI!'
    assert std_dev > 0, 'Standard deviation for Gaussian noise must me greater than 0 for PFI!'

    if not random_samples:
        torch.manual_seed(42)
    steps = torch.normal(mean=0, std=std_dev, size=(perturb_size, 1))
    while (steps > 0).sum() != perturb_size//2: # ensure that positive and negative perturbations are equal
        steps = torch.normal(mean=0, std=std_dev, size=(perturb_size, 1))

    batch_size, pred_size, forecast = pred[0].shape if (model.Pred_Type == 'Mean_Var' or model.Ensemble) else pred.shape
    flatten_size = inputs.shape[1]
    input_channels, window_size = model.input_channels, model.window_size
    rec_start_idx = input_channels * window_size
    rnn_window = model.rnn_window if model.Type in ['AR', 'AR_RNN'] else 0
    ch_size = (input_channels + pred_size) if model.Type in ['AR', 'AR_RNN'] else input_channels

    def auto_correlation(signal):
        """Compute the auto-correlation of a channel signal"""
        signal = signal.detach().numpy()
        signal = signal - np.mean(signal) # zero-mean to avoid bias and make intervals more comparable
        auto_cov = np.correlate(signal, signal, mode='full')
        return auto_cov / (np.max(auto_cov) + 1e-6)
    
    def get_autocorrelations(inputs):
        """Get the auto-correlations of input and recurrent signals."""
        input_ = inputs[:,:rec_start_idx].reshape(input_channels, window_size)
        if model.Type in ['AR', 'AR_RNN']:
            x_rec = inputs[:,rec_start_idx:].reshape(pred_size, rnn_window)
        
        auto_corrs = []
        for ch_idx in range(ch_size):
            if ch_idx < input_channels:
                auto_cor = auto_correlation(input_[ch_idx])
            else:
                auto_cor = auto_correlation(x_rec[ch_idx-input_channels])
            auto_corrs.append(torch.tensor(auto_cor))
        return auto_corrs

    def get_decay(auto_corrs, idx):
        """
        Get the decay values from the auto-correlations, including the indices from
        the slicing operation at the current position within the sliding window.
        """
        if model.Type in ['AR', 'AR_RNN']:
            ch = idx//window_size if idx < rec_start_idx else (idx-rec_start_idx)//rnn_window + input_channels
            pos = idx % window_size if idx < rec_start_idx else (idx-rec_start_idx) % rnn_window
        else:
            ch, pos = idx // window_size, idx % window_size
        win_size = window_size if ch < input_channels else rnn_window
        
        if correlated:
            auto_cor = auto_corrs[ch][win_size-1-pos : 2*win_size-1-pos]
            zero_intersecs = np.where(np.diff(np.sign(auto_cor)) != 0)[0]
            p = np.min(np.abs(zero_intersecs - pos)) if len(zero_intersecs) > 0 else win_size-idx
            start, stop = max(0, pos-p), min(pos+p+1, win_size)
            decay = auto_cor[start:stop]
        else:
            decay, (start, stop) = torch.ones(1), (pos, pos+1)
        return decay, (start, stop), ch, win_size

    def perturb_input(batched_input, decay, win_size, ch, start, stop):
        """
        Perturb the input signal with Gaussian noise, over the length of the
        auto-correlation period of the signal (until first intersection with x-axis).
        """
        if ch < input_channels: # within input channels
            input_ = batched_input[:,:rec_start_idx].reshape(-1, input_channels, win_size)
            input_[:, ch, start:stop] *= decay
            return torch.cat((input_.flatten(start_dim=1), batched_input[:,rec_start_idx:]), dim=1)
        else: # within recurrent channels
            x_rec = batched_input[:,rec_start_idx:].reshape(-1, pred_size, win_size)
            x_rec[:, ch-input_channels, start:stop] *= decay
            return torch.cat((batched_input[:,:rec_start_idx], x_rec.flatten(start_dim=1)), dim=1)

    def perturbation(model, inputs, pred, steps):
        """
        Compute the RMS score of differences, coming from all perturbations
        of the inputs w.r.t. the reference output for one timestep.
        """
        sens_temp = torch.full((batch_size*forecast, pred_size, flatten_size), float('nan'))
        for i in np.arange(batch_size):
            batched_inp = inputs[i:i+1].repeat(perturb_size, 1) # shape = [perturb_size, flatten_size]
            auto_corrs = get_autocorrelations(inputs[i:i+1]) if correlated else torch.ones(ch_size)
            
            for j in np.arange(flatten_size):
                decay, start_stop, ch, win_size = get_decay(auto_corrs, j)
                decay = 1 + steps * decay
                perturbed_inp = batched_inp.clone()
                perturbed_inp = perturb_input(perturbed_inp, decay, win_size, ch, *start_stop)
                perturbed_pred = model.forward_sens(perturbed_inp)  # shape = [perturb_size, pred_size, forecast]
                del perturbed_inp

                # compute a sensitivity metric of all perturbations in each feature variation
                if model.Pred_Type == 'Mean_Var' or model.Ensemble:
                    differences = (perturbed_pred[0] - pred[i:i+1,...]).detach()
                else:
                    differences = (perturbed_pred - pred[i:i+1,...]).detach()
                differences = torch.mean(differences, dim=0)
                sens_temp[i*forecast:(i+1)*forecast,:,j] = differences.T # shape = [forecast, pred_size]
        return sens_temp
        
    # Compute and return the differences from the perturbations
    if model.Pred_Type == 'Point':
        return perturbation(model, inputs, pred, steps)
    elif model.Pred_Type == 'Mean_Var' or model.Ensemble:
        mean_pred, var_pred = pred
        sens_temp_mean = perturbation(model, inputs, mean_pred, steps)
        sens_temp_var = torch.zeros_like(sens_temp_mean)

        if random_samples:
            permutations = torch.full((random_samples, batch_size*forecast, pred_size, flatten_size), float('nan'))
            samples = torch.full((random_samples, pred_size), float('nan'))
            for i in range(random_samples): # not direct sampling, but reparametrization trick: mean + eps*std, with eps ~ N(0,1)
                eps = torch.randn(1, pred_size, 1) * amplification # used for later up-scaling
                sampled_pred = mean_pred + eps * torch.sqrt(var_pred)
                permutations[i] = perturbation(model, inputs, sampled_pred[0], steps)
                samples[i] = eps[0,:,0].detach()
            sens_temp_var = permutations.mean(dim=0)
            return sens_temp_mean, sens_temp_var, permutations.mean(dim=1), samples
        return sens_temp_mean, sens_temp_var


def _comp_sensitivity(method, model, inp, pred, num_samples=10, std_dev=0.2, correlated=True, random_samples=0, amplification=1):
    """
    Abatracted mathod that computes the sensitivity analysis for each step in the dataloader
    based on the given method.

    Parameters
    ----------
    method : The method to use for sensitivity analysis.
    model : Model consisting of nn.Modules
    inputs : torch.Tensor
        Input tensor with already concatenated external excitation and recurrent state signals,
        shape=[batch_size, input_channels*window_size + pred_size*rnn_window]
    pred : torch.Tensor
        Output tensor that contains the predictions,
        shape=[batch_size, pred_size, forecast]
    num_samples : int, optional
        Number of permutations per input feature. The default is 4.
    std_dev : float, optional
        Standard deviation of the Gaussian noise as form of permutation.
        The default is 0.2.
    correlated : bool, optional
        If True, the perturbations are based on the auto-correlation of the signals.
        The default is True.
    random_samples : int, optional
        Number of random samples, drawn from a standard normal distribution, to approximate the
        expected sensitivity range across the aleatoric uncertainty of MVE models.
        The default is 0 samples, i.e. no sampling is performed.
    amplification : float, optional
        Factor to amplify the sensitivity gradients when performing the sensitivity analysis
        under re-sampling for MVE models. The default is 1, i.e. no amplification.

    Returns
    -------
    The computed sensitivity analysis result. In case of an MVE model, the result is a tuple,
    containing the mean and variance of the sensitivity analysis.
    
    Raises
    ------
    ValueError: If an invalid method is given for the sensitivity analysis.
    """
    method = method.lower()
    if method == SensitivityMethods.GRADIENT.value:
        return _comp_grad_sens(inp, pred, model.Pred_Type, model.Ensemble, random_samples, amplification)
    elif method == SensitivityMethods.SMOOTH_GRAD.value:
        return _comp_smooth_grad_sens(model, inp, pred, model.Pred_Type, model.Ensemble, num_samples, std_dev)
    elif method == SensitivityMethods.INTEGRATED_GRADIENT.value:
        return _comp_integrated_grad_sens(model, inp, pred, model.Pred_Type, model.Ensemble, num_samples)
    elif method == SensitivityMethods.PERTURBATION.value:
        return _comp_perturb_sens(model, inp, pred, num_samples, std_dev, correlated, random_samples, amplification)
    else:
        raise ValueError((f"Given method '{method}' is not implemented! Choose from: "
                          f"{[x.lower() for x in list(SensitivityMethods.__members__)]}"))


def _check_sens_params_pred(sens_params):
    """
    Check the sensitivity parameters for all AR and RNN models.

    Parameters
    ----------
    sens_params : dict
        Dictionary that contains the parameters for the sensitivity analysis.
    
    Returns
    -------
    if sens_params is given / not None:
        tuple : method, comp_sens, verbose, sens_length, num_samples, std_dev,
        correlated, random_samples, amplification
    elif sens_params is None:
        bool : comp_sens
    """
    if sens_params:
        method = sens_params.get('method', '')
        comp_sens = sens_params.get('comp', False)
        verbose = sens_params.get('verbose', False)
        sens_length = sens_params.get('sens_length', None)
        num_samples = sens_params.get('num_samples', 10)
        std_dev = sens_params.get('std_dev', 0.2)
        correlated = sens_params.get('correlated', True)
        random_samples = sens_params.get('random_samples', 0)
        amplification = sens_params.get('amplification', 1)
        return method, comp_sens, verbose, sens_length, num_samples, std_dev, correlated, random_samples, amplification
    else:
        comp_sens = False
        verbose = False
        return comp_sens, verbose


def _random_subset_sens_indices(sens_length, forecast, m_type, dataloader, batched=False):
    """
    Create random indices for sensitivity analysis that
    are a subset of the dataloader indices, allowing for faster prediction.
    
    Parameters
    ----------
    sens_length : int
        Desired size for the reduced points that the sensitivity analysis is computed on
    forecast : int
        The model's forecast length
    m_type : str
        The model type, specified as object attribute in "model.Type"
    dataloader : Dataloader
        Dataloader for the test dataset to predict the output for.
    batched : bool, optional
        If True, the dataloader is batched, i.e. only used for "_pred_ARNN_batch" function.
        The default is False.


    Returns
    -------
    num_timesteps : int
        Number of timesteps for sensitivity analysis
    sens_indices : list
        List of indices that are used for the sensitivity analysis
    
    Raises
    ------
    AssertionError
        If the given sensitivity length is smaller than forecast length times batch size.
    AssertionError
        If the given sensitivity length exceeds the maximum length of the dataloader.
    """
    sens_length = int(sens_length)
    batch_size = next(iter(dataloader))[0].shape[0] if m_type == 'RNN' else 1
    fc, bs = forecast, batch_size

    assert sens_length >= fc*bs, f'Given sensitivity length of {sens_length} must be at least of size {fc*bs}!'
    if not batched:
        num_timesteps = len(dataloader.dataset)*fc
        loader_length = num_timesteps - dataloader.dataset.subclass.add_zero
        assert sens_length <= loader_length, f'Given sensitivity length of {sens_length} exceeds maximum dataloader length of {loader_length}!'
    else:
        num_timesteps = len(dataloader)*forecast
        assert sens_length <= num_timesteps, f'Given sensitivity length of {sens_length} exceeds maximum dataloader length of {num_timesteps}!'
        add_zeros = [sw.add_zero for sw in dataloader.sws]
        min_length = min([le*forecast-zeros for le, zeros in zip(dataloader.__lengths__(), add_zeros)])

    # round sens_length to the closest multiple of forecast*batch_size if needed
    if sens_length % (fc*bs) != 0:
        sens_length = np.around(sens_length/(fc*bs)) if m_type == 'RNN' else np.ceil(sens_length/(fc*bs))
        sens_length = int(sens_length * fc*bs)
        print(f'INFO: Given sensitivity length was rounded to {sens_length} as closest multiple of batch_size={bs} * forecast={fc}.')
    num_timesteps = sens_length
    sampler = LatinHypercube(d=1)
    samples = sampler.random(n=sens_length//(fc*bs)).flatten()

    # fill up sens_indices in case of duplications until desired sens_length is reached
    min_length = len(dataloader) if not batched else min_length
    sens_indices = np.floor(samples * min_length).astype(int)
    unique_indices = set(sens_indices)
    while len(unique_indices) < (sens_length//(fc*bs)):
        additional_samples = sampler.random(n=(sens_length//(fc*bs)) - len(unique_indices)).flatten()
        additional_indices = np.floor(additional_samples * min_length).astype(int)
        unique_indices.update(additional_indices)
    sens_indices = sorted(unique_indices)

    return num_timesteps, sens_indices


def _compress_model(old_model, state_dict_path, new_window_size=None, new_rnn_window=None, retrain=False, retrain_params=None):
    """
    Compress the first layer of an ARNN model by reducing the input and recurrent window sizes.
    Additional option to fine-tune the compressed model after compression.
    INFO: Currently NOT working for ensemble and RNN models!

    Parameters
    ----------
    old_model : nn.Module
        The old NN model to compress
    state_dict_path : str
        Path to the saved state dict of the old model
    new_window_size : int
        New window size for the input features
    new_rnn_window : int
        New window size for the recurrent features
    retrain : bool, optional
        If True, fine-tune the new model after compression. The default is False.
    data_handle : data_handle object, optional
        Object from Meas_handling module to load and prepare the data for retraining.
        Only needed if retrain=True. The default is None, i.e. not provided.

    Returns
    -------
    new_model : nn.Module
        The compressed model with reduced input and recurrent window sizes.
    """
    # Get the name of all first weight tensors
    weight_names = []
    for name, _ in old_model.named_parameters():
        if '0.weight' in name:
            weight_names.append(name)

    # Load the saved weights
    state_dict = torch.load(state_dict_path)

    # Extract and reduce the first weight matrix
    for w_name in weight_names:
        w_mat = state_dict[w_name]
        rec_start_idx = old_model.input_channels * old_model.window_size
        if new_window_size:
            assert new_window_size <= old_model.window_size, 'New window_size must be smaller or equal than the old one!'
            temp_inp = w_mat[:, :rec_start_idx].view(w_mat.shape[0], -1, old_model.window_size)
            temp_inp = temp_inp[...,-new_window_size:].flatten(start_dim=1) # take only the _last_ new_window_size columns
        else:
            new_window_size = old_model.window_size
            temp_inp = w_mat[:, :rec_start_idx]
        
        if new_rnn_window:
            assert new_rnn_window <= old_model.rnn_window, 'New rnn_window_size must be smaller or equal than the old one!'
            temp_rec = w_mat[:, rec_start_idx:].view(w_mat.shape[0], -1, old_model.rnn_window)
            temp_rec = temp_rec[...,-new_rnn_window:].flatten(start_dim=1)
        else:
            new_rnn_window = old_model.rnn_window
            temp_rec = w_mat[:, rec_start_idx:]
        state_dict[w_name] = torch.cat((temp_inp, temp_rec), dim=1)

    new_params = {'window_size': new_window_size, 'rnn_window': new_rnn_window}
    # Extract __init__ parameters and create a new model with modified window sizes
    cls = old_model.__class__
    parameters = inspect.signature(cls.__init__).parameters
    init_params = {param: getattr(old_model, param) for param in parameters if param != 'self' and hasattr(old_model, param)}
    model_temp = old_model.DNN if hasattr(old_model, 'DNN') else old_model
    parameters = inspect.signature(model_temp.__class__.__init__).parameters
    wrong_keys = ['input_size', 'output_size']
    init_params_dnn = {param: getattr(model_temp, param) for param in parameters if param != 'self'
                   and param not in wrong_keys and hasattr(model_temp, param)}
    init_params.update(init_params_dnn)
    if cls.__name__ == 'SeparateMVEARNN':
        init_params['var_hidden_size'] = init_params.pop('hidden_size')
    
    for name, value in init_params.items():
        wrapped_model = value if isinstance(value, nn.Module) else None
        if wrapped_model:
            wrapped_params = inspect.signature(wrapped_model.__init__).parameters
            wrapped_init_params = {param: getattr(wrapped_model, param) for param in wrapped_params \
                                   if param != 'self' and hasattr(wrapped_model, param)}

            wrapped_init_params.update({k: v for k, v in new_params.items() if k in wrapped_init_params})
            model_temp = wrapped_model.DNN if hasattr(wrapped_model, 'DNN') else wrapped_model
            parameters = inspect.signature(model_temp.__class__.__init__).parameters
            init_params_dnn = {param: getattr(model_temp, param) for param in parameters if param != 'self'
                        and param not in wrong_keys and hasattr(model_temp, param)}
            wrapped_init_params.update(init_params_dnn)

            new_wrapped_model = wrapped_model.__class__(**wrapped_init_params)
            init_params[name] = new_wrapped_model
            # print('wrapped_init_params:', wrapped_init_params, '\n')
            break
    
    if 'window_size' in init_params: # indicates that the outer model is of type nn.Module
        init_params.update({k: v for k, v in new_params.items() if k in init_params})
    # print('init_params:', init_params, '\n')

    # Instantiate a new model with updated parameters and load the modified state dict
    compressed_model = cls(**init_params)
    compressed_model.load_state_dict(state_dict)

    params_old = summary(old_model).total_params
    params_new = summary(compressed_model).total_params
    print((f'Reduction by {params_old - params_new} parameters in the Input layer, resulting '
           f'in a total model compression ratio of {(1-params_new/params_old):.1%}\n'))
    
    if retrain and retrain_params:
        lr = retrain_params.get('lr', 1e-4) / 2 # reduce learning rate for fine-tuning
        patience = retrain_params.get('patience', 5)
        max_epochs = retrain_params.get('max_epochs', 100)
        stab = retrain_params.get('stabilizer', 5e-3)
        data_handle = retrain_params.get('data_handle', None)
        if data_handle is None:
            raise ValueError('No data_handle object provided for re-training the model!')
        
        ## V1: fine-tuning the compressed model
        train_loader, val_loader = data_handle.give_torch_loader(window_size=new_window_size,
                                                                 rnn_window=new_rnn_window, keyword='short')
        opt = torch.optim.Adam(compressed_model.parameters(), lr=lr)
        crit = nn.MSELoss() if compressed_model.Pred_Type == 'Point' else GaussianNLLLoss()
        print((f'Start fine-tuning the compressed model with lr={lr:.2e}, patience={patience}, '
               f'max_epochs={max_epochs} and stab={stab:.2e} ...'))
        res_df = train_model(model=compressed_model, train_loader=train_loader, max_epochs=max_epochs, optimizer=opt, device='cpu',
                        criterion=crit, stabelizer=stab, val_loader=val_loader, patience=patience, print_results=True)
        print('Fine-tuning finished!\n')

        if wrapped_model:
            return compressed_model, res_df, init_params, wrapped_init_params
        else:
            return compressed_model, res_df, init_params
    
    if wrapped_model:
        return compressed_model, init_params, wrapped_init_params
    else:
        return compressed_model, init_params
