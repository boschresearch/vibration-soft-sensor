# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:52:41 2022

@author: WET2RNG
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm

from softsensor.model import _LSTM, _GRU, _RNN, Feed_ForwardNN, _filter_parameters
from softsensor.autoreg_models import SensitivityMethods, _prediction, _comp_sensitivity, _check_sens_params_pred, _random_subset_sens_indices


class AR_RNN(nn.Module):
    """
    Autoregressive Recurrent Network that utilises the past outputs in
    combination with hidden cells

    Parameters
    ----------
    input_channels : int
        Number of input channels
    pred_size : int
        Number of predicted values
    window_size : int
        Size of the sliding window applied to the time series
    rnn_window : int
        Window Size of the Recurent Connection
    blocks : int
        Number of parallel recurrent blocks.
    num_layers : int
        nunmber of recurernt blocks.(depth of the recurretn part)
    blocktype : str, optional
        blocktype, options are: 'RNN', 'GRU' and 'LSTM'. The default is 'LSTM'.
    hidden_size : list of int or None, optional
        List gives the size of hidden units. The default is None.
    activation : str, optional
        Activation function to activate the feature space.
        The default is 'relu'.
    bias : bool, optional
        If True, bias weights are used. The default is True.
    dropout : float [0,1], optional
        Adds dropout layers after each Linear Layer. The default is None.

    Returns
    -------
    None.
    
    Examples
    --------
    
    >>> import softsensor.recurrent_models
    >>> import torch
    >>> m = softsensor.recurrent_models.AR_RNN(2, 1, 10, 10, 16, 1)
    >>> print(m)
    AR_RNN(
      (RecBlock): _LSTM(
        (lstm): LSTM(30, 16, batch_first=True)
      )
      (DNN): Feed_ForwardNN(
        (DNN): Sequential(
          (0): Linear(in_features=16, out_features=1, bias=True)
        )
      )
    )
    >>> input = torch.randn(32, 2, 10)
    >>> rec_input = torch.randn(32, 1, 10)
    >>> output = m(input, rec_input)
    >>> print(output.shape)
    torch.Size([32, 1, 1])
    """

    def __init__(self, input_channels, pred_size, window_size, rnn_window,
                 blocks, num_layers, blocktype='LSTM', hidden_size=None,
                 activation='relu', bias=True, dropout=None, forecast=1,
                 Pred_Type='Point'):

        super().__init__()

        self.params = _filter_parameters(locals().copy())

        self.input_channels = input_channels
        self.pred_size = pred_size
        self.window_size = window_size
        self.rnn_window = rnn_window
        self.blocktype = blocktype
        self.activation = activation
        self.forecast = forecast

        self.Type = 'AR_RNN'
        self.Pred_Type = Pred_Type
        self.Ensemble = False
        if self.Pred_Type == 'Point':
            preds = 1
        elif self.Pred_Type == 'Mean_Var':
            preds = 2
        else:
            raise ValueError('No valid Pred_Type given!')

        # Define Long-short Term Memory Network
        self.RecBlock = _get_rnn_block(
            window_size*input_channels + rnn_window*pred_size, blocks,
            num_layers, blocktype, bias, dropout)

        # Define Linear Network
        self.DNN = Feed_ForwardNN(blocks, pred_size*forecast*preds, hidden_size,
                                  activation, bias, dropout)

    def forward(self, inp, x_rec, device='cpu'):
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
        device : str, optional
            device to compute on. Needed because of the storage of the hidden
            cells. The default is 'cpu'.
        Returns
        -------
        output: torch.tensor dtype=torch.float()
            shape=[batch size, pred_size, forecast]
        """
        inp = torch.flatten(inp, start_dim=1)
        x_rec = torch.flatten(x_rec, start_dim=1)
        inp = torch.cat([inp, x_rec], dim=1)
        inp = self.RecBlock(inp, device)
        inp = self.DNN(inp)
        if self.Pred_Type == 'Point':
            return inp.reshape(-1, self.pred_size, self.forecast)
        elif self.Pred_Type == 'Mean_Var':
            pred = inp.reshape(-1, self.pred_size, self.forecast, 2)
            mean, hidden_std = pred[:,:,:,0], pred[:,:,:,1]
            var = F.softplus(hidden_std)
            return mean, var
    
    def forward_sens(self, inp, device='cpu'):
        """
        Forward function to propagate through the network, but only with one input tensor
        that is already concatenated to allow for gradient-based sensitivity analysis

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation,
            shape=[batch size, flatten_size]
        device : str, optional
            device to compute on. Needed because of the storage of the hidden
            cells. The default is 'cpu'.
        Returns
        -------
        output: torch.tensor dtype=torch.float()
            shape=[batch size, pred_size, forecast]
        """
        inp = self.RecBlock(inp, device)
        inp = self.DNN(inp)
        if self.Pred_Type == 'Point':
            return inp.reshape(-1, self.pred_size, self.forecast)
        elif self.Pred_Type == 'Mean_Var':
            pred = inp.reshape(-1, self.pred_size, self.forecast, 2)
            mean, hidden_std = pred[:,:,:,0], pred[:,:,:,1]
            var = F.softplus(hidden_std)
            return mean, var
    
    def estimate_uncertainty_mean_std(self, inp, x_rec, device='cpu'):
        return self(inp, x_rec, device)
    
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
        self.RecBlock.init_hidden()
        comp_sens, random_samples = _check_sens_params_RNN(sens_params)

        if comp_sens:
            pred_result = _prediction(self, dataloader, device, sens_params=sens_params)
            if random_samples:
                pred, sens_dict, sens_uq, eps = pred_result
            else:
                pred, sens_dict = pred_result
        else:
            pred = _prediction(self, dataloader, device)

        if self.Pred_Type == 'Point':
            if comp_sens:
                return pred[0], sens_dict
            else:
                return pred[0]
        
        elif self.Pred_Type == 'Mean_Var':
            if comp_sens:
                return (pred[0], pred[1]), sens_dict if not random_samples \
                    else ((pred[0], pred[1]), sens_dict, sens_uq, eps)
            else:
                return (pred[0], pred[1])

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
        torch.Size([64, 16])
        >>> print(rec_w[1].shape)
        torch.Size([64, 10])
        """

        rec_weights = self.RecBlock.get_recurrent_weights()
        if self.blocktype == 'GRU':
            rec_weights.append(
                self.RecBlock.gru.weight_ih_l0[
                    :, -self.rnn_window*self.pred_size:])
        if self.blocktype == 'LSTM':
            rec_weights.append(
                self.RecBlock.lstm.weight_ih_l0[
                    :, -self.rnn_window*self.pred_size:])
        return rec_weights


class RNN_DNN(nn.Module):
    """
    Recurrent Network that utilises a hidden state

    Parameters
    ----------
    input_channels : int
        Number of input channels
    pred_size : int
        Number of predicted values
    window_size : int
        Size of the sliding window applied to the time series
    blocks : int
        Number of parallel recurrent blocks.
    num_layers : int
        nunmber of recurernt blocks.(depth of the recurretn part)
    blocktype : str, optional
        blocktype, options are: 'RNN', 'GRU' and 'LSTM'. The default is 'LSTM'.
    hidden_size : list of int or None, optional
        List gives the size of hidden units. The default is None.
    activation : str, optional
        Activation function to activate the feature space.
        The default is 'relu'.
    bias : bool, optional
        If True, bias weights are used. The default is True.
    dropout : float [0,1], optional
        Adds dropout layers after each Linear Layer. The default is None.

    Returns
    -------
    None.

    Examples
    --------
    
    >>> import softsensor.recurrent_models
    >>> import torch
    >>> m = softsensor.recurrent_models.RNN_DNN(2, 1, 10, 16, 1)
    >>> print(m)
    RNN_DNN(
      (RecBlock): _LSTM(
        (lstm): LSTM(20, 16, batch_first=True)
      )
      (DNN): Feed_ForwardNN(
        (DNN): Sequential(
          (0): Linear(in_features=16, out_features=1, bias=True)
        )
      )
    )
    >>> input = torch.randn(32, 2, 10)
    >>> output = m(input)
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
                                   batch_size=32, Add_zeros=True)
    >>> pred = m.prediction(loader[0])
    >>> print(pred.shape)
    torch.Size([1, 101])

    """

    def __init__(self, input_channels, pred_size, window_size, blocks,
                 num_layers, blocktype='LSTM', hidden_size=None,
                 activation='relu', bias=True, dropout=None, forecast=1,
                 Pred_Type='Point'):

        super().__init__()

        self.params = _filter_parameters(locals().copy())

        self.input_channels = input_channels
        self.pred_size = pred_size
        self.window_size = window_size
        self.activation = activation
        self.forecast = forecast
        self.precomp = False
        self.rnn_window = None

        self.Type = 'RNN'
        self.Ensemble = False
        self.Pred_Type = Pred_Type
        if self.Pred_Type == 'Point':
            preds = 1
        elif self.Pred_Type == 'Mean_Var':
            preds = 2
        else:
            print('No valid Pred_Type given')

        # Define Long-short Term Memory Network
        self.RecBlock = _get_rnn_block(window_size*input_channels, blocks,
                                       num_layers, blocktype, bias, dropout)

        # Define Linear Network
        self.DNN = Feed_ForwardNN(blocks, pred_size*forecast*preds, hidden_size,
                                  activation, bias, dropout)

    def forward(self, inp, device='cpu'):
        """
        Forward function to probagate through the network

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation,
            shape=[batch size, external channels, window_size]
        device : str, optional
            device to compute on. Needed because of the storage of the hidden
            cells. The default is 'cpu'.
        Returns
        -------
        output: torch.tensor dtype=torch.float()
            shape=[batch size, pred_size, forecast]
        """
        inp = self.RecBlock(inp, device)
        inp = self.DNN(inp)
        if self.Pred_Type == 'Point':
            return inp.reshape(-1, self.pred_size, self.forecast)
        elif self.Pred_Type == 'Mean_Var':
            pred = inp.reshape(-1, self.pred_size, self.forecast, 2)
            mean, hidden_std = pred[:,:,:,0], pred[:,:,:,1]
            var = F.softplus(hidden_std)
            return mean, var
    
    def forward_sens(self, inp, device='cpu'):
        return self(inp, device)
    
    def estimate_uncertainty_mean_std(self, inp, device='cpu'):
        return self(inp, device)

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
        self.RecBlock.init_hidden()
        comp_sens, _ = _check_sens_params_RNN(sens_params)

        if comp_sens:
            pred, sensitivity = _pred_lstm(self, dataloader, device, sens_params=sens_params)
        else:
            pred = _pred_lstm(self, dataloader, device)

        if self.Pred_Type == 'Point':
            if comp_sens:
                return pred, sensitivity
            else:
                return pred
        
        elif self.Pred_Type == 'Mean_Var':
            if comp_sens:
                return (pred[0], pred[1]), sensitivity
            else:
                return (pred[0], pred[1])

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
        torch.Size([64, 16])
        """
        return self.RecBlock.get_recurrent_weights()

class parr_RNN_DNN(nn.Module):
    def __init__(self, input_channels, pred_size, blocks, hidden_window=1,
                 num_layers=1, blocktype='LSTM', hidden_size=None,
                 activation='relu', bias=True, dropout=None,
                 forecast=1, Pred_Type='Point'):
        super().__init__()

        self.params = _filter_parameters(locals().copy())
    
        self.input_channels = input_channels        
        self.pred_size = pred_size
        self.window_size = 1
        self.forecast = forecast
        self.hidden_window = hidden_window
        self.hidden_values = []
        self.precomp = True
        self.rnn_window = None

        self.Type = 'RNN'
        self.Pred_Type = Pred_Type
        self.Ensemble = False
        if self.Pred_Type == 'Point': 
            preds = 1
        elif self.Pred_Type == 'Mean_Var':
            preds = 2

        # Define Long-short Term Memory Network
        self.RecBlock = _parallel_RNN(input_channels, pred_size, blocks, hidden_window,
                                      num_layers, blocktype, bias, dropout,
                                      forecast)
        
        self.RecBlock.precomp_hidden_states()

            
        # Define Linear Network
        self.DNN = Feed_ForwardNN(blocks*hidden_window, pred_size*forecast*preds, hidden_size,
                                  activation, bias, dropout)
        
    def forward(self, inp, device='cpu'):
        inp = self.RecBlock(inp, device)
        inp = inp.flatten(start_dim=1)
        inp=self.DNN(inp)
        if self.Pred_Type == 'Point':
            return inp.reshape(-1, self.pred_size, self.forecast)
        elif self.Pred_Type == 'Mean_Var':
            pred = inp.reshape(-1, self.pred_size, self.forecast, 2)
            mean, hidden_std = pred[:,:,:,0], pred[:,:,:,1]
            var = F.softplus(hidden_std)
            return mean, var

    def estimate_uncertainty_mean_std(self, inp, device='cpu'):
        return self(inp, device)

    def prediction(self, dataloader, device='cpu'):
        self.RecBlock.RecBlock.init_hidden()
        self.RecBlock.precomp_hidden_states()
        return _pred_lstm(self, dataloader, device)

class _parallel_RNN(nn.Module):
    def __init__(self, input_channels, pred_size, blocks, hidden_window=1,
                 num_layers=1, blocktype='LSTM', bias=True, dropout=None,
                 forecast=1, Pred_Type='Point'):

        super().__init__()

        self.params = _filter_parameters(locals().copy())
    
        self.input_channels = input_channels        
        self.pred_size = pred_size
        self.window_size = 1
        self.forecast = forecast
        self.hidden_window = hidden_window
        self.hidden_values = []

        self.Type = 'RNN'
        self.Pred_Type = Pred_Type

        # Define Long-short Term Memory Network
        self.RecBlock = _get_rnn_block(input_channels, blocks,
                                      num_layers, blocktype, bias, dropout)

        self.precomp_hidden_states()

    def forward(self, inp, device='cpu'):
        """
        Forward function to probagate through the network

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation,
            shape=[batch size, external channels, window_size]
        device : str, optional
            device to compute on. Needed because of the storage of the hidden
            cells. The default is 'cpu'.
        Returns
        -------
        output: torch.tensor dtype=torch.float()
            shape=[batch size, pred_size]

        """
        inp = self.RecBlock(inp, device)
        self.hidden_values = torch.cat((self.hidden_values, inp), dim=0)
        
        dnn_inp =[]
        for i in range(inp.shape[0]):
            start = -inp.shape[0] - self.hidden_window + i + self.forecast
            end = -inp.shape[0] + i + self.forecast
            if end == 0:
                dnn_inp.append(self.hidden_values[start:, :])
            else:
                dnn_inp.append(self.hidden_values[start:end, :]) 
            
        
        inp = torch.stack(dnn_inp, dim=0)
        return inp
    
    def init_hidden(self):
        self.RecBlock.init_hidden()
    
    def precomp_hidden_states(self, device='cpu'):
        inp = torch.zeros(self.hidden_window - self.forecast,
                          self.input_channels,
                          1).to(device)
        self.hidden_values = self.RecBlock(inp, device)


'''
helpers
'''
def _get_rnn_block(input_size, blocks, num_layers, blocktype, bias, dropout):
    """
    Return Recurrent block from model class

    Parameters
    ----------
    input_size : int
        Input size of the block
    blocks : int
        Number of parallel recurrent blocks.
    num_layers : int
        number of recurrent blocks.(depth of the recurretn part)
    blocktype : str, optional
        blocktype, options are: 'RNN', 'GRU' and 'LSTM'. The default is 'LSTM'.
    bias : bool, optional
        If True, bias weights are used. The default is True.
    dropout : float [0,1], optional
        Adds dropout layers after each Linear Layer. The default is None.

    Returns
    -------
    nn.Module
        Recurrent block that will be used in the model.

    """
    if blocktype == 'LSTM':
        return _LSTM(input_size, hidden_size=blocks, num_layers=num_layers,
                    bias=bias, dropout=dropout)
    elif blocktype == 'GRU':
        return _GRU(input_size, hidden_size=blocks, num_layers=num_layers,
                   bias=bias, dropout=dropout)
    elif blocktype == 'RNN':
        return _RNN(input_size, hidden_size=blocks, num_layers=num_layers,
                   bias=bias, dropout=dropout)


def _check_sens_params_RNN(sens_params):
    """
    Check the sensitivity parameters for AR_RNN and RNN_DNN models for validity

    Parameters
    ----------
    sens_params : dict
        Dictionary that contains the parameters for the sensitivity analysis.
    
    Returns
    -------
    comp_sens : bool
        Boolean that defines whether sensitivity analysis is computed.
    """
    if sens_params:
        comp_sens = sens_params.get('comp', False)
        random_samples = sens_params.get('random_samples', None)
    else:
        comp_sens = False
        random_samples = None
    return comp_sens, random_samples


def _pred_lstm(model, dataloader, device, sens_params=None):
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
        (If not a multiple of the model's forecast*batch_size, the number will be
        rounded to the closest multiple.)
        The default is None, i.e. no sensitivity analysis is computed.

    Returns
    -------
    if loss_ft=None:
        torch.Tensor
            Torch Tensor of same langth as input
    if loss_ft=torch loss funciton:
        (torch.Tensor, loss)
            tuple of Torch Tensor of same langth as input and loss
    if comp_gradients=True:
        (torch.Tensor, sensitivity dict)
            tuple of Torch Tensor of same langth as input and sensitivity dictionary
    """
    fc = model.forecast
    prediction_type = model.Pred_Type
    if prediction_type == 'Point':
        num_outputs = 1
    elif prediction_type == 'Mean_Var':
        num_outputs = 2
    prediction = torch.full((num_outputs, model.pred_size, model.window_size-1), 0)
    original_out = torch.full((model.pred_size, model.window_size-1), 0)

    model.to(device)
    prediction = prediction.to(device)
    original_out = original_out.to(device)

    checks = _check_sens_params_pred(sens_params)
    if sens_params:
        method, comp_sens, verbose, sens_length, num_samples, std_dev, correlated = checks[:-2]
    else:
        comp_sens, verbose = checks # False

    # Initialise 3D tensor for sensitivity analysis
    if comp_sens:
        flatten_size = model.window_size*model.input_channels
        num_timesteps = len(dataloader.dataset)*fc
        loader_length = num_timesteps - dataloader.dataset.subclass.add_zero
        sens_indices = np.arange(len(dataloader.dataset))

        if sens_length:
            num_timesteps, sens_indices = _random_subset_sens_indices(sens_length, fc, model.Type, dataloader)
        
        sens_mean = torch.full((num_timesteps, model.pred_size, flatten_size), float('nan'))
        if prediction_type == 'Mean_Var':
            sens_var = sens_mean.clone()
        if verbose:
            print(f'Shape of sensitivity tensor: {sens_mean.shape}')
            print(f'Start {method.upper()}-based Sensitivity Analysis...')

    idx = 0
    for i, data in enumerate(tqdm(dataloader) if verbose else dataloader):
        inputs, output = data
        inputs, output = inputs.to(device), output.to(device)

        # get batch size from inputs
        bs = inputs.size(0)

        # Prepare input for model to allow autograd computing gradients of outputs w.r.t. inputs
        inputs = Variable(torch.flatten(inputs, start_dim=1), requires_grad=True)
        if comp_sens:
            pred = model.forward_sens(inputs, device)
        elif prediction_type == 'Mean_Var' or model.Ensemble:
            pred = model.estimate_uncertainty_mean_std(inputs, device)
        else:
            pred = model(inputs, device)

        if comp_sens and i in sens_indices:
            sens_temp = _comp_sensitivity(method, model, inputs, pred, num_samples, std_dev, correlated)
            if prediction_type == "Mean_Var" or model.Ensemble:
                sens_mean[idx:idx+bs*fc], sens_var[idx:idx+bs*fc] = sens_temp
            else:
                sens_mean[idx:idx+bs*fc] = sens_temp
            idx += bs*fc
        
        if type(pred) == tuple:
            pred = torch.vstack(pred)

        prediction = torch.cat((prediction, pred.detach().reshape(num_outputs, model.pred_size, -1)), 2)
        original_out = torch.cat((original_out, output.reshape(model.pred_size, -1)), 1)

    cut_zeros = dataloader.dataset.subclass.add_zero
    if cut_zeros != 0:
        prediction = prediction[..., :-cut_zeros]
        original_out = original_out[:, :-cut_zeros]
    
    if prediction_type == 'Point':
        prediction = prediction[0, ...]
    
    # shape = [num_outputs, pred_size, len(dataset)] = [num_outputs, pred_size, forecast*len(dataloader)]
    prediction = prediction[..., model.window_size-1:]
    original_out = original_out[:, model.window_size-1:]
    
    if comp_sens:
        sens_mean = sens_mean[:loader_length]
        if prediction_type == "Mean_Var" or model.Ensemble:
            sens_var = sens_var[:loader_length]
            sensitivity_dict = {'Mean': sens_mean.cpu(), 'Aleatoric_UQ': sens_var.cpu()}
        else:
            sensitivity_dict = {'Point': sens_mean.cpu()}
        if verbose:
            print(f'{method.upper()}-based Sensitivity Analysis completed!\n')
    
        return prediction, sensitivity_dict
    else:
        return prediction
