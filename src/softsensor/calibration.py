# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

from scipy.optimize import minimize_scalar
from softsensor.autoreg_models import _prediction
from softsensor.recurrent_models import _pred_lstm
from softsensor.eval_tools import comp_batch
from softsensor.metrics import ece


def _optimize_scale_factor(
    mean,
    var,
    targets,
    criterion = ece,
    worst_cal = 0.5,
    optimizer_bounds = (1e-2, 5)
):
    """
    Optimizes the calibration temperature for a single output on the prediction and targets
    
    Uses Brent's method (https://en.wikipedia.org/wiki/Brent%27s_method) for black-box optimization
    
    Parameters
    ----------
    mean : torch.tensor
        mean prediction of the uncalibrated model
    var : torch.tensor
        var prediction of the uncalibrated model
    targets : torch.tensor
        ground truth of the dataset
    criterion: function with inputs (mean, targets, var), optional
        criterion to evaluate for optimization. The default is ECE
    worst_cal: float, optional
        worst score for criterion. The default is 0.5 (worst ECE)
    optimizer_bounds: (float, float), optional
        lower bound and higher bound for temperature value. The default is (0.01, 2)

    Returns
    -------
    opt_ratio: float
        Scaling factor that is used for calibration of the predictive std
    """
    def obj(ratio):
        # If ratio is 0, return worst-possible calibration metric
        if ratio == 0:
            return worst_cal
        
        curr_cal = criterion(mean, targets, ratio**2 * var)
        return curr_cal

    result = minimize_scalar(fun=obj, bounds=optimizer_bounds)
    opt_ratio = result.x

    if not result.success:
        print("Optimization did not succeed")
        original_cal = criterion(mean, targets, var)
        ratio_cal = criterion(mean, targets, opt_ratio**2 * var)
        if ratio_cal > original_cal:
            print(
                "No better calibration found, no recalibration performed and returning original uncertainties"
            )
            opt_ratio = 1.0

    return opt_ratio


def optimize_temperature(model, data_handle, val_tracks, criterion=ece, device='cpu', worst_cal=0.5, optimizer_bounds=(.01, 5)):
    """
    Optimizes the calibration temperature for each output on the calibration set
    
    Parameters
    ----------
    model : uncertainty model 
        model to calibrate
    calibration_loader : List[Dataloader]
        dataset to fit the scaling factor
    criterion: function with inputs (mean, targets, var), optional
        criterion to evaluate for optimization. The default is ECE
    worst_cal: float, optional
        worst score for criterion. The default is 0.5 (worst ECE)

    Returns
    -------
    temperature: torch.tensor[float], shape=[model.pred_size]
    """
    num_outputs = model.pred_size
    cal_mean = torch.tensor([])
    cal_var = torch.tensor([])
    cal_targets = torch.tensor([])
    
    dfs = comp_batch([model], data_handle, val_tracks, device=device, names=['model'])
    for d in dfs:
        mean_sens = [f'{s}_model' for s in data_handle.output_sensors]
        cal_mean = torch.cat([cal_mean, torch.tensor(np.array(d[mean_sens]))], dim=0)
        
        var_sens = [f'{s}_model_var' for s in data_handle.output_sensors]
        cal_var = torch.cat([cal_var, torch.tensor(np.array(d[var_sens]))], dim=0)
        
        sens = [f'{s}' for s in data_handle.output_sensors]
        cal_targets = torch.cat([cal_targets, torch.tensor(np.array(d[sens]))], dim=0)
        
    # Determine calibration temparature
    temperature = [_optimize_scale_factor(cal_mean[:, output], cal_var[:, output],
                                         cal_targets[:, output], criterion, worst_cal, optimizer_bounds)
                   for output in range(num_outputs)]
    return torch.tensor(temperature).float().to(device)


class TemperatureScaling(nn.Module):
    """
    Wrapper class for uncertainty models that applies temperature scaling after prediction

    Parameters
    ----------
    model : uncertainty model 
        model that is used for prediction
    temperature : tensor[float]
        factors for std scaling (squared for)

    Returns
    -------
    None.
    
    Examples
    --------
    
    Define Model

    >>> import softsensor.autoreg_models
    >>> from softsensor.calibration import TemperatureScaling
    >>> import torch
    >>> m = softsensor.autoreg_models.DensityEstimationARNN(2, 1, 10, 10, [16, 8])
    >>> temps = torch.tensor([.5])
    >>> scaled_model = TemperatureScaling(m, temps)
    >>> input = torch.randn(32, 2, 10)
    >>> rec_input = torch.randn(32, 1, 10)
    >>> output = scaled_model(input, rec_input)
    >>> print(output[0].shape) # mean
    torch.Size([32, 1, 1])
    >>> print(output[1].shape) # new var
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

    Check model prediction scaled and unscaled

    >>> loader = handler.give_list(window_size=10, keyword='training',
                                   rnn_window=10, batch_size=1)

    >>> mean, var = m.prediction(loader[0])
    >>> print(var[0][:5]*0.25)
    tensor([0.1769, 0.1831, 0.1769, 0.1770, 0.1798])
    >>> mean, var_scaled = scaled_model.prediction(loader[0])
    >>> print(var[0][:5])
    tensor([0.1769, 0.1831, 0.1769, 0.1770, 0.1798])


    """

    def __init__(self, model, temperature):
        super().__init__()

        self.model = model
        self.temperature = nn.Parameter(temperature)
        self.temperature.requires_grad = False
        
        self.window_size = model.window_size
        self.rnn_window = model.rnn_window
        self.forecast = model.forecast
        self.input_channels = model.input_channels
        self.pred_size = model.pred_size
        try: # needed for computation of stability score
            self.activation = model.activation
        except Exception: # for models that do not have an activation function
            pass
    
        self.Pred_Type = model.Pred_Type
        self.Type = model.Type
        self.Ensemble = model.Ensemble
        
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
        mean, var = self.model(*args)
        #var = [torch.ones([mean.shape[0], 1, mean.shape[2]]) * v for v in self.homoscedastic_var]
        temps = self.temperature.square()[:, None]
        for i, v in enumerate(temps):
            var[:, i, :] = var[:, i, :]*v
        return mean, var

    def forward_sens(self, *args):
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
        mean, var = self.model.forward_sens(*args)
        #var = [torch.ones([mean.shape[0], 1, mean.shape[2]]) * v for v in self.homoscedastic_var]
        temps = self.temperature.square()[:, None]
        for i, v in enumerate(temps):
            var[:, i, :] = var[:, i, :]*v
        return mean, var

    def estimate_uncertainty_mean_std(self, inp, x_rec):
        return self(inp, x_rec)
    
    def get_recurrent_weights(self):
        """
        Function that returns the weight that effect the Recurrent input of the
        underlying Network (mean network)

        Returns
        -------
        recurrent_weights : list of weight Tensors
            List of the Weights that effect the Recurrent input of the Network.
        """
        return self.model.get_recurrent_weights()

    def prediction(self, dataloader, device="cpu", sens_params=None):
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
        if loss_ft=None:
            (torch.Tensor, list[torch.Tensor])
                tuple of Torch Tensor of same length as input and var
        if loss_ft=torch loss funciton:
            (torch.Tensor, list[torch.Tensor], loss)
                tuple of Torch Tensor of same length as input, var and loss
        """
        if self.Type == 'RNN':
            if self.model.precomp:
                self.model.to(device)
                self.model.RecBlock.RecBlock.init_hidden()
                self.model.RecBlock.precomp_hidden_states(device)
                return _pred_lstm(self, dataloader, device, sens_params=sens_params)
            else:
                self.model.RecBlock.RecBlock.init_hidden()
                return _pred_lstm(self, dataloader, device, sens_params=sens_params)
        else:
            return _prediction(self, dataloader, device, sens_params=sens_params)
