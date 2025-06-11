# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from softsensor.autoreg_models import _async_prediction, _prediction


class SyncEnsemble(nn.Module):
    """ Sync ensemble wrapper for point prediction models.
        All ensemble models have the same recurrent state at each point.

    Parameters
    ----------
    model : uncertainty model 
        model that is used for prediction
    ensemble_weights : list[torch state_dict]
        weights of the individual base models

    Returns
    -------
    None.
    
    Examples
    --------
    
    >>> import softsensor.ensemble_wrappers as ew
    >>> import softsensor.autoreg_models as am
    >>> import torch
    >>> m = []
    >>> for i in range(5):
    >>>     m_temp = am.ARNN(2, 1, 10, 10, [16, 8])
    >>>     m.append(m_temp.state_dict())
    >>> ensemble = ew.SyncEnsemble(am.ARNN(2, 1, 10, 10, [16, 8]), m)
    >>> input = torch.randn(32, 2, 10)
    >>> rec_input = torch.randn(32, 1, 10)
    >>> out = ensemble.estimate_uncertainty_mean_std(input, rec_input)
    >>> print(out[0]) # Mean
    >>> print(out[1]) # Std
    """
    def __init__(self, model, ensemble_weights):
        super().__init__()

        self.Type = model.Type
        self.window_size = model.window_size
        self.rnn_window = model.rnn_window
        self.forecast = model.forecast
        self.input_channels = model.input_channels
        self.pred_size = model.pred_size
        self.activation = model.activation

        self.model = model
        self.ensemble_weights = ensemble_weights
        self.Type = 'AR'
        self.Pred_Type = 'Point'
        self.Ensemble = True

    def load_and_fire(self, weights, inputs, x_rec):
        """
        Loads the state_dict of a base model and computes forward

        Parameters
        ----------
        weights : torch state_dict
            Weights of a base model
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation,
            shape=[batch size, external channels, window_size]
        x_rec : torch.tensor, dtype=torch.float
            Recurrent Input for forward Propagation.
            shape=[batch size, pred_size, rnn_window]

        Returns
        -------
        pred: torch.tensor dtype=torch.float()
            shape=[batch size, pred_size]
        """
        self.model.load_state_dict(weights)
        pred = self.model(inputs, x_rec)
        return pred

    def estimate_uncertainty_mean_std(self, inp, x_rec):
        """
        Computes the mean prediction and epistemic uncertainty of the ensemble for one step

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
        (mean, std)
            mean: torch.tensor dtype=torch.float()
                shape=[batch size, pred_size]

            std: torch.tensor dtype=torch.float() in [0,1]
                shape=[batch size, pred_size]
        """
        preds = [self.load_and_fire(w, inp, x_rec) for w in self.ensemble_weights]
        mean = torch.stack(preds).mean(axis=0).detach()
        std = torch.stack(preds).std(axis=0).detach()
        return (mean, std)
    
    def forward_sens(self, inp):
        """
        Computes the mean prediction and epistemic uncertainty of the ensemble for one step,
        but only with one input tensor that is already concatenated to allow for
        gradient-based sensitivity analysis

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation that is already concatenated,
            shape=[batch size, external channels*window_size + pred_size*rnn_window]

        Returns
        -------
        (mean, std)
            mean: torch.tensor dtype=torch.float()
                shape=[batch size, pred_size]

            std: torch.tensor dtype=torch.float() in [0,1]
                shape=[batch size, pred_size]
        """
        preds = []
        for w in self.ensemble_weights:
            self.model.load_state_dict(w)
            pred = self.model.forward_sens(inp)
            preds.append(pred)
        
        mean = torch.stack(preds).mean(axis=0)
        std = torch.stack(preds).std(axis=0)
        return (mean, std)

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
                
        Examples
        --------
        Based on the Example from initialisation
        
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
        >>> pred = ensemble.prediction(loader[0]) # Prediction of Mean and Variance
        >>> print(pred.shape)
        torch.Size([2, 1, 101])
        """
        return _prediction(self, dataloader, device, sens_params=sens_params)

class AsyncEnsemble(SyncEnsemble):
    """
    Async ensemble wrapper for point prediction models.
    Ensemble members predict the entire time series independently.

    Parameters
    ----------
    model : uncertainty model 
        model that is used for prediction
    ensemble_weights : list[torch state_dict]
        weights of the individual base models

    Returns
    -------
    None.
    

    Examples
    --------
    
    >>> import softsensor.ensemble_wrappers as ew
    >>> import softsensor.autoreg_models as am
    >>> import torch
    >>> m = []
    >>> for i in range(5):
    >>>     m_temp = am.ARNN(2, 1, 10, 10, [16, 8])
    >>>     m.append(m_temp.state_dict())
    >>> ensemble = ew.AsyncEnsemble(am.ARNN(2, 1, 10, 10, [16, 8]), m)
    >>> input = torch.randn(32, 2, 10)
    >>> rec_input = torch.randn(32, 1, 10)
    >>> out = ensemble.estimate_uncertainty_mean_std(input, rec_input)
    >>> print(out[0]) # Mean
    >>> print(out[1]) # Std
        
    """

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

        Examples
        --------
        Based on the Example from initialisation
        
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
        >>> pred = ensemble.prediction(loader[0]) # Prediction of Mean and Variance
        >>> print(pred[0].shape)
        torch.Size([1, 101])
    
        """
        return _async_prediction(self.model, dataloader, device,
                                 ensemble_weights=self.ensemble_weights, sens_params=sens_params)

class AsyncMVEEnsemble(AsyncEnsemble):
    """ Async ensemble wrapper for MVE models
    
    Parameters
    ----------
    model : uncertainty model 
        model that is used for prediction
    ensemble_weights : list[torch state_dict]
        weights of the individual base models

    Returns
    -------
    None.
    

    Examples
    --------
    
    >>> import softsensor.ensemble_wrappers as ew
    >>> import softsensor.autoreg_models as am
    >>> import torch
    >>> params = {'input_channels': 2,
                  'pred_size': 1,
                  'window_size': 10,
                  'rnn_window': 10}
    >>> m = []
    >>> for i in range(5):
    >>>     mean_model = am.ARNN(**params, hidden_size=[16, 8])
    >>>     m_temp = am.SeparateMVEARNN(**params,mean_model=mean_model,
                                        var_hidden_size=[16, 8])
    >>>     m.append(m_temp.state_dict())
    >>> ensemble = ew.AsyncMVEEnsemble(m_temp, m)
    >>> input = torch.randn(32, 2, 10)
    >>> rec_input = torch.randn(32, 1, 10)
    >>> out = ensemble.estimate_uncertainty_mean_stds(input, rec_input)
    >>> print(len(out))
    3
        
    """
    def __init__(self, model, ensemble_weights):
        AsyncEnsemble.__init__(self, model, ensemble_weights)
        self.Pred_Type = 'Mean_Var'
    
    def estimate_uncertainty_mean_stds(self, inp, x_rec):
        """
        Computes the mean prediction and uncertainty of the ensemble for one step

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
        (mean, aleatoric std, epistemic std)
            mean: torch.tensor dtype=torch.float()
                shape=[batch size, pred_size]

            std: torch.tensor dtype=torch.float() in [0,1]
                shape=[batch size, pred_size]
        """
        preds = [self.load_and_fire(w, inp, x_rec) for w in self.ensemble_weights]
        means = torch.stack([p[0] for p in preds])
        vars = torch.stack([p[1] for p in preds])
        mean = means.mean(axis=0).detach()
        epist_var = means.square().mean(axis=0).detach() - mean.square() # corresponds to Var(X) = E[X^2] - E[X]^2
        aleat_var = vars.mean(axis=0).detach()
        return (mean, epist_var, aleat_var)

    def prediction(self, dataloader, device='cpu', reduce=False, sens_params=None):
        """
        Prediction of a whole Time Series

        Parameters
        ----------
        dataloader : Dataloader
            Dataloader to predict output
        device : str, optional
            device to compute on. The default is 'cpu'.
        reduce: bool, optional
            Whether the combined uncertainty or both uncertainties should be returned. The default is True.
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
            Key of that dict is the prediction type (str), value is the sensitivity tensor.
        
        Examples
        --------
        Based on the Example from initialisation
        
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
        >>> pred = ensemble.prediction(loader[0], reduce=False) # Prediction of Mean and Variance
        >>> print(pred[0].shape) # Mean
        >>> print(pred[1].shape) # aletoric Variance 
        >>> print(pred[2].shape) # epistemic Variance
        """
        return _async_prediction(self.model, dataloader, device, n_samples=1, reduce=reduce,
                                 ensemble_weights=self.ensemble_weights, sens_params=sens_params)

class AsyncMCDropout(nn.Module):
    """ Async MCDO wrapper for probabilistic point prediction models

    Parameters
    ----------
    model : uncertainty model 
        probabilistic model that is used for prediction 
    n_samples : int
        amount of samples to draw for monte carlo estimation

    Returns
    -------
    None.
    
    
    Examples
    --------
    
    >>> import softsensor.autoreg_models as am
    >>> import softsensor.ensemble_wrappers as ew
    >>> import torch
    >>> m = am.ARNN(2, 1, 10, 10, [16, 8], dropout=.5, concrete_dropout=True)
    >>> ensemble = ew.AsyncMCDropout(m, n_samples=10)
    >>> input = torch.randn(32, 2, 10)
    >>> rec_input = torch.randn(32, 1, 10)
    >>> out = ensemble.estimate_uncertainty_mean_std(input, rec_input)
    >>> print(out[0].shape) # Mean
    >>> print(out[1].shape) # Std
    """
    def __init__(self, model, n_samples): 
        super().__init__()
        
        self.Type = model.Type
        self.window_size = model.window_size
        self.rnn_window = model.rnn_window
        self.forecast = model.forecast
        self.pred_size = model.pred_size

        self.model = model
        self.n_samples = n_samples
        self.Type = model.Type
        self.Pred_Type = model.Pred_Type
        self.Ensemble = True

    def estimate_uncertainty_mean_std(self, inp, x_rec):
        """
        Computes the mean prediction and epistemic uncertainty of the ensemble for one step

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
        (mean, std)
            mean: torch.tensor dtype=torch.float()
                shape=[batch size, pred_size]

            std: torch.tensor dtype=torch.float() in [0,1]
                shape=[batch size, pred_size]
        """
        preds = [self.model(inp, x_rec) for i in range(self.n_samples)]
        mean = torch.stack(preds).mean(axis=0).detach()
        std = torch.stack(preds).std(axis=0).detach()
        return (mean, std)

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
            (torch.Tensor, list[torch.Tensor])
                tuple of Torch Tensor of same length as input and variances
        if loss_ft=torch loss function:
            (torch.Tensor, list[torch.Tensor], loss)
                tuple of Torch Tensor of same length as input, variances and loss
        
        Examples
        --------
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
        >>> pred = ensemble.prediction(loader[0]) # Prediction of Mean and Variance

        """
        return _async_prediction(self.model, dataloader, device, self.n_samples)
        
class AsyncMCDropoutMVE(AsyncMCDropout):
    """ Async MCDO wrapper for MVE models
    
    Parameters
    ----------
    model : uncertainty model 
        probabilistic model that is used for prediction 
    n_samples : int
        amount of samples to draw for monte carlo estimation

    Returns
    -------
    None.
    
    Examples
    --------
    
    >>> import softsensor.autoreg_models as am
    >>> import softsensor.ensemble_wrappers as ew
    >>> import torch
    >>> params = {'input_channels': 2,
                  'pred_size': 1,
                  'window_size': 10,
                  'rnn_window': 10}
    >>> mean_model = am.ARNN(**params, hidden_size=[16, 8],
                             dropout=.5, concrete_dropout=True)
    >>> m = am.SeparateMVEARNN(**params,mean_model=mean_model,
                               var_hidden_size=[16, 8])
    >>> ensemble = ew.AsyncMCDropoutMVE(m, n_samples=10)
    >>> input = torch.randn(32, 2, 10)
    >>> rec_input = torch.randn(32, 1, 10)
    >>> out = ensemble.estimate_uncertainty_mean_stds(input, rec_input)
    >>> print(len(out))
    """
    def __init__(self, model, n_samples):
        AsyncMCDropout.__init__(self, model, n_samples)
        self.Pred_Type = 'Mean_Var'
    
    def estimate_uncertainty_mean_stds(self, inp, x_rec):
        """
        Computes the mean prediction and uncertainty of the ensemble for one step

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
        (mean, aleatoric std, epistemic std)
            mean: torch.tensor dtype=torch.float()
                shape=[batch size, pred_size]

            std: torch.tensor dtype=torch.float() in [0,1]
                shape=[batch size, pred_size]
        """
        preds = [self.model(inp, x_rec) for i in range(self.n_samples)]
        means = torch.stack([p[0] for p in preds])
        vars = torch.stack([p[1] for p in preds])
        mean = means.mean(axis=0).detach()
        epist_var = means.square().mean(axis=0).detach() - mean.square() # corresponds to Var(X) = E[X^2] - E[X]^2
        aleat_var = vars.mean(axis=0).detach()
        return (mean, epist_var, aleat_var)

    def prediction(self, dataloader, device='cpu', reduce=False):
        """
        Prediction of a whole Time Series

        Parameters
        ----------
        dataloader : Dataloader
            Dataloader to predict output
        device : str, optional
            device to compute on. The default is 'cpu'.
        reduce: bool, optional
            Whether the combined uncertainty or both uncertainties should be returned. The default is True.

        Returns
        -------
        if loss_ft=None:
            (torch.Tensor, torch.Tensor) if reduce else (torch.Tensor, torch.Tensor, torch.Tensor)
                tuple of Torch Tensor of same length as input (prediction, uncertainties)
                where uncertainties = aleatoric_var + epistemic_var if reduce else (aleatoric_var, epistemic_var)
        if loss_ft=torch.loss:
            ((torch.Tensor, torch.Tensor), float) if reduce else ((torch.Tensor, torch.Tensor, torch.Tensor), float)
                adds the computed loss as a second output ((prediction, uncertainties), loss)
        
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
        >>> pred = ensemble.prediction(loader[0], reduce=False) # Prediction of Mean and Variance
        >>> print(pred[0].shape) # Mean
        >>> print(pred[1].shape) # aletoric Variance 
        >>> print(pred[2].shape) # epistemic Variance
    
        """
        return _async_prediction(self.model, dataloader, device, self.n_samples, reduce)
