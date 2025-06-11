# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:00:36 2022

@author: WET2RNG
"""
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import inspect
import random
import sys
import math

from softsensor.autoreg_models import ARNN, SeparateMVEARNN, DensityEstimationARNN, QuantileARNN
from softsensor.recurrent_models import AR_RNN, RNN_DNN, parr_RNN_DNN
from softsensor.train_model import train_model
from softsensor.stab_scheduler import get_scheduler
from softsensor.eval_tools import comp_batch, _comp_ARNN_batch
from softsensor.ensemble_wrappers import AsyncMCDropout, AsyncMCDropoutMVE
from softsensor.calibration import optimize_temperature, TemperatureScaling
from softsensor.temporal_fusion_transformer import TFT

from sklearn.model_selection import ParameterGrid
from hyperopt import fmin, tpe, STATUS_OK, Trials


def random_search(data_handle, criterion, model_type,
                parameters, grid_params, max_iterations=3, pretrained_model=None,
                reconfigure_criterion=False, val_criterion=None, val_prediction=False, device='cpu',
                key='training', print_results=False, calibration=None, n_samples=[5000, 1000]):
    """

    random search to efficiently optimize hyperparameters. 
    Returns parameters and best model for evaluation

    Parameters
    ----------
    
    data_handle : Meas Handling class
        used for getting the training and evaluation data
    criterion : nn.Loss function
        Loss function e.g. nn.MSELoss()
    model_type : str
        Describing the Model Type: currently implemented
        ['ARNN', 'MCDO', 'MVE', 'Sep_MVE', 'MVE_MCDO', 'MVE_Student_Forced',
        'QR', 'QR_Student_Forced', 'BNN', 'MVE_BNN', 'RNN'].
    parameters : dict
        dictionary of static parameters in the grid search.
    grid_params : dict
        dict of grid parameters with grid options as list or scipy stats function.
        See examples:
    max_iterations : int, optional
        number of iterations. The default is 3
    pretrained_model: str, optional
        path to pretrained model to load as base model. The default is None
    reconfigure_criterion: bool, optional
        if True, the criterion is reconfigured with params from the grid. The default is False
    val_criterion: nn.Loss function, optional
        val_criterion to be used for validation instead of criterion. The default is None
    val_prediction : bool, optional
        if True, prediction on testing tracks in data_handle is used for
        hyperparameter evaluation
    device : str, optional
        device to run training on. The default is 'cpu'.
    key : str, optional
        'training' or 'short'. Training uses whole dataloader, short just
        subset for training. default is training
    print_results : bool, optional
        True prints results for every epoch. default is False
    
    Returns
    -------
    result_df : pd.DataFrame
        parameters and corrosponding results for each grid search step.
    best_model : torch Model
        best performing model.
        
    Examples
    --------
    
    Data Preprocessing
    
    >>> import softsensor.meas_handling as ms
    >>> import numpy as np
    >>> import pandas as pd
    >>> t = np.linspace(0, 1.0, 101)
    >>> d = {'sine_inp': np.sin(2 * np.pi * 100 * t) ,
             'cos_inp': np.cos(2 * np.pi * 50 * t),
             'out': np.linspace(0, 1.0, 101)}
    >>> list_of_df = [pd.DataFrame(d), pd.DataFrame(d)]
    >>> test_df = {'sine_inp': np.sin(2 * np.pi * 100 * t),
                   'cos_inp': np.cos(2 * np.pi * 50 * t),
                   'out': np.linspace(0, 1.0, 101)}
    >>> test_df = [pd.DataFrame(test_df)]
    >>> handler = ms.Meas_handling(list_of_df, train_names=['sine1', 'sine2'],
                                   input_sensors=['sine_inp', 'cos_inp'],
                                   output_sensors=['out'], fs=100,
                                   test_dfs=test_df, test_names=['test'])

    Optimize an ARNN

    >>> from softsensor.hyperparameter_optimization import random_search
    >>> import scipy.stats as stats
    >>> import torch.nn as nn
    >>> grid_params = {'lr': stats.loguniform(1e-4, 1e-1),
                       'optimizer': ['Adam', 'SGD']}
    >>> model_type = 'ARNN'
    >>> model_params = {'input_channels': 2,
                        'pred_size': 1,
                        'window_size': 50,
                        'rnn_window': 10,
                        'max_epochs': 3,
                        'patience': 3,
                        'hidden_size': [8],
                        }
    >>> criterion = nn.MSELoss()
    >>> df, model = random_search(handler, criterion,
                                  model_type, model_params, grid_params,
                                  max_iterations=4, val_prediction=True)
    run 1/4 finishes with loss 0.1794743835926056 and parameters {'lr': 0.0009228490219458666, 'optimizer': 'Adam'}, time=0s
    run 2/4 finishes with loss 0.16934655606746674 and parameters {'lr': 0.00040497789386739904, 'optimizer': 'SGD'}, time=0s
    run 3/4 finishes with loss 0.09789121896028519 and parameters {'lr': 0.0033839123896820455, 'optimizer': 'Adam'}, time=0s
    run 4/4 finishes with loss 0.0789249911904335 and parameters {'lr': 0.00022746356317548106, 'optimizer': 'Adam'}, time=0s
    
    Optimize an ARNN with Mean Variance Estimation
    
    >>> from softsensor.hyperparameter_optimization import grid_search
    >>> import torch.nn as nn
    >>> from softsensor.losses import DistributionalMSELoss
    >>> model_type = 'MVE'
    >>> model_params = {'input_channels': 2,
                        'pred_size': 1,
                        'window_size': 10,
                        'rnn_window': 10,
                        'max_epochs': 3,
                        'patience': 3,
                        'hidden_size': [8],
                        'var_hidden_size': [8],
                        }
    >>> df, model = random_search(handler, DistributionalMSELoss(),
                                  model_type, model_params, grid_params, max_iterations=4,
                                  val_prediction=True, val_criterion=DistributionalMSELoss())
    run 1/4 finishes with loss 0.11428132653236389 and parameters {'lr': 0.00036427297324473465, 'optimizer': 'SGD'}, time=0s
    run 2/4 finishes with loss 0.6519314646720886 and parameters {'lr': 0.0023511068144571627, 'optimizer': 'SGD'}, time=0s
    run 3/4 finishes with loss 0.14868338406085968 and parameters {'lr': 0.0003292040157588834, 'optimizer': 'Adam'}, time=0s
    run 4/4 finishes with loss 0.0994466096162796 and parameters {'lr': 0.01498455784730249, 'optimizer': 'Adam'}, time=0s
    """

    grid = []
    for i in range(max_iterations):
        temp_dict = {}
        for entry in grid_params:
            if isinstance(grid_params[entry], list):
                nd = {entry: random.choice(grid_params[entry])}
            else:
                nd = {entry: grid_params[entry].rvs(size=1).item()}
            temp_dict = {**temp_dict, **nd}

        grid.append(temp_dict)

    param_dict = []
    result_list = []
    loss_list = []
    model_list = []
    
    for i, gp in enumerate(grid):
        start = time.time()
        params = {**parameters, **gp}
        print(f'Run {i+1}/{len(grid)} started with parameters {gp}...')
        (val_loss, model, params, results) = _eval_grid_params(params, data_handle, model_type, device, key, criterion,
                                                               pretrained_model, reconfigure_criterion, print_results,
                                                               calibration, val_prediction, val_criterion, n_samples)
        
        # Add parameters to list
        loss_list.append(val_loss)
        model_list.append(model)
        param_dict.append({**params, **results})
        result_list.append(results)

        end = time.time()
        print(f'Run {i+1}/{len(grid)} finished with loss {val_loss}' +
              f' and parameters {gp}, time={int(end - start)}s \n')
        sys.stdout.flush()

    loss_list = np.asarray(loss_list)

    # return best model and results
    result_df = pd.DataFrame(param_dict)
    result_df['loss'] = loss_list
    best_model = model_list[np.nanargmin(loss_list)]

    return result_df, best_model


def grid_search(data_handle, criterion, model_type,
                parameters, grid_params, pretrained_model=None, reconfigure_criterion=False,
                val_criterion=None, val_prediction=False, device='cpu',
                key='training', print_results=False, calibration=None, n_samples=[5000, 1000]):
    """

    Grid search class to efficiently optimize hyperparameters. 
    Returns parameters and best model for evaluation

    Parameters
    ----------
    
    data_handle : Meas Handling class
        used for getting the training and evaluation data

    criterion : nn.Loss function
        Loss function e.g. nn.MSELoss()
    model_type : str
        Describing the Model Type: currently implemented.
        ['ARNN', 'MCDO', 'MVE', 'Sep_MVE', 'MVE_MCDO', 'MVE_Student_Forced',
        'QR', 'QR_Student_Forced', 'BNN', 'MVE_BNN', 'RNN'].
    parameters : dict
        dictionary of static parameters in the grid search.
    grid_params : dict
        dict of grid parameters with grid options as list.
    pretrained_model: str, optional
        path to pretrained model to load as base model. The default is None
    reconfigure_criterion: bool, optional
        if True, the criterion is reconfigured with params from the grid. The default is False
    val_criterion: nn.Loss function, optional
        val_criterion to be used for validation instead of criterion. The default is None
    val_prediction : bool, optional
        if True, prediction on testing tracks in data_hanlde is used for
        hyperparameter evaluation
    device : str, optional
        device to run training on. The default is 'cpu'.
    key : str, optional
        'training' or 'short'. Training uses whole dataloader, short just
        subset for training. default is training
    print_results : bool, optional
        True prints results for every epoch. default is False
    
    Returns
    -------
    result_df : pd.DataFrame
        parameters and corrosponding results for each grid search step.
    best_model : torch Model
        best performing model.
        
    Examples
    --------
    
    Data Preprocessing
    
    >>> import softsensor.meas_handling as ms
    >>> import numpy as np
    >>> import pandas as pd
    >>> t = np.linspace(0, 1.0, 101)
    >>> d = {'sine_inp': np.sin(2 * np.pi * 100 * t) ,
             'cos_inp': np.cos(2 * np.pi * 50 * t),
             'out': np.linspace(0, 1.0, 101)}
    >>> list_of_df = [pd.DataFrame(d), pd.DataFrame(d)]
    >>> test_df = {'sine_inp': np.sin(2 * np.pi * 100 * t),
                   'cos_inp': np.cos(2 * np.pi * 50 * t),
                   'out': np.linspace(0, 1.0, 101)}
    >>> test_df = [pd.DataFrame(test_df)]
    >>> handler = ms.Meas_handling(list_of_df, train_names=['sine1', 'sine2'],
                                   input_sensors=['sine_inp', 'cos_inp'],
                                   output_sensors=['out'], fs=100,
                                   test_dfs=test_df, test_names=['test'])

    Optimize an ARNN

    >>> from softsensor.hyperparameter_optimization import grid_search
    >>> import torch.nn as nn
    >>> grid_params = {'lr': [0.0001, 0.001],
                       'optimizer': ['Adam', 'SGD']}
    >>> model_type = 'ARNN'
    >>> model_params = {'input_channels': 2,
                        'pred_size': 1,
                        'window_size': 50,
                        'rnn_window': 10,
                        'max_epochs': 3,
                        'patience': 3,
                        'hidden_size': [8],
                        }
    >>> criterion = nn.MSELoss()
    >>> df, model = grid_search(handler, criterion,
                                model_type, model_params, grid_params,
                                val_prediction=True)
    run 1/4 finishes with loss 0.06058402732014656 and parameters {'lr': 0.0001, 'optimizer': 'Adam'}, time=0s
    run 2/4 finishes with loss 0.155076265335083 and parameters {'lr': 0.0001, 'optimizer': 'SGD'}, time=0s
    run 3/4 finishes with loss 0.14059486985206604 and parameters {'lr': 0.001, 'optimizer': 'Adam'}, time=0s
    run 4/4 finishes with loss 0.542301595211029 and parameters {'lr': 0.001, 'optimizer': 'SGD'}, time=0s
    
    Optimize an ARNN with Mean Variance Estimation
    
    >>> from softsensor.hyperparameter_optimization import grid_search
    >>> import torch.nn as nn
    >>> from softsensor.losses import DistributionalMSELoss, HeteroscedasticNLL
    >>> model_type = 'MVE'
    >>> model_params = {'input_channels': 2,
                        'pred_size': 1,
                        'window_size': 10,
                        'rnn_window': 10,
                        'max_epochs': 3,
                        'patience': 3,
                        'hidden_size': [8],
                        'var_hidden_size': [8],
                        }
    >>> df, model = grid_search(handler, DistributionalMSELoss(),
                                model_type, model_params, grid_params,
                                val_prediction=True, val_criterion=HeteroscedasticNLL())
    run 1/4 finishes with loss 0.07786498963832855 and parameters {'lr': 0.0001, 'optimizer': 'Adam'}, time=0s
    run 2/4 finishes with loss -0.11223804950714111 and parameters {'lr': 0.0001, 'optimizer': 'SGD'}, time=0s
    run 3/4 finishes with loss 0.06112978607416153 and parameters {'lr': 0.001, 'optimizer': 'Adam'}, time=0s
    run 4/4 finishes with loss -0.05090484023094177 and parameters {'lr': 0.001, 'optimizer': 'SGD'}, time=0s

    """

    grid = list(ParameterGrid(grid_params))

    param_dict = []
    result_list = []
    loss_list = []
    model_list = []
    
    for i, gp in enumerate(grid):
        start = time.time()
        params = {**parameters, **gp}
        print(f'Run {i+1}/{len(grid)} started with parameters {gp}...')
        (val_loss, model, params, results) = _eval_grid_params(params, data_handle, model_type, device, key, criterion,
                                                               pretrained_model, reconfigure_criterion,
                                                               print_results, calibration, val_prediction, val_criterion, n_samples)
        
        
        
        # Add parameters to list
        loss_list.append(val_loss)
        model_list.append(model)
        param_dict.append({**params, **results})
        result_list.append(results)

        end = time.time()
        print(f'Run {i+1}/{len(grid)} finished with loss {val_loss}' +
              f' and parameters {gp}, time={int(end - start)}s \n')
        sys.stdout.flush()

    loss_list = np.asarray(loss_list)

    # return best model and results
    result_df = pd.DataFrame(param_dict)
    result_df['loss'] = loss_list
    best_model = model_list[np.nanargmin(loss_list)]

    return result_df, best_model


def _eval_grid_params(params, data_handle, model_type, device, key, criterion,
                      pretrained_model=None, reconfigure_criterion=False,
                      print_results=False, calibration=None, val_prediction=False, val_criterion=None,
                      n_samples=[5000, 1000]):

    # define data parameters
    list_dp = ['window_size', 'batch_size', 'rnn_window', 'forecast']
    data_params = list(set(params).intersection(list_dp))
    data_params = {key: params[key] for key in data_params}

    # define model and data
    model, train_loader, val_loader = _def_model(data_handle, model_type,
                                                params, device, key,
                                                data_params, pretrained_model, n_samples)
     # reconfigure criterion
    if reconfigure_criterion:
        signature = inspect.signature(type(criterion).__init__).parameters
        criterion_params = _get_model_params(signature, params)
        criterion.__init__(**criterion_params)

    # init optimizer
    if model_type in ["MVE_Student_Forced", "QR_Student_Forced"]:
        opt = _get_optimizer(model.DNN, params)
    else:
        opt = _get_optimizer(model, params)

    # train model
    model, results = _train_Network(params, model, train_loader, val_loader,
                                    criterion, opt, device, print_results)

# calibrate variance prediction
    if calibration is not None:
        model = _calibrate_model(model, data_handle, data_handle.test_names,
                                calibration, device, optimizer_bounds=(.01, 5))
    # additional prediction to eval
    if val_prediction:
        model.eval()

        if val_criterion:
            val_loss = _calc_pred_loss(data_handle, model, model_type,
                                data_params, device, val_criterion)
        else:
            val_loss = _calc_pred_loss(data_handle, model, model_type,
                                data_params, device, criterion)

        model.train()
    else:
        val_loss = results['loss']
    
    return (val_loss, model, params, results)


def hyperopt_search(data_handle, criterion, model_type,
                    parameters, grid_params, max_iterations=3, pretrained_model=None, reconfigure_criterion=False,
                    val_criterion=None, val_prediction=False, device='cpu', key='training',
                    print_results=False, calibration=None, n_samples=[5000, 1000]):
    """
    Hyperparameter optimization using bayesian optimization. Returns
    parameters and best model for evaluation.
    Algorithm:
    https://www.researchgate.net/publication/216816964_Algorithms_for_Hyper-Parameter_Optimization
    
    Parameters
    ----------
    data_handle : Meas Handling class
        used for getting the training and evaluation data
    optimizer : str
        Algorithm to use for training of the models possibilities are
        ['Adam', 'SGD'].
    criterion : nn.Loss function
        Loss function e.g. nn.MSELoss()
    model_type : str
        Describing the Model Type: currently implemented.
        ['ARNN', 'MCDO', 'MVE', 'Sep_MVE', 'MVE_MCDO', 'MVE_Student_Forced',
        'QR', 'QR_Student_Forced', 'BNN', 'MVE_BNN', 'RNN'].
    parameters : dict
        dictionary of static parameters in the grid search.
    grid_params : dict
        dict of grid parameters with hyperopt distribution for each parameter.
    max_iterations : int, optional
        number of iterations. The default is 3
    pretrained_model: str, optional
        path to pretrained model to load as base model. The default is None
    reconfigure_criterion: bool, optional
        if True, the criterion is reconfigured with params from the grid. The default is False
    val_criterion: nn.Loss function, optional
        val_criterion to be used for validation instead of criterion. The default is None
    val_prediction : bool, optional
        if True, prediction on testing tracks in data_handle is used for
        hyperparameter evaluation
    device : str, optional
        device to run training on. The default is 'cpu'.
    key : str, optional
        'training' or 'short'. Training uses whole dataloader, short just
        subset for training. default is training
    print_results : bool, optional
        True prints results for every epoch. default is False
    
    Returns
    -------
    result_df : pd.DataFrame
        parameters and corresponding results for each grid search step.
    best_model : torch Model
        best performing model.
    
    
    Examples
    --------
    
    Data Preprocessing
    
    >>> import softsensor.meas_handling as ms
    >>> import numpy as np
    >>> import pandas as pd
    >>> t = np.linspace(0, 1.0, 101)
    >>> d = {'sine_inp': np.sin(2 * np.pi * 100 * t) ,
             'cos_inp': np.cos(2 * np.pi * 50 * t),
             'out': np.linspace(0, 1.0, 101)}
    >>> list_of_df = [pd.DataFrame(d), pd.DataFrame(d)]
    >>> test_df = {'sine_inp': np.sin(2 * np.pi * 100 * t),
                   'cos_inp': np.cos(2 * np.pi * 50 * t),
                   'out': np.linspace(0, 1.0, 101)}
    >>> test_df = [pd.DataFrame(test_df)]
    >>> handler = ms.Meas_handling(list_of_df, train_names=['sine1', 'sine2'],
                                   input_sensors=['sine_inp', 'cos_inp'],
                                   output_sensors=['out'], fs=100,
                                   test_dfs=test_df, test_names=['test'])
    

    Optimize an ARNN
    
    >>> from softsensor.hyperparameter_optimization import hyperopt_search
    >>> import torch.nn as nn
    >>> from hyperopt import hp
    >>> grid_params = {'lr': hp.uniform('lr', 1e-5, 1e-3),
                       'activation': hp.choice('actiavtion', ['relu', 'sine'])}
    >>> model_type = 'ARNN'
    >>> model_params = {'input_channels': 2,
                        'pred_size': 1,
                        'window_size': 50,
                        'rnn_window': 10,
                        'max_epochs': 3,
                        'patience': 3,
                        'hidden_size': [8],
                        'optimizer': 'SGD'
                        }
    >>> criterion = nn.MSELoss()
    >>> df, model = hyperopt_search(handler, criterion, model_type, model_params,
                                    grid_params, max_iterations=3,
                                    val_prediction=True)
    100%|██████████| 3/3 [00:00<00:00, 10.76trial/s, best loss: 0.25191113352775574]
    

    Optimize an ARNN with Mean Variance Estimation
    
    >>> from softsensor.hyperparameter_optimization import grid_search
    >>> import torch.nn as nn
    >>> from softsensor.losses import DistributionalMSELoss
    >>> model_type = 'MVE_ARNN'
    >>> grid_params = {'lr': hp.uniform('lr', 1e-5, 1e-3),
                       'activation': hp.choice('actiavtion', ['relu', 'sine'])}
    >>> model_params = {'input_channels': 2,
                        'pred_size': 1,
                        'window_size': 10,
                        'rnn_window': 10,
                        'max_epochs': 3,
                        'patience': 3,
                        'hidden_size': [8],
                        'var_hidden_size': [8],
                        'optimizer': 'SGD'
                        }
    >>> criterion = DistributionalMSELoss()
    >>> df, model = hyperopt_search(handler, criterion, model_type, model_params,
                                    grid_params, max_iterations=3,
                                    val_prediction=True)
    100%|██████████| 3/3 [00:00<00:00,  7.42trial/s, best loss: -0.4127456843852997]

    """

    train_class = train_hyperopt(data_handle, criterion, model_type,
                                 parameters, pretrained_model, reconfigure_criterion, val_criterion,
                                 val_prediction, device, key, print_results, calibration, n_samples)

    trials = Trials()
    _ = fmin(train_class.comp,
             space=grid_params,
             algo=tpe.suggest,
             max_evals=max_iterations,
             trials=trials)

    results = pd.DataFrame([t['result'] for t in trials])
    best_model = results['Model'][np.nanargmin(results['loss'])]
    return results.drop(['Model'], axis=1), best_model


def _calibrate_model(model, data_handle, val_tracks, criterion, device, optimizer_bounds=(.01, 5)):
    temps = optimize_temperature(model, data_handle, val_tracks, criterion, device, optimizer_bounds)
    calibrated_model = TemperatureScaling(model, temps)
    return calibrated_model

def _def_dtype(model_params):
    """
    changes the type of Parameters if needed

    Parameters
    ----------
    model_params : dict
        parameters

    Returns
    -------
    model_params : dict
        changed parameters.

    """

    int_params = ['input_channels', 'pred_size', 'window_size', 'filters',
                  'kernel_size', 'depth', 'pooling_size', 'rnn_window',
                  'batch_size', 'num_layers', 'blocks', 'oscillations',
                  'max_dilation', 'forecast', 'hidden_window', 'n_heads']

    temp_p = list(set(model_params).intersection(int_params))
    temp_p = {key: model_params[key] for key in temp_p}
    temp = {i: int(model_params[i]) for i in temp_p}
    model_params.update(**temp)

    list_params = ['hidden_size']
    temp_p = list(set(model_params).intersection(list_params))
    temp_p = {key: model_params[key] for key in temp_p}
    temp = {i: [int(param) for param in model_params[i]]
            if model_params[i] is not None else None for i in temp_p}
    model_params.update(**temp)
    return model_params


def _def_model(data_handle, model_type, params, device, key, data_params, pretrained_model=None, n_samples=[5000, 1000]):
    """
    Defines Model, training loader and validation loader for training

    Parameters
    ----------
    data_handle : Meas Handling class
        used for getting the trainign and evaluation data
    model_type : str
        Describing the Model Type: currently implemented
        ['ARNN', 'MCDO', 'MVE', 'Sep_MVE', 'MVE_MCDO', 'MVE_Student_Forced',
        'QR', 'QR_Student_Forced', 'BNN', 'MVE_BNN', 'RNN'].
    params : dict
        parameters for model initialization.
    device : str, optional
        device to run training on. The default is 'cpu'.
    key : str
        'training' or 'short'.
    data_params: dict
        dict that contains necessary data parameters for classes
        data_handle.give_dataloader or data_handle.give_list
    pretrained_model: str, optional
        path to pretrained model to load as base model. The default is None
        
    Returns
    -------
    model : torch neural Network
    train_loader : DataLoader
        or list of Dataloader for RNN Models
    val_loader : DataLoader
        or list of Dataloader for RNN Models

    """
    # init model and data
    if model_type in ['ARNN', 'MCDO']:
        signature = inspect.signature(ARNN.__init__).parameters
        model_params = _get_model_params(signature, params)
        model = ARNN(**model_params)
        train_loader, val_loader = data_handle.give_torch_loader(
            keyword=key, n_samples=n_samples, **data_params)

    elif model_type in ['SepMVE', 'MVE_MCDO']:
        signature = inspect.signature(ARNN.__init__).parameters
        model_params = _get_model_params(signature, params)
        mean_model = ARNN(**model_params)
        signature = inspect.signature(SeparateMVEARNN.__init__).parameters
        model_params = _get_model_params(signature, params)
        model = SeparateMVEARNN(**model_params, mean_model=mean_model)
        train_loader, val_loader = data_handle.give_torch_loader(
            keyword=key, n_samples=n_samples, **data_params)

    elif model_type == 'MVE_Student_Forced':
        signature = inspect.signature(ARNN.__init__).parameters
        model_params = _get_model_params(signature, params)
        mean_model = ARNN(**model_params)

        signature = inspect.signature(SeparateMVEARNN.__init__).parameters
        model_params = _get_model_params(signature, params)
        model = SeparateMVEARNN(**model_params, mean_model=mean_model)
        train_loader, val_loader = data_handle.give_torch_loader(
            keyword=key, n_samples=n_samples, **data_params, pre_comp=True)
        #model.Type = "AR_Student_Forced"

    elif model_type == 'QR':
        signature = inspect.signature(ARNN.__init__).parameters
        model_params = _get_model_params(signature, params)
        mean_model = ARNN(**model_params)
        signature = inspect.signature(QuantileARNN.__init__).parameters
        model_params = _get_model_params(signature, params)
        model = QuantileARNN(**model_params, mean_model=mean_model)
        train_loader, val_loader = data_handle.give_torch_loader(
            keyword=key, n_samples=n_samples, **data_params)

    elif model_type == 'QR_Student_Forced':
        signature = inspect.signature(ARNN.__init__).parameters
        model_params = _get_model_params(signature, params)
        mean_model = ARNN(**model_params)
        signature = inspect.signature(QuantileARNN.__init__).parameters
        model_params = _get_model_params(signature, params)
        model = QuantileARNN(**model_params, mean_model=mean_model)
        train_loader, val_loader = data_handle.give_torch_loader(
            keyword=key, n_samples=n_samples, **data_params, pre_comp=True)
        #model.Type = "AR_Student_Forced"
        
    elif model_type == 'MVE':
        signature = inspect.signature(DensityEstimationARNN.__init__).parameters
        model_params = _get_model_params(signature, params)
        model = DensityEstimationARNN(**model_params)
        train_loader, val_loader = data_handle.give_torch_loader(
            keyword=key, n_samples=n_samples, **data_params)

    elif model_type == 'RNN':
        signature = inspect.signature(RNN_DNN.__init__).parameters
        model_params = _get_model_params(signature, params)
        model = RNN_DNN(**model_params)
        train_loader = data_handle.give_list(keyword=key, **data_params,
                                             full_ds=False, Add_zeros=True)
        val_loader = data_handle.give_list(keyword='testing', **data_params,
                                           full_ds=False, Add_zeros=True)
        
    elif model_type == 'parr_RNN':
        signature = inspect.signature(parr_RNN_DNN.__init__).parameters
        model_params = _get_model_params(signature, params)
        model = parr_RNN_DNN(**model_params)
        train_loader = data_handle.give_list(keyword=key, **data_params,
                                             full_ds=False, Add_zeros=True,
                                             window_size=1)
        val_loader = data_handle.give_list(keyword='testing', **data_params,
                                           full_ds=False, Add_zeros=True,
                                           window_size=1)
    elif model_type == 'TFT':
        signature = inspect.signature(TFT.__init__).parameters
        model_params = _get_model_params(signature, params)
        model = TFT(**model_params)
        train_loader = data_handle.give_list(keyword=key, **data_params,
                                             full_ds=False, Add_zeros=True,
                                             window_size=1)
        val_loader = data_handle.give_list(keyword='testing', **data_params,
                                           full_ds=False, Add_zeros=True,
                                           window_size=1)
    elif model_type == 'AR_RNN':
        signature = inspect.signature(AR_RNN.__init__).parameters
        model_params = _get_model_params(signature, params)
        model = AR_RNN(**model_params)        
        train_loader = data_handle.give_list(
            keyword=key, **data_params, full_ds=False)
        val_loader = data_handle.give_list(
            keyword='testing', **data_params, full_ds=False)

    else:
        raise KeyError(f'{model_type} is not a valid model name')

    if pretrained_model:
        model.mean_model.load_state_dict(torch.load(pretrained_model))
        #model.load_state_dict(torch.load(pretrained_model)) # for DensityEstimationARNN models

    # if model.Type == "AR_Student_Forced":
    #     train_loader, val_loader = data_handle.precompute_predictions(model.mean_model, data_params["batch_size"])
    
    model.to(device)
    model.float()   
    return model, train_loader, val_loader


def _get_model_params(signature, params):
    """
    Returns model parameters from parameters and model signature

    Parameters
    ----------
    signature : dict
        model signature parameters.
    params : dict
        dict of parameters.

    Returns
    -------
    dict
        returns parameters for model definition.

    """
    model_params = list(signature.keys())[1:]
    model_params = list(set(params).intersection(model_params))
    model_params = {key: params[key] for key in model_params}
    return _def_dtype(model_params)


def _train_Network(params, model, train_loader, val_loader,
                  criterion, opt, device, print_results):
    """
    Training procedure for the different Models

    Parameters
    ----------
    params : dict
        dictionary of the training parameters.
    model : torch neural Network
    train_loader : DataLoader
        or list of Dataloader for RNN Models
    val_loader : DataLoader
        or list of Dataloader for RNN Models
    criterion : nn.Loss function
        Loss function e.g. nn.MSELoss()
    opt : torch.optim
        optimizer with trainable parameters of the model.
    device : str, optional
        device for computation. The default is 'cpu'.
    print_results : bool, optional
        Prints results if True in every epoch. The default is True.

    Returns
    -------
    model
        Trained Neural Network.
    results
        results as dict with 'loss' as key

    """
    Model_type = model.Type

    if Model_type == 'Feed_Forward':
        list_tp = ['max_epochs', 'patience', 'rel_perm']

    elif Model_type == 'AR': # or Model_type == 'AR_Student_Forced':
        list_tp = ['max_epochs', 'patience', 'stabelizer', 'rel_perm',
                   'local_wd']
    elif Model_type == 'RNN':
        list_tp = ['max_epochs', 'patience', 'rel_perm', 'local_wd']
    elif Model_type == 'AR_RNN':
        list_tp = ['max_epochs', 'patience', 'rel_perm', 'local_wd']
    else:
        raise KeyError('No valid model name given')

    train_params = list(set(params).intersection(list_tp))
    train_params = {key: params[key] for key in train_params}
    
    if 'stab_method' in list(params.keys()):
        list_stab_params = ['stab_method', 's0', 's1', 'm']
        stab_params = list(set(params).intersection(list_stab_params))
        stab_params = {key: params[key] for key in stab_params}
        stabelizer = get_scheduler(model=model, track=True, **stab_params)
    else:
        if 'stabelizer' in list(params.keys()):
            stabelizer = params['stabelizer']
        else:
            stabelizer = None
    results = train_model(
        model, train_loader, optimizer=opt, device=device,
        criterion=criterion, val_loader=val_loader, give_results=True,
        print_results=print_results, stabelizer=stabelizer, **train_params)

    return model, _comp_results(results)


def _calc_pred_loss(data_handle, model, model_Type, data_params, device, criterion):
    """
    Calculates loss from prediction

    Parameters
    ----------
    data_handle : Meas Handling class
        used for getting the training and evaluation data
    model : torch neural Network
    model_type : str
        Describing the Model Type: currently implemented:
        ['CNN_DNN', 'ARNN', 'Point', 'Point_MCDO', 
        'MVE', 'MVE_ARNN', 'MVE_MCDO', 'MVE_Student_Forced', 
        'QR', 'QR_Student_Forced', 'Point_BNN', 'MVE_BNN', 
        'MVE', 'Evidential_ARNN', 'CNN_ARNN', 'AR_CNN', 'AR_Freq_CNN', 'AR_RNN'].
    data_params : dict
        model pull parameters from data handle.
    device : str, optional
        device for computation. The default is 'cpu'.
    criterion : nn.Loss function
        Loss function e.g. nn.MSELoss()

    Returns
    -------
    loss : float
        validation loss for hyperparameter optimization.

    """

    # if 'AR' in model_Type or ('Point' in model_Type):
    if (model.Type == 'AR') or (model.Type == 'AR_RNN'): # or (model.Type == 'AR_Student_Forced'):
        data_params['rnn_window'] = model.rnn_window
        data_params['window_size'] = model.window_size
        data_params['batch_size'] = 1
    
    # if ('ARNN' in model_Type) or ('AR' in model_Type) or ('Point' in model_Type) or ('MVE' in model_Type) or ('QR' in model_Type):
    # if (model.Type == 'AR') or (model.Type == 'AR_RNN'):

    
    if model_Type in ['parr_RNN', 'TFT']:
        data_params['window_size'] = 1
    if model_Type == 'MCDO':
        eval_model = AsyncMCDropout(model, 5)
    elif model_Type == 'MVE_MCDO':
        eval_model = AsyncMCDropoutMVE(model, 5)
    else:
        eval_model = model

    pred = comp_batch([eval_model], data_handle, data_handle.test_names, ['pred'], device)

    if eval_model.Pred_Type == 'Point' and not eval_model.Ensemble:
        pred_sens = [f'{s}_pred' for s in data_handle.output_sensors] 
        loss = [criterion(torch.tensor(df[pred_sens].values.transpose(1,0)), torch.tensor(df[data_handle.output_sensors].values.transpose(1, 0))) for df in pred]
        
    elif eval_model.Pred_Type == 'Mean_Var' or eval_model.Ensemble:
        pred_sens = [f'{s}_pred' for s in data_handle.output_sensors]
        var_sens = [f'{s}_pred_var' for s in data_handle.output_sensors]
        loss = [criterion([torch.tensor(df[pred_sens].values.transpose(1, 0)), torch.tensor(df[var_sens].values.transpose(1, 0))], torch.tensor(df[data_handle.output_sensors].values.transpose(1, 0))) for df in pred]
    elif eval_model.Pred_Type == 'Quantile':
        loss = []
        for out_sens in data_handle.output_sensors:
            labels = [f'{out_sens}_median']
            for i in range(math.floor(eval_model.n_quantiles / 2)):
                labels = labels + [f'{out_sens}_pred_lb{i}', f'{out_sens}_pred_ub{i}']
            loss_temp = [criterion(torch.tensor(df[labels].values), torch.tensor(df[out_sens].values)) for df in pred]
            loss.append(torch.mean(torch.stack(loss_temp)))
    else:
        print('unknown prediction type')
    
    loss = torch.mean(torch.stack(loss))
    return loss.to('cpu')


class train_hyperopt():
    """
    Class for Hyperparameter optimization

    Parameters
    ----------
    data_handle : Meas Handling class
        used for getting the training and evaluation data
    criterion : nn.Loss function
        Loss function e.g. nn.MSELoss()
    model_type : str
        Describing the Model Type: currently implemented:
        ['CNN_DNN', 'ARNN', 'Point', 'Point_MCDO', 
        'MVE', 'MVE_ARNN', 'MVE_MCDO', 'MVE_Student_Forced', 
        'QR', 'QR_Student_Forced', 'Point_BNN', 'MVE_BNN', 
        'MVE', 'Evidential_ARNN', 'CNN_ARNN', 'AR_CNN', 'AR_Freq_CNN', 'AR_RNN'].
    parameters : dict
        dictionary of static parameters in the grid search.
    pretrained_model: str, optional
        path to pretrained model to load as base model. The default is None
    reconfigure_criterion: bool, optional
        if True, the criterion is reconfigured with params from the grid. The default is False
    val_criterion: nn.Loss function, optional
        val_criterion to be used for validation instead of criterion. The default is None
    val_prediction : torch dataloader, optional
        list of dataloaders for prediction as an alternative to simple
        validation loss from the training. The default is None.
    device : str, optional
        device to run training on. The default is 'cpu'.
    key : str, optional
        'training' or 'short'. Training uses whole dataloader, short just
        subset for training
    print_results : bool, optional
        True prints results for every epoch

    Returns
    -------
    None.

    """

    def __init__(self, data_handle, criterion, model_type,
                 parameters, pretrained_model, reconfigure_criterion,
                 val_criterion, val_prediction, device, key, print_results,
                 calibration, n_samples):

        self.data_handle = data_handle
        self.model_type = model_type
        self.criterion = criterion
        self.parameters = parameters
        self.pretrained_model = pretrained_model
        self.reconfigure_criterion = reconfigure_criterion
        self.val_criterion = val_criterion
        self.val_prediction = val_prediction
        self.device = device
        self.key = key
        self.print_results = print_results
        self.calibration = calibration
        self.n_samples = n_samples

    def comp(self, gp):
        """
        compute Hyperparameter optimization with specific grid params

        Parameters
        ----------
        gp : dict
            dict of grid parameters with grid options as list.

        Returns
        -------
        dict
            results including loss and parameters.

        """
        params = {**self.parameters, **gp}
        list_dp = ['window_size', 'batch_size', 'rnn_window', 'forecast']
        data_params = list(set(params).intersection(list_dp))
        data_params = {key: params[key] for key in data_params}
        data_params = _def_dtype(data_params)

        # define model and data
        model, train_loader, val_loader = _def_model(
            self.data_handle, self.model_type, params,
            self.device, self.key, data_params, self.pretrained_model, self.n_samples)

        # reconfigure criterion
        if self.reconfigure_criterion:
            signature = inspect.signature(type(self.criterion).__init__).parameters
            criterion_params = _get_model_params(signature, params)
            self.criterion.__init__(**criterion_params)

        # init optimizer
        if self.model_type in ["MVE_Student_Forced", "QR_Student_Forced"]:
            opt = _get_optimizer(model.DNN, params)
        else:
            opt = _get_optimizer(model, params)

        # train model
        model, results = _train_Network(params, model,
                                       train_loader, val_loader,
                                       self.criterion, opt, self.device,
                                       self.print_results)
        
        # calibrate variance prediction
        if self.calibration is not None:
            model = _calibrate_model(model, self.data_handle, self.data_handle.test_names,
                                     self.calibration, self.device, optimizer_bounds=(.01, 5))


        # additional prediction to eval
        if self.val_prediction:
            model.eval()

            if self.val_criterion:
                val_loss = _calc_pred_loss(self.data_handle, model, self.model_type,
                                      data_params, self.device, self.val_criterion)
            else:
                val_loss = _calc_pred_loss(self.data_handle, model, self.model_type,
                                      data_params, self.device, self.criterion)

            model.train()
            results['loss'] = val_loss

        params = _def_dtype(params)
        return {'status': STATUS_OK,
                'Model': model,
                **params,
                **results}


def _get_optimizer(model, params):
    """
    Returns optimizer with parameters

    Parameters
    ----------
    model : torch neural network with parameters
    params : dict
        list of params.

    Returns
    -------
    torch.optim
        Optimizer for subsequent training.

    """
    if params['optimizer'] == 'Adam':
        adam_params = ['lr', 'weight_decay']
        list_of_params = list(set(params).intersection(adam_params))
        init_params = {key: params[key] for key in list_of_params}
        return optim.Adam(model.parameters(), **init_params)

    if params['optimizer'] == 'SGD':
        adam_params = ['lr', 'momentum', 'dampening', 'weight_decay']
        list_of_params = list(set(params).intersection(adam_params))
        init_params = {key: params[key] for key in list_of_params}
        return optim.SGD(model.parameters(), **init_params)


def _comp_results(struct):
    """
    Returns results according to dict

    Parameters
    ----------
    struct : dict
        dict of results from train_model.

    Returns
    -------
    results : dict
        results including loss and stability score if defined.

    """
    if len(struct['val_loss']) != 0:
        results = {'loss': min(struct['val_loss'])}
    else:
        results = {'loss': min(struct['train_loss'])}

    if len(struct['stability_score']) != 0:
        sc = struct['stability_score'][np.argmin(struct['val_loss'])]
        results = {**results, **{'stabelize score': sc}}

    return {**results, **struct}
