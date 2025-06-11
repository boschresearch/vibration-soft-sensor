# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:16:16 2022

@author: WET2RNG
"""

from softsensor.meas_handling import Meas_handling
from softsensor.losses import DistributionalMSELoss, GaussianNLLLoss, BetaNLL, PinballLoss, PSDLoss
from softsensor.hyperparameter_optimization import grid_search, hyperopt_search, random_search
from softsensor.hyperparameter_optimization import _comp_results, _get_optimizer
from softsensor.metrics import quantile_ece, nll, ece

import scipy.stats as stats
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from hyperopt import hp


def Sine_df():
    t = np.linspace(0, 1.0, 101)
    xlow = np.sin(2 * np.pi * 100 * t)       # 100Hz Signal
    xhigh = np.sin(2 * np.pi * 3000 * t)     # 3000Hz Signal
    d = {'sine_inp': xlow + xhigh,
         'cos_inp': np.cos(2 * np.pi * 50 * t),
         'out': np.linspace(0, 1.0, 101),
         'pre_out': np.linspace(0, 1.0, 101)}
    list_of_df = [pd.DataFrame(d), pd.DataFrame(d)]

    test_df = {'sine_inp': 10*xlow + xhigh,
               'cos_inp': np.cos(2 * np.pi * 50 * t),
               'out': np.linspace(0, 1.0, 101),
               'pre_out': np.linspace(0, 1.0, 101)}

    test_df = [pd.DataFrame(test_df)]

    handler = Meas_handling(list_of_df, train_names=['sine1', 'sine2'],
                            input_sensors=['sine_inp', 'cos_inp'],
                            output_sensors=['out'], fs=100,
                            test_dfs=test_df, test_names=['test'], 
                            pre_comp_cols=['pre_out'])
    return handler

def test_random_search():
    handler = Sine_df()
    
    grid_params = {'lr': stats.loguniform(1e-6, 1e-4),
                   'rel_perm': stats.loguniform(1e-2, 1e-1),
                   'optimizer': ['Adam', 'SGD']}
    
    model_type = 'ARNN'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 50,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    }
    criterion = nn.MSELoss()
    df, model = random_search(handler, criterion,
                            model_type, model_params, grid_params,
                            val_prediction=True)
    
    grid_params = {'lr': stats.loguniform(1e-6, 1e-4),
                   'optimizer': ['Adam', 'SGD'], 
                   'hidden_size': [[16, 8], [32, 4]]}
    
    model_type = 'MCDO'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 50,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'dropout': 0.1,
                    'concrete_dropout': True,
                    'n_samples': 5
                    }
    
    df, model = random_search(handler, criterion,
                            model_type, model_params, grid_params,
                            val_prediction=True, val_criterion=GaussianNLLLoss())
    
    
    grid_params = {'lr': stats.loguniform(1e-5, 1e-3),
                   'optimizer': ['Adam', 'SGD']}
    
    
    model_type = 'SepMVE'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'var_hidden_size': [8],
                    }
    
    df, model = random_search(handler, GaussianNLLLoss(),
                            model_type, model_params, grid_params,
                            val_prediction=True, val_criterion=GaussianNLLLoss())
    
    grid_params['beta'] = [0.1, 0.2]
    df, model = random_search(handler, BetaNLL(),
                            model_type, model_params, grid_params, pretrained_model=None, reconfigure_criterion=True,
                            val_prediction=True, val_criterion=GaussianNLLLoss())
    
    grid_params = {'lr': stats.loguniform(1e-6, 1e-4),
                   'optimizer': ['Adam', 'SGD']}
    
    model_type = 'MVE_MCDO'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'dropout': 0.1,
                    'hidden_size': [8],
                    'var_hidden_size': [8],
                    'n_samples': 2
                    }
    
    df, model = random_search(handler, DistributionalMSELoss(),
                            model_type, model_params, grid_params,
                            val_prediction=True, val_criterion=GaussianNLLLoss())
    
    model_type = 'MVE'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    }
    
    df, model = random_search(handler, GaussianNLLLoss(),
                            model_type, model_params, grid_params,
                            val_prediction=True, val_criterion=GaussianNLLLoss())
    
    quantiles = [0.5, 0.05, 0.95]
    model_type = 'QR'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'Adam',
                    'n_quantiles': 3,
                    }
    df, model = random_search(handler, PinballLoss(quantiles),
                            model_type, model_params, grid_params)
    
    quantiles = [0.5]
    for q in np.arange(0.025, 0.5, 0.025):
        quantiles += [q, 1-q]
    
    model_type = 'QR_Student_Forced'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'Adam',
                    'n_quantiles': 39,
                    }
    model_params["batch_size"] = 200
    print(len(quantiles))
    df, model = random_search(handler, PinballLoss(quantiles),
                            model_type, model_params, grid_params,
                            val_prediction=True, val_criterion=PinballLoss(quantiles))
    
    model_type = 'RNN'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'blocks': 5,
                    'num_layers': 1,
                    'patience': 3,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'SGD'
                    }
    
    df, model = random_search(handler, criterion,
                            model_type, model_params, grid_params,
                            val_prediction=False)
    
    
    model_type = 'TFT'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'hidden_window': 10,
                    'blocks': 5,
                    'num_layers': 1,
                    'patience': 3,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'SGD',
                    'Pred_Type': 'Mean_Var'
                    }
    
    df, model = random_search(handler, GaussianNLLLoss(),
                              model_type, model_params, grid_params,
                              val_prediction=True, calibration=nll)

def test_grid_search():
    handler = Sine_df()

    criterion = nn.MSELoss()

    grid_params = {'lr': [0.00001, 0.0001],
                   'stab_method': ['log_lin', 0.001],
                   'optimizer': ['Adam', 'SGD']}

    model_type = 'ARNN'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 50,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    }

    df, model = grid_search(handler, criterion,
                            model_type, model_params, grid_params,
                            val_prediction=True)
    print(df['loss'])

    grid_params = {'lr': [0.000001, 0.00001],
                   'optimizer': ['Adam', 'SGD']}

    model_type = 'MCDO'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 50,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'dropout': 0.1,
                    'stab_method': 'const',
                    's1': 0.00001, 
                    'concrete_dropout': True,
                    'n_samples': 5
                    }
    

    df, model = grid_search(handler, nn.MSELoss(),
                            model_type, model_params, grid_params,
                            val_prediction=True, val_criterion=GaussianNLLLoss())

    model_type = 'ARNN'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'var_hidden_size': [8],
                    }

    df, model = grid_search(handler, nn.MSELoss(),
                            model_type, model_params, grid_params,
                            val_prediction=True, val_criterion=None)
    torch.save(model.state_dict(), 'tmp_model.pt')
    
    model_type = 'SepMVE'
    grid_params['beta'] = [0.1, 0.2]
    df, model = grid_search(handler, BetaNLL(),
                            model_type, model_params, grid_params, pretrained_model=None, reconfigure_criterion=True,
                            val_prediction=True, val_criterion=GaussianNLLLoss())
    model_type = 'MVE_Student_Forced'
    model_params["batch_size"] = 512
    df, model = grid_search(handler, GaussianNLLLoss(),
                            model_type, model_params, grid_params, pretrained_model="tmp_model.pt",
                            val_prediction=True, val_criterion=GaussianNLLLoss())
    os.remove("tmp_model.pt")

    model_type = 'MVE_MCDO'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'dropout': 0.1,
                    'hidden_size': [8],
                    'var_hidden_size': [8],
                    'n_samples': 2
                    }

    df, model = grid_search(handler, DistributionalMSELoss(),
                            model_type, model_params, grid_params,
                            val_prediction=True, val_criterion=GaussianNLLLoss())

    model_type = 'MVE'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    }

    df, model = grid_search(handler, GaussianNLLLoss(),
                            model_type, model_params, grid_params,
                            val_prediction=True, val_criterion=GaussianNLLLoss())

    quantiles = [0.5, 0.05, 0.95]
    model_type = 'QR'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'Adam',
                    'n_quantiles': 3,
                    }
    df, model = grid_search(handler, PinballLoss(quantiles),
                            model_type, model_params, grid_params)

    quantiles = [0.5]
    for q in np.arange(0.025, 0.5, 0.025):
        quantiles += [q, 1-q]

    model_type = 'QR_Student_Forced'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'Adam',
                    }
    model_params["batch_size"] = 200
    df, model = grid_search(handler, PinballLoss(quantiles),
                            model_type, model_params, grid_params,
                            val_prediction=True, val_criterion=PinballLoss(quantiles))

    model_type = 'RNN'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'blocks': 5,
                    'num_layers': 1,
                    'patience': 3,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'SGD'
                    }

    df, model = grid_search(handler, criterion,
                            model_type, model_params, grid_params,
                            val_prediction=False)
    
    model_type = 'RNN'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'blocks': 5,
                    'num_layers': 1,
                    'patience': 3,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'SGD',
                    'Pred_Type': 'Mean_Var'
                    }

    df, model = grid_search(handler, GaussianNLLLoss(),
                            model_type, model_params, grid_params,
                            val_prediction=False)
    

    model_type = 'parr_RNN'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'hidden_window': 10,
                    'blocks': 5,
                    'num_layers': 1,
                    'patience': 3,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'SGD'
                    }

    df, model = grid_search(handler, criterion,
                            model_type, model_params, grid_params,
                            val_prediction=False)
    
    model_type = 'parr_RNN'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'hidden_window': 10,
                    'blocks': 5,
                    'num_layers': 1,
                    'patience': 3,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'SGD',
                    'Pred_Type': 'Mean_Var'
                    }

    df, model = grid_search(handler, GaussianNLLLoss(),
                            model_type, model_params, grid_params,
                            val_prediction=True)

def test_hyperopt_search():
    handler = Sine_df()


    grid_params = {'lr': hp.uniform('lr', 1e-5, 1e-3),
                   'activation': hp.choice('activation', ['relu', 'sine']),
                   'forecast': hp.quniform('forecast', 1, 5, 2)}

    criterion = nn.MSELoss()

    model_type = 'AR_RNN'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'blocks': 5,
                    'num_layers': 1,
                    'patience': 3,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'Adam'
                    }

    grid_params = {'lr': hp.uniform('lr', 1e-5, 1e-3),
                   'activation': hp.choice('actiavtion', ['relu', 'sine'])}

    df, model = hyperopt_search(handler, criterion, model_type,
                                model_params, grid_params, max_iterations=2,
                                val_prediction=False, device='cpu',
                                key='training')
    
    model_type = 'AR_RNN'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'blocks': 5,
                    'num_layers': 1,
                    'patience': 3,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'Adam',
                    'Pred_Type': 'Mean_Var'
                    }

    grid_params = {'lr': hp.uniform('lr', 1e-5, 1e-3),
                   'activation': hp.choice('actiavtion', ['relu', 'sine'])}

    df, model = hyperopt_search(handler, GaussianNLLLoss(), model_type,
                                model_params, grid_params, max_iterations=2,
                                val_prediction=False, device='cpu',
                                key='training')

    print(df)

    model_type = 'ARNN'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'var_hidden_size': [8],
                    'optimizer': 'Adam'
                    }

    df, model = hyperopt_search(handler, nn.MSELoss(), model_type,
                                model_params, grid_params, max_iterations=2,
                                val_prediction=True, val_criterion=nn.MSELoss(),
                                device='cpu', key='training')
    torch.save(model.state_dict(), 'tmp_model.pt')
    
    model_type = 'SepMVE'
    grid_params['beta'] = hp.loguniform('beta', 0, 1)
    df, model = hyperopt_search(handler, BetaNLL(), model_type,
                                    model_params, grid_params, pretrained_model=None, reconfigure_criterion=True,
                                    max_iterations=2, val_prediction=True, val_criterion=GaussianNLLLoss(),
                                    device='cpu', key='training')
    model_type = 'MVE_Student_Forced'
    model_params["batch_size"] = 512
    df, model = hyperopt_search(handler, GaussianNLLLoss(), model_type,
                                    model_params, grid_params, pretrained_model="tmp_model.pt",
                                    max_iterations=2, val_prediction=True, val_criterion=GaussianNLLLoss(),
                                    device='cpu', key='training')
    os.remove("tmp_model.pt")


    model_type = 'MVE_MCDO'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'dropout': 0.1,
                    'hidden_size': [8],
                    'var_hidden_size': [8],
                    'optimizer': 'Adam',
                    'n_samples': 2
                    }

    df, model = hyperopt_search(handler, DistributionalMSELoss(), model_type,
                                    model_params, grid_params,
                                    max_iterations=2, val_prediction=True, val_criterion=GaussianNLLLoss(),
                                    device='cpu', key='training')


    model_type = 'MVE'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'Adam'
                    }

    df, model = hyperopt_search(handler, GaussianNLLLoss(), model_type,
                                    model_params, grid_params,
                                    max_iterations=2, val_prediction=True, val_criterion=GaussianNLLLoss(),
                                    device='cpu', key='training')
    
    quantiles = [0.5, 0.05, 0.95]
    model_type = 'QR'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'Adam',
                    'n_quantiles': 3,
                    }
    df, model = hyperopt_search(handler, PinballLoss(quantiles),
                                model_type, model_params, grid_params,
                                max_iterations=2, device='cpu', key='training')    

    quantiles = [0.5]
    for q in np.arange(0.025, 0.5, 0.025):
        quantiles += [q, 1-q]

    model_type = 'QR_Student_Forced'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'Adam',
                    }
    model_params["batch_size"] = 200
    df, model = hyperopt_search(handler, PinballLoss(quantiles),
                                model_type, model_params, grid_params,
                                val_prediction=True, val_criterion=PinballLoss(quantiles),
                                max_iterations=2, device='cpu', key='training')  


    try:
        df, model = hyperopt_search(handler, criterion, 'not_valid_type',
                                    model_params, grid_params, max_iterations=2,
                                    val_prediction=False, device='cpu',
                                    key='training')
        assert False
    except KeyError:
        assert True


def test_train_optim():

    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 50,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'forecast': 3,
                    'optimizer': 'Adam'}

    grid_params = {'lr': hp.uniform('lr', 1e-5, 1e-3),
                   'activation': hp.choice('actiavtion', ['relu', 'sine'])}
    model_type = 'ARNN'
    handler = Sine_df()
    criterion = nn.MSELoss()
    df, model = hyperopt_search(handler, criterion, model_type,
                                model_params, grid_params, max_iterations=2,
                                val_prediction=False, device='cpu',
                                key='training')

    weights = [[8, 110], [8], [3, 8], [3]]
    for (name, W), shape in zip(model.named_parameters(), weights):
        assert W.shape == torch.Size(shape)

def test_calibration():
    handler = Sine_df()

    grid_params = {'lr': hp.uniform('lr', 1e-5, 1e-3),
                   'activation': hp.choice('activation', ['relu', 'sine']),
                   'forecast': hp.quniform('forecast', 1, 5, 2)}   

    model_type = 'MVE'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'hidden_size': [8],
                    'optimizer': 'Adam'
                    }


    df, model = hyperopt_search(handler, GaussianNLLLoss(), model_type,
                                    model_params, grid_params,
                                    max_iterations=2, val_prediction=True, val_criterion=GaussianNLLLoss(),
                                    device='cpu', key='training', calibration=nll)

    
    grid_params = {'lr': [0.0001, 0.001],
                   'optimizer': ['Adam', 'SGD']}
    
    model_type = 'MVE_MCDO'
    model_params = {'input_channels': 2,
                    'pred_size': 1,
                    'window_size': 10,
                    'rnn_window': 10,
                    'max_epochs': 3,
                    'patience': 3,
                    'dropout': 0.1,
                    'hidden_size': [8],
                    'var_hidden_size': [8],
                    'n_samples': 2
                    }

    df, model = grid_search(handler, DistributionalMSELoss(),
                            model_type, model_params, grid_params,
                            val_prediction=True, val_criterion=GaussianNLLLoss(),
                            calibration=ece)


#def test_psd():
handler = Sine_df()

criterion = nn.MSELoss()

grid_params = {'lr': [0.00001, 0.0001],
               'stab_method': ['log_lin', 0.001],
               'optimizer': ['Adam', 'SGD']}

model_type = 'ARNN'
model_params = {'input_channels': 2,
                'pred_size': 1,
                'window_size': 50,
                'rnn_window': 10,
                'max_epochs': 3,
                'patience': 3,
                'hidden_size': [8],
                }

df, model = grid_search(handler, criterion,
                        model_type, model_params, grid_params,
                        val_prediction=True, val_criterion=PSDLoss(fs=handler.fs, window=32))
df, model = grid_search(handler, criterion,
                        model_type, model_params, grid_params,
                        val_prediction=True, val_criterion=PSDLoss(fs=handler.fs, window=32, type='log_area'))

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(40, 1)

    def forward(self, inp):
        inp = inp.view(-1, 40)
        return self.lin1(inp)


def test_get_optimizer():
    params = {'lr': 0.001,
              'optimizer': 'Adam'}
    optimizer = _get_optimizer(model(), params)
    assert isinstance(optimizer, optim.Adam)

    params = {'lr': 0.001,
              'optimizer': 'SGD'}
    optimizer = _get_optimizer(model(), params)
    assert isinstance(optimizer, optim.SGD)


def test_comp_results():
    results = {'train_loss': [5, 1, 2],
               'val_loss': [],
               'stability_score': []}
    res = _comp_results(results)
    assert res['loss'] == 1

    results = {'train_loss': [5, 1, 2],
               'val_loss': [2, 2, 6],
               'stability_score': []}
    res = _comp_results(results)
    assert res['loss'] == 2

    results = {'train_loss': [5, 1, 2],
               'val_loss': [3, 2, 6],
               'stability_score': [-1, 2, -1]}
    res = _comp_results(results)
    assert res['loss'] == 2
    assert res['stabelize score'] == 2
