# -*- coding: utf-8 -*-
import sys

import torch
import torch.nn as nn

from softsensor.stab_scheduler import const_stab, none_stab, Stability_Criterion
from softsensor.stab_scheduler import Stability_Loss


def train_model(model, train_loader, max_epochs, optimizer,
                device='cpu', criterion=nn.MSELoss(), val_loader=None,
                patience=None, print_results=False, stabelizer=None,
                local_wd=None, give_results=True, rel_perm=0):
    """
    training Function for Autoregressive Modelling of Time Series

    Parameters
    ----------
    model : nn.Module
        model must have a forward function to predict the output
    train_loader : dataloader or list of dataloader
        dataloader for training if Model_Type is Feed_Forward or AR otherwise
        list of dataloader.
    max_epochs : int
        Maximum number rof training epochs.
    optimizer : torch.optim
        optimizer with trainable parameters of the model.
    device : str, optional
        device for computation. The default is 'cpu'.
    criterion : nn.Loss, optional
        Loss function for training. The default is nn.MSELoss().
    val_loader : dataloader or list of dataloader, optional
        dataloader for training if Model_Type is Feed_Forward or AR otherwise
        list of dataloader. The default is None.
    patience : int, optional
        patience for the val loader (only needed if val_loader is not None).
        The default is None.
    print_results : bool, optional
        True prints result of every epoch. The default is False.
    stabelizer : float, optional
        stability score for Model_Type 'AR'. The default is None.
    local_wd : float, optional
        Applies a local weight decay on all weights that interact with the
        recurrent input for Model_Type 'AR'. The default is None.
    give_results : bool, optional
        Prints results if True in every epoch. The default is True.
    rel_perm : float, optional
        relative permutation applied to the input to prevent overfitting.
        The default is 0.

    Returns
    -------
    results : dict
        dictionary with arrays for train_loss, val_loss and stability_score.
        
    Examples
    --------
    
    Data Preprocessing
    
    >>> import softsensor.meas_handling as ms
    >>> import numpy as np
    >>> import pandas as pd
    >>> t = np.linspace(0, 1.0, 1001)
    >>> d = {'inp1': np.random.randn(1001),
             'inp2': np.random.randn(1001),
             'out': np.random.randn(1001)}
    >>> handler = ms.Meas_handling([pd.DataFrame(d, index=t)], ['train'],
                                   ['inp1', 'inp2'], ['out'], fs=100)
    

    Train an ARNN
    
    >>> import softsensor.autoreg_models as am
    >>> from softsensor.train_model import train_model
    >>> import torch.optim as optim
    >>> import torch.nn as nn
    >>> model = am.ARNN(2, 1, 10, 10 , [16])
    >>> train_dat, val_dat = handler.give_torch_loader(10, 'training', rnn_window=10,
                                                       shuffle=True)
    >>> opt = optim.Adam(model.parameters(), lr=1e-4)
    >>> crit = nn.MSELoss()
    >>> results = train_model(model=model, train_loader=train_dat, max_epochs=5,
                              optimizer=opt, device='cpu', criterion=crit, stabelizer=5e-3,
                              val_loader=val_dat, print_results=False)
    >>> print(results['val_loss'])

    Train an ARNN with stabilits scheduling
    
    >>> import softsensor.autoreg_models as am
    >>> from softsensor.train_model import train_model
    >>> from softsensor.stab_scheduler import get_scheduler
    >>> import torch.optim as optim
    >>> import torch.nn as nn
    >>> model = am.ARNN(2, 1, 10, 10 , [16])
    >>> stab = get_scheduler('log_lin', model, track_n=30)
    >>> train_dat, val_dat = handler.give_torch_loader(10, 'training', rnn_window=10,
                                                       shuffle=True)
    >>> opt = optim.Adam(model.parameters(), lr=1e-4)
    >>> crit = nn.MSELoss()
    >>> results = train_model(model=model, train_loader=train_dat, max_epochs=5,
                              optimizer=opt, device='cpu', criterion=crit, stabelizer=stab,
                              val_loader=val_dat, print_results=False)
    >>> print(results['stabelizer'])


    Train an ARNN with Mean Variance Estimation (MVE)
    
    >>> import softsensor.autoreg_models as am
    >>> from softsensor.train_model import train_model
    >>> from softsensor.losses import HeteroscedasticNLL
    >>> import torch.optim as optim
    >>> mean_model = am.ARNN(2, 1, 10, 10 , [16])
    >>> model = am.SeparateMVEARNN(2, 1, 10, 10, mean_model, [16])
    >>> train_dat, val_dat = handler.give_torch_loader(10, 'training', rnn_window=10,
                                                       shuffle=True)
    >>> opt = optim.Adam(model.parameters(), lr=1e-4)
    >>> crit = HeteroscedasticNLL()
    >>> results = train_model(model=model, train_loader=train_dat, max_epochs=5,
                              optimizer=opt, device='cpu', criterion=crit, stabelizer=5e-3,
                              val_loader=val_dat, print_results=False)
    >>> print(results['val_loss'])
    
    
    Train an RNN
    
    >>> import softsensor.recurrent_models as rm
    >>> from softsensor.train_model import train_model
    >>> import torch.optim as optim
    >>> import torch.nn as nn
    >>> model = rm.RNN_DNN(2, 1, 10, 16, 1)
    >>> train_dat = handler.give_list(10, 'training')
    >>> opt = optim.Adam(model.parameters(), lr=1e-4)
    >>> crit = nn.MSELoss()
    >>> results = train_model(model=model, train_loader=train_dat, max_epochs=5,
                              optimizer=opt, device='cpu', criterion=crit,
                              val_loader=train_dat, print_results=False)
    >>> print(results['val_loss'])

    """

    model.to(device)
    model.train()

    if val_loader is not None:
        es = early_stopping(patience)
    else:
        es = early_stopping(patience=None)

    results = {'train_loss': [],
               'val_loss': [],
               'stability_score': [],
               'stabelizer': []
               }

    epoch = 0
    stop = False
    
    # check weather stabelizer is scheduled or float value
    if isinstance(stabelizer, float):
        stabelizer = const_stab(stabelizer)
    if stabelizer is None:
        stabelizer = none_stab()

    Model_Type = model.Type
    
    # Eval stability
    if (Model_Type == 'AR') and any([stabelizer, local_wd]):
        results['stability_score'].append(
            Stability_Criterion(model).detach().to('cpu').item())
        
    # Check if model is Bayesian (Bayesian Models need to be trained using the ELBO)
    train_crit = criterion

    # Training Process
    while (epoch < max_epochs) and (stop is False):
        epoch += 1
        
        # Training Step
        if Model_Type == 'AR':
            temp_loss = _AR_train(train_loader, model, optimizer, train_crit,
                                  device, rel_perm, stabelizer, local_wd)
        elif Model_Type == 'RNN':
            temp_loss = _RNN_train(train_loader, model, optimizer, train_crit,
                                   device, rel_perm, stabelizer, local_wd,
                                   AR=False)
        elif Model_Type == 'AR_RNN':
            temp_loss = _RNN_train(train_loader, model, optimizer, train_crit,
                                   device, rel_perm, stabelizer, local_wd,
                                   AR=True)
        else:
            raise KeyError("invalid Model Type given. Possible Types are" +
                           "['Feed_Forward', 'AR', 'AR_Student_Forced', 'RNN', 'AR_RNN']")

        results['train_loss'].append(temp_loss)

        # Eval Val Loader
        if val_loader is not None:
            if Model_Type in ['Feed_Forward', 'AR']:
                temp_loss = _ff_eval(val_loader, model, criterion, device)
            # elif Model_Type == 'AR_Student_Forced':
            #    temp_loss = _ff_eval_batch(val_loader, model, criterion, device)
            elif 'RNN' in Model_Type:
                temp_loss = _RNN_eval(val_loader, model, device, criterion)

            results['val_loss'].append(temp_loss)

        # Eval stability
        if Model_Type in ['AR', 'AR_Student_Forced'] and any([stabelizer, local_wd]):
            results['stability_score'].append(
                Stability_Criterion(model).detach().to('cpu').item())

        # Print results if wanted
        if print_results:
            _print_results(max_epochs, results)

        # check early stopping
        stop = es.call(results['val_loss'], model)
        if stop and print_results:
            print('Training Process finished by early stopping')

    # load best parameter combination for network
    model.load_state_dict(es._load_best_dict())

    model.to('cpu')
    # returns results if wanted
    if give_results:
        if stabelizer.track:
            results['stabelizer'] = stabelizer.stabs
        return results


def _AR_train(dataloader, model, optimizer, criterion, device,
              rel_perm=0, stabelizer=None, local_wd=None):
    """
    Training for an Autoregressive Network

    Parameters
    ----------
    dataloader : Dataloader
        Training data.
    model : Torch Neural Network
        Network to train.
    optimizer : torch.optim()
        Optimizer for training Process.
    criterion : torch Loss function
        Criterion  e.g. nn.MSELoss()
    device : torch.device
        Device for computation.
    rel_perm : float, optional
        relative permutation applied to the input to prevent overfitting.
        The default is 0.
    stabelizer : float, optional
        stability score for Model_Type 'AR'. The default is None.
    local_wd : float, optional
        Applies a local weight decay on all weights that interact with the
        recurrent input for Model_Type 'AR'. The default is None.

    Returns
    -------
    Torch.Tensor
        Single entry tensor with loss

    """

    running_loss_train = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, output = data
        optimizer.zero_grad()

        x_rec = inputs[1].to(device)
        inputs, output = inputs[0].to(device), output.to(device)
        inputs = _relative_permutation(inputs, rel_perm).detach()
        x_rec = _relative_permutation(x_rec, rel_perm).detach()
        
        prediction = model(inputs, x_rec)

        stab = stabelizer.get_stab(model)
        if any([stab, local_wd]):
            crit = Stability_Loss(criterion, model, stab, local_wd)

        else:
            crit = criterion
        loss = crit(prediction, output)

        loss.backward()
        optimizer.step()

        running_loss_train += loss.detach().item()
    return running_loss_train/len(dataloader)


def _RNN_train(dataloader, model, optimizer, criterion, device,
               rel_perm=0, stabelizer=None, local_wd=None, AR=False):
    """
    Training for an Autoregressive Network

    Parameters
    ----------
    dataloader : list of Dataloader (unshuffled)
        Training data.
    model : Torch Neural Network
        Network to train.
    optimizer : torch.optim()
        Optimizer for training Process .
    criterion : torch Loss function
        Criterion  e.g. nn.MSELoss()
    device : torch.device
        Device for computation.
    rel_perm : float, optional
        relative permutation applied to the input to prevent overfitting.
        The default is 0.
    stabelizer : float, optional
        stability score for Model_Type 'AR'. The default is None.
    local_wd : float, optional
        Applies a local weight decay on all weights that interact with the
        recurrent input for Model_Type 'AR'. The default is None.
    AR : bool, optional
        True if recurrent model is autoregressive, default is False
    Returns
    -------
    Torch.Tensor
        Single entry tensor with loss

    """

    running_loss_train = 0.0
    for loader in dataloader:
        model.RecBlock.init_hidden()
        if hasattr(model.RecBlock, 'hidden_values'):
            model.RecBlock.precomp_hidden_states(device)
        rl = 0.0
        for i, data in enumerate(loader, 0):

            inputs, output = data
            optimizer.zero_grad()

            if AR:
                x_rec = inputs[1].to(device)
                inputs, output = inputs[0].to(device), output.to(device)
                inputs = _relative_permutation(inputs, rel_perm).detach()
                x_rec = _relative_permutation(x_rec, rel_perm).detach()
                prediction = model(inputs, x_rec, device)
            else:
                inputs, output = inputs.to(device), output.to(device)
                prediction = model(inputs, device)
                
            stab = stabelizer.get_stab(model)
            if any([stab, local_wd]):
                crit = Stability_Loss(criterion, model, stab, local_wd)
            else:
                crit = criterion
            loss = crit(prediction, output)
            
            loss.backward()
            optimizer.step()
            if hasattr(model.RecBlock, 'hidden_values'):
                model.RecBlock.hidden_values = model.RecBlock.hidden_values.detach() 
                   

            rl += loss.detach().item()
        
        running_loss_train += rl/len(loader)
    return running_loss_train/len(dataloader)


def _RNN_eval(val_loader, model, device, criterion):
    """
    Evaluation for a Recurrent Neural Network

    Parameters
    ----------
    dataloader : list of Dataloader (unshuffled)
        Evaluation data.
    model : Torch Neural Network
        Network to train.
    device : torch.device
        Device for computation.
    criterion : torch Loss function
        Criterion e.g. nn.MSELoss()

    Returns
    -------
    running_loss_val : Torch.Tensor
        Single entry tensor with loss

    """
    vl = 0
    for loader in val_loader:
        model.RecBlock.init_hidden()
        temp_loss = _ff_eval(loader, model, criterion, device)
        vl = vl + temp_loss
    return (vl/len(val_loader))


def _ff_eval(dataloader, model, criterion, device):
    """
    Evaluation class for defined model classes.

    Parameters
    ----------
    dataloader : Dataloader
        Training data.
    model : Torch Neural Network
        Network to train.
    criterion : torch Loss function
        Criterion e.g. nn.MSELoss()
    device : torch.device
        Device for computation.
    Returns
    -------
    running_loss_val : Torch.Tensor
        Single entry tensor with loss

    """
    model.eval()
    running_loss_val = 0.0
    Type = model.Type
    for i, data in enumerate(dataloader, 0):

        inputs, output = data
        if Type == 'Feed_Forward':
            inputs, output = inputs.to(device), output.to(device)
            prediction = model(inputs)

        if Type == 'AR':
            x_rec = inputs[1].to(device)
            inputs, output = inputs[0].to(device), output.to(device)

            prediction = model(inputs, x_rec)

        if Type == 'AR_RNN':
            x_rec = inputs[1].to(device)
            inputs, output = inputs[0].to(device), output.to(device)
            prediction = model(inputs, x_rec, device)

        if Type == 'RNN':
            inputs, output = inputs.to(device), output.to(device)
            prediction = model(inputs, device)

        loss = criterion(prediction, output)

        running_loss_val += loss.item()
    model.train()
    return running_loss_val/len(dataloader)


class early_stopping():
    """
    Early Stopping funtion to prevent overfitting in the training process

    Parameters
    ----------
    model : pytorch Network
        Model from which parameters are temporarily stored to keep best models
    patience : int
        Patience of the early stopping, precisely how many epochs without
        an increase in performance are allowed

    Returns
    -------
    None.

    """

    def __init__(self, patience):

        self.patience = patience
        self.min = None
        self.parameters = None

    def call(self, loss, model):
        """
        Calling function, for given val_loss and model

        Parameters
        ----------
        loss : list
            list of val_loss, individual elements have to be torch dtype
        model : pytorch Network
            Network to store parameters from in case of improvement.

        Returns
        -------
        bool
            True to stop training, False to continue training.

        """
        if self.patience is None:
            self._save_dict(model)
            return False

        loss = torch.tensor(loss)
        if torch.argmin(loss).item() == loss.shape[0]-1:
            self._save_dict(model)
            self.min = loss[-1]

        if loss.shape[0] < self.patience:
            return False
        else:
            temp_min = torch.min(loss[-self.patience:])
            if temp_min <= self.min:
                return False
            else:
                return True

    def _save_dict(self, model):
        """
        Internal function to store network parameters

        Parameters
        ----------
        model : pytorch Network
            Network to store parameters from in case of improvement.

        Returns
        -------
        None.

        """
        self.parameters = model.state_dict()

    def _load_best_dict(self):
        """
        Loading function to get best performing parameter combination

        Returns
        -------
        model_state_dict()
            Best performing parameter combination to load into Network

        """
        return self.parameters


def _print_results(max_e, results):
    """
    Printing function for training

    Parameters
    ----------
    max_e : int
        maximum number of epochs to run.
    train_loss : list
        list of train_loss, individual elements have to be torch dtype
    val_loss : list
        list of val_loss, individual elements have to be torch dtype

    Returns
    -------
    None.

    """
    if len(results['val_loss']) == 0:
        le = results['train_loss']
    else:
        le = results['val_loss']
    e = len(results['train_loss'])

    if len(results['stability_score']) != 0:
        sys.stdout.write("[%-60s] %d%%, epoch %d, loss: %f SC: %f \r"
                         % ('='*int((60*(e)/max_e)), (100*(e)/max_e),
                            e, le[-1], results['stability_score'][-1]))
    else:
        sys.stdout.write("[%-60s] %d%%, epoch %d, loss: %f \r"
                         % ('='*int((60*(e)/max_e)), (100*(e)/max_e),
                            e, le[-1]))


def _relative_permutation(x, sigma):
    """
    Permute input with gaussian noise

    Parameters
    ----------
    x : torch tensor
        Tensor to be permuted.
    sigma : float
        realtive Std to permute input.

    Returns
    -------
    x : torch tensor
        permuted tensor.

    """
    scale = x * sigma
    x = torch.randn_like(x).detach()*scale.detach() + x
    return x
