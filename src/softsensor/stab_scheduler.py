# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:12:56 2023

@author: WET2RNG
"""

import math
import warnings
import numpy as np
import torch.nn as nn

def Stability_Criterion(Network):
    """
    Defined stability criterion for Autoregressive Neural Networks.
    If criterion < 0 Network is Input to state stable according to:
    https://www.researchgate.net/publication/346701375_Stability_of_discrete-time_feed-forward_neural_networks_in_NARX_configuration

    Parameters
    ----------
    Network : torch Network
        Neural Network with function Network.get_recurretn_weights.

    Raises
    ------
    Warning
        if activation function is not defined with Lipschitz constant
        explicitly.

    Returns
    -------
    float
        stabiliyt criterion for the Network.

    """
    mult_weight_norm = local_weight_product(Network)
    const = arnn_threshold(Network)
    return (mult_weight_norm - const)


def local_weight_product(Network):
    """
    Defines the product of the norm of the recurrent wegiths in a defined
    Network

    Parameters
    ----------
    Network : Torch Neural Network
        Neural Network with function Network.get_recurretn_weights.

    Returns
    -------
    mult_weight_norm : float
        product of the recurrent weight norm.

    """
    mult_weight_norm = 1
    
    for W in Network.get_recurrent_weights():
        mult_weight_norm = mult_weight_norm * W.norm(p='fro')
    return mult_weight_norm
    

def local_weight_decay(Network):
    """
    Defined local weight decay for Autoregressive Neural Networks.

    Parameters
    ----------
    Network : torch Network
        Neural Network with function Network.get_recurretn_weights.

    Returns
    -------
    mult_weight_norm : float
        local weigth decay of the Network.

    """
    mult_weight_norm = 0
    for W in Network.get_recurrent_weights():
        mult_weight_norm = mult_weight_norm + W.norm()
    return mult_weight_norm


class Stability_Loss(nn.Module):
    """
    Callable loss function to train for accuracy and stability of an
    autoregressive Network at the same time

    Parameters
    ----------
    criterion : nn.Loss, optional
        Loss function for training. e.g. nn.MSELoss().
    DNN : torch Network
        Neural Network with function Network.get_recurretn_weights.
    stabelizer : float, optional
        added stability parameter applied to recurrent weights.
        The default is 0.
    local_wd : float, optional
        added local weight decay parameter applied to recurrent weights.
        The default is 0.

    Returns
    -------
    None.

    """

    def __init__(self, criterion, DNN, stabelizer=0, local_wd=0):

        super(Stability_Loss, self).__init__()
        self.DNN = DNN
        self.criterion = criterion

        if stabelizer is None:
            self.stabl = 0
        else:
            self.stabl = stabelizer
        if local_wd is None:
            self.local_wd = 0
        else:
            self.local_wd = local_wd

    def forward(self, y_pred, y):
        """
        Forward function for the defined loss between the prediction y_pred and
        y with added stabelization term

        Parameters
        ----------
        y_pred : torch.tensor
            predicted values.
        y : torch.tensor
            true values.

        Returns
        -------
        loss : torch.tensor
            loss between y_pred and y.

        """
        
        crit_loss = self.criterion(y_pred, y)

        if (self.stabl != 0) and (self.local_wd == 0):
            loss = self.stabl * local_weight_product(self.DNN)

        elif (self.stabl == 0) and (self.local_wd != 0):
            loss = self.local_wd * local_weight_decay(self.DNN)

        elif (self.stabl != 0) and (self.local_wd != 0):
            loss = self.local_wd * local_weight_decay(self.DNN) + \
            self.stabl * local_weight_product(self.DNN)
    
        else:
            loss = 0

        return (crit_loss + loss)


class const_stab():
    """
    Stability score scheduler that returns a constant value

    Parameters
    ----------
    s1 : float, optional
        the constant weight to be returned. The default is 0.1.
    track : Bool, optional
        if True the values are tracked in a dict. The default is True.

    Returns
    -------
    None.

    """
    def __init__(self, s1=0.1, track=True, track_n=1): 

        self.stab = s1
        self.track = track
        self.stabs = {'sc': [],
                      'eta': [s1]}
        self.track_n = track_n
        self.n = 0

    def get_stab(self, model=None):
        """
        call method to get stability parameter

        Parameters
        ----------
        model : ARNN Model, optional
            Not needed, optional Parameter to be consistant with other
            stability schedulers. The default is None.

        Returns
        -------
        float
            s1 value.

        """
            
        return self.stab

class none_stab():
    """
    Stability score scheduler that returns a None

    Parameters
    ----------
    track : Bool, optional
        if True the values are tracked in a dict. The default is True.

    Returns
    -------
    None.

    """
    def __init__(self, track=True, track_n=1):
        self.track = track
        self.stabs = {'sc': [],
                      'eta': [None]}
        self.track_n = track_n
        self.n = 0

    def get_stab(self, model=None):
        """
        call method to get stability parameter
        
        Parameters
        ----------
        model : ARNN Model, optional
            Not needed, optional Parameter to be consistant with other
            stability schedulers. The default is None.

        Returns
        -------
        None

        """
            
        return None
        

class linear_stab():
    """
    Stability score scheduler designed as a linear funciton with value s1 at
    stability score zero 
    eta = m*x + s1
    m = s1 / (Network specific threshold)
    If criterion < 0 Network is Input to state stable according to:
    https://www.researchgate.net/publication/346701375_Stability_of_discrete-time_feed-forward_neural_networks_in_NARX_configuration

    Parameters
    ----------
    model : ARNN
        ARNN according to softsensor.autoregr_models design
    s1 : float, optional
        stability parameter at stability score zero
    track : Bool, optional
        if True the values are tracked in a dict. The default is True.

    Returns
    -------
    None.

    """
    def __init__(self, model, s1=1e-2, track=True, track_n=1):
        self.track = track
        self.stabs = {'sc': [],
                      'eta': []}
        self.track_n = track_n
        self.n = 0
        kappa = arnn_threshold(model)
        m = s1/kappa
        self.stab = lambda x: m*x + s1

    def get_stab(self, model, sc=None):
        """
        call method to get stability parameter
        
        Parameters
        ----------
        model : ARNN Model
            ARNN according to softsensor.autoregr_models design 
        sc : float or None, optional
            stability score to evaluate function at, if None is given computes
            stability with method Stability_Criterion(model)
            The default is None.

        Returns
        -------
        st : float
            stablizer according to the linear function

        """
        if sc is None:
            sc = Stability_Criterion(model).item()
        st = self.stab(sc)
        if (self.track) and (self.n % self.track_n == 0):
            self.stabs['sc'].append(sc)
            self.stabs['eta'].append(st)
        self.n = self.n+1
        return st
        
    
class log_lin_stab():    
    """
    Stability score scheduler designed as an exponential function up to a
    stability criterion of zero and a linear function afterwards
    If criterion < 0 Network is Input to state stable according to:
    https://www.researchgate.net/publication/346701375_Stability_of_discrete-time_feed-forward_neural_networks_in_NARX_configuration

    Parameters
    ----------
    model : ARNN
        ARNN according to softsensor.autoregr_models design
    s0 : float, optional
        stability parameter at the lower Networkspecific threshold,
        The default is 1e-6.
    s1 : float, optional
        stability parameter at stability score zero,The default is 1e-3.
    m : float, optional
        gradient of the linear function between [0, inf].
        If None is given, use the gradient of the exponential function.
        The default is None.
    track : Bool, optional
        if True the values are tracked in a dict. The default is True.

    Returns
    -------
    None.

    """
    def __init__(self, arnn, s0=1e-6, s1=1e-3, m=None, track=True, track_n=1):

        self.track = track
        self.stabs = {'sc': [],
                      'eta': []}
        self.track_n = track_n
        self.n = 0
        kappa = arnn_threshold(arnn)
        x0 = [-kappa, s0]
        x1 = [0, s1]
        b = (x0[1]/x1[1])**(1/(x0[0] - x1[0]))
        a = x0[1]/(b**x0[0])
        if m is None:
            m = b**x1[0] * math.log(b) * a
        y_lim = x1[1] - m*x1[0]
        
        self.log_stab = lambda x: a * b**x
        self.lin_stab = lambda x: m*x + y_lim

    def get_stab(self, model, sc=None):
        """
        call method to get stability parameter
        
        Parameters
        ----------
        model : ARNN Model
            ARNN according to softsensor.autoregr_models design 

        Returns
        -------
        st : float
            stablizer according to the linear function
        sc : float or None, optional
            stability score to evaluate function at, if None is given computes
            stability with method Stability_Criterion(model)
            The default is None.

        """
        if sc is None:
            sc = Stability_Criterion(model).item()
        
        if sc >= 0:
            st = self.lin_stab(sc)
        else:
            st = self.log_stab(sc)
        
        if (self.track) and (self.n % self.track_n == 0):
            self.stabs['sc'].append(sc)
            self.stabs['eta'].append(st)
        self.n = self.n+1
        return st


class heaviside_stab():
    """
    Stability score scheduler designed as a heaviside funciton with value s1 at
    stability score greater zero 
    If criterion < 0 Network is Input to state stable according to:
    https://www.researchgate.net/publication/346701375_Stability_of_discrete-time_feed-forward_neural_networks_in_NARX_configuration
    
    Parameters
    ----------
    model : ARNN
        ARNN according to softsensor.autoregr_models design
    s1 : float, optional
        stability parameter at stability score zero, The default is 1e-2.
    track : Bool, optional
        if True the values are tracked in a dict. The default is True.
    
    Returns
    -------
    None.
    
    """
    def __init__(self, s1=1e-2, track=True, track_n=1):
        self.track = track
        self.stabs = {'sc': [],
                      'eta': []}
        self.track_n = track_n
        self.n = 0
        self.m = s1
        
    def get_stab(self, model, sc=None):
        """
        call method to get stability parameter
        
        Parameters
        ----------
        model : ARNN Model
            ARNN according to softsensor.autoregr_models design 
        sc : float or None, optional
            stability score to evaluate function at, if None is given computes
            stability with method Stability_Criterion(model)
            The default is None.

        Returns
        -------
        st : float
            stablizer according to the linear function

        """
        if sc is None:
            sc = Stability_Criterion(model).item()
        if sc > 0:
            st = self.m
        else:
            st = 0

        if (self.track) and (self.n % self.track_n == 0):
            self.stabs['sc'].append(sc)
            self.stabs['eta'].append(st)
        self.n = self.n+1
        
        return st

def arnn_threshold(arnn):
    """
    function to compute the specific lower limit for the stability of an arnn

    Parameters
    ----------
    arnn : ARNN Model
        ARNN according to softsensor.autoregr_models design 

    Raises
    ------
    Warning
        if the activation function is not implemendet. currently implemented are:
        ['tanh', 'relu', 'leaky_relu', 'sine']

    Returns
    -------
    kappa : float
        lower limit for the stability of an arnn.

    """
    if arnn.activation in ['tanh', 'relu', 'leaky_relu', 'sine']:
        Lipschitz = 1
    else:
        Lipschitz = 1
        warnings.warn('Warning: Lipschitz constant for activation function is not ' +
                      'known, assuming Lipschitz = 1')

    kappa = 1/(Lipschitz * np.sqrt(arnn.rnn_window))
    return kappa


def get_scheduler(stab_method, model, s1=1e-2, s0=1e-8, m=None, track=True, track_n=100):
    """
    wrapper function to get the specific scheduler by name

    Parameters
    ----------
    stab_method : str or float or None
        defines the specific Method. 
        implemented str: ['const', 'lin', 'log_lin', 'heaviside']
        float value returnes the same scheduler as 'const'
    arnn : torch Network
        Neural Network with function Network.get_recurretn_weights.
    s1 : float, optional
        scheduler specific parameter. The default is 1e-2.
    s0 : float, optional
        scheduler specific parameter. The default is 1e-8.
    m : float, optional
        scheduler specific parameter. The default is None.
    track : Bool, optional
        if True the values are tracked in a dict. The default is True.

    Raises
    ------
    Warning
        if invalid stab_method is given

    Returns
    -------
    class 
        stab scheduler Method.

    """
    if isinstance(stab_method, float):
        return const_stab(s1, track, track_n)
    elif stab_method == 'const':
        return const_stab(s1, track, track_n)
    elif stab_method is None:
        return none_stab(track, track_n)
    elif stab_method == 'lin':
        return linear_stab(model, s1, track, track_n)
    elif stab_method == 'log_lin':
        return log_lin_stab(model, s0, s1, m, track, track_n)
    elif stab_method == 'heaviside':
        return heaviside_stab(s1, track, track_n)
    else:
        raise Warning('No Valid scheduler method given')