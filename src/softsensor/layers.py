# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter


class ConcreteDropout(nn.Module):
    """ 
    Dropout layer that uses continuous relaxation of dropout's discrete masks
    This allows for automatic tuning of the dropout probability during training, resulting in a more robust method

    See "Concrete Dropout"
    [Gal et al. 2017 https://arxiv.org/pdf/1705.07832.pdf]

    Parameters
    ----------
    layer: Preceding layer that the weight dropout is applied to
    weight_regularizer: Penalty for large weights that considers dropout probability
    dropout_regularizer: Penalty for small dropout rate (entropy of dropout)
    init_min: Minimum of dropout distribution
    init_max: Maximum of dropout distribution

    Returns
    -------
    None
    
    Note
    ------
    
    See "Concrete Dropout"
    [Gal et al. 2017 https://arxiv.org/pdf/1705.07832.pdf]

    """

    def __init__(self, layer, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()

        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.layer = layer
        
    def forward(self, x):
        """
        Forward function to apply concrete dropout to the outputs of self.layer 

        Parameters
        ----------
        x : torch.tensor dtype=torch.float
            Input tensor for forward propagation

        Returns
        -------
        out: torch.tensor dtype=torch.float()
            Output tensor

        """
        p = torch.sigmoid(self.p_logit)
        out = self.layer(self._concrete_dropout(x, p))
        return out
        
    def _concrete_dropout(self, x, p):
        """
        Concrete dropout

        Parameters
        ----------
        x : torch.tensor dtype=torch.float
            Input tensor for forward propagation
        p : torch.tensor dtype=torch.float
            Trainable dropout probability parameter

        Returns
        -------
        x: torch.tensor dtype=torch.float()
            Output tensor

        """
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        
        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        
        return x