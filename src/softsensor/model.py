# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from softsensor.layers import ConcreteDropout

class Feed_ForwardNN(nn.Module):
    """
    Deep Neural Network with Fully Connected Layers

    Parameters
    ----------
    input_size : int
        Size of input array.
    output_size : int
        Size of output array.
    hidden_size : list of int or None, optional
        List gives the size of hidden units. The default is None.
    activation : str, optional
        Activation function to activate the feature space.
        The default is 'relu'.
    bias : bool, optional
        If True, bias weights are used. The default is True.
    dropout : float [0,1], optional
        Adds dropout layers after each Linear Layer. The default is None.
    concrete_dropout: bool, optional
        Whether to use normal or concrete dropout layers if dropout is not None. The default is False
    Returns
    -------
    None.
    
    Example
    -------
    
    >>> import softsensor.model as model
    >>> import torch
    >>> model = model.Feed_ForwardNN(input_size=40, output_size=2,
                                     hidden_size=None, bias=True)
    >>> print(model)
    Feed_ForwardNN(
      (DNN): Sequential(
        (0): Linear(in_features=40, out_features=2, bias=True)
      )
    )
    >>> input = torch.randn(32, 40)
    >>> output = model(input)
    >>> print(output.size())
    torch.Size([32, 2])
    
    """

    def __init__(self, input_size, output_size, hidden_size=None,
                 activation='relu', bias=True, dropout=None, concrete_dropout=False, bn=False):
        super().__init__()

        self.act = activation
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.output_size = output_size
        self.dropout = dropout

        self.DNN = []
        in_features = self.input_size

        if hidden_size:
            for i, hidden in enumerate(self.hidden_size):
                layer = nn.Linear(in_features, hidden, bias=bias)
                if concrete_dropout and i>=1:
                    self.DNN.append(ConcreteDropout(layer, init_min=dropout, init_max=dropout))
                    self.DNN.append(_activation_function(activation))
                else:
                    self.DNN.append(layer)
                    self.DNN.append(_activation_function(activation))
                
                if bn:
                    self.DNN.append(nn.LayerNorm(hidden))

                if dropout and not (concrete_dropout and i>=1):
                    self.DNN.append(nn.Dropout(dropout))
    
                in_features = hidden

        self.DNN.append(nn.Linear(in_features, self.output_size,
                                  bias=bias))

        self.DNN = nn.Sequential(*self.DNN)
    
        #self.DNN = build_dnn(input_size, output_size, hidden_size, activation, bias, dropout, concrete_dropout)

    def forward(self, inp):
        """
        Forward function to probagate through the network

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation

        Returns
        -------
        output: torch.tensor dtype=torch.float()
            Output tensor

        """
        inp = self.DNN(inp)
        return inp


class CNN(nn.Module):
    """
    Convolutional Model

    Parameters
    ----------
    input_channels : int
        Number of input channels
    window_size : int
        Size of the sliding window applied to the time series
    filters : int or list of int
        Number of filters used in the convolution
    kernel_size : int or list of int
        Width of the filter, (needs to be uneven)
    depth : int,
        Depth of the Network,
        (how often is Convolution, Activation and Pooling repeated)
        The default is 1.
    pooling : str, optional
        Pooling Variant to pool filtered time series. options are: 'average'
        and 'max'
        The default is None.
    pooling_size : int, optional
        Kernel size of the pooling layer. The default is 2.
    activation : str, optional
        Activation function to activate the feature space.
        The default is 'relu'.
    bias : bool, optional
        If True, bias weights are used. The default is True.
    dropout : float [0,1], optional
        Adds dropout layers after each activation. The default is None.
    Returns
    -------
    None.

    Example
    -------
    
    >>> import softsensor.model as model
    >>> import torch
    >>> model = model.CNN(input_channels=4, window_size=50, filters=8,
                          kernel_size=5, depth=2, pooling='average')
    >>> print(model)
    CNN(
      (ConvNet): Sequential(
        (0): conv_block(
          (Conv): Conv1d(4, 8, kernel_size=(5,), stride=(1,))
          (pool): AvgPool1d(kernel_size=(2,), stride=(2,), padding=(0,))
          (activation): ReLU()
        )
        (1): conv_block(
          (Conv): Conv1d(8, 8, kernel_size=(5,), stride=(1,))
          (pool): AvgPool1d(kernel_size=(2,), stride=(2,), padding=(0,))
          (activation): ReLU()
        )
      )
    )
    >>> input = torch.randn(32, 4, 50)
    >>> output = model(input)
    >>> print(output.size())
    torch.Size([32, 8, 10])
    
    
    To get the length of the output time series:
    >>> print(model.ts_length)
    10
    
    """

    def __init__(self, input_channels, window_size, filters, kernel_size,
                 depth=1, pooling=None, pooling_size=2, activation='relu',
                 bias=True, dropout=None):

        super().__init__()
        self.act = activation
        self.window_size = window_size
        self.dropout = dropout

        if isinstance(filters, int):
            self.filters = [filters] * depth
        else:
            self.filters = filters

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * depth
        else:
            self.kernel_size = kernel_size

        # Define Convlutional Network
        self.ConvNet = []

        in_channels = input_channels
        self.ts_length = window_size
        for kern, filt in zip(self.kernel_size, self.filters):

            self.ConvNet.append(_conv_block(in_channels, self.ts_length,  filt,
                                           kern, pooling, pooling_size,
                                           self.act, bias=bias))
            in_channels = filt
            self.ts_length = self.ConvNet[-1].output_size
            if dropout is not None:
                self.ConvNet.append(nn.Dropout(self.dropout))

        self.ConvNet = nn.Sequential(*self.ConvNet)

    def forward(self, inp):
        """
        Forward function to probagate through the network

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation

        Returns
        -------
        output: torch.tensor dtype=torch.float()
            Output tensor

        """
        inp = self.ConvNet(inp)
        return inp


class Freq_Att_CNN(nn.Module):
    """
    Convolutional  Model build on the parallel convolutions of the input
    with delated convolutions and subsequent convolution of the resulting
    parallel time series

    Parameters
    ----------
    input_channels : int
        Number of channels of the external excitation signal
    filters : int or list of int
        Number of filters used in the convolution.
    kernel_size : int or list of int
        Width of the filter
    max_dilation : int
        maximum dilation in the parallel convolution. The derfault is 4
    oscillation : int
        Number of full oscillations for each parallel convolution is
        processed. The default is 4
    depth : int, optional
        Depth of the Convolutional Network. The depth is applied to the
        convolutions
        (how often is Convolution, Activation and Pooling repeated)
        The default is 1.
    bypass: bool, optional
        if True bypass is applied for faster convergence.
        The default is True.
    activation : str, optional
        Activation function to activate the feature space.
        The default is 'relu'.
    bias : bool, optional
        If True, bias weights are used. The default is True.
    dropout : float [0,1], optional
        Adds dropout layers after each activation. The default is None.
    Returns
    -------
    None.
    
    Example
    -------
    
    >>> import softsensor.model as model
    >>> import torch
    >>> model = model.Freq_Att_CNN(input_channels=2, filters=4, kernel_size=5,
                                   max_dilation=2, oscillations=2)
    >>> print(model)
    Freq_Att_CNN(
      (stage1): _Parallel_Conv(
        (Convs): ModuleList(
          (0): _delated_conv(
            (ConvNet): Sequential(
              (0): Conv1d(2, 4, kernel_size=(5,), stride=(1,))
              (1): ReLU()
            )
          )
          (1): _delated_conv(
            (ConvNet): Sequential(
              (0): Conv1d(2, 4, kernel_size=(5,), stride=(2,), dilation=(2,))
              (1): ReLU()
            )
          )
        )
      )
      >>> ws = model.window_size
      >>> input = torch.randn(32, 2, ws)
      >>> output = model(input)
      >>> print(output.size())
      torch.Size([32, 6, 2])

    """

    def __init__(self, input_channels, filters, kernel_size, max_dilation=4,
                 oscillations=4, depth=1, bypass=True, activation='relu',
                 bias=True, dropout=None):

        super().__init__()
        self.stage1 = _Parallel_Conv(input_channels, filters, kernel_size,
                                    max_dilation, oscillations, activation,
                                    bias)
        self.stage2 = CNN(filters*max_dilation, self.stage1.Lout,
                          filters, kernel_size, depth, activation=activation,
                          bias=bias, dropout=dropout)
        self.ts_length = self.stage2.ts_length
        self.window_size = self.stage1.window_size
        self.bypass = bypass

    def forward(self, x):
        """
        Forward function to probagate through the network

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation

        Returns
        -------
        output: torch.tensor dtype=torch.float()
            Output tensor

        """
        out = self.stage1(x)
        out = self.stage2(out)
        if self.bypass:
            return torch.cat((out, x[:, :, -self.ts_length:]), dim=1)
        else:
            return out


'''
helpers
'''


class _conv_block(nn.Module):
    """
    Convolutional Block usng  the combination of Convolution,
    Activation and Pooling

    Parameters
    ----------
    in_channels : int
        Number of input channels
    window_size : int
        Length of the input time window
        (needed for output shape calculation)
    filters : int
        Number of filters used in the convolution
    kernel_size : int
        Filter width.
    pooling : str
        Pooling Variant to pool filtered time series. ['max'], ['average']
        The default is None.
    pooling_size : int
        Kernel size of the pooling layer.
        The default is 1.
    activation : str, optional
        Activation function e.g. ['relu'], ['sigmoid']. The default is 'relu'
    bias : bool, optional
        If True, bias weights are used. The default is True.

    Returns
    -------
    None.

    """

    def __init__(self, in_channels, window_size, filters, kernel_size,
                 pooling=None, pooling_size=1, activation='relu', bias=True):

        super().__init__()

        self.Conv = nn.Conv1d(in_channels, filters, kernel_size, bias=bias)
        self.output_size = window_size - kernel_size + 1
        self.pooling = pooling

        if pooling is not None:
            if pooling == 'max':
                self.pool = nn.MaxPool1d(kernel_size=pooling_size,
                                         ceil_mode=True)
            elif pooling == 'average':
                self.pool = nn.AvgPool1d(kernel_size=pooling_size,
                                         ceil_mode=True)
            self.output_size = math.ceil((self.output_size - pooling_size) /
                                         pooling_size + 1)
        self.activation = _activation_function(activation)

    def forward(self, inp):
        """
        Forward Propagation of the Convolutional Block

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation

        Returns
        -------
        output: torch.tensor dtype=torch.float()
            Output tensor

        """
        inp = self.Conv(inp)
        if self.pooling is not None:
            inp = self.pool(inp)
        return self.activation(inp)


class _RNN(nn.Module):
    """
    RNN Block to be used in mode advanced models

    Parameters
    ----------
    in_channels : int
        Number of individual inputs
    blocks : int
        width of the RNN Layer.
    num_layers : int
        depth of the RNN Layer.
    dropout : float [0, 1], optional
        dropout between the RNN Layers (only valid if num_layers > 1).
        The default is None.

    Returns
    -------
    None.

    """

    def __init__(self, input_size, hidden_size, num_layers, bias=True,
                 dropout=None):

        super(_RNN, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        if dropout is None:
            dropout = 0

        self.rnn = nn.RNN(input_size=input_size, hidden_size=self.hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout, bias=bias)
        self.init_hidden()

    def forward(self, x, device='cpu'):
        """
        Forward Propagation of the RNN Block

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation
        device : str, optional
            Device the hidden state is stored on. The default is 'cpu'

        Returns
        -------
        output: torch.tensor dtype=torch.float()
            Output tensor of shape [batch_size, hidden_size]

        """
        x = torch.flatten(x, start_dim=1)
        x = torch.unsqueeze(x, 0)

        # Propagate input through LSTM
        self.h_0 = self.h_0.to(device)
        out, self.h_0 = self.rnn(x, self.h_0)
        self.h_0 = self.h_0.detach()

        self.init_hidden(self.h_0)
        return torch.squeeze(out, dim=0)

    def get_recurrent_weights(self):
        """
        Function that returns the weight that effect the Recurrent input of the
        Network

        Returns
        -------
        recurrent_weights : list of weight Tensors
            List of the Weights that effect the Recurren input of the Network.

        """
        return _get_rec_weights_rnn(self)

    def init_hidden(self, hidden=None):
        """
        Initialize the hidden state

        Parameters
        ----------
        hidden : None or tensor, optional
            if None, hidden state is initialized as zero.
            if hidden is Tensor the hidden state is initialized as the given
            TensorThe default is None.

        Returns
        -------
        None.

        """
        if torch.is_tensor(hidden):
            self.h_0 = Variable(hidden)
        else:
            self.h_0 = Variable(torch.zeros(
                self.num_layers, 1, self.hidden_size))


class _GRU(nn.Module):
    """
    GRU Block to be used in mode advanced models

    Parameters
    ----------
    in_channels : int
        Number of individual inputs
    blocks : int
        width of the RNN Layer.
    num_layers : int
        depth of the RNN Layer.
    dropout : float [0, 1], optional
        dropout between the RNN Layers (only valid if num_layers > 1).
        The default is None.

    Returns
    -------
    None.

    """

    def __init__(self, input_size, hidden_size, num_layers, bias=True,
                 dropout=None):
        super(_GRU, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        if dropout is None:
            dropout = 0

        self.gru = nn.GRU(input_size=input_size, hidden_size=self.hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout, bias=bias)
        self.init_hidden()

    def forward(self, x, device='cpu'):
        """
        Forward Propagation of the GRU Block

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation
        device : str, optional
            Device the hidden state is stored on. The default is 'cpu'

        Returns
        -------
        output: torch.tensor dtype=torch.float()
            Output tensor of shape [batch_size, hidden_size]

        """
        x = torch.flatten(x, start_dim=1)
        x = torch.unsqueeze(x, 0)

        # Propagate input through GRU
        self.h_0 = self.h_0.to(device)
        out, self.h_0 = self.gru(x, self.h_0)
        self.h_0 = self.h_0.detach()
        self.init_hidden(self.h_0)
        return torch.squeeze(out, dim=0)

    def get_recurrent_weights(self):
        """
        Function that returns the weight that effect the Recurrent input of the
        Network

        Returns
        -------
        recurrent_weights : list of weight Tensors
            List of the Weights that effect the Recurren input of the Network.

        """
        return _get_rec_weights_rnn(self)

    def init_hidden(self, hidden=None):
        """
        Initialize the hidden state

        Parameters
        ----------
        hidden : None or tensor, optional
            if None, hidden state is initialized as zero.
            if hidden is Tensor the hidden state is initialized as the given
            TensorThe default is None.

        Returns
        -------
        None.

        """
        if torch.is_tensor(hidden):
            self.h_0 = Variable(hidden)
        else:
            self.h_0 = Variable(torch.zeros(
                self.num_layers, 1, self.hidden_size))


class _LSTM(nn.Module):
    """
    LSTM Block to be used in mode advanced models

    Parameters
    ----------
    in_channels : int
        NUmbe rof individual inputs
    blocks : int
        width of the RNN Layer.
    num_layers : int
        depth of the RNN Layer.
    dropout : float [0, 1], optional
        dropout between the RNN Layers (only valid if num_layers > 1).
        The default is None.

    Returns
    -------
    None.

    """

    def __init__(self, input_size, hidden_size, num_layers, bias=True,
                 dropout=None):

        super(_LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        if dropout is None:
            dropout = 0

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=self.hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout, bias=bias)
        self.init_hidden()

    def forward(self, x, device='cpu'):
        """
        Forward Propagation of the LSTM Block

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation
        device : str, optional
            Device the hidden state is stored on. The default is 'cpu'

        Returns
        -------
        output: torch.tensor dtype=torch.float()
            Output tensor of shape [batch_size, hidden_size]

        """
        x = torch.flatten(x, start_dim=1)
        x = torch.unsqueeze(x, 0)

        # Propagate input through LSTM
        self.h_0, self.c_0 = self.h_0.to(device), self.c_0.to(device)
        out, (self.h_0, self.c_0) = self.lstm(x, (self.h_0, self.c_0))
        self.h_0 = self.h_0.detach()
        self.c_0 = self.c_0.detach()

        self.init_hidden((self.h_0, self.c_0))
        return torch.squeeze(out, dim=0)

    def get_recurrent_weights(self):
        """
        Function that returns the weight that effect the Recurrent input of the
        Network

        Returns
        -------
        recurrent_weights : list of weight Tensors
            List of the Weights that effect the Recurren input of the Network.

        """
        return _get_rec_weights_rnn(self)

    def init_hidden(self, hidden=(None, None)):
        """
        Initialize the hidden state

        Parameters
        ----------
        hidden : (None , None) or (tensor, tensor), optional
            if None, hidden state is initialized as zero.
            if hidden is Tensor the hidden state is initialized as the given
            TensorThe default is None.

        Returns
        -------
        None.

        """
        if torch.is_tensor(hidden[0]):
            self.h_0 = Variable(hidden[0])
            self.c_0 = Variable(hidden[1])
        else:
            self.h_0 = Variable(torch.zeros(
                self.num_layers, 1, self.hidden_size))
            self.c_0 = Variable(torch.zeros(
                self.num_layers, 1, self.hidden_size))


class _Parallel_Conv(nn.Module):
    """
    Parallel convolution with increasing dilation, the resultingwindow size
    is computed using:

        .. math:: target_window = kernel_size * oscillations - (kernel_size - 1)
        .. math:: window_size = (target_window-1)*dil + dil*(kernel-1) + 1
        .. math:: Lenght_out = kernel_size * oscillations - (kernel_size - 1)*depth

    target window and window_size are furthermore saved as internal
    variables

    Parameters
    ----------
    input_channels : int
        Number of channels of the external excitation signal
    filters : int or list of int
        Number of filters used in the convolution
    kernel_size : int
        Width of the filter
    max_dilation : int
        maximum dilation in the parallel convolution. The default is 4
    oscillations : int
        Number of full oscillations for each parallel convolution is
        processed. The default is 4
    activation : str, optional
        Activation function to activate the feature space.
        The default is 'relu'.
    bias : bool, optional
        If True, bias weights are used. The default is True.

    Returns
    -------
    None.

    """

    def __init__(self, in_channels, filters, kernel_size, max_dilation=4,
                 oscillations=4, activation='relu', bias=True):

        super(_Parallel_Conv, self).__init__()
        self.Lout = kernel_size * oscillations - (kernel_size - 1)
        self.window_size = self._comp_ws(max_dilation, kernel_size)

        self.ws = []
        self.Convs = nn.ModuleList()
        for dil in range(1, max_dilation+1):
            self.ws.append(self._comp_ws(dil, kernel_size))
            self.Convs.append(_delated_conv(in_channels, filters, kernel_size,
                                           dil, activation, bias))

    def forward(self, x):
        """
        Forward Propagation of the Convolutional Block

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation

        Returns
        -------
        output: torch.tensor dtype=torch.float()
            Output tensor shape [batch_size, filters, self.window_size]

        """

        outs = []
        for conv, window in zip(self.Convs, self.ws):
            outs.append(conv(x[:, :, :window]))
        out = torch.cat(outs, dim=1)
        return out

    def _comp_ws(self, dil, kernel):
        """
        Internal function to compute window size

        Parameters
        ----------
        dil : int
            dilation between the individual elements of the kernel.
        kernel : int
            kernel_size.

        Returns
        -------
        int
            window size to archieve self.target_window as the output of the
            convolutions.

        """
        return (self.Lout-1)*dil + dil*(kernel-1) + 1


class _delated_conv(nn.Module):
    """
    Convolutional Network with time dilated first Layer

    Parameters
    ----------
    input_channels : int
        Number of channels of the external excitation signal
    kernel_size : int or list of int
        Width of the filter
    dlation : int
        dilation of the first layer
    filters : int or list of int
        Number of filters used in the convolution
    activation : str
        Activation function to activate the feature space.
    bias : bool, optional
        If True, bias weights are used. The default is True.

    Returns
    -------
    None.
    """

    def __init__(self, in_channels, filters, kernel_size, dilation,
                 activation='relu', bias=True):

        super(_delated_conv, self).__init__()
        self.activation = activation
        self.ConvNet = []

        self.ConvNet.append(nn.Conv1d(in_channels, filters, kernel_size,
                                      dilation=dilation, stride=dilation,
                                      bias=bias))
        self.ConvNet.append(_activation_function(self.activation))

        self.ConvNet = nn.Sequential(*self.ConvNet)

    def forward(self, x):
        """
        Forward Propagation of the Convolutional Block

        Parameters
        ----------
        inp : torch.tensor dtype=torch.float
            Input tensor for forward propagation

        Returns
        -------
        output: torch.tensor dtype=torch.float()

        """
        x = self.ConvNet(x)
        return x


def _get_rec_weights_rnn(model):
    """
    Function that returns the weight that effect the Recurrent input of the
    Network

    Parameters
    ----------
    model : model
        Recurretn Model with recurretn weights.

    Returns
    -------
    recurrent_weights : list of weight Tensors
        List of the Weights that effect the Recurrent input of the Network.

    """
    recurrent_weights = []
    for name, W in model.named_parameters():
        if 'weight_hh' in name:
            recurrent_weights.append(W)
    return recurrent_weights


def _filter_parameters(params):
    """
    Filters 'self' and '__class__' from parameterdict

    Parameters
    ----------
    params : dict
        dict with named parameters.

    Returns
    -------
    params : dict
        dict with named parameters.

    """
    [params.pop(key) for key in ['self', '__class__'] if key in params.keys()]
    return params


def _activation_function(act):
    """
    Activate latent features, activation function ist choosen in __init__()

    Parameters
    ----------
    inp : torch.tensor dtype=torch.float()
        features to be activated

    Returns
    -------
    output: torch.tensor dtype=torch.float()
        activated input

    """
    if act == 'relu':
        return nn.ReLU()
    if act == 'sigmoid':
        return nn.Sigmoid()
    if act == 'tanh':
        return nn.Tanh()
    if act == 'sine':
        return Sine()
    if act == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise ValueError('No valid activation function name given')


class Sine(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, Tensor):
        return torch.sin(Tensor)
