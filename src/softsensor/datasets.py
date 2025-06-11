# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:16:57 2023

The methods and classes considered define the type of data set used. They are
Sliding Windows over the individual time series out of pd.DataFrames
@author: Tobias Westmeier CR/AMP4
"""
import torch
import math
import numpy as np
from torch.utils.data import Dataset


class SlidingWindow(Dataset):
    """
    Initialize a sliding window class from a pandas Dataframe in
    torch.utils.Dataset format with a tuple of input and output data
    The dataframe is split into individual tensors with a length of windowsize
    and width if len(input_columns) using a sliding window approach for the
    input data
    if rnn_window is used, two tensors are generated for each time step, where
    the additional tensor defines the past output data with length of
    rnn_window and width of len(output_columns)

    Parameters
    ----------
    df : pd.DataFrame
        pandas DataFrame with named columns and time dependent data.
    windowsize : int
        sliding window length.
    output_columns : list of str
        list of columns used for output.
    input_columns : list of str
        list of columns used for input.
    Add_zeros : TYPE, optional
        Adds zeros at the beginning of the time series to generate a dataset
        with as many inputs as the length of the windowsize.
        The default is False.
    rnn_window : int, optional
        use an additional sliding window as input in which the past output
        values are saved. The default is None.
    pre_comp : list of str, optional
        precomputed solution for models with student forcing used for second
        input instead of output columns. The default is None.

    Returns
    -------
    None

    Notes
    -----
    The use of a recurrent time window (rnn_window) ensures that zeros are
    artificially placed at the beginning of the time series to keep the
    dimension of the time series constant in the output
    SlidingWindow.__len__() returns length of df). If a rnn_window is
    specified, the time series is shortened. If the output length is to be
    kept, Add_zeros=True must be used.
    

    Examples
    --------
    Example of a pure feed forward SlidingWindow dataset 
    
    >>> import softsensor.datasets as ds
    >>> import pandas as pd
    >>> import numpy as np
    >>> d = {'in_col': np.linspace(0, 100, 101),
             'out_col': np.linspace(100, 0, 101)}
    >>> df = pd.DataFrame(d)
    >>> sw = ds.SlidingWindow(df, 10, ['out_col'], ['in_col'])
    >>> print(sw.__len__())
    92
    >>> print(sw.__getitem__(1))
    (tensor([[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]]), tensor([[91.]]))

    Example of a  SlidingWindow dataset with recurrent connection
    
    >>> import softsensor.datasets as ds
    >>> import pandas as pd
    >>> import numpy as np
    >>> d = {'in_col': np.linspace(0, 100, 101),
             'in_col2': np.linspace(0, 100, 101),
             'out_col': np.linspace(100, 0, 101)}
    >>> df = pd.DataFrame(d)
    >>> sw = ds.SlidingWindow(df, 3, ['out_col'], ['in_col'], rnn_window=3)
    >>> print(sw.__len__())
    101
    >>> print(sw.__getitem__(2))
    ((tensor([[0., 1., 2.]]), tensor([[  0.,   100., 99.]])), tensor([[98.]]))
    
    
    Example of a  SlidingWindow dataset with recurrent connection and
    precomputed prediction
    
    >>> import softsensor.datasets as ds
    >>> import pandas as pd
    >>> import numpy as np
    >>> d = {'in_col': np.linspace(0, 100, 101),
             'in_col2': np.linspace(0, 100, 101),
             'out_col': np.linspace(100, 0, 101), 
             'out_col_precomp': np.linspace(100, 0, 101) + 100}
    >>> df = pd.DataFrame(d)
    >>> sw = ds.SlidingWindow(df, 3, ['out_col'], ['in_col'], rnn_window=3, 
                              pre_comp=['out_col_precomp'])
    >>> print(sw.__len__())
    101
    >>> print(sw.__getitem__(2))
    ((tensor([[0., 1., 2.]]), tensor([[  0., 200., 199.]])), tensor([[98.]]))
    """

    def __init__(self, df, windowsize, output_columns, input_columns,
                 Add_zeros=False, rnn_window=None, forecast=1, full_ds=True,
                 pre_comp=None):

        self.windowsize = windowsize
        self.rnn_window = rnn_window
        self.num_inp = len(input_columns)
        self.num_out = len(output_columns)
        self.forecast = forecast  
        self.pre_comp = pre_comp
                
        self.pre = True if pre_comp is not None else False

        # Read in important data as tensors
        self.data_y = df[output_columns]
        self.data_y = torch.transpose(torch.tensor(self.data_y.values), 0, 1)
        self.data_x = df[input_columns]
        self.data_x = torch.transpose(torch.tensor(self.data_x.values), 0, 1)
        
        if self.pre:
            self.data_y_pre = df[pre_comp]
            self.data_y_pre = torch.transpose(torch.tensor(self.data_y_pre.values), 0, 1)
        else:
            self.data_y_pre = None

        if rnn_window is None:
            self.subclass = ff_SlidingWindow(self.windowsize, self.data_x,
                                             self.data_y, Add_zeros, forecast,
                                             full_ds)
        else:
            self.subclass = rec_SlidingWindow(self.windowsize, self.data_x,
                                              self.data_y, rnn_window,
                                              forecast, full_ds, self.data_y_pre, self.pre)

    def __getitem__(self, index):
        """
        Takes index and gives back time window for input and subsequent target
        output

        Parameters
        ----------
        index : int: the starting point for the time window

        Returns
        -------
        x : torch.Tensor
            if rnn_window is None: Tensor of shape
            [len(input_columns), windowsize]
            if rnn_windwo is not None: tuple of Tensors with shape
            ([len(input_columns), windowsize], [len(output_columns), rnn_window])
        y : torch.Tensor
            Tensor of shape [output_channels, 1]

        """
        return self.subclass.__getitem__(index)

    def __len__(self):
        """

        Returns
        -------
        number_of_samples : int
            Samples in Dataset

        """
        return self.subclass.__len__()


class rec_SlidingWindow(Dataset):
    """
    Subclass for SlidingWindow class
    two tensors are generated for each time step, where one tensor defines the
    input x  and the second one the past output data with length of
    rnn_window and width of len(output_columns)

    Parameters
    ----------
    windowsize : int
        sliding window length.
    data_x : torch.tensor
        input data
    data_y : torch.tensor
        output data as well as data that serves as second input 
    rnn_window : int, optional
        use an additional sliding window as input in which the past output
        values are saved. The default is None.

    Returns
    -------
    None

    """
    def __init__(self, windowsize, data_x, data_y, rnn_window, forecast,
                 full_ds, data_y_pre=None, pre=False):
        self.windowsize = windowsize
        self.rnn_window = rnn_window
        self.forecast = forecast
        self.data_x = data_x
        self.data_y = data_y
        self.full_ds = full_ds
        self.orig_length = self.data_x.shape[1]
        
        self.pre = pre
        
        if self.pre:
            self.data_y_pre = data_y_pre

        # use modulus to define datast length that is divisable by forecast
        if rnn_window > (windowsize-forecast):
            self.size = self.rnn_window
            self.offset = self.rnn_window - windowsize + forecast
        else:
            self.size = self.windowsize - forecast
            self.offset = 0

        # Add Zeros to the beginning of input and output data
        zeros_inp_begin = torch.zeros((self.data_x.shape[0], self.size))
        zeros_out_begin = torch.zeros((self.data_y.shape[0], self.size))

        # Add zeros at the end to match dimensions for forcasting horizon
        self.mod = self.data_x.shape[1] % forecast
        if self.mod == 0:
            self.add_zero = 0
            self.data_x = torch.cat((zeros_inp_begin, self.data_x), dim=1)
            self.data_y = torch.cat((zeros_out_begin, self.data_y), dim=1)
            
            if self.pre:
                self.data_y_pre = torch.cat((zeros_out_begin, self.data_y_pre), dim=1)
        else:
            self.add_zero = forecast - self.mod
            zeros_inp_end = torch.zeros((self.data_x.shape[0], self.add_zero))
            zeros_out_end = torch.zeros((self.data_y.shape[0], self.add_zero))

            self.data_x = torch.cat((zeros_inp_begin,
                                     self.data_x,
                                     zeros_inp_end), dim=1)
            self.data_y = torch.cat((zeros_out_begin,
                                     self.data_y,
                                     zeros_out_end), dim=1)
            if self.pre:
                self.data_y_pre = torch.cat((zeros_out_begin,
                                         self.data_y_pre,
                                         zeros_out_end), dim=1)
    def __getitem__(self, index):
        """
        Takes index and gives back time window for input and subsequent target
        output

        Parameters
        ----------
        index : int: the starting point for the time window

        Returns
        -------
        x : torch.Tensor
            tuple of Tensors with shape
            ([len(input_columns), windowsize], [len(output_columns), rnn_window])
        y : torch.Tensor
            Tensor of shape [output_channels, 1]
        """
        if not self.full_ds:
            index = index * self.forecast

        index = index + self.offset
        x1 = self.data_x[:, index:index+self.windowsize]
        
        if self.pre:
            x2 = self.data_y_pre[:,
                             (index+self.windowsize-self.rnn_window-self.forecast):
                             (index+self.windowsize-self.forecast)]
        else:
            x2 = self.data_y[:,
                             (index+self.windowsize-self.rnn_window-self.forecast):
                             (index+self.windowsize-self.forecast)]

        x = (x1.float(), x2.float())
        y = self.data_y[:, index + self.windowsize - self.forecast:
                        index + self.windowsize]
        y = y.float()

        return x, y

    def __len__(self):
        """

        Returns
        -------
        number_of_samples : int
            Samples in Dataset

        """

        if self.mod == 0:
            samples = self.orig_length - self.forecast + 1
        else:
            samples = self.orig_length - self.mod + 1
        if not self.full_ds:
            samples = math.ceil(samples / self.forecast)
        return samples


class ff_SlidingWindow(Dataset):
    """
    Subclass for SlidingWindow class (feed forward)
    two tensors are generated for each time step, where one tensor defines the
    input x  and the second one the past output data with length of
    rnn_window and width of len(output_columns)

    Parameters
    ----------
    windowsize : int
        sliding window length.
    data_x : torch.tensor
        input data
    data_y : torch.tensor
        output data
    Add_zeros : bool
        Adds zeros at the beginning of the time series with length of
        windowsize-1.
    Returns
    -------
    None

    """

    def __init__(self, windowsize, data_x, data_y, Add_zeros, forecast,
                 full_ds):
        self.windowsize = windowsize
        self.forecast = forecast
        self.data_x = data_x
        self.data_y = data_y
        self.orig_length = self.data_x.shape[1]
        self.Add_zeros = Add_zeros
        self.full_ds = full_ds

        if Add_zeros:
            # Add Zeros to the beginning of input and data
            zeros_inp_begin = torch.zeros((self.data_x.shape[0],
                                           windowsize - forecast))
            zeros_out_begin = torch.zeros((self.data_y.shape[0],
                                           windowsize - forecast))
            self.data_x = torch.cat((zeros_inp_begin, self.data_x), dim=1)
            self.data_y = torch.cat((zeros_out_begin, self.data_y), dim=1)

        # Add zeros at the end to match dimensions for forcasting horizon
        self.mod = (self.data_x.shape[1] - windowsize + forecast) % forecast
        if self.mod != 0:
            self.add_zero = forecast - self.mod
            zeros_inp_end = torch.zeros((self.data_x.shape[0], self.add_zero))
            zeros_out_end = torch.zeros((self.data_y.shape[0], self.add_zero))

            self.data_x = torch.cat((self.data_x,
                                     zeros_inp_end), dim=1)
            self.data_y = torch.cat((self.data_y,
                                     zeros_out_end), dim=1)
        else:
            self.add_zero = 0

    def __getitem__(self, index):
        """
        Takes index and gives back time window for input and subsequent target
        output

        Parameters
        ----------
        index : int: the starting point for the time window

        Returns
        -------
        x : torch.Tensor
            shape: [len(input_columns), windowsize]
        y : torch.Tensor
            Tensor of shape [output_channels, 1]

        """
        if not self.full_ds:
            index = index * self.forecast
        x = self.data_x[:, index:index+self.windowsize]
        y = self.data_y[:, index + self.windowsize - self.forecast:
                        index + self.windowsize]

        x, y, = x.float(), y.float()

        return x, y

    def __len__(self):
        """

        Returns
        -------
        number_of_samples : int
            Samples in Dataset

        """
        if self.Add_zeros:
            if self.mod == 0:
                samples = self.orig_length - self.forecast + 1
            else:
                samples = self.orig_length - self.mod + 1
        else:
            if self.mod == 0:
                samples = self.orig_length - self.windowsize + 1
            else:
                samples = self.orig_length - self.windowsize + 1 + self.forecast - self.mod
        if not self.full_ds:
            samples = math.ceil(samples / self.forecast)
        return samples


class batch_rec_SW(Dataset):
    """
    Batching of multiple sliding window classes for parallelisation purposes

    Parameters
    ----------
    list_of_sw: list of SlidingWindow
        list of individual SlidingWindow classes for batching

    Returns
    -------
    None.

    """

    def __init__(self, list_of_sw):
        # directly use the subclass instead of the wrapper
        self.sws = [sw.subclass for sw in list_of_sw]

        # Get some important parameters from class
        self.size = self.sws[0].size
        self.forecast = self.sws[0].forecast
        self.num_inp = list_of_sw[0].num_inp
        self.num_out = list_of_sw[0].num_out
        self.offset = self.sws[0].offset
        self.windowsize = self.sws[0].windowsize
        self.rnn_window = self.sws[0].rnn_window

        self.lengths = [sw.__len__() for sw in list_of_sw]

        self.sws = [x for _, _, x in sorted(zip(self.lengths,
                                                enumerate(self.sws),
                                                self.sws))][::-1]

        self.original_ind = np.flip(np.argsort(self.lengths))

        self.lengths = sorted(self.lengths)[::-1]
        self.valid_sws = []
        num = len(self.lengths)
        ii = 0

        for i in self.lengths[::-1]:
            self.valid_sws = self.valid_sws + (i - ii) * [num]
            ii = i
            num = num - 1

        self.data_x = torch.full((self.valid_sws[0], self.num_inp,
                                   self.lengths[0]*self.forecast+ self.size),
                                 float('nan'))
        self.data_y = torch.full((self.valid_sws[0], self.num_out,
                                  self.lengths[0]*self.forecast + self.size),
                                 float('nan'))

        for i, sw in enumerate(self.sws):
            self.data_x[i, :,
                        :sw.__len__()*self.forecast + self.size] = sw.data_x
            self.data_y[i, :,
                        :sw.__len__()*self.forecast + self.size] = sw.data_y

    def __getitem__(self, index):
        """
        Takes index and gives back time window for input and subsequent target
        output. Ouput is batches for individual time series.
        
        batch_size depends on the numbr of SlidingWindow datasaets that have
        .__len__() >= index

        Parameters
        ----------
        index : int: the starting point for the time window

        Returns
        -------
        
        x : (torch.Tensor, torch.Tensor) 
                Tensor with shape
                ([batch_size, len(input_columns), windowsize], [batch_size, len(output_columns), rnn_window])

        y : torch.Tensor
                Tensor of shape [batch_size, output_channels, 1]

        """
        ind = index
        index = index * self.forecast + self.offset
        x1 = self.data_x[:self.valid_sws[ind], :,
                         index:index+self.windowsize]

        x2 = self.data_y[:self.valid_sws[ind], :,
                         (index+self.windowsize-self.rnn_window-self.forecast):
                             (index+self.windowsize - self.forecast)]

        x = (x1.float(), x2.float())

        y = self.data_y[:self.valid_sws[ind], :,
                        index + self.windowsize - self.forecast:
                        index + self.windowsize]
        y = y.float()

        return x, y

    def __len__(self):
        """

        Returns
        -------
        number_of_samples : int
            Samples in Dataset

        """
        return self.lengths[0]

    def __lengths__(self):
        """


        Returns
        -------
        list of int
            sorted list with length of each time series considered.

        """
        return self.lengths

    def __widths__(self):
        """


        Returns
        -------
        list of int dim:[self.__len__()]
            number of SlidingWindow classes for which the index is valid.

        """
        return self.valid_sws

    def permutation(self):
        """


        Returns
        -------
        list of int dim:[len(list_of_sw)]
            returns original indizes before permutation

        """
        return self.original_ind
