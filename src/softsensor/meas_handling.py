# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:08:01 2021

@author: WET2RNG
"""

import pandas as pd
import numpy as np
from torch.utils.data import ConcatDataset, random_split, DataLoader
from torch.utils.data import Subset
from sklearn.preprocessing import StandardScaler
from pylife.stress.timesignal import butter_bandpass
from scipy import interpolate

from softsensor.datasets import SlidingWindow


class Meas_handling:
    """
    Measurement handling class that can be used for the
    whole data preprocessing

    Parameters
    ----------
    train_dfs : List of pd.DataFrames
        List of pd.DataFrame for every Measurement used as Training data
        for subsequent models.
    train_names : List of str
        List of Track Names corresponding to the train_dfs. Need to have the
        same length as train_dfs
    input_sensors : List of str
        Input sensors for the subsequent models, The order of the str defines
        the order in which the data is present inside the loader
    output_sensors : List of str
        Output sensors for the subsequent models, The order of of the str
        defines the order in which the data is present inside the loader
    fs : int
        sample frequency.
    test_dfs : List of pd.DataFrames, optional
        List of pd.DataFrame for every Measurement used as Testing data
        for subsequent models. The default is None
    test_names : List of str, optional
        List of Track Names corresponding to the test_dfs. Need to have the
        same length as test_dfs. The default is None
    pre_comp_cols : List of str, optional
        Defines wheather and which a precomputed solution is in the dataset.
        Precomputed solutions might be helpfull in certain training tasks
        The default is None

    Returns
    -------
    None.

    Examples
    -------
    Define a Measurment Handling class

    >>> import softsensor.meas_handling as ms
    >>> import pandas as pd
    >>> import numpy as np
    >>> t = np.linspace(0, 1.0, 10001)
    >>> xlow = np.sin(2 * np.pi * 100 * t)
    >>> xhigh = 0.2 * np.sin(2 * np.pi * 3000 * t)
    >>> d = {'sine_inp': xlow + .1 * xhigh,
             'cos_inp': np.cos(2 * np.pi * 50 * t),
             'out': np.linspace(0, 1.0, 10001)}
    >>> t = np.linspace(0, 1.0, 10001)
    >>> test_df = {'sine_inp': 10*xlow + .1 * xhigh,
                   'cos_inp': np.cos(2 * np.pi * 50 * t),
                   'out': np.linspace(0, 1.0, 10001)}
    >>> handler = ms.Meas_handling([pd.DataFrame(d, index=t), pd.DataFrame(d, index=t)],
                                ['sine1', 'sine2'],
                                ['sine_inp', 'cos_inp'], ['out'], 10000,
                                [pd.DataFrame(test_df, index=t)], ['test'])

    """

    def __init__(
        self,
        train_dfs,
        train_names,
        input_sensors,
        output_sensors,
        fs,
        test_dfs=None,
        test_names=None,
        pre_comp_cols=None,
    ):

        self.train_names = train_names
        self.test_names = test_names
        self.input_sensors = input_sensors
        self.output_sensors = output_sensors
        self.pre_comp_cols = pre_comp_cols
        self.fs = fs

        self.train_df = train_dfs
        self.test_df = test_dfs

        self.scaler = None
        self.Scaled = False

        # check that input is valid
        self._check_input(train_dfs, train_names, input_sensors + output_sensors)
        if (test_dfs is not None) or (test_names is not None):
            self._check_input(test_dfs, test_names, input_sensors + output_sensors)
            names = train_names + test_names
            self.test_data = True
        else:
            names = train_names
            self.test_data = False

        self._check_naming(names)

    def Scale(self, scaler=StandardScaler(), predef_scaler=False):
        """
        Scale the Data. The scaler is fitted only on the traindata. Afterwards
        train and testdata is transformed.

        Parameters
        ----------
        scaler : sklearn.preprocessing scaler, optional
            The default is StandardScaler().
        predef_scaler : bool, optional
            if True, scaler needs to be fitted already and will be used to
            scale data. This might come in handy if the data used for scaling
            is no longer available or special scaling procedures are required
            The default is False
        Returns
        -------
        None.

        Examples
        -------
        Based on the Example from class initialisation

        >>> df = handler.give_dataframe('sine1')
        >>> print(np.var(df['sine_inp'].values))
        >>> handler.Scale()
        >>> print(np.var(df['sine_inp'].values))
        1.0

        """

        if predef_scaler:
            self.scaler = scaler
        else:
            dfs = [d[self.input_sensors + self.output_sensors] for d in self.train_df]
            self.scaler = scaler.fit(pd.concat(dfs))

        for i, df in enumerate(self.train_df):
            df = df[self.input_sensors + self.output_sensors]
            self.train_df[i][self.input_sensors + self.output_sensors] = (
                self.scaler.transform(df)
            )

        if self.test_data:
            for i, df in enumerate(self.test_df):
                df = df[self.input_sensors + self.output_sensors]
                self.test_df[i][self.input_sensors + self.output_sensors] = (
                    self.scaler.transform(df)
                )

        self.Scaled = True

    def Resample(self, fs, kind="linear"):
        """
        Resample self.train_df and self.test_df to fs using Fourier method
        along the given axis.

        Parameters
        ----------
        fs : float or int
            fs to resample data to.
        kind : str
            Specifies the kind of interpolation as a string or as an integer
            specifying the order of the spline interpolator to use. The string
            has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’,
            ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’,
            ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation
            of zeroth, first, second or third order; ‘previous’ and ‘next’
            simply return the previous or next value of the point; ‘nearest-up’
            nd ‘nearest’ differ when interpolating half-integers
            (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’
            down. Default is ‘linear’. see interpolate.interp1d docu
        Returns
        -------
        None.

        """
        self.fs = fs
        for i, df in enumerate(self.train_df):
            new_t = np.arange(df.index[0], df.index[-1], 1 / fs)
            new_data = {
                col: interpolate.interp1d(df.index, df[col])(new_t) for col in df
            }
            self.train_df[i] = pd.DataFrame(new_data, index=new_t)

        if self.test_data:
            for i, df in enumerate(self.test_df):
                new_t = np.arange(df.index[0], df.index[-1], 1 / fs)
                new_data = {
                    col: interpolate.interp1d(df.index, df[col])(new_t) for col in df
                }
                self.test_df[i] = pd.DataFrame(new_data, index=new_t)

    def Filter(self, freq_lim):
        """
        Function to bandpass Filter the Sensor data

        Parameters
        ----------
        freq_lim : (low_cut, high cut)
            Defining the low and high cut for bandpass filtering

        Returns
        -------
        None.

        Examples
        -------
        Based on the Example from class initialisation,
        Result should be roughly zero

        >>> handler.Filter(freq_lim=(10, 700))
        >>> filtered_sine = handler.train_df[0]['sine_inp'].values
        >>> dev = xlow - filtered_sine
        >>> print(np.mean(dev))
        0.0
        """

        for i, df in enumerate(self.train_df):
            self.train_df[i] = butter_bandpass(df, freq_lim[0], freq_lim[1])

        if self.test_data:
            for i, df in enumerate(self.test_df):
                self.test_df[i] = butter_bandpass(df, freq_lim[0], freq_lim[1])

    def fade_in(self, window_length, window_type="hanning", columns=None):
        """
        Apply the defined window to the train and test data to fade in.

        Parameters
        ----------
        window_type: str
            Defining the window function type.
            Supported window types are: hanning, hamming, blackman, bartlette
        window_length : int
            Defining the window length of the half window.
        Returns
        -------

        Examples
        -------
        Based on the Example from class initialisation,
        Result should be roughly zero

        >>> print(handler.test_df[0]['cos_inp'][0])
        1.0
        >>> handler.fade_in(10)
        >>> print(handler.test_df[0]['cos_inp'][0])
        0.0
        """
        if columns is None:
            columns = [
                self.train_df[0].columns.get_loc(c)
                for c in self.input_sensors + self.output_sensors
            ]

        window = self._get_window(window_type, window_length * 2)
        # use only the half window
        window_in = window[:window_length]
        window_out = window[:-window_length]

        for i, df in enumerate(self.train_df):
            self.train_df[i].iloc[:window_length, columns] = df.iloc[
                :window_length, columns
            ].mul(window_in, axis=0)
            # self.train_df[i].iloc[-window_length:, columns] = df.iloc[-window_length:, columns].mul(window_out, axis=0)

        if self.test_df is None:
            return None

        for i, df in enumerate(self.test_df):
            self.test_df[i].iloc[:window_length, columns] = np.array(
                df.iloc[:window_length, columns].mul(window_in, axis=0)
            )
            # elf.test_df[i].iloc[-window_length:, columns] = np.array(df.iloc[-window_length:, columns].mul(window_out, axis=0))

    @staticmethod
    def _get_window(window_type: str, window_length: int) -> np.ndarray:
        """
        Function to return a discrete window function.

        Parameters
        ----------
        window_type: str
            Defining the window function type.
            Supported window types are: Hanning, Hamming, Blackman
        window_length : int
            Defining the window length of the window.

        Returns
        -------
        Output: np.ndarray
            Returns the discrete window function.
        """
        window_type = window_type.lower()
        if window_type == "hanning":
            return np.hanning(window_length)
        if window_type == "hamming":
            return np.hamming(window_length)
        if window_type == "blackman":
            return np.blackman(window_length)
        if window_type == "bartlett":
            return np.bartlett(window_length)

        print(
            f'Unsupported window type: {window_type}! Use the default window type "Hanning"'
        )
        return np.hanning(window_length)

    def give_torch_loader(
        self,
        window_size,
        keyword="training",
        train_ratio=0.8,
        batch_size=32,
        rnn_window=None,
        shuffle=False,
        Add_zeros=True,
        forecast=1,
        full_ds=False,
        pre_comp=False,
        n_samples=[5000, 1000],
    ):
        """
        Gives back a torch dataloader for training, evaluation or testing
        purpose

        Parameters
        ----------
        window_size : int
            Window size for input and output series.
        keyword : str, optional
            possibilities are 'training', 'testing', 'short' or ['Name'].
            'training' gives a training and validation dataloader using
            all training data.
            'testing' gives a single dataloader using all test data.
            'short' gives a training loader with 5000 samples and a
            validation loader with 1000 samples.
            ['Name'], list of names, names must be present in either
            train_names or test_names
            The default is 'training'.
        train_ratio : float, optional
            only needed for keyword 'training'. Defines the ration of training
            data compared to validation data. The default is .8.
        batch_size : int, optional
            Batchsize for the dataloader. The default is 32.
        rnn_window : int, optional
            Window size of the recurrent window. The default is None.
        shuffle : bool, optional
            Loader is shuffled or not. The default is False.
        Add_zeros : bool, optional
            only needed if rnn_window is False, Adds zeros to the beginning of
            the time series. The default is True
        forecast : int, optional
            forcasting horizon
        pre_comp : bool, optional
            using precomputed solution or not. The default is False.

        Returns
        -------
        torch.dataloader
            one dataloader if train_ratio  = 1, otherwise two dataloaders for
            training and validation

        """

        # get data from keyword
        if keyword == "short":
            set_list = self.give_Datasets(
                window_size,
                "training",
                rnn_window,
                Add_zeros,
                forecast,
                full_ds,
                pre_comp,
            )
        else:
            set_list = self.give_Datasets(
                window_size, keyword, rnn_window, Add_zeros, forecast, full_ds, pre_comp
            )

        dataset = ConcatDataset(set_list)

        # Define data loaders
        if keyword == "short":
            return self._split_dataset(
                dataset,
                train_ratio,
                batch_size,
                shuffle,
                short=True,
                n_samples=n_samples,
            )
        else:
            return self._split_dataset(
                dataset, train_ratio, batch_size, shuffle, short=False
            )

    def _split_dataset(
        self, dataset, ratio, batch_size, shuffle, short=False, n_samples=[5000, 1000]
    ):
        """
        Internal class to split a given dataset in training and evaluation sets

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            dataset to split.
        ratio : float. (0, 1]
            ratio to split dataset in training and validation.
            ratio = 1: dataset isn't split at all
            ratio < 1: dataset is split with ratio as the percentage of data
            in the first loader
        batch_size : int
            Batchsize for the dataloader.
        shuffle : bool
            Loader is shuffled or not.
        short : bool, optional
            gives back a short dataloader with 500 samples for training and
            1000 for validation. The default is False.

        Returns
        -------
        torch.dataloader
            one dataloader if train_ratio  = 1, otherwise two dataloaders for
            training and validation

        """
        if ratio != 1:
            length_of_set = dataset.__len__()

            if short:
                train_samples = n_samples[0]
                val_samples = n_samples[1]
                _samples = length_of_set - train_samples - val_samples
                train_set, val_set, _ = random_split(
                    dataset, (train_samples, val_samples, _samples)
                )
            else:
                train_samples = int(length_of_set * ratio)
                val_samples = length_of_set - train_samples
                train_set, val_set = random_split(dataset, (train_samples, val_samples))

            train_loader = DataLoader(train_set, batch_size, shuffle=shuffle)
            val_loader = DataLoader(val_set, batch_size, shuffle=shuffle)
            return (train_loader, val_loader)
        else:
            if short:
                dataset = Subset(dataset, np.arange(n_samples[0]))
            train_loader = DataLoader(dataset, batch_size, shuffle=shuffle)
            return train_loader

    def give_list(
        self,
        window_size,
        keyword="testing",
        batch_size=32,
        Add_zeros=False,
        rnn_window=None,
        forecast=1,
        full_ds=False,
    ):
        """
        Gives List of DataLoader

        Parameters
        ----------
        window_size : int
            Window size for input and output series.
        keyword : str, optional
            possibilities are 'training', 'testing', 'short' or [Name].
            'training' gives a training and validation dataloader using
            all training data.
            'testing' gives a single dataloader using all test data.
            'short' gives a training loader with 5000 samples and a
            validation loader with 1000 samples.
            [Name], gives back a unshuffled loader corresponding to the track
            name define in init
            The default is 'training'.
        batch_size : int, optional
            Batchsize for the dataloader. The default is 32.
        Add_zeros : bool, optional
            Appends zeros at the beginning for Autoregressive models.
            The default is False.
        rnn_window : int, optional
            Window size of the recurrent window. The default is None.
        forecast : int, optional
            forecasting horizon
        Returns
        -------
        list_loader : list of torch.dataloader
            list of dataloaders with individual Measurements.

        """

        set_list = self.give_Datasets(
            window_size, keyword, rnn_window, Add_zeros, forecast, full_ds
        )

        list_loader = [DataLoader(s, batch_size, shuffle=False) for s in set_list]
        return list_loader

    def give_Datasets(
        self,
        window_size,
        keyword="training",
        rnn_window=None,
        Add_zeros=True,
        forecast=1,
        full_ds=False,
        pre_comp=False,
    ):
        """
        Gives List of DataSets

        Parameters
        ----------
        window_size : int
            Window size for input and output series.
        keyword : str, optional
            possibilities are 'training', 'testing', 'short' or [Name].
            'training' gives a training and validation dataloader using
            all training data.
            'testing' gives a single dataloader using all test data.
            'short' gives a training loader with 5000 samples and a
            validation loader with 1000 samples.
            [Name], gives back a unshuffled loader corresponding to the track
            name define in init
            The default is 'training'.
        Add_zeros : bool, optional
            Appends zeros at the beginning for Autoregressive models.
            The default is False.
        rnn_window : int, optional
            Window size of the recurrent window. The default is None.
        forecast : int, optional
            forcasting horizon
        Returns
        -------
        set_list : list of SlidingWindow Datasets
            list of Datsets with individual Measurements.

        """
        if keyword == "testing":
            if self.test_data:
                temp_df = [self.give_dataframe(k) for k in self.test_names]
            else:
                raise ValueError(
                    "keyword testing given, but no test_df is " + "defined"
                )
        elif keyword == "training":
            temp_df = [self.give_dataframe(k) for k in self.train_names]
        else:
            temp_df = [self.give_dataframe(k) for k in keyword]
        set_list = []

        pre = self.pre_comp_cols if pre_comp else None

        for df in temp_df:
            set_list.append(
                SlidingWindow(
                    df,
                    window_size,
                    self.output_sensors,
                    self.input_sensors,
                    Add_zeros,
                    rnn_window,
                    forecast,
                    full_ds,
                    pre,
                )
            )
        return set_list

    def give_dataframes(self, keywords):
        """
        Returns list f dataframes

        Parameters
        ----------
        keywords : list of str or 'training' or 'testing'
            rdefines which dataframes will be returned.
            list of str return list of same length with dataframes
            'training' or 'testing' return list of dfs in training / testing

        Returns
        -------
        dfs : list of dfs
            List of Dataframes.

        """

        if keywords == "testing":
            dfs = [self.give_dataframe(k) for k in self.test_names]
        elif keywords == "training":
            dfs = [self.give_dataframe(k) for k in self.train_names]
        else:
            dfs = [self.give_dataframe(k) for k in keywords]
        return dfs

    def give_dataframe(self, Name):
        """
        Gives back Dataframe corresponding to the specific name

        Parameters
        ----------
        Name : str
            String that matches train or test name defined in init.

        Returns
        -------
        df : pd.DataFrame
            DataFrame that corresponds to the given Name.

        """
        idx, Train = self._get_idx(Name)
        if Train:
            df = self.train_df[idx]
        else:
            df = self.test_df[idx]
        return df

    def _get_idx(self, name):
        """
        Gives index of df name in df list

        Parameters
        ----------
        Name : str
            Name of the df the index is needed.

        Returns
        -------
        idx : int
            index in df list.
        train : bool
            True if name is in training names, False if in test names.

        """

        idx = None
        if name in self.train_names:
            idx = self.train_names.index(name)
            train = True
            return idx, train

        if self.test_data:
            if name in self.test_names:
                idx = self.test_names.index(name)
                train = False
                return idx, train

        raise ValueError(f"{name} not in train_names or test_names")

    def _check_input(self, df_list, name_list, columns):
        """
        Internal class to check whether the input is valid for preprocessing.
        Checks whether:
            df_list is a list of pd.DataFrames
            names_list is a list of str with the same length as df_list
            columns is a list of str
            all columns are present in all DataFrames

        Parameters
        ----------
        df_list : list
            valid input is a list of pandas DataFrames
        name_list : list
            valid input is a list of str with the same length as df_list.
        columns : list
            valid input is a list of str with column names.

        Raises
        ------
        ValueError
            raises Error if one of the conditions is not met.

        Returns
        -------
        None.

        """
        # check list of dfs
        if type(df_list) is list:
            for df in df_list:
                if not isinstance(df, pd.DataFrame):
                    raise ValueError("Not all list entries are pd.DataFrame" + "type")
                else:
                    if not set(columns).issubset(set(df.columns)):
                        raise ValueError(
                            "columns are not present in all" + " dataframes"
                        )
        else:
            raise ValueError("Entry is not a list")

        # check list of names
        if type(name_list) is not list:
            raise ValueError("Entry is not a list")
        else:
            for name in name_list:
                if not isinstance(name, str):
                    raise ValueError("not all list entries are type str")

        # check if ist have same length
        if not len(df_list) == len(name_list):
            raise ValueError("length of lists does not match")

    def _check_naming(self, names):
        """
        Internal class to check for duplicated names. Duplication can lead to
        problems when specific Measurements need to be accessed

        Parameters
        ----------
        names : list of str
            list of str to check for duplication.

        Raises
        ------
        Warning
            if there are duplicated names, a warning is raised.

        Returns
        -------
        None.

        """
        if not len(names) == len(set(names)):
            raise Warning(
                "there are dublicated names in train_names and/or" + "test_names"
            )
