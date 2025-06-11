# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 09:59:14 2021

@author: KRD2RNG
"""
import pandas as pd
import numpy as np
from scipy import io as sio


def _get_time_index(df, fs=None):
    """
    Change Time index to fs

    Parameters
    ----------
    df : pandas.DataFrame
        pandas Dataframe to change time index.
    fs : float or None, optional
        sampling rate of the dataframe. The default is None.

    Returns
    -------
    df : pandas.DataFrame
        pandas Dataframe with changed time index.
        if fs = None, index is just changed to "time"

    """
    if fs is not None:
        # time = np.arange(start=0, stop=len(df) * 1 / fs, step=1 / fs)
        time = np.linspace(0, stop=(len(df) - 1) / fs, num=len(df))
        df.index = time
    df.index.name = "time"
    return df


def readmatfile(file, sensor_names, fs=None):
    """
    reads matlab file format into pandas DataFrame using scipy.io functionality

    Parameters
    ----------
    file : TYPE
        mat file
    sensor_names : list
        list of sensors (keys of the mat file)

    Returns
    -------
    data : DataFrame

    """
    mat = sio.loadmat(file)
    data = pd.DataFrame()
    for sens_act in sensor_names:
        if mat[sens_act].shape[1] == 3:
            df = pd.DataFrame(mat[sens_act], columns=[sens_act + "_x",
                                                      sens_act + "_y",
                                                      sens_act + "_z"])
            data = pd.concat((data, df), axis=1)
        else:
            data[sens_act] = mat[sens_act].flatten()
    data = _get_time_index(data, fs)
    return data


def readcsvfile(file_name, t_col='Unnamed: 0'):
    """
    reads csv file format into pandas DataFrame using pandas functionality

    Parameters
    ----------
    file : TYPE
        csv file.
    t_col : str, optional
        time series column name. The default is 'Unnamed: 0'.

    Returns
    -------
    data : DataFrame

    """
    data = pd.read_csv(file_name, index_col=t_col)
    return data


def readnpyfile(file_name, sensor_names, fs=None):
    """
    reads npy file format into pandas DataFrame using numpy functionality

    Parameters
    ----------
    file : TYPE
        npy file
    sensor_names : list
        list of sensors
    fs : int, optional
        sample frequency of the time series. The default is None.

    Returns
    -------
    data : DataFrame
        DESCRIPTION.

    """
    data = pd.DataFrame(np.load(file_name),
                        columns=sensor_names)
    data = _get_time_index(data, fs)
    return data
