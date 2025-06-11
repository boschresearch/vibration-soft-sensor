# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 08:49:30 2021

@author: KRD2RNG
"""

import numpy as np
import pandas as pd
from scipy import io as sio
import softsensor.data_prep as data_prep
import os
import pytest

@pytest.fixture
def path_data():
    path = os.path.abspath('')
    if "tests" not in path:
        path = os.path.join(path, 'tests')
    path = os.path.join(path, 'test_data')
    return path

@pytest.fixture
def create_input_df():
    fs = 2048
    t = np.arange(0, 2, 1/fs)
    ts_df = pd.DataFrame({"sine": np.sin(10* 2 * np.pi * t),
                          "cos": 10 * np.cos(5* 2 * np.pi * t),
                          "wn": np.random.randn(len(t))},
                         index=t)
    ts_df.index.name = "t"
    return ts_df

def test_readmatfile(path_data, create_input_df):
    ts_true = create_input_df
    ts_df = data_prep.readmatfile(os.path.join(path_data, "test_read_mat.mat"), ts_true.columns)
    ts_df.index = ts_true.index
    pd.testing.assert_frame_equal(ts_df.drop('wn', axis=1), ts_true.drop('wn', axis=1))
    return

def test_readmatfile_3d(path_data, create_input_df):
    ts_true = create_input_df
    ts_true.columns = ["S_1_x", "S_1_y", "S_1_z"]
    ts_df = data_prep.readmatfile(os.path.join(path_data, "test_read_mat_3d.mat"), ["S_1"])
    ts_df.index = ts_true.index
    pd.testing.assert_frame_equal(ts_df[["S_1_x", "S_1_y"]], ts_true[["S_1_x", "S_1_y"]])

    return

def test_readcsvfile(path_data, create_input_df):
    ts_true = create_input_df
    ts_df = data_prep.readcsvfile(os.path.join(path_data, "test_read_csv.csv"), t_col="t")
    ts_df.index = ts_true.index
    pd.testing.assert_frame_equal(ts_df.drop('wn', axis=1), ts_true.drop('wn', axis=1))
    return

def test_readnpyfile(path_data, create_input_df):
    ts_true = create_input_df
    ts_df = data_prep.readnpyfile(
        os.path.join(path_data, "test_read_npy.npy"),
        ts_true.columns,
        fs=np.mean(1 / np.diff(ts_true.index.values)),
    )
    ts_df.index = ts_true.index
    pd.testing.assert_frame_equal(ts_df.drop('wn', axis=1), ts_true.drop('wn', axis=1))
    return
