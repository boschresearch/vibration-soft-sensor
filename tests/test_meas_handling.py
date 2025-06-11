# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:19:11 2021

@author: KRD2RNG
"""

from softsensor.meas_handling import Meas_handling
from softsensor.autoreg_models import ARNN

import torch
import pandas as pd
import pytest
import numpy as np

"""
Testing of Meas handling
"""


@pytest.fixture()
def train_data():
    t = np.linspace(0, 1.0, 10001)
    xlow = np.sin(2 * np.pi * 100 * t)  # 100Hz Signal
    xhigh = 0.2 * np.sin(2 * np.pi * 3000 * t)  # 3000Hz Signal
    d = {
        "sine_inp": xlow + xhigh,
        "cos_inp": np.cos(2 * np.pi * 50 * t),
        "out": np.linspace(0, 1.0, 10001),
    }

    return pd.DataFrame(d, index=t), pd.DataFrame(d, index=t)


@pytest.fixture()
def test_data():
    t = np.linspace(0, 1.0, 10001)
    xlow = np.sin(2 * np.pi * 100 * t)  # 100Hz Signal
    xhigh = 0.2 * np.sin(2 * np.pi * 3000 * t)  # 3000Hz Signal
    test_df = {
        "sine_inp": 10 * xlow + xhigh,
        "cos_inp": np.cos(2 * np.pi * 50 * t),
        "out": np.linspace(0, 1.0, 10001),
    }

    return pd.DataFrame(test_df, index=t)


@pytest.fixture()
def handler(train_data, test_data):
    handler = Meas_handling(
        [train_data[0], train_data[1]],
        ["sine1", "sine2"],
        ["sine_inp", "cos_inp"],
        ["out"],
        10000,
        [test_data],
        ["test"],
    )

    return handler


def test_raiseError():
    data = {"inp1": [0, 1, 2, 3], "inp2": [2, 3, 4, 6], "out1": [0, 0, 0, 0]}
    train_df = pd.DataFrame(data)
    train_names = None

    try:
        Meas_handling(train_df, train_names, ["inp1"], ["out1"], fs=10)
        assert False
    except ValueError:
        assert True

    train_df = [pd.DataFrame(data)]
    train_names = None

    try:
        Meas_handling(train_df, train_names, ["inp1"], ["out1"], fs=10)
        assert False
    except ValueError:
        assert True

    train_df = [pd.DataFrame(data), pd.DataFrame(data)]
    train_names = ["train1", "train2"]

    try:
        Meas_handling(train_df, train_names, ["not valid sensor"], ["out1"], fs=10)
        assert False
    except ValueError:
        assert True

    train_df = [pd.DataFrame(data), pd.DataFrame(data)]
    train_names = ["train1", "train2"]

    test_df = [pd.DataFrame(data), pd.DataFrame(data)]
    test_names = ["test1", "test2"]

    try:
        Meas_handling(
            train_df, train_names, ["inp1", "inp2"], ["out1"], 10, test_df, test_names
        )
        assert True
    except ValueError:
        assert False

    train_df = [pd.DataFrame(data), pd.DataFrame(data)]
    train_names = ["train1", "train2"]

    test_df = [pd.DataFrame(data), pd.DataFrame(data)]

    try:
        Meas_handling(
            train_df, train_names, ["not valid sensor"], ["out1"], 10, test_df
        )
        assert False
    except ValueError:
        assert True

    train_df = [pd.DataFrame(data), None]
    train_names = ["train1", "train2"]

    test_df = [pd.DataFrame(data), pd.DataFrame(data)]

    try:
        Meas_handling(
            train_df, train_names, ["not valid sensor"], ["out1"], 10, test_df
        )
        assert False
    except ValueError:
        assert True

    train_df = [pd.DataFrame(data), pd.DataFrame(data)]
    train_names = ["train1", None]

    test_df = [pd.DataFrame(data), pd.DataFrame(data)]

    try:
        Meas_handling(
            train_df, train_names, ["not valid sensor"], ["out1"], 10, test_df
        )
        assert False
    except ValueError:
        assert True

    train_df = [pd.DataFrame(data), pd.DataFrame(data)]
    train_names = ["train1"]

    test_df = [pd.DataFrame(data), pd.DataFrame(data)]

    try:
        Meas_handling(
            train_df, train_names, ["not valid sensor"], ["out1"], 10, test_df
        )
        assert False
    except ValueError:
        assert True

    try:
        Meas_handling(
            train_df, train_names, ["inp1", "inp2"], ["out1"], 10, test_df, train_names
        )
        assert False
    except ValueError:
        assert True


@pytest.mark.skip(reason="To be fixed for linux")
def test_Measurment_handling_filter(handler):
    t = np.linspace(0, 1.0, 10001)
    xlow = np.sin(2 * np.pi * 100 * t)
    handler.Filter(freq_lim=(10, 700))
    filtered_sine = handler.train_df[0]["sine_inp"].values
    dev = xlow - filtered_sine
    assert np.mean(dev) < 5e-3

    filtered_sine = handler.test_df[0]["sine_inp"].values
    dev = 10 * xlow[200:9800] - filtered_sine[200:9800]
    assert np.mean(dev) < 5e-3


def test_Measurment_handling_resample(handler):
    fs = 1 / np.mean(np.diff(handler.train_df[0].index))
    assert fs == 10000

    handler.Resample(fs=1000)

    fs = 1 / np.mean(np.diff(handler.train_df[0].index))
    assert int(fs) == 1000
    assert handler.fs == 1000

    t = np.linspace(0, 1.0, 1001)[:-1]
    xlow = np.sin(2 * np.pi * 100 * t)  # 100Hz Signal
    xhigh = 0.2 * np.sin(2 * np.pi * 3000 * t)  # 3000Hz Signal
    d = {
        "sine_inp": xlow + xhigh,
        "cos_inp": np.cos(2 * np.pi * 50 * t),
        "out": np.linspace(0, 1.0, 1001)[:-1],
    }
    test_df = pd.DataFrame(d, index=t)

    pd.testing.assert_frame_equal(handler.train_df[0], test_df)

    fs = 1 / np.mean(np.diff(handler.test_df[0].index))
    assert int(fs) == 1000

    handler.Resample(fs=577)
    fs = 1 / np.mean(np.diff(handler.train_df[0].index))
    assert int(fs) == 577
    assert handler.fs == 577

    fs = 1 / np.mean(np.diff(handler.test_df[0].index))
    assert int(fs) == 577


def test_resample_offset():
    t = np.linspace(0, 1.0, 10001)
    xlow = np.sin(2 * np.pi * 100 * t)  # 100Hz Signal
    xhigh = 0.2 * np.sin(2 * np.pi * 3000 * t)  # 3000Hz Signal
    d = {
        "sine_inp": xlow + xhigh,
        "cos_inp": np.cos(2 * np.pi * 50 * t),
        "out": np.linspace(0, 1.0, 10001),
    }

    handler = Meas_handling(
        [pd.DataFrame(d, index=t + 1)],
        ["sine1"],
        ["sine_inp", "cos_inp"],
        ["out"],
        fs=10001,
    )
    t = np.linspace(0, 1.0, 1001)[:-1]
    xlow = np.sin(2 * np.pi * 100 * t)  # 100Hz Signal
    xhigh = 0.2 * np.sin(2 * np.pi * 3000 * t)  # 3000Hz Signal
    d = {
        "sine_inp": xlow + xhigh,
        "cos_inp": np.cos(2 * np.pi * 50 * t),
        "out": np.linspace(0, 1.0, 1001)[:-1],
    }
    test_df = pd.DataFrame(d, index=t)

    handler.Resample(fs=1000)
    for c in ["sine_inp", "cos_inp", "out"]:
        np.testing.assert_allclose(handler.train_df[0][c], test_df[c], atol=1e-5)


def test_Measurment_handling_scale(handler):
    # scaled_sine = handler.train_df[0]['sine_inp'].values
    handler.Scale()

    scaled_sine = handler.train_df[0]["sine_inp"].values
    assert np.var(scaled_sine) - 1 < 1e-3

    scaled_cos = handler.train_df[0]["cos_inp"].values
    assert np.var(scaled_cos) - 1 < 1e-3

    scaled_sine = handler.test_df[0]["sine_inp"].values
    assert np.var(scaled_sine) > 10

    scaled_cos = handler.test_df[0]["cos_inp"].values
    assert np.var(scaled_cos) - 1 < 1e-3

    scaler = handler.scaler
    handler.Scale(scaler, True)

    scaled_sine = handler.train_df[0]["sine_inp"].values
    assert np.var(scaled_sine) > 1.5


def test_Measurment_handling_loader(handler, train_data):
    window_size = 200
    train_loader, val_loader = handler.give_torch_loader(
        window_size,
        keyword="training",
        train_ratio=0.8,
        batch_size=32,
        rnn_window=None,
        shuffle=False,
        Add_zeros=True,
    )

    assert len(train_loader.dataset) == 16001
    assert len(val_loader.dataset) == 4001

    train_loader, val_loader = handler.give_torch_loader(
        window_size,
        keyword="testing",
        train_ratio=0.5,
        batch_size=32,
        rnn_window=None,
        shuffle=True,
        Add_zeros=False,
    )
    assert len(train_loader.dataset) == 4901
    assert len(val_loader.dataset) == 4901

    train_loader = handler.give_torch_loader(
        window_size=100,
        keyword="short",
        train_ratio=1,
        batch_size=32,
        rnn_window=120,
        shuffle=False,
    )
    assert len(train_loader.dataset) == 5000
    train_loader, val_loader = handler.give_torch_loader(
        window_size=100, keyword="short", rnn_window=120
    )
    assert len(train_loader.dataset) == 5000
    assert len(val_loader.dataset) == 1000

    train_loader = handler.give_torch_loader(
        window_size=5,
        keyword="testing",
        train_ratio=1,
        batch_size=32,
        rnn_window=5,
        shuffle=False,
    )

    t = np.linspace(0, 1.0, 10001)
    xlow = np.sin(2 * np.pi * 100 * t)
    xhigh = 0.2 * np.sin(2 * np.pi * 3000 * t)
    test_df = {
        "sine_inp": 10 * xlow + xhigh,
        "cos_inp": np.cos(2 * np.pi * 50 * t),
        "out": np.linspace(0, 1.0, 10001),
    }

    data, _ = train_loader.dataset[5]
    np.testing.assert_almost_equal(
        [test_df["sine_inp"][1:6], test_df["cos_inp"][1:6]], data[0]
    )
    np.testing.assert_almost_equal([test_df["out"][:5]], data[1])

    data, _ = train_loader.dataset[0]
    np.testing.assert_almost_equal([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]], data[0])
    np.testing.assert_almost_equal([[0, 0, 0, 0, 0]], data[1])

    handler = Meas_handling([train_data[0]], ["train"], ["sine_inp"], ["out"], 10000)
    try:
        loader = handler.give_torch_loader(
            window_size=5,
            keyword="testing",
            train_ratio=1,
            batch_size=32,
            rnn_window=5,
            shuffle=False,
        )
        assert False
    except ValueError:
        assert True


def test_Measurment_handling_loader_length(handler, train_data):
    window_size = 200
    train_loader, val_loader = handler.give_torch_loader(
        window_size,
        keyword="short",
        train_ratio=0.8,
        batch_size=32,
        rnn_window=None,
        shuffle=False,
        Add_zeros=True,
        n_samples=[5000, 1000],
    )

    assert len(train_loader.dataset) == 5000
    assert len(val_loader.dataset) == 1000

    train_loader, val_loader = handler.give_torch_loader(
        window_size,
        keyword="short",
        train_ratio=0.8,
        batch_size=32,
        rnn_window=None,
        shuffle=False,
        Add_zeros=True,
        n_samples=[1000, 100],
    )

    assert len(train_loader.dataset) == 1000
    assert len(val_loader.dataset) == 100

    train_loader, val_loader = handler.give_torch_loader(
        window_size,
        keyword="short",
        train_ratio=0.8,
        batch_size=32,
        rnn_window=None,
        shuffle=False,
        Add_zeros=True,
        n_samples=[1000, 100],
    )

    assert len(train_loader.dataset) == 1000
    assert len(val_loader.dataset) == 100


def test_get_idx(handler):
    idx, train = handler._get_idx("sine1")
    assert idx == 0
    assert train is True

    idx, train = handler._get_idx("test")
    assert idx == 0
    assert train is False

    try:
        idx, train = handler._get_idx("not_valid_inp")
        assert False
    except ValueError:
        assert True

    t = np.linspace(0, 1.0, 101)
    d = {"sine_inp": np.sin(2 * np.pi * t), "out": np.linspace(0, 1.0, 101)}
    list_of_df = [pd.DataFrame(d, index=t), pd.DataFrame(d, index=t)]
    names = ["sine1", "sine2"]

    handler = Meas_handling(list_of_df, names, ["sine_inp"], ["out"], 100)
    try:
        idx, train = handler._get_idx("not_valid_inp")
        assert False
    except ValueError:
        assert True


def test_Measurment_handling_dataframe(handler, train_data, test_data):

    df = handler.give_dataframe("sine1")
    pd.testing.assert_frame_equal(train_data[0], df)

    df = handler.give_dataframe("test")
    pd.testing.assert_frame_equal(test_data, df)

    df = handler.give_dataframes("training")
    pd.testing.assert_frame_equal(train_data[0], df[0])
    pd.testing.assert_frame_equal(train_data[1], df[1])

    df = handler.give_dataframes("testing")
    pd.testing.assert_frame_equal(test_data, df[0])

    df = handler.give_dataframes(["sine1", "test"])
    pd.testing.assert_frame_equal(train_data[0], df[0])
    pd.testing.assert_frame_equal(test_data, df[1])


def test_give_list(handler, train_data, test_data):

    list_df = handler.give_list(
        window_size=10,
        keyword="training",
        batch_size=32,
        Add_zeros=False,
        rnn_window=None,
    )

    assert len(list_df) == 2
    data, _ = list_df[0].dataset[0]
    np.testing.assert_allclose(
        [train_data[0]["sine_inp"].values[:10], train_data[0]["cos_inp"].values[:10]],
        data,
        atol=1e-7,
    )
    data, _ = list_df[1].dataset[0]
    np.testing.assert_allclose(
        [train_data[1]["sine_inp"].values[:10], train_data[1]["cos_inp"].values[:10]],
        data,
        atol=1e-7,
    )

    list_df = handler.give_list(
        window_size=10,
        keyword="testing",
        batch_size=32,
        Add_zeros=False,
        rnn_window=None,
    )
    assert len(list_df) == 1
    data, _ = list_df[0].dataset[0]
    np.testing.assert_allclose(
        [test_data["sine_inp"].values[:10], test_data["cos_inp"].values[:10]],
        data,
        atol=1e-7,
    )

    list_df = handler.give_list(
        window_size=10,
        keyword=["sine1", "test"],
        batch_size=32,
        Add_zeros=False,
        rnn_window=None,
    )

    assert len(list_df) == 2
    data, _ = list_df[0].dataset[0]
    np.testing.assert_allclose(
        [train_data[0]["sine_inp"].values[:10], train_data[0]["cos_inp"].values[:10]],
        data,
    )
    data, _ = list_df[1].dataset[0]
    np.testing.assert_allclose(
        [test_data["sine_inp"].values[:10], test_data["cos_inp"].values[:10]],
        data,
        atol=1e-7,
    )

    try:
        list_df = handler.give_list(
            window_size=10,
            keyword="not_valid_key",
            batch_size=32,
            Add_zeros=False,
            rnn_window=None,
        )
        assert False
    except ValueError:
        assert True

    handler = Meas_handling([train_data[0]], ["train"], ["sine_inp"], ["out"], 10000)
    try:
        list_df = handler.give_list(
            window_size=10,
            keyword="testing",
            batch_size=32,
            Add_zeros=False,
            rnn_window=None,
        )
        assert False
    except ValueError:
        assert True


def test_Measurement_handling_get_window():
    assert np.allclose(Meas_handling._get_window("Hanning", 10), np.hanning(10))
    assert np.allclose(Meas_handling._get_window("HaMming", 18), np.hamming(18))
    assert np.allclose(Meas_handling._get_window("blackman", 2), np.blackman(2))
    assert np.allclose(Meas_handling._get_window("bartleTT", 7), np.bartlett(7))


def Sine_df():
    t = np.linspace(0, 1.0, 10001)
    xlow = np.sin(2 * np.pi * 100 * t)  # 100Hz Signal
    xhigh = np.sin(2 * np.pi * 3000 * t)  # 3000Hz Signal
    d = {
        "sine_inp": xlow + xhigh,
        "cos_inp": np.cos(2 * np.pi * 50 * t),
        "out": np.linspace(0, 1.0, 10001),
    }
    list_of_df = [pd.DataFrame(d, index=t), pd.DataFrame(d, index=t)]

    test_df = {
        "sine_inp": 10 * xlow + xhigh,
        "cos_inp": np.cos(2 * np.pi * 50 * t),
        "out": np.linspace(0, 1.0, 10001),
    }

    test_df = [pd.DataFrame(test_df, index=t)]

    handler = Meas_handling(
        list_of_df,
        train_names=["sine1", "sine2"],
        input_sensors=["sine_inp", "cos_inp"],
        output_sensors=["out"],
        fs=10000,
        test_dfs=test_df,
        test_names=["test"],
    )
    return handler, xlow, xhigh


def test_Measurement_handling_fade_in():
    handler, _, _ = Sine_df()
    handler_test, _, _ = Sine_df()

    handler.fade_in(10, "Hanning")

    for df, df_test in zip(handler.train_df, handler_test.train_df):
        assert np.allclose(df.iloc[0, :].to_numpy(), np.zeros_like(df.iloc[0, :]))
        assert np.allclose(
            df.iloc[10:-10, :].to_numpy(),
            df_test.iloc[10:-10, :].to_numpy(),
            rtol=1e-1,
            atol=1e-1,
        )

    if handler.test_df is not None and handler_test.test_df is not None:
        for df, df_test in zip(handler.test_df, handler_test.test_df):
            assert np.allclose(df.iloc[0, :].to_numpy(), np.zeros_like(df.iloc[0, :]))
            assert np.allclose(
                df.iloc[10:-10, :].to_numpy(),
                df_test.iloc[10:-10, :].to_numpy(),
                rtol=1e-1,
                atol=1e-1,
            )
