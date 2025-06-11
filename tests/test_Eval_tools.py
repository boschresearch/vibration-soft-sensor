# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:16:16 2022

@author: WET2RNG
"""
# %%
from softsensor.meas_handling import Meas_handling
import pandas as pd
import os
import numpy as np
import pytest
import torch
from softsensor.eval_tools import (
    _check_sens_analysis,
    comp_error,
    comp_pred,
    comp_mean_metrics,
)
from softsensor.eval_tools import _comp_ARNN_batch, _ARNN_dataframe_pred, comp_batch
from softsensor.arx import ARX
from softsensor.linear_methods import tf
from softsensor.autoreg_models import ARNN, SeparateMVEARNN, DensityEstimationARNN
from softsensor.autoreg_models import QuantileARNN
from softsensor.recurrent_models import AR_RNN, RNN_DNN
from softsensor.homoscedastic_model import HomoscedasticModel, fit_homoscedastic_var
from softsensor.ensemble_wrappers import AsyncEnsemble, AsyncMCDropout, AsyncMVEEnsemble
from softsensor.temporal_fusion_transformer import TFT


@pytest.fixture
def Sine_df():
    t = np.linspace(0, 1.0, 101)
    xlow = np.sin(2 * np.pi * 100 * t)  # 100Hz Signal
    xhigh = np.sin(2 * np.pi * 3000 * t)  # 3000Hz Signal
    d = {
        "sine_inp": xlow + xhigh,
        "cos_inp": np.cos(2 * np.pi * 50 * t),
        "out": np.linspace(0, 1.0, 101),
    }
    list_of_df = [pd.DataFrame(d), pd.DataFrame(d)]

    test_df = {
        "sine_inp": 10 * xlow + xhigh,
        "cos_inp": np.cos(2 * np.pi * 50 * t),
        "out": np.linspace(0, 1.0, 101),
    }

    test_df = [pd.DataFrame(test_df)]

    handler = Meas_handling(
        list_of_df,
        train_names=["sine1", "sine2"],
        input_sensors=["sine_inp", "cos_inp"],
        output_sensors=["out"],
        fs=100,
        test_dfs=test_df,
        test_names=["test"],
    )
    return handler


@pytest.fixture
def result_df():
    d = {
        "out": np.random.randn(1000),
        "out2": np.random.randn(1000),
        "out_random": np.random.randn(1000),
        "out_zeros": np.zeros((1000)),
        "out2_random": np.random.randn(1000),
        "out2_zeros": np.zeros((1000)),
    }
    d["out_copy"] = d["out"]
    d["out2_copy"] = d["out2"]
    df = pd.DataFrame(d)
    return df


def test_comp_pred(Sine_df):
    model1 = ARNN(input_channels=2, pred_size=1, window_size=10, rnn_window=10)
    model2 = RNN_DNN(
        input_channels=2,
        pred_size=1,
        window_size=10,
        blocks=4,
        num_layers=2,
        blocktype="GRU",
    )
    model3 = tf(window_size=16, fs=100, hop=8)
    model3.fit(Sine_df.train_df, Sine_df.input_sensors, Sine_df.output_sensors)

    order = (2, 6)
    arx = ARX(order=order)
    arx.fit(
        Sine_df.train_df, Sine_df.input_sensors, Sine_df.output_sensors, verbose=False
    )
    results = comp_pred(
        [model1, model2, model3, arx], Sine_df, "test", names=None, batch_size=16
    )
    keys = list(results.filter(regex="out_").columns)
    true_keys = ["AR", "RNN", "TF", "ARX"]
    for i, tk in enumerate(true_keys):
        assert keys[i] == f"out_{tk}"

    results = comp_pred(
        [model1, model2], Sine_df, "test", names=["M1, M2"], batch_size=16
    )

    result_df = comp_error(
        results,
        out_sens=["out"],
        fs=100,
        names=["TF", "ARX"],
        metrics=["MSE", "MAE", "MAPE"],
        freq_range=(5, 25),
    )

    result_df = comp_error(
        results,
        out_sens=["out"],
        fs=100,
        names=["TF", "ARX"],
        metrics=["MSE", "MAE", "MAPE"],
        freq_range=None,
    )

    keys = list(result_df.columns)
    true_keys = ["TF", "ARX"]
    for i, tk in enumerate(true_keys):
        assert keys[i] == tk

    keys = list(result_df.index)
    true_keys = ["MSE", "MAE", "MAPE"]
    for i, tk in enumerate(true_keys):
        assert keys[i] == f"out_{tk}"


def test_comp_pred_grad_sens_True(Sine_df):
    model1 = ARNN(input_channels=2, pred_size=1, window_size=10, rnn_window=10)
    model2 = RNN_DNN(
        input_channels=2,
        pred_size=1,
        window_size=10,
        blocks=4,
        num_layers=2,
        blocktype="GRU",
    )
    model3 = tf(window_size=16, fs=100, hop=8)
    model3.fit(Sine_df.train_df, Sine_df.input_sensors, Sine_df.output_sensors)

    order = (2, 6)
    arx = ARX(order=order)
    arx.fit(
        Sine_df.train_df, Sine_df.input_sensors, Sine_df.output_sensors, verbose=False
    )

    # assertions with grad_sens dict given
    res_df, sensitivities = comp_pred(
        [model1, model2, model3, arx],
        Sine_df,
        "test",
        names=None,
        batch_size=16,
        sens_analysis={"method": "gradient", "params": {"comp": True, "plot": False}},
    )
    keys = list(res_df.filter(regex="out_").columns)
    true_keys = ["AR", "RNN", "TF", "ARX"]
    for i, tk in enumerate(true_keys):
        assert keys[i] == f"out_{tk}"

    assert isinstance(sensitivities, list)
    assert len(sensitivities) == 4
    assert sensitivities[0]["Point"].shape == (101, 1, 30)
    assert sensitivities[1]["Point"].shape == (101, 1, 20)
    for sens in sensitivities[:2]:
        assert sens.keys() == {"Point"}
        assert isinstance(sens["Point"], torch.Tensor)
    for sens in sensitivities[2:]:
        assert sens is None


def test_comp_pred_grad_sens_False(Sine_df):
    model1 = ARNN(input_channels=2, pred_size=1, window_size=10, rnn_window=10)
    model2 = RNN_DNN(
        input_channels=2,
        pred_size=1,
        window_size=10,
        blocks=4,
        num_layers=2,
        blocktype="GRU",
    )
    model3 = tf(window_size=16, fs=100, hop=8)
    model3.fit(Sine_df.train_df, Sine_df.input_sensors, Sine_df.output_sensors)
    order = (2, 6)
    arx = ARX(order=order)
    arx.fit(
        Sine_df.train_df, Sine_df.input_sensors, Sine_df.output_sensors, verbose=False
    )

    # assertions with grad_sens dict given
    res_df, sensitivities = comp_pred(
        [model1, model2, model3, arx],
        Sine_df,
        "test",
        names=None,
        batch_size=16,
        sens_analysis={"method": "gradient", "params": {"comp": False}},
    )
    keys = list(res_df.filter(regex="out_").columns)
    true_keys = ["AR", "RNN", "TF", "ARX"]
    for i, tk in enumerate(true_keys):
        assert keys[i] == f"out_{tk}"

    assert isinstance(sensitivities, list)
    assert len(sensitivities) == 4
    for sens in sensitivities:
        assert sens is None


def test_comp_pred_exception(Sine_df):
    model1 = ARNN(input_channels=2, pred_size=1, window_size=10, rnn_window=10)
    model2 = RNN_DNN(
        input_channels=2,
        pred_size=1,
        window_size=10,
        blocks=4,
        num_layers=2,
        blocktype="GRU",
    )
    model3 = tf(window_size=16, fs=100, hop=8)
    model3.fit(Sine_df.train_df, Sine_df.input_sensors, Sine_df.output_sensors)
    order = (2, 6)
    arx = ARX(order=order)
    arx.fit(
        Sine_df.train_df, Sine_df.input_sensors, Sine_df.output_sensors, verbose=False
    )

    # assertions with grad_sens dict given
    try:
        _, sensitivities = comp_pred(
            [model1, model2, model3, arx],
            Sine_df,
            "test",
            names=None,
            batch_size=16,
            sens_analysis={"method": "foo", "params": {"comp": False, "plot": False}},
        )
    except ValueError as e:
        assert (
            str(e)
            == "Given method 'foo' is not implemented! Choose from: ['gradient', 'smooth_grad', 'integrated_gradient', 'perturbation']"
        )


def test_comp_error(result_df):
    errors = comp_error(
        result_df,
        out_sens=["out", "out2"],
        fs=100,
        names=["random", "copy"],
        metrics=["MSE", "MAE", "MAPE"],
        freq_metrics=["MSE", "MSLE"],
        freq_range=(5, 25),
    )

    keys = list(errors.columns)
    true_keys = ["random", "copy"]
    for i, tk in enumerate(true_keys):
        assert keys[i] == tk

    keys = list(errors.index)
    true_keys = [
        "out_MSE",
        "out2_MSE",
        "out_MAE",
        "out2_MAE",
        "out_MAPE",
        "out2_MAPE",
        "out_PSD_MSE",
        "out2_PSD_MSE",
        "out_PSD_MSLE",
        "out2_PSD_MSLE",
    ]

    for i, tk in enumerate(true_keys):
        assert keys[i] == tk
        assert errors["random"][tk] != 0
        assert errors["copy"][tk] == 0


def test_comp_error2(result_df):
    errors = comp_error(
        result_df,
        out_sens=["out"],
        fs=100,
        names=["copy"],
        metrics=["KLD", "JSD", "Wasserstein"],
        freq_metrics=None,
        freq_range=(5, 25),
        bins=5,
    )

    keys = list(errors.columns)
    true_keys = ["copy"]
    for i, tk in enumerate(true_keys):
        assert keys[i] == tk

    keys = list(errors.index)
    true_keys = ["KLD", "JSD", "Wasserstein"]
    for i, tk in enumerate(true_keys):
        assert keys[i] == f"out_{tk}"
        assert errors["copy"][f"out_{tk}"] == 0

    errors = comp_error(
        result_df,
        out_sens=["out"],
        fs=100,
        names=["zeros"],
        metrics=["KLD"],
        freq_metrics=None,
        freq_range=(5, 25),
    )

    keys = list(errors.columns)
    true_keys = ["zeros"]
    for i, tk in enumerate(true_keys):
        assert keys[i] == tk

    keys = list(errors.index)
    true_keys = ["KLD"]
    for i, tk in enumerate(true_keys):
        assert keys[i] == f"out_{tk}"
        assert np.isinf(errors["zeros"][f"out_{tk}"])

    errors = comp_error(
        result_df,
        out_sens=["out"],
        fs=100,
        names=["zeros"],
        metrics=None,
        freq_metrics=None,
        freq_range=(5, 25),
    )
    assert errors is None


def dataframe(le):
    t = np.linspace(0, 1.0, le)
    xlow = np.sin(2 * np.pi * 100 * t)
    xhigh = np.sin(2 * np.pi * 3000 * t)
    d = {
        "sine_inp": xlow + xhigh,
        "cos_inp": np.cos(2 * np.pi * 50 * t),
        "out1": np.linspace(0, 1.0, le),
        "out2": np.linspace(0, 1.0, le),
    }
    return pd.DataFrame(d)


def test_comp_error_distributional_metrics():
    lengths = [577]
    list_of_df = [dataframe(le) for le in lengths]
    names = [f"{le}" for le in lengths]

    handler = Meas_handling(
        list_of_df,
        train_names=names,
        input_sensors=["sine_inp", "cos_inp"],
        output_sensors=["out1", "out2"],
        fs=100,
    )

    model = DensityEstimationARNN(2, 2, 10, 10, [16, 8])

    result_df = comp_batch(
        [model], handler, handler.train_names, names=["MVE"], batch_size=16
    )

    metrics = ["RMSE", "R2", "Corr", "NLL", "RMV", "CRPS", "Het", "PICP", "MPIW", "ECE"]

    errors = comp_error(
        result_df[0],
        out_sens=["out1"],
        fs=100,
        names=["MVE"],
        metrics=metrics,
        freq_metrics=None,
    )

    keys = list(errors.index)
    true_keys = [f"out1_{m}" for m in metrics]

    for i, tk in enumerate(true_keys):
        assert keys[i] == tk

    keys = list(errors.columns)
    true_keys = ["MVE"]

    for i, tk in enumerate(true_keys):
        assert keys[i] == tk


def test_quantile_metrics():
    lengths = [577]
    list_of_df = [dataframe(le) for le in lengths]
    names = [f"{le}" for le in lengths]

    handler = Meas_handling(
        list_of_df,
        train_names=names,
        input_sensors=["sine_inp", "cos_inp"],
        output_sensors=["out1", "out2"],
        fs=100,
    )

    model = QuantileARNN(2, 2, 10, 10, hidden_size=[8], n_quantiles=11)

    result_df = comp_batch(
        [model], handler, handler.train_names, names=["Q_ARNN"], batch_size=16
    )
    metrics = ["RMSE", "MAE", "R2", "Corr", "PICP", "MPIW", "ECE"]


def test_batch_pred(Sine_df):
    model1 = ARNN(input_channels=2, pred_size=1, window_size=10, rnn_window=10)
    model2 = RNN_DNN(
        input_channels=2,
        pred_size=1,
        window_size=10,
        blocks=4,
        num_layers=2,
        blocktype="GRU",
    )
    model3 = tf(window_size=16, fs=100, hop=8)
    model3.fit(Sine_df.train_df, Sine_df.input_sensors, Sine_df.output_sensors)

    dataframes = Sine_df.give_dataframes("training")

    labels = ["out_model"]
    pred = _comp_ARNN_batch(model1, Sine_df, "training")
    dataframes = _ARNN_dataframe_pred(dataframes, pred, labels)


def test_comp_batch():
    lengths = [577, 168, 1000, 455]
    list_of_df = [dataframe(le) for le in lengths]
    names = [f"{le}" for le in lengths]

    handler = Meas_handling(
        list_of_df,
        train_names=names,
        input_sensors=["sine_inp", "cos_inp"],
        output_sensors=["out1", "out2"],
        fs=100,
    )

    model1 = ARNN(input_channels=2, pred_size=2, window_size=10, rnn_window=10)
    model2 = RNN_DNN(
        input_channels=2,
        pred_size=2,
        window_size=10,
        blocks=4,
        num_layers=2,
        blocktype="GRU",
    )
    model3 = tf(fs=100, window_size=48, hop=16)
    model3.fit(handler.train_df, handler.input_sensors, handler.output_sensors)

    models = [model1, model2, model3]
    dataframes = comp_batch(
        models,
        handler,
        handler.train_names,
        ["ARNN", "RNN", "TF"],
        device="cpu",
        batch_size=256,
    )


def test_comp_batch_sens_true_grad():
    lengths = [577, 168]
    list_of_df = [dataframe(le) for le in lengths]
    names = [f"{le}" for le in lengths]

    handler = Meas_handling(
        list_of_df,
        train_names=names,
        input_sensors=["sine_inp", "cos_inp"],
        output_sensors=["out1", "out2"],
        fs=100,
    )

    model1 = ARNN(input_channels=2, pred_size=2, window_size=10, rnn_window=10)
    model2 = RNN_DNN(
        input_channels=2,
        pred_size=2,
        window_size=10,
        blocks=4,
        num_layers=2,
        blocktype="GRU",
    )
    model3 = tf(window_size=16, fs=100, hop=8)
    model3.fit(handler.train_df, handler.input_sensors, handler.output_sensors)

    models = [model1, model2, model3]
    dataframes, sensitivities = comp_batch(
        models,
        handler,
        handler.train_names,
        ["ARNN", "RNN", "TF"],
        device="cpu",
        batch_size=256,
        sens_analysis={"method": "gradient", "params": {"comp": True, "plot": False}},
    )

    assert len(sensitivities) == 3
    for i, sens in enumerate(sensitivities):
        if i <= 1:
            assert sens.keys() == {"Point"}
            assert isinstance(sens["Point"], list)
            assert len(sens["Point"]) == 2
            assert all(isinstance(x, torch.Tensor) for x in sens["Point"])
            assert all(not any(torch.isnan(torch.flatten(x))) for x in sens["Point"])
        else:
            assert sens is None
    assert sensitivities[0]["Point"][0].shape == (577, 2, 40)
    assert sensitivities[0]["Point"][1].shape == (168, 2, 40)
    assert sensitivities[1]["Point"][0].shape == (577, 2, 20)
    assert sensitivities[1]["Point"][1].shape == (168, 2, 20)


def test_comp_batch_sens_true_perturb():
    lengths = [577, 168]
    list_of_df = [dataframe(le) for le in lengths]
    names = [f"{le}" for le in lengths]

    handler = Meas_handling(
        list_of_df,
        train_names=names,
        input_sensors=["sine_inp", "cos_inp"],
        output_sensors=["out1", "out2"],
        fs=100,
    )

    model1 = ARNN(input_channels=2, pred_size=2, window_size=10, rnn_window=10)
    model2 = RNN_DNN(
        input_channels=2,
        pred_size=2,
        window_size=10,
        blocks=4,
        num_layers=2,
        blocktype="GRU",
    )
    model3 = tf(window_size=16, fs=100, hop=8)
    model3.fit(handler.train_df, handler.input_sensors, handler.output_sensors)

    models = [model1, model2, model3]
    dataframes, sensitivities = comp_batch(
        models,
        handler,
        handler.train_names,
        ["ARNN", "RNN", "TF"],
        device="cpu",
        batch_size=256,
        sens_analysis={
            "method": "perturbation",
            "params": {"comp": True, "plot": False},
        },
    )

    assert len(sensitivities) == 3
    for i, sens in enumerate(sensitivities):
        if i <= 1:
            assert sens.keys() == {"Point"}
            assert isinstance(sens["Point"], list)
            assert len(sens["Point"]) == 2
            assert all(isinstance(x, torch.Tensor) for x in sens["Point"])
            assert all(not any(torch.isnan(torch.flatten(x))) for x in sens["Point"])
        else:
            assert sens is None
    assert sensitivities[0]["Point"][0].shape == (577, 2, 40)
    assert sensitivities[0]["Point"][1].shape == (168, 2, 40)
    assert sensitivities[1]["Point"][0].shape == (577, 2, 20)
    assert sensitivities[1]["Point"][1].shape == (168, 2, 20)


def test_comp_batch_sens_length_true():
    lengths = [577, 168]
    list_of_df = [dataframe(le) for le in lengths]
    names = [f"{le}" for le in lengths]

    handler = Meas_handling(
        list_of_df,
        train_names=names,
        input_sensors=["sine_inp", "cos_inp"],
        output_sensors=["out1", "out2"],
        fs=100,
    )

    model = ARNN(input_channels=2, pred_size=2, window_size=10, rnn_window=10)

    dataframes, sensitivities = comp_batch(
        [model],
        handler,
        handler.train_names,
        ["ARNN"],
        device="cpu",
        batch_size=256,
        sens_analysis={
            "method": "gradient",
            "params": {"comp": True, "plot": False, "sens_length": 100},
        },
    )

    assert sensitivities.keys() == {"Point"}
    assert isinstance(sensitivities["Point"], list)
    assert len(sensitivities["Point"]) == 2
    assert all(isinstance(sens, torch.Tensor) for sens in sensitivities["Point"])
    assert all(
        not any(torch.isnan(torch.flatten(sens))) for sens in sensitivities["Point"]
    )
    assert all(sens.shape == (100, 2, 40) for sens in sensitivities["Point"])


def test_check_sens_analysis():
    sens_params = {
        "comp": True,
        "num_samples": 50,
        "foo": False,  # invalid key
    }
    sens_analysis = {"method": "gradient", "params": sens_params}

    checks = _check_sens_analysis(sens_analysis)
    gt_dict = {
        "method": "gradient",
        "comp": True,
        "plot": False,
        "num_samples": 50,
        "foo": False,
    }
    assert checks == gt_dict

    # assert empty inner dict
    sens_analysis = {"method": "perturbation", "params": {}}

    checks = _check_sens_analysis(sens_analysis)
    gt_dict = {
        "method": "perturbation",
        "comp": False,
        "plot": False,
    }
    assert checks == gt_dict

    # assert empty outer dict
    sens_analysis = {}
    checks = _check_sens_analysis(sens_analysis)
    assert isinstance(checks, tuple)
    assert len(checks) == 3
    assert checks[0] is None
    assert checks[1] == ""
    assert checks[2] is False


def test_comp_batch_forecast_sens_false():
    lengths = [577, 168]
    list_of_df = [dataframe(le) for le in lengths]
    names = [f"{le}" for le in lengths]

    handler = Meas_handling(
        list_of_df,
        train_names=names,
        input_sensors=["sine_inp", "cos_inp"],
        output_sensors=["out1", "out2"],
        fs=100,
    )

    model1 = ARNN(
        input_channels=2, pred_size=2, window_size=10, rnn_window=10, forecast=3
    )
    model2 = RNN_DNN(
        input_channels=2,
        pred_size=2,
        window_size=10,
        blocks=4,
        num_layers=2,
        blocktype="GRU",
        forecast=3,
    )
    model3 = tf(window_size=16, fs=100, hop=8)
    model3.fit(handler.train_df, handler.input_sensors, handler.output_sensors)

    models = [model1, model2, model3]
    _, sensitivities = comp_batch(
        models,
        handler,
        handler.train_names,
        ["ARNN", "RNN", "TF"],
        device="cpu",
        batch_size=256,
        sens_analysis={"method": "gradient", "params": {"comp": False, "plot": False}},
    )
    assert len(sensitivities) == 3
    assert all([sens is None for sens in sensitivities])


def test_comp_batch_forecast_sens_true():
    lengths = [577, 168]
    list_of_df = [dataframe(le) for le in lengths]
    names = [f"{le}" for le in lengths]

    handler = Meas_handling(
        list_of_df,
        train_names=names,
        input_sensors=["sine_inp", "cos_inp"],
        output_sensors=["out1", "out2"],
        fs=100,
    )

    model1 = ARNN(
        input_channels=2, pred_size=2, window_size=10, rnn_window=10, forecast=3
    )
    model2 = RNN_DNN(
        input_channels=2,
        pred_size=2,
        window_size=10,
        blocks=4,
        num_layers=2,
        blocktype="GRU",
        forecast=3,
    )

    models = [model1, model2]
    dataframes, sensitivities = comp_batch(
        models,
        handler,
        handler.train_names,
        ["ARNN", "RNN"],
        device="cpu",
        batch_size=256,
        sens_analysis={"method": "gradient", "params": {"comp": True, "plot": False}},
    )

    assert len(sensitivities) == 2
    for i, sens in enumerate(sensitivities):
        if i <= 1:
            assert sens.keys() == {"Point"}
            assert isinstance(sens["Point"], list)
            assert len(sens["Point"]) == 2
            assert all(isinstance(x, torch.Tensor) for x in sens["Point"])
            assert all(not any(torch.isnan(torch.flatten(x))) for x in sens["Point"])
        else:
            assert sens is None
    assert sensitivities[0]["Point"][0].shape == (577, 2, 40)
    assert sensitivities[0]["Point"][1].shape == (168, 2, 40)
    assert sensitivities[1]["Point"][0].shape == (577, 2, 20)
    assert sensitivities[1]["Point"][1].shape == (168, 2, 20)


def test_comp_batch_exception():
    lengths = [577, 168]
    list_of_df = [dataframe(le) for le in lengths]
    names = [f"{le}" for le in lengths]

    handler = Meas_handling(
        list_of_df,
        train_names=names,
        input_sensors=["sine_inp", "cos_inp"],
        output_sensors=["out1", "out2"],
        fs=100,
    )

    model1 = ARNN(input_channels=2, pred_size=2, window_size=10, rnn_window=10)
    model2 = RNN_DNN(
        input_channels=2,
        pred_size=2,
        window_size=10,
        blocks=4,
        num_layers=2,
        blocktype="GRU",
    )
    model3 = tf(window_size=16, fs=100, hop=8)
    model3.fit(handler.train_df, handler.input_sensors, handler.output_sensors)

    models = [model1, model2, model3]
    try:
        _, sensitivities = comp_batch(
            models,
            handler,
            handler.train_names,
            ["ARNN", "RNN", "TF"],
            device="cpu",
            batch_size=256,
            sens_analysis={"method": "foo", "params": {"comp": False, "plot": False}},
        )
    except ValueError as e:
        assert (
            str(e)
            == "Given method 'foo' is not implemented! Choose from: ['gradient', 'perturbation']"
        )


def test_comp_batch_quantiles():
    lengths = [577, 168]
    list_of_df = [dataframe(le) for le in lengths]
    names = [f"{le}" for le in lengths]

    handler = Meas_handling(
        list_of_df,
        train_names=names,
        input_sensors=["sine_inp", "cos_inp"],
        output_sensors=["out1", "out2"],
        fs=100,
    )

    model1 = QuantileARNN(
        input_channels=2,
        pred_size=2,
        window_size=10,
        hidden_size=[8, 4],
        activation="leaky_relu",
        rnn_window=10,
        n_quantiles=5,
    )

    dataframes = comp_batch([model1], handler, handler.train_names, ["Quantile_ARNN"])


def test_comp_batch_forecast():
    lengths = [577, 168, 1000, 455]
    list_of_df = [dataframe(le) for le in lengths]
    names = [f"{le}" for le in lengths]

    handler = Meas_handling(
        list_of_df,
        train_names=names,
        input_sensors=["sine_inp", "cos_inp"],
        output_sensors=["out1", "out2"],
        fs=100,
    )

    model1 = ARNN(
        input_channels=2, pred_size=2, window_size=10, rnn_window=10, forecast=2
    )
    model2 = RNN_DNN(
        input_channels=2,
        pred_size=2,
        window_size=10,
        blocks=4,
        num_layers=2,
        blocktype="GRU",
        Pred_Type="Mean_Var",
    )
    model3 = tf(window_size=16, fs=100, hop=8)
    model3.fit(handler.train_df, handler.input_sensors, handler.output_sensors)

    model4 = SeparateMVEARNN(
        input_channels=2,
        pred_size=2,
        window_size=10,
        rnn_window=10,
        mean_model=model1,
        var_hidden_size=[4, 4],
        forecast=2,
    )

    pred_df = comp_batch(
        [model1], handler, handler.train_names, names=["ARNN"], batch_size=16
    )

    var = fit_homoscedastic_var(pred_df, ["out1_ARNN", "out2_ARNN"], ["out1", "out2"])
    homosc_m = HomoscedasticModel(model1, var)

    m = []
    for i in range(3):
        m_temp = ARNN(2, 2, 10, 10, [16, 8])
        m.append(m_temp.state_dict())
    ensemble = AsyncEnsemble(ARNN(2, 2, 10, 10, [16, 8]), m)

    m = ARNN(2, 2, 10, 10, [16, 8], dropout=0.5, concrete_dropout=True)
    mcdo = AsyncMCDropout(m, n_samples=5)

    m = []
    for i in range(3):
        m_temp = DensityEstimationARNN(2, 2, 10, 10, [16, 8])
        m.append(m_temp.state_dict())
    mve_ensemble = AsyncMVEEnsemble(m_temp, m)

    tft = TFT(
        input_channels=2,
        pred_size=2,
        hidden_window=10,
        blocks=4,
        num_layers=2,
        n_heads=2,
        blocktype="GRU",
        Pred_Type="Mean_Var",
    )

    models = [
        model1,
        model2,
        model3,
        model4,
        homosc_m,
        ensemble,
        mcdo,
        mve_ensemble,
        tft,
    ]

    dataframes = comp_batch(
        models,
        handler,
        handler.train_names,
        ["ARNN", "RNN", "TF", "MVE", "ARNN_hom", "ARNN_ens", "MCDO", "MVE_ens", "TFT"],
        device="cpu",
        batch_size=256,
    )
    for n in list(dataframes[0].head()):
        print(n)


def test_comp_MVE_ensemble():
    lengths = [577, 168, 1000, 455]

    list_of_df = [dataframe(le) for le in lengths]
    names = [f"{le}" for le in lengths]

    handler = Meas_handling(
        list_of_df,
        train_names=names,
        input_sensors=["sine_inp", "cos_inp"],
        output_sensors=["out1", "out2"],
        fs=100,
    )
    m = []
    for i in range(5):
        mean_model = ARNN(2, 2, 10, 10, [16, 8])
        m_temp = SeparateMVEARNN(
            input_channels=2,
            pred_size=2,
            window_size=10,
            rnn_window=10,
            mean_model=mean_model,
            var_hidden_size=[4, 4],
        )
        m.append(m_temp.state_dict())
    ensemble = AsyncMVEEnsemble(m_temp, m)

    models = [ensemble]

    dataframes = comp_batch(
        models,
        handler,
        handler.train_names,
        ["MVEEnsemble"],
        device="cpu",
        batch_size=256,
    )
    print(list(dataframes[0].head()))


def test_uncertainty_comp_pred(Sine_df):
    mean_model = ARNN(
        input_channels=2, pred_size=1, window_size=10, rnn_window=10, hidden_size=[8, 4]
    )
    mve = SeparateMVEARNN(
        input_channels=2,
        pred_size=1,
        window_size=10,
        rnn_window=10,
        mean_model=mean_model,
        var_hidden_size=[4, 4],
    )

    hom = HomoscedasticModel(mean_model, torch.tensor([0.3]))

    results_df = comp_batch([mve, hom], Sine_df, ["test"], names=["MVE", "Hom"])
    keys = list(results_df[0].filter(regex="out_").columns)

    assert keys[0] == "out_MVE"
    assert keys[1] == "out_MVE_var"
    assert keys[2] == "out_Hom"
    assert keys[3] == "out_Hom_var"


def test_comp_mean_metrics(Sine_df):
    mean_model = ARNN(
        input_channels=2, pred_size=1, window_size=10, rnn_window=10, hidden_size=[8, 4]
    )
    mve = SeparateMVEARNN(
        input_channels=2,
        pred_size=1,
        window_size=10,
        rnn_window=10,
        mean_model=mean_model,
        var_hidden_size=[4, 4],
    )

    hom = HomoscedasticModel(mean_model, torch.tensor([0.3]))

    errors = comp_mean_metrics(
        [mve, hom],
        Sine_df,
        fs=100,
        model_names=["MVE", "Hom"],
        metrics=["NLL", "CRPS", "ECE"],
    )

    keys = list(errors.columns)
    assert keys[0] == "MVE"
    assert keys[1] == "Hom"

    keys = list(errors.index)
    true_keys = ["out_NLL", "out_ECE", "out_CRPS"]

    for i, tk in enumerate(true_keys):
        assert keys[i] == tk
        assert errors["MVE"][tk] != 0


# %%
