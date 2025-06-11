# %%
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from softsensor.metrics import *
from softsensor.datasets import SlidingWindow
from softsensor.autoreg_models import QuantileARNN


# %%
def dataframe(le):
    d = {
        "in_col1": np.linspace(0, 100, le),
        "in_col2": np.sin(np.linspace(0, 100, le)),
        "out_col1": np.cos(np.linspace(0, 100, le)),
        "out_col2": np.linspace(0, 20, le),
    }

    return pd.DataFrame(d)


def test_compute_quantile_metrics():
    windowsize = 20
    input_col = ["in_col1", "in_col2"]
    output_col = ["out_col1", "out_col2"]

    data = SlidingWindow(
        dataframe(le=1000), windowsize, output_col, input_col, rnn_window=20
    )
    loader = [DataLoader(data, shuffle=True, batch_size=1)]

    torch.random.manual_seed(0)

    model = QuantileARNN(
        input_channels=2,
        pred_size=2,
        window_size=windowsize,
        hidden_size=[8, 4],
        activation="leaky_relu",
        rnn_window=20,
    )

    scores = compute_quantile_metrics(model, loader)
    assert list(scores.keys()) == [
        "x_RMSE",
        "x_MAE",
        "x_R2",
        "x_Corr",
        "x_PICP",
        "x_MPIW",
        "x_ECE",
    ]


def test_distance_calc():
    # Create a simple inverse CDF dataframe
    index = np.linspace(0, 1, 10)
    data = {
        "A": np.linspace(0, 10, 10),
        "B": np.linspace(0, 5, 10),
        "C": np.linspace(5, 15, 10),
    }
    inverse_cdf = pd.DataFrame(data, index=index)

    # Test with p=1
    result_p1 = distance_calc(inverse_cdf, 1)

    # Check dimensions and types
    assert isinstance(result_p1, pd.DataFrame)
    assert result_p1.shape == (3, 3)
    assert list(result_p1.index) == ["A", "B", "C"]
    assert list(result_p1.columns) == ["A", "B", "C"]

    # Check specific values
    assert np.isclose(result_p1.loc["A", "A"], 0.0)
    assert np.isclose(result_p1.loc["A", "B"], 2.5)
    assert np.isclose(result_p1.loc["B", "C"], 7.5)

    # Test with p=2
    result_p2 = distance_calc(inverse_cdf, 2)

    # Test symmetry property
    for col1 in result_p1.columns:
        for col2 in result_p1.columns:
            assert np.isclose(result_p1.loc[col1, col2], result_p1.loc[col2, col1])
            assert np.isclose(result_p2.loc[col1, col2], result_p2.loc[col2, col1])


def test_wasserstein_class():
    np.random.seed(42)
    # Create test data
    n = 100
    data = pd.DataFrame(
        {
            "A": np.random.normal(0, 1, n),
            "B": np.random.normal(2, 1, n),
            "C": np.random.normal(5, 2, n),
        }
    )

    # Create weights (optional)
    weights = pd.DataFrame({"A": np.ones(n), "B": np.ones(n), "C": np.ones(n)})

    # Initialize WassersteinDistance object
    wd = WassersteinDistance(data, weights)

    # Test sort_data method
    indices, sorted_data, sorted_weights = wd.sort_data()
    pd.testing.assert_series_equal(
        data.min(), sorted_data.iloc[0, :], check_names=False
    )
    pd.testing.assert_series_equal(
        data.max(), sorted_data.iloc[-1, :], check_names=False
    )

    # # Test weighted_hist method
    wd.weighted_hist(bins=20, nfft=50, nperseg=50)
    assert np.isclose(wd.hist["A"][0].sum(), 4.472248)
    assert np.isclose(wd.psd.index.max(), 0.5)
    assert np.isclose(wd.psd.index.min(), 0.02)
    np.testing.assert_almost_equal(
        wd.NPSD.sum().to_numpy(), np.array([1.0, 1.0, 1.0]), decimal=1
    )
    assert np.isclose(wd.wsfourier_distribution["A"].mean(), 0.2676841)
    pdf_result = wd.pdf(n_points=500)
    np.testing.assert_almost_equal(
        np.trapezoid(pdf_result, axis=0, x=pdf_result.index),
        np.array([1.0, 1.0, 1.0]),
        decimal=1,
    )

    cdf_result = wd.cdf(n_points=500)
    np.testing.assert_almost_equal(
        cdf_result.max().to_numpy(),
        np.array([1.0, 1.0, 1.0]),
        decimal=2,
    )

    inv_cdf_result = wd.inverse_cdf(n_points=100)
    assert np.isclose(inv_cdf_result.index.max(), 1.0)

    wsf_pdf = wd.wsf_pdf(n_points=500)
    np.testing.assert_almost_equal(
        np.trapezoid(wsf_pdf, axis=0, x=wsf_pdf.index),
        np.array([1.0, 1.0, 1.0]),
        decimal=1,
    )

    wsf_cdf = wd.wsf_cdf(n_points=500)
    np.testing.assert_almost_equal(
        wsf_cdf.max().to_numpy(),
        np.array([1.0, 1.0, 1.0]),
        decimal=1,
    )

    wsf_inv_cdf = wd.wsf_inverse_cdf(n_points=100)
    assert np.isclose(wsf_inv_cdf.index.max(), 1.0)


def test_wasserstein_distance():
    signal = dataframe(1000)
    wd = WassersteinDistance(signal)
    w_distance = wd.wasserstein_distance_p(p=1)

    np.testing.assert_almost_equal(
        w_distance.iloc[:, 0].to_numpy(),
        np.array([0.0, 50.0, 50.0, 40.0]),
        decimal=1,
    )


def test_wasserstein_fourier_distance():
    N = 256  # number of samples
    f = [2, 4]
    t = np.linspace(0, 8, N + 1)
    signal = pd.DataFrame(
        {f"{f}": np.sin(2 * np.pi * f * t) for f in f}, index=pd.Index(t, name="time")
    )
    wd = WassersteinDistance(signal)
    wsf_distance = wd.wasserstein_fourier_distance(p=2)
    assert np.isclose(wsf_distance.iloc[0, 1], 2.0, atol=1e-2)
