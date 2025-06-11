# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:13:14 2021

@author: KRD2RNG
"""
# %%
import numpy as np
from pylife.stress.timesignal import psd_df
import scipy.signal as sg
import pandas as pd
from scipy.integrate import simpson
from softsensor.frequency_methods import get_amplitude
from softsensor.frequency_methods import get_phase
from softsensor.frequency_methods import psd_smoothing
from softsensor.frequency_methods import scale_PSD
from softsensor.frequency_methods import (
    psd_moment,
    psd_parameters_from_moments,
    psd_statistics,
)
from softsensor.frequency_methods import interpolate_log_series
from softsensor.frequency_methods import reshape_stpsd
from softsensor.frequency_methods import FDS
from softsensor.frequency_methods import calculate_coherence
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming
from pylife.stress.timesignal import psd_df
import pytest


# %%
def create_input_DF():
    fs = 1024
    t_end = 100
    t = np.arange(0, t_end, 1 / fs)
    ts_df = pd.DataFrame(
        {
            "Data_M_sin1": np.sin(100 * 2 * np.pi * t),
            "Data_M_sin2": 2 * np.sin(10 * 2 * np.pi * t + 0.05),
            "Data_M_cos": 10 * np.cos(20 * 2 * np.pi * t),
        },
        index=t,
    )
    ts_df.index.name = "t"
    vel_df = pd.DataFrame(
        55 + 55 * sg.sawtooth(2 * np.pi * np.arange(0, t_end, 0.01), width=0.5),
        index=np.arange(0, t_end, 0.01),
        columns=["Data_v_ts"],
    )
    return ts_df, vel_df


def test_scale_PSD():
    df_saw = pd.DataFrame(sg.sawtooth(16 * 2 * np.pi * np.linspace(1, 2, 4096))).abs()
    saw_scaled = scale_PSD(df_saw, 10, 100, 7)
    pd.testing.assert_frame_equal(saw_scaled, df_saw * (10 / 100) ** (2 / 7))


#


def test_psd_smooting_saw():
    df_saw = pd.DataFrame(sg.sawtooth(16 * 2 * np.pi * np.linspace(1, 2, 4096))).abs()
    saw_smoothed = psd_smoothing(df_saw, 512)[256:-256]
    pd.testing.assert_frame_equal(saw_smoothed, 0 * saw_smoothed + 0.5, rtol=1.0e-3)


def test_psd_smooting_dirac():
    df_dirac = pd.DataFrame(sg.unit_impulse(128, idx="mid"))
    dirac_smoothed = psd_smoothing(df_dirac, 4)
    assert dirac_smoothed.max()[0] == 0.5


#


def test_psd_moment():
    acc_df, _ = create_input_DF()
    psd_acc = psd_df(acc_df, 1024, 1024)
    moments = pd.concat([psd_moment(psd_acc, n_moment=n_m) for n_m in range(2)])
    pd.testing.assert_frame_equal(
        moments,
        pd.DataFrame(
            np.array([[0.5, 2.0, 50.0], [50.0, 20.0, 1000.0]]),
            index=pd.Index(["Moment_0", "Moment_1"], name="stats"),
            columns=acc_df.columns,
        ),
        rtol=1e-2,
    )


def test_psd_parameters_from_moments():
    acc_df, _ = create_input_DF()
    psd = psd_df(acc_df, 1024, 1024)
    moments = pd.concat([psd_moment(psd, n_moment=n_m) for n_m in range(5)])
    stats_para = psd_parameters_from_moments(moments)
    pd.testing.assert_frame_equal(
        stats_para,
        pd.DataFrame(
            np.array(
                [acc_df.std().to_numpy().T, [100, 10, 20], [100, 10, 20], [1, 1, 1]]
            ),
            columns=acc_df.columns,
            index=stats_para.index,
        ),
        rtol=1e-2,
    )


def test_psd_statistics():
    acc_df, _ = create_input_DF()
    psd = psd_df(acc_df, 1024)
    stats_data = psd_statistics(psd, n_moments=5)
    res = pd.Index(
        ["Moment_" + str(ii) for ii in range(5)]
        + ["rms", "v_0_plus", "v_p", "irregularity_factor"],
        name="stats",
    )

    pd.testing.assert_index_equal(stats_data.index.get_level_values("stats"), right=res)


def test_psd_statistics_mom_3():
    acc_df, _ = create_input_DF()
    psd = psd_df(acc_df, 1024)
    stats_data = psd_statistics(psd, n_moments=3)
    res = pd.Index(
        ["Moment_" + str(ii) for ii in range(5)]
        + ["rms", "v_0_plus", "v_p", "irregularity_factor"],
        name="stats",
    )

    pd.testing.assert_index_equal(stats_data.index.get_level_values("stats"), right=res)


@pytest.fixture
def sine_wave():
    fs = 1000  # Sampling frequency
    t = np.linspace(0, 1, fs)  # Time vector
    f = 10  # Frequency of the signal
    x = np.sin(2 * np.pi * f * t)  # Signal
    return fs, t, x


def test_get_phase_2d(sine_wave):
    nperseg = 1000
    fs, _, x = sine_wave
    stfft = ShortTimeFFT(
        win=hamming(nperseg), fs=fs, hop=nperseg // 2, scale_to="magnitude"
    )

    # Get complex spectrum
    c_spec = stfft.stft(x)

    # Calculate amplitude and max index
    _, max_idx = get_amplitude(stfft, c_spec)

    # Calculate phase
    phase_df = get_phase(stfft, c_spec, max_idx)

    np.testing.assert_almost_equal(phase_df.loc[0, 0], 0, decimal=1)
    np.testing.assert_almost_equal(phase_df.iloc[-1, 0], np.pi, decimal=1)


def test_get_phase_3d():
    acc, _ = create_input_DF()
    fs = 1024
    nperseg = 1024
    stfft = ShortTimeFFT(
        win=hamming(nperseg), fs=fs, hop=nperseg // 2, scale_to="magnitude"
    )
    c_spec = stfft.stft(acc.to_numpy(), axis=0)
    _, max_idx = get_amplitude(stfft, c_spec, columns=acc.columns)
    phase_df = get_phase(stfft, c_spec, max_idx, columns=acc.columns)
    np.testing.assert_almost_equal(phase_df.iloc[0, 0], np.pi, decimal=1)
    np.testing.assert_almost_equal(phase_df.iloc[0, 1], 0, decimal=1)


def test_get_amplitude(sine_wave):
    # Create test data

    nperseg = 1000
    fs, _, x = sine_wave
    # define the ShortTimeFFT object
    stfft = ShortTimeFFT(
        win=hamming(nperseg), fs=fs, hop=nperseg // 2, scale_to="magnitude"
    )
    # Complex spectrum
    c_spec = stfft.stft(x)
    # Calculate amplitude and max index
    amplitude, max_idx = get_amplitude(stfft, c_spec, columns=["sine"])

    np.testing.assert_almost_equal(amplitude.max().max(), 0.5, decimal=1)
    assert amplitude.shape[0] == nperseg // 2 + 1


def test_reshape_stpsd():
    stpsd = np.random.rand(1025, 2, 25)
    windows_new = 10

    stpsd_reshaped = reshape_stpsd(stpsd, windows_new)

    assert stpsd_reshaped.shape == (1025, 2, windows_new)
    np.testing.assert_allclose(
        stpsd_reshaped.mean(axis=-1), stpsd.mean(axis=-1), rtol=5e-1
    )


def test_reshape_stpsd_zeros():
    stpsd = np.zeros((1025, 2, 25))
    windows_new = 10

    stpsd_reshaped = reshape_stpsd(stpsd, windows_new)

    assert stpsd_reshaped.shape == (1025, 2, windows_new)
    assert stpsd_reshaped.sum() == stpsd.sum()


def test_interpolate_log_series():
    # Create a sample series with logarithmic spacing
    series = pd.Series(
        [1, 10, 100, 1000],
        index=pd.Index([1, 10, 100, 1000], name="original_index"),
        name="test_series",
    )
    # Define a new index for interpolation
    new_index = pd.Index([2, 20, 200, 2000], name="new_index")

    # Perform interpolation
    interpolated_series = interpolate_log_series(series, new_index)

    # Expected values based on logarithmic interpolation
    expected_values = [2.0, 20.0, 200.0, 1]
    expected_series = pd.Series(expected_values, index=new_index, name="test_series")

    # Assert the interpolated series matches the expected series
    pd.testing.assert_series_equal(interpolated_series, expected_series, rtol=1e-3)


def test_interpolate_log_series_with_left_right():
    # Create a sample series with logarithmic spacing
    series = pd.Series(
        [1, 10, 100, 1000],
        index=pd.Index([1, 10, 100, 1000], name="original_index"),
        name="test_series",
    )
    # Define a new index with values outside the original range
    new_index = pd.Index([0.1, 2, 20, 200, 2000, 5000], name="new_index")

    # Perform interpolation with left and right fill values
    interpolated_series = interpolate_log_series(
        series, new_index, left=None, right=None
    )

    # Expected values with left=0 and right=0
    expected_values = [1.0, 2.0, 20.0, 200.0, 1000.0, 1000.0]
    expected_series = pd.Series(expected_values, index=new_index, name="test_series")

    # Assert the interpolated series matches the expected series
    pd.testing.assert_series_equal(interpolated_series, expected_series, rtol=1e-3)


def test_fds_calculate_fds_disp():
    psd = pd.Series(
        np.ones(12),
        index=[1, 10, 19, 28, 37, 46, 55, 64, 73, 82, 91, 100],
    )
    fds = FDS(response_type="disp")
    expected_fds = pd.Series(
        [
            2.232e-7,
            7.697e-12,
            3.122e-13,
            4.494e-14,
            1.114e-14,
            3.737e-15,
            1.519e-15,
            7.033e-16,
            3.558e-16,
            1.889e-16,
            9.579e-17,
            2.211e-17,
        ],
        index=[1, 10, 19, 28, 37, 46, 55, 64, 73, 82, 91, 100],
        name="None_FDS_disp",
    )
    pd.testing.assert_series_equal(
        fds.calculate_fds(psd).iloc[1:], expected_fds.iloc[1:], rtol=1e-2
    )


def test_fds_calculate_fds_vel():
    psd = pd.Series(
        np.ones(12),
        index=[1, 10, 19, 28, 37, 46, 55, 64, 73, 82, 91, 100],
    )
    fds = FDS(response_type="vel")
    expected_fds = pd.Series(
        [
            0,
            1.27e-04,
            6.67e-05,
            4.52e-05,
            3.42e-05,
            2.75e-05,
            2.30e-05,
            1.98e-05,
            1.74e-05,
            1.55e-05,
            1.39e-05,
            1.27e-05,
        ],
        index=[1, 10, 19, 28, 37, 46, 55, 64, 73, 82, 91, 100],
        name="None_FDS_vel",
    )
    pd.testing.assert_series_equal(
        fds.calculate_fds(psd).iloc[1:], expected_fds.iloc[1:], rtol=1e-2
    )


def test_fds_calculate_psd_vel():

    fds = FDS(response_type="vel")
    fds_val = pd.Series(
        [
            1.27e-04,
            6.67e-05,
            4.52e-05,
            3.42e-05,
            2.75e-05,
            2.30e-05,
            1.98e-05,
            1.74e-05,
            1.55e-05,
            1.39e-05,
            1.27e-05,
        ],
        index=[10, 19, 28, 37, 46, 55, 64, 73, 82, 91, 100],
    )
    psd_expected = pd.Series(
        np.ones(11),
        index=[10, 19, 28, 37, 46, 55, 64, 73, 82, 91, 100],
        name="None_PSD_vel_based",
    )
    pd.testing.assert_series_equal(fds.calculate_psd(fds_val), psd_expected, rtol=1e-2)


def test_fds_calculate_psd_disp():

    fds = FDS(response_type="disp")
    fds_val = pd.Series(
        [
            8.13e-12,
            3.28e-13,
            4.72e-14,
            1.17e-14,
            3.95e-15,
            1.62e-15,
            7.57e-16,
            3.92e-16,
            2.19e-16,
            1.30e-16,
            8.13e-17,
        ],
        index=[10, 19, 28, 37, 46, 55, 64, 73, 82, 91, 100],
    )
    psd_expected = pd.Series(
        np.ones(11),
        index=[10, 19, 28, 37, 46, 55, 64, 73, 82, 91, 100],
        name="None_PSD_disp_based",
    )
    pd.testing.assert_series_equal(fds.calculate_psd(fds_val), psd_expected, rtol=1e-2)


def test_calculate_coherence():

    # Use existing test data generator
    acc_df, _ = create_input_DF()

    # Calculate coherence
    coherence_df = calculate_coherence(acc_df, acc_df, nperseg=1024, fs=1024)
    assert np.isclose(coherence_df.iloc[:, 0].mean(), 1)


# %%
