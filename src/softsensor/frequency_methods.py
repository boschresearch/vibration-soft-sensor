# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings
from scipy.special import gamma
from scipy.signal import coherence


def scale_PSD(psd, t_original, t_new=60, b=8):
    """
    scaling PSD to new test duration time

    Parameters
    ----------
    t_new : float, optional
        time new of the PSD. The default is 60.
    b : float, optional
        SN slope. The default is 8.


    Returns
    -------
    new scaled PSDs as Series

    """
    PSD_scaled = psd.multiply((t_original / t_new) ** (2 / b), axis=0)
    return PSD_scaled


def psd_smoothing(psd_df, window_size=16):
    """
    smoothing of PSD using convolution in frequency domain

    Parameters
    ----------
    psd_df : DataFrame
        Raw PSD
    window_size : int, optional
        number of frequency points for window function. The default is 16.

    Returns
    -------
    smooth_df : DataFrame
        DESCRIPTION.

    """

    conv_window = (
        1
        + np.cos(
            2 * np.pi * np.arange(-window_size / 2 + 1, window_size / 2) / window_size
        )
    ) / window_size
    smooth_df = psd_df.apply(np.convolve, args=(conv_window, "same"))
    return smooth_df


def get_amplitude(sft, complex_stft, method=np.max, columns=None, t_axis=-1):
    """
    Calculate the amplitude of the complex Short-Time Fourier Transform (STFT).

    Parameters:
        sft (object): The Short-Time Fourier Transform object from scipy.signal.
        complex_stft (ndarray): The complex STFT array.
        columns (list or ndarray, optional): The column labels for the resulting DataFrame.

    Returns:
        amplitude (DataFrame): The amplitude of the complex STFT.
        max_idx (ndarray): The indices of the maximum values in the complex STFT.
    """
    complex_stft = np.nan_to_num(complex_stft)
    if len(complex_stft.shape) == 2:
        complex_stft = complex_stft[:, np.newaxis, :]
    amplitude = pd.DataFrame(
        np.apply_along_axis(method, t_axis, np.abs(complex_stft)),
        index=pd.Index(sft.f, name="frequency"),
        columns=pd.Index(
            columns if columns is not None else np.arange(complex_stft.shape[-2]),
            name="index",
        ),
    ).dropna(axis=0)
    max_idx = np.argmax(np.abs(complex_stft), axis=-1, keepdims=True)
    return amplitude, max_idx


def get_phase(sft, complex_stft, max_idx, columns=None):
    """
    Calculate the phase of the complex STFT at the given maximum index.

    Parameters:
    - sft: The Short-Time Fourier Transform object.
    - complex_stft: The complex STFT matrix.
    - max_idx: The maximum index to take along the last axis of the complex STFT matrix.
    - columns: Optional parameter specifying the column names for the resulting DataFrame.

    Returns:
    - phase: DataFrame containing the phase values of the complex STFT.

    """
    if len(complex_stft.shape) == 2:
        complex_stft = complex_stft[:, np.newaxis, :]

    phase = pd.DataFrame(
        np.angle(np.take_along_axis(complex_stft, max_idx, axis=-1)[:, :, 0]),
        index=pd.Index(sft.f, name="frequency"),
        columns=pd.Index(
            columns if columns is not None else np.arange(complex_stft.shape[-2]),
            name="index",
        ),
    ).dropna(axis=0)
    return phase


def psd_moment(psd_df, n_moment=0):
    """
    calculate the moment of a given psd

    Parameters
    ----------
    psd_df : DataFrame
        index is the corresponding frequency
    n_moment : int, optional
        nth moment. The default is 0.

    Returns
    -------
    df
        DataFrame.

    """
    # ToDo: make it valid for single index "freq"
    frequency = psd_df.index
    moment = np.trapezoid(
        psd_df.multiply(np.power(frequency, n_moment), axis=0), frequency, axis=0
    )
    moment_df = pd.DataFrame(
        moment, index=psd_df.columns, columns=["Moment_" + str(n_moment)]
    ).T

    moment_df.index.rename("stats", inplace=True)
    return moment_df


def psd_parameters_from_moments(psd_stats):
    """
    Calculate various parameters from the power spectral density (PSD) moments.

    Args:
        psd_stats (pd.DataFrame): DataFrame containing PSD moments.

    Returns:
        pd.DataFrame: DataFrame containing calculated parameters including RMS, v_0_plus,
                      v_p, and irregularity_factor.
    """
    psd_stats = psd_stats.set_index(
        pd.Index(np.ones(len(psd_stats)), name="dummy"), append=True
    )

    # psd_stats = psd_stats.drop("freq", axis=1)#.filter(regex="Data_")
    rms = (
        psd_stats.rename(index={"Moment_0": "rms"}).xs(
            "rms", level=-2, drop_level=False
        )
        ** 0.5
    )

    v_0_plus = (
        psd_stats.xs("Moment_2", level=-2, drop_level=False).divide(
            psd_stats.xs("Moment_0", level=-2, drop_level=False).to_numpy()
        )
        ** 0.5
    )

    v_0_plus = v_0_plus.rename(index={"Moment_2": "v_0_plus"})

    v_p = (
        psd_stats.xs("Moment_4", level=-2, drop_level=False).divide(
            psd_stats.xs("Moment_2", level=-2, drop_level=False).to_numpy()
        )
    ) ** 0.5
    v_p = v_p.rename(index={"Moment_4": "v_p"})
    irregularity_factor = (
        v_0_plus.rename(index={"v_0_plus": "irregularity_factor"}) / v_p.to_numpy()
    )
    stats_df = pd.concat((rms, v_0_plus, v_p, irregularity_factor))
    stats_df = stats_df.droplevel("dummy")
    return stats_df


def psd_statistics(psd_df, n_moments=5):
    """
    Calculate statistics of Power Spectral Density (PSD) data.

    Args:
        psd_df (DataFrame): Input DataFrame containing PSD data.
        n_moments (int, optional): Number of moments to calculate. Defaults to 5.

    Returns:
        DataFrame: DataFrame containing the calculated PSD statistics.

    Raises:
        UserWarning: If n_moments is less than or equal to 4, it is set to 5.

    """
    if n_moments <= 4:
        n_moments = 5
        warnings.warn("n_moments must be >= 5")
    psd_stats = pd.concat((psd_moment(psd_df, ii) for ii in range(n_moments)))
    return pd.concat((psd_stats, psd_parameters_from_moments(psd_stats)))


def _interpolate_slice(slice, x_original, x_new):
    interp_func = np.interp(x_new, x_original, slice)
    return interp_func


def reshape_stpsd(stpsd, windows_new):
    """
    Reshapes the Short-Time Power Spectral Density (STPSD) array to match the desired number of windows.

    Parameters:
        stpsd (ndarray): The input STPSD array.
        windows_new (int): The desired number of windows.

    Returns:
        ndarray: The reshaped STPSD array.

    """
    stpsd_reshaped = np.apply_along_axis(
        _interpolate_slice,
        -1,
        stpsd,
        np.linspace(0, 1, stpsd.shape[-1]),
        np.linspace(0, 1, windows_new),
    )
    return stpsd_reshaped


# %%
def interpolate_log_series(series, new_index, left=0, right=0):
    """
    Interpolate the log values of a series based on the log values of a new index.

    Parameters:
    - series (pd.Series): The input series.
    - new_index (pd.Index): The new index.

    Returns:
    - pd.Series: The interpolated log series.

    """
    log_series = np.log(series)
    log_interp_func = np.interp(
        np.log(new_index.values),
        np.log(series.index),
        log_series,
        left=left,
        right=right,
    )
    interpolated_series = np.exp(log_interp_func)
    return pd.Series(interpolated_series, index=new_index, name=series.name)


class FDS:
    def __init__(
        self,
        Q=10,
        K=1.0,
        b=4.0,
        C=1e3,
        duration=1.0,
        response_type="disp",
        scaled=False,
    ):
        """
        Initialize the FDS class with the given parameters.

        Args:
            Q (float): Dynamic amplification factor. Default is 10.
            K (float): The stiffness of the SDOF system. Default is 1.0.
            b (float): The fatigue strength exponent. Default is 4.0.
            C (float): Basquin coefficient. Default is 1e3.
            duration (float): The duration of the excitation in seconds. Default is 1.0.
            response_type (str): Type of response ('disp', 'vel', 'acc'). Default is 'disp'.
            scaled (bool): Whether to scale the FDS. Default is False.
        """
        self.Q = Q
        self.K = K
        self.b = b
        self.C = C
        self.duration = duration
        self.response_type = response_type
        self.scaled = scaled

    def calculate_fds(self, psd):
        """
        Calculate the Fatigue Damage Spectrum (FDS) from the Power Spectral Density (PSD). According to the formula in the paper:
        https://http://www.vibrationdata.com/tutorials_alt/fatigue_damage_spectra.pdf

        Parameters:
        -----------
        psd : pandas.Series
            The power spectral density data. The index should represent the frequency values.
        Returns:
        --------
        pandas.Series
            The calculated FDS values with the same index as the input PSD and a modified name indicating the response type.
        Notes:
        ------
        - The method modifies the PSD based on the response type ('vel' for velocity or 'acc' for acceleration).
        - The FDS calculation involves several parameters:
            - self.duration: Duration of the signal.
            - self.K, self.b, self.C, self.Q: Constants used in the FDS calculation.
            - gamma: Gamma function from the scipy library.
        - If self.scaled is True, the FDS is scaled by the integral of the FDS over the frequency range.
        """

        self.psd = psd.copy()
        f = psd.index.values
        if self.response_type == "vel":
            psd = psd * (2 * np.pi * f) ** 2
        if self.response_type == "acc":
            psd = psd * (2 * np.pi * f) ** 4
        fds = (
            f
            * self.duration
            * (self.K**self.b / self.C)
            * (self.Q * psd.to_numpy() / (2 * (2 * np.pi * f) ** 3)) ** (self.b / 2)
            * gamma(1 + self.b / 2)
        )
        if self.scaled:
            fds = fds / np.trapezoid(fds, f)
        name = str(psd.name) if not isinstance(psd.name, str) else fds.name
        return pd.Series(
            fds, index=self.psd.index, name=name + "_FDS_" + self.response_type
        )

    def calculate_psd(self, fds):
        """
        Calculate the Power Spectral Density (PSD) for a given Fatigue Damage Spectrum (FDS).
        Parameters:
        -----------
        fds : array-like
            The Fatigue Damage Spectrum (FDS) values.
        Returns:
        --------
        pd.Series
            A pandas Series containing the calculated PSD values, indexed by the same index as `self.psd` and named
            with the original name appended with "_PSD_" and the response type.
        Notes:
        ------
        - The PSD is calculated for a displacement-based FDS.
        - The PSD is rescaled based on the response type:
            - If `self.response_type` is "vel", the PSD is divided by (2 * np.pi * f)^2.
            - If `self.response_type` is "acc", the PSD is divided by (2 * np.pi * f)^4.
        """

        f = fds.index.values
        psd = (
            (
                fds
                / (
                    f
                    * self.duration
                    * (self.K**self.b / self.C)
                    * gamma(1 + self.b / 2)
                )
            )
            ** (2 / self.b)
            * (2 * (2 * np.pi * f) ** 3)
            / self.Q
        )  # that is the psd for a displacement based FDS
        # rescaling
        if self.response_type == "vel":
            psd = psd / (2 * np.pi * f) ** 2
        if self.response_type == "acc":
            psd = psd / (2 * np.pi * f) ** 4

        name = str(fds.name) if not isinstance(fds.name, str) else fds.name
        name += "_PSD_" + self.response_type + "_based"
        return pd.Series(psd, index=fds.index, name=name)


def calculate_coherence(df1, df2, nperseg=2048, fs=None):
    """
    Calculate the coherence between all input and all output sensors.

    Parameters:
    df1 (pd.DataFrame): DataFrame containing the first set of signals.
    df2 (pd.DataFrame): DataFrame containing the second set of signals.
    fs (int): Sampling frequency. Default is 4096.
    nperseg (int): Length of each segment for the coherence calculation. Default is 2048.

    Returns:
    pd.DataFrame: DataFrame containing the coherence values with frequency as the index.
    """
    if fs == None:
        fs = np.round(np.mean((1 / df1.index.diff().total_seconds().dropna())))
    coherence_dict = {}
    for col1 in df1.columns:
        for col2 in df2.columns:
            f, Cxy = coherence(df1[col1], df2[col2], fs=int(fs), nperseg=nperseg)
            coherence_dict[(col1, col2)] = Cxy

    coherence_df = pd.DataFrame(coherence_dict, index=f)
    coherence_df.index.name = "frequency"
    return coherence_df
