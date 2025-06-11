# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:04:39 2022

@author: Daniel Kreuter
"""
import numpy as np
import pandas as pd
from scipy.signal import csd
import scipy.linalg as la
from scipy.signal import ShortTimeFFT, get_window


class tf:
    """
    linear methods based on FFT and short term FFT
    Calculate the linear MiMo Transferfunction similar to Matlab tfestimate:
    https://uk.mathworks.com/help/signal/ref/tfestimate.html#bufqg8e

    Parameters
    ----------
    NFFT : int
        Length of the fft. The default ist 512.
    fs : int
        sample frequency. The default ist 1024.
    no_overlap : int
        Points overlapping. If None, than it is defined by no_overlap = np.fix(0.67 * NFFT)
        The default is None.
    spectrum_type : str:
        'spectrum' or 'psd'. The default is 'spectrum'.

    Returns
    -------
    None.

    Example
    -------
    Computing the linear transfer function of a linear system with white noise
    excitation

    >>> import softsensor.data_gen as dg
    >>> import softsensor.linear_methods as lm
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> time = np.linspace(0, 100, 1000)
    >>> params = {'D': 0.05, 'c_nlin': 0}
    >>> F = dg.white_noise(time)
    >>> df = dg.get_academic_data(time, Model='Duffing', F=F, params=params)
    >>> model = lm.tf(NFFT=512, fs=fs, no_overlap=256)
    >>> model.fit([df], ['F(t)'], ['x'])
    >>> output = model.prediction(df, ['x'])
    >>> plt.plot(df.index, df['x'], label='original')
    >>> plt.plot(df.index, output, label='prediciton')

    .. image:: C:/Users/wet2rng/Desktop/Coding/SoftSensor/doc/img/lienar_tf.png

    """

    def __init__(self, window_size=512, fs=1024, hop=256, spectrum_type="spectrum"):
        super().__init__()
        self.Type = "TF"
        self.window_size = window_size
        self.fs = fs
        self.hop = hop
        self.spectrum_type = spectrum_type
        # Short-Time Fourier Transformation Object
        self.window = get_window("hann", self.window_size)

        self.STFT = ShortTimeFFT(
            self.window,
            hop=self.hop,
            fs=self.fs,
            scale_to="magnitude" if spectrum_type == "spectrum" else "psd",
        )

    def fit(self, df_list, inp_sens, out_sens):
        """
        Fit linear TF to list of dfs by taking the mean transfer function

        Parameters
        ----------
        df_list : list of pd.DataFrame
            list of training Files.
        inp_sens : list of str
            input sensors.
        out_sens : list of str
            output sensors.

        Returns
        -------
        None.

        """
        self.inp_sens = inp_sens
        self.out_sens = out_sens

        tf_list = []
        for df in df_list:
            tf, freq = self.tf_mimo(df[self.inp_sens], df[self.out_sens])
            tf_list.append(tf)
        self.tf = np.mean(np.stack(tf_list, axis=1), axis=1)
        self.frequency = freq

    def prediction(self, test_df, columns=None, boundary=True):
        """
        Predict test_df

        Parameters
        ----------
        test_df : pd.DataFrame
            dtaFrame for the precition, needs columns that have the same name
            as in fit.
        columns : list of str, optional
            names for the prediction. The default is None.

        Returns
        -------
        ts_pred : pd.DataFrame
            returns dataframe with linear_TF prediction as column.

        """
        out_sfft = self.sfft(test_df[self.inp_sens])
        out_sfft = self.tf @ np.moveaxis(out_sfft, 0, 1)
        ts_pred = self.isfft(
            np.moveaxis(out_sfft, 1, 0), columns=columns, boundary=boundary
        )
        if columns is None:
            ts_pred = ts_pred[: len(test_df), :]
        else:
            ts_pred = ts_pred.iloc[: len(test_df), :]
        return ts_pred

    def sfft(self, ts):
        """
        computes the short term fast Fourier transformation (stft) for each input column of the
        DataFrame

        Parameters
        ----------
        ts : pd.DataFrame
            pandas data frame.
        Returns
        -------
        STFT : array
            The Fourier transformed signal.

        """

        self.signal_len = ts.shape[0]

        Zxx = self.STFT.stft(ts.T.to_numpy())

        Zxx = 2 * Zxx  # one sided spectrum !!!
        self.frequency = self.STFT.f
        return Zxx

    def _get_rayleigh_ampl(self, Zxx):
        rg = np.random.default_rng()
        ampl = np.abs(Zxx)
        phase = np.angle(Zxx)
        ampl_rayleigh = rg.rayleigh(scale=ampl)
        return ampl_rayleigh * np.exp(1j * phase)

    def isfft(self, Zxx, columns=None, ampl_dist=None, boundary=True):
        """
        Computes the inverse Fourier transformation

        Parameters
        ----------
        Zxx : Array,
            Fourier transformed signal.
        columns : list of col names, optional
            Column names of the output df. The default is None.
        ampl_dist : None, str, optional
            if STFT is a PSD, you can define the amplitude distribution with a Raleigh or mixed distribution.
            Options are *None*, *'rayleigh'* or *'mixed'*.
            The default is None.

        Returns
        -------
        TS_IFFT : TYPE
            DESCRIPTION.

        """
        if ampl_dist == "rayleigh":
            Zxx = self._get_rayleigh_ampl(Zxx)
        if ampl_dist == "mixed":
            Zxx = 0.5 * (Zxx + self._get_rayleigh_ampl(Zxx))

        out = self.STFT.istft(
            0.5 * Zxx
        )  # Rescaling  with 0.5: We have stored the stft with factor 2

        out = out.T
        if columns is not None:
            out = pd.DataFrame(
                out,
                columns=columns,
                index=pd.Index(
                    np.linspace(
                        0, out.shape[0] / self.fs, out.shape[0], endpoint=False
                    ),
                    name="time",
                ),
            )
            out = out.iloc[: self.signal_len]
        else:
            out = out[: self.signal_len]

        return out

    def tf_mimo(self, inp_df, out_df):
        """
        ToDo: replace with scipy.signal.csd
        Calculate the linear MiMo Transferfunction similar to Matlab tfestimate:
        https://uk.mathworks.com/help/signal/ref/tfestimate.html#bufqg8e

        Parameters
        ----------
        inp_df: pandas DataFrame
            input x(t) (index is time stamp),
        out_df: pandas DataFrame
            output y(t) (index is time stamp),
        Returns
        -------
        tf: numpy array
            transfer function of shape (freq steps,no of out signals,
                                        no of input signal), complex
        f: numpy array
        corresponding frequency array
        """

        csd_tot = np.empty(
            (int(self.STFT.mfft / 2 + 1), len(out_df.columns), len(inp_df.columns)),
            dtype=np.complex128,
        )
        csd_tot_in = np.empty(
            (int(self.STFT.mfft / 2 + 1), len(inp_df.columns), len(inp_df.columns)),
            dtype=np.complex128,
        )
        ii = 0
        for colin in inp_df:
            oo = 0
            ii_in = 0
            for colout in out_df:
                freq, csd_akt = csd(
                    inp_df[colin].to_numpy(),
                    out_df[colout].to_numpy(),
                    nfft=self.STFT.mfft,
                    fs=self.fs,
                    nperseg=self.STFT.mfft,
                    detrend=False,
                    noverlap=int(self.window_size - self.hop),
                )  # with hanning window
                csd_tot[:, oo, ii] = csd_akt
                oo += 1
            for colin_psd in inp_df:
                _, csd_in = csd(
                    inp_df[colin].to_numpy(),
                    inp_df[colin_psd].to_numpy(),
                    nfft=self.STFT.mfft,
                    fs=self.fs,
                    nperseg=self.STFT.mfft,
                    detrend=False,
                    noverlap=int(self.window_size - self.hop),
                )  # with hanning window
                csd_tot_in[:, ii_in, ii] = csd_in
                ii_in += 1
            ii += 1
        csd_tot_in_inv = np.empty_like(csd_tot_in)
        for ii in range(int(self.STFT.mfft / 2 + 1)):
            csd_tot_in_inv[ii, :, :] = la.pinv(csd_tot_in[ii, :, :])
        return csd_tot @ csd_tot_in_inv, freq
