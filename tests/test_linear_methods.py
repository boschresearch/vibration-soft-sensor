# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:19:11 2021

@author: WET2RNG"
"""
# %%
from softsensor.linear_methods import tf
import numpy as np
import pandas as pd
from scipy import signal as sg
from softsensor.data_gen import white_noise, sine, sweep, get_academic_data


# %%
def create_input_DF():
    fs = 1024
    t_end = 10
    t = np.arange(0, t_end, 1 / fs)
    ts_df = pd.DataFrame(
        {
            "Data_M_sin1": np.sin(100 * 2 * np.pi * t),
            "Data_M_sin2": 2 * np.sin(3 * 2 * np.pi * t + 0.05),
            "Data_M_cos": 10 * np.cos(5 * 2 * np.pi * t),
        },
        index=t,
    )
    ts_df.index.name = "time"
    return ts_df


# %%
def test_sfft():
    ts_df = create_input_DF()
    tf_class = tf(window_size=1024, fs=1024, hop=512)
    sfft = tf_class.sfft(ts_df)
    spectrum = pd.DataFrame(
        np.abs(sfft).max(axis=-1).T, columns=ts_df.columns, index=tf_class.frequency
    )
    pd.testing.assert_series_equal(spectrum.max(), ts_df.max())


def test_isfft():
    ts_df = create_input_DF()
    tf_class = tf(window_size=512, fs=1024, hop=256)
    sfft = tf_class.sfft(ts_df)
    t_isfft = tf_class.isfft(sfft, columns=ts_df.columns)
    pd.testing.assert_frame_equal(t_isfft.describe(), ts_df.describe())


def test_isfft_psd_normal():
    fs = 1024
    t_end = 10
    t = np.arange(0, t_end, 1 / fs)
    F = white_noise(t)
    ts_df = pd.DataFrame(F.F, index=F.time, columns=["wn"])
    tf_class = tf(window_size=512, fs=fs, spectrum_type="psd", hop=256)
    sfft = tf_class.sfft(ts_df)
    t_isfft = tf_class.isfft(sfft, columns=ts_df.columns)
    pd.testing.assert_frame_equal(t_isfft.describe(), ts_df.describe())


def test_isfft_psd_rayleigh():
    fs = 1024
    t_end = 10
    t = np.arange(0, t_end, 1 / fs)
    F = white_noise(t)
    ts_df = pd.DataFrame(F.F, index=F.time, columns=["wn"])
    tf_class = tf(window_size=512, fs=fs, spectrum_type="psd", hop=256)
    sfft = tf_class.sfft(ts_df)
    t_isfft = tf_class.isfft(sfft, columns=ts_df.columns, ampl_dist="rayleigh")
    pd.testing.assert_series_equal(ts_df.count(), t_isfft.count())


def test_isfft_psd_mixed():
    fs = 1024
    t_end = 10
    t = np.arange(0, t_end, 1 / fs)
    F = white_noise(t)
    ts_df = pd.DataFrame(F.F, index=F.time, columns=["wn"])
    tf_class = tf(window_size=512, fs=fs, spectrum_type="psd", hop=256)
    sfft = tf_class.sfft(ts_df)
    t_isfft = tf_class.isfft(sfft, columns=ts_df.columns, ampl_dist="mixed")
    pd.testing.assert_series_equal(ts_df.count(), t_isfft.count())


def test_mimo_simple_series():
    ts_df = create_input_DF()
    tf_class = tf(window_size=512, fs=1024, hop=256)
    tf_class.fit([ts_df], ["Data_M_sin1"], ["Data_M_sin1"])
    ts_pred = tf_class.prediction(ts_df, ["Data_M_sin1"])
    pd.testing.assert_frame_equal(ts_pred.describe(), ts_df[["Data_M_sin1"]].describe())


def test_mimo_simple_df():
    ts_df = create_input_DF()
    tf_class = tf(window_size=512, fs=1024, hop=256)
    tf_class.fit([ts_df], ts_df.columns, ts_df.columns)
    ts_pred = tf_class.prediction(ts_df, ts_df.columns)
    pd.testing.assert_frame_equal(
        ts_pred.describe(), ts_df.describe(), atol=1e-1, rtol=1e-2
    )


# %%
def test_mimo():
    ems_params = {"D": 0.5, "c_lin": 2e3, "c_nlin": 0}

    ode_type = "Duffing"

    fs = 1024
    end_t = 10
    steps = end_t * fs
    time = np.linspace(0, end_t, steps)

    F = white_noise(time)
    train_df = get_academic_data(time, ode_type, ems_params, F, x0=[0, 0])

    end_t = 60
    steps = end_t * fs
    time = np.linspace(0, end_t, steps)
    F = sweep(time, f0=1, f1=15, t1=end_t)
    # test_df = get_academic_data(time, ode_type, ems_params, F, x0=[0, 0])
    test_df = get_academic_data(
        time, ode_type, ems_params, white_noise(time), x0=[0, 0]
    )

    inp_sens = ["F(t)"]
    out_sens = ["x"]

    model = tf(window_size=512, fs=fs, hop=256)
    model.fit([train_df], inp_sens, out_sens)
    # pd.Series(np.abs(model.tf[:,0,0])).plot()

    output = model.prediction(test_df, ["x"])
    pd.testing.assert_frame_equal(
        test_df["x"].to_frame().describe(), output.describe(), atol=1e-1, rtol=1e-1
    )


# %%


def time_evolution(A, B, C, D, u, x, t):
    """calculate time response"""
    y = np.zeros((len(D), len(t)))
    for k in range(0, len(t) - 1):
        y[:, k] = C @ x.ravel() + D @ u[:, k]
        x = A @ x.ravel() + B @ u[:, k]
    return pd.DataFrame(data=y.T, index=t)


def test_tf_mimo_dof_1():
    stiffness = 1e4
    damping = 18
    mass = 1
    SampleFrequency = 1024
    t_end = 10
    t = np.linspace(0, t_end, t_end * SampleFrequency)
    #  Generate some data

    Ac = np.array([[0, 1], [-stiffness / mass, -damping / mass]])
    Bc = np.array([[0], [1 / mass]])
    Cc = np.array([[1, 0]])
    Dc = np.array([[0]])

    # Discrete
    A, B, C, D, _ = sg.cont2discrete((Ac, Bc, Cc, Dc), dt=1 / SampleFrequency)

    # u = np.zeros_like(t)
    # u[0:int(len(u)/2)] = np.random.randn(int(len(u)/2))

    u = np.array([sg.unit_impulse(len(t), idx="mid")])
    x = np.array([[0], [0]])
    y = time_evolution(A, B, C, D, u, x, t)

    #  TF estimation
    inp_df = pd.DataFrame(data=u.T, index=t, columns=["inp"])

    # TF estimation using Model
    mimo = tf(window_size=1024, fs=SampleFrequency, hop=256)
    mimo.fit(
        [inp_df.join(y), inp_df.join(y)],
        inp_df.columns,
        y.columns,
    )

    #  Vergroesserungsfunktion
    omega_0 = np.sqrt(stiffness / mass)
    D_lehr = damping / (2 * omega_0 * mass)
    eta = 2 * np.pi * mimo.frequency / omega_0
    V1 = 1 / np.sqrt((1 - eta**2) ** 2 + (2 * D_lehr * eta) ** 2)

    #  SS2TF
    num, den = sg.ss2tf(A, B, C, D)
    z = np.exp(2j * np.pi * np.linspace(0, 0.5, len(mimo.frequency)))
    frf = np.polyval(num[0, :], z) / np.polyval(den, z)

    #  Compare results

    df_compare_log = pd.DataFrame(
        data=np.log10(np.array([V1 / omega_0**2, abs(mimo.tf[:, 0, 0]), abs(frf)]).T),
        index=eta,
        columns=["V1", "tf", "frf"],
    )

    df_compare_log = df_compare_log[df_compare_log.index <= 3]
    pd.testing.assert_series_equal(
        df_compare_log["tf"], df_compare_log["V1"], rtol=1e-1, check_names=False
    )


def test_tf_mimo_dof_2():
    """MIMO according to
    https://link.springer.com/content/pdf/10.1007%2F978-3-8348-9030-6_4.pdf
    """
    SampleFrequency = 2048
    N = int(50e3)
    t = np.linspace(0, N / SampleFrequency, N + 1)
    m1 = 1e1
    m2 = 5e1
    c1 = 1e6
    c2 = 5e7
    c3 = 5e6
    d1 = 1e3
    d2 = 0.5e3
    d3 = 3e3

    Ac = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-(c1 + c2) / m1, c2 / m1, -(d1 + d2) / m1, d2 / m1],
            [c2 / m2, -(c2 + c3) / m2, d2 / m2, -(d2 + d3) / m2],
        ]
    )
    Bc = np.array([[0, 0], [0, 0], [1 / m1, 0], [0, 1 / m2]])

    Cc = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    Dc = np.array([[0, 0], [0, 0]])
    # Discrete
    A, B, C, D, _ = sg.cont2discrete((Ac, Bc, Cc, Dc), dt=1 / SampleFrequency)
    # # Create TS
    u = np.random.randn(2, N + 1)
    x = np.array([[0], [0], [0], [0]])
    y = time_evolution(A, B, C, D, u, x, t)
    #  TF estimation
    inp_df = pd.DataFrame(data=u.T, index=t, columns=["inp1", "inp2"])
    mimo = tf(window_size=2048, fs=SampleFrequency, hop=1024)
    mimo.fit([inp_df.join(y)], inp_df.columns, y.columns)
    df_tf = pd.DataFrame()
    for inp in range(D.shape[0]):  # inputs
        for out in range(D.shape[1]):  # response
            df_tf["inp_" + str(inp) + "_out_" + str(out)] = np.log10(
                abs(mimo.tf[:, out, inp])
            )
    #  SS2TF
    z = np.exp(2j * np.pi * np.linspace(0, 0.5, len(mimo.frequency)))
    df_frf = pd.DataFrame()
    for inp in range(D.shape[0]):  # inputs
        num, den = sg.ss2tf(A, B, C, D, input=inp)
        for out in range(D.shape[1]):  # response
            frf = np.polyval(num[out, :], z) / np.polyval(den, z)
            df_frf["inp_" + str(inp) + "_out_" + str(out)] = np.log10(abs(frf))
    #  Compare Data
    pd.testing.assert_frame_equal(df_tf, df_frf, rtol=1e-2)


# %%
