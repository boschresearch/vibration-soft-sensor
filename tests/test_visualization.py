import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from softsensor.visualization import plot_uncertainty, plot_uncertainty_both, plot_calibration_curve

@pytest.fixture(autouse=True)
def disable_plot_blocking():
    plt.ion() # disable plot blocking by turning on interactive mode
    yield


def result_df():
    d = {'out': np.random.randn(100),
         'out2': np.random.randn(100),
         'out_random': np.random.randn(100),
         'out_zeros': np.zeros((100)),
         'out_random_var': np.absolute(np.random.randn(100)),
         'out_zeros_var': np.absolute(np.random.randn(100)),
         'out_random_ep_var': np.absolute(np.random.randn(100)),
         'out_zeros_ep_var': np.absolute(np.random.randn(100)),
         'out_random_al_var': np.absolute(np.random.randn(100)),
         'out_zeros_al_var': np.absolute(np.random.randn(100)),
         }

    d['out_copy'] = d['out']
    d['out2_copy'] = d['out2']
    t = np.linspace(0, 1, 100)
    df = pd.DataFrame(d, index=t)
    return df

def test_plot():
    df = result_df()
    fig, ax = plot_uncertainty(df, 'out_random', 'out_random_var', ground_truth='out', t_start=.1, t_end=.3,
                               n_std=2)
    fig, ax = plot_uncertainty_both(df, 'out_random', 'out_random_ep_var', 'out_random_al_var', ground_truth='out',
                                    t_start=.1, t_end=.3, n_std=2)

def test_plot_calibration_curve():
    df = result_df()
    fig, ax = plot_calibration_curve(df, 'out', ['random', 'zeros'], quantiles=np.arange(0.0, 1.1, 0.1), show_legend=True)
