# -*- coding: utf-8 -*-

import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as st
from scipy.stats.qmc import LatinHypercube
import torch

from softsensor.autoreg_models import (_postprocess_sens, _reshape_array)
from softsensor.metrics import picp

plt.rcParams.update({
    # "font.family": "CMU Serif",
    "axes.xmargin": 0.01,
    })

# change this to the desired base path of the project
BASE_PATH = 'c:/Users/FSL5FE/MA/Coding/SoftSensorACC/use_cases/Fischer_MA'


def plot_uncertainty(dataframe, mean, var, ground_truth=None, t_start=None, t_end=None,
                     n_std=2, show_legend=False):
    """
    Plots the prediction with distributional uncertainty

    Parameters
    ----------
    mean : torch.Tensor
        Point prediction
    std : torch.Tensor
        Uncertainty estimate
    t_start: int, optional
        Start of plotting window
    t_end: int, optional
        End of plotting window
    ground_truth : torch.Tensor, optional
        Ground truth of point prediction
    fs : int, optional
        Sampling rate. The default is 10
    n_std : int, optional
        Amount of standard deviations to plot. The default is 2
    show_legend : bool, optional   
        Show legend in the plot. The default is False
    title: string, optional
        Title for the plot. The default is None
    title_info: dict[string,string], optional
        Must contain keys ['dataset', 'model', 'track', 'sensor']. The default is None.
    fig_path : string, optional   
        Path to save fig. The default is None
    is_duffing: bool, optional
        Whether it is position prediction (like Duffing dataset) or acceleration prediction. The default is True.
    show: bool, optional
        If the plot should be displayed. The default is True.

    Returns
    -------
    None.

    """
    df = dataframe[t_start:t_end]
    mean_val = np.array(df[[mean]])
    var_val = np.array(df[[var]])
    std = np.sqrt(var_val)

    fig, ax = plt.subplots(1, 1)
    if ground_truth is not None:
        ax.plot(df.index, df[[ground_truth]], c='k', label=ground_truth)
    
    ax.plot(df.index, mean_val, c='b', label=mean)

    for i in range(1,n_std+1):
        alpha = 0.1+0.4*(n_std+1-i)/(n_std+1)
        ax.fill_between(df.index, (mean_val-i*std).squeeze(), (mean_val+i*std).squeeze(), alpha=alpha, color='b',
                        label=rf"Uncertainty {i}$\sigma$")
        if show_legend:
            ax.legend()
    return fig, ax


def plot_uncertainty_both(dataframe, mean, epistemic_var, aleatoric_var, ground_truth=None, t_start=None, t_end=None,
                          n_std=2, show_legend=False):
    """
    Plots the prediction with aleatoric and epistemic uncertainty

    Parameters
    ----------
    mean : torch.Tensor
        Point prediction
    epistemic_std : torch.Tensor
        Epistemic uncertainty estimate (e.g. ensemble, mcdo)
    aleatoric_std : torch.Tensor
        Aleatoric uncertainty estimate (e.g. mve)
    t_start: int, optional
        Start of plotting window
    t_end: int, optional
        End of plotting window
    ground_truth : torch.Tensor, optional
        Ground truth of point prediction
    fs : int, optional
        Sampling rate. The default is 10
    n_std : int, optional
        Amount of standard deviations to plot. The default is 2
    show_legend : bool, optional   
        Show legend in the plot. The default is False
    title: string, optional
        Title for the plot. The default is None
    title_info: dict[string,string], optional
        Must contain keys ['dataset', 'model', 'track', 'sensor']. The default is None.
    fig_path : string, optional   
        Path to save fig

    Returns
    -------
    None.

    """
    df = dataframe[t_start:t_end]
    mean_val = np.array(df[[mean]])
    ep_var = np.array(df[[epistemic_var]])
    al_var = np.array(df[[aleatoric_var]])
    
    ep_std = np.sqrt(ep_var)
    al_std = np.sqrt(al_var)

    fig, ax = plt.subplots(1, 1)
    if ground_truth is not None:
        ax.plot(df.index, df[[ground_truth]], c='k', label=ground_truth)
    
    ax.plot(df.index, mean_val, c='b', label=mean)
    
    std = ep_std
    for i in range(1,n_std+1):
        alpha = 0.1+0.4*(n_std+1-i)/(n_std+1)
        ax.fill_between(df.index, (mean_val-i*std).squeeze(), (mean_val+i*std).squeeze(), alpha=alpha, color='r',
                        label=rf"Epistemic {i}$\sigma$")
    std = ep_std + al_std
    for i in range(1,n_std+1):
        alpha = 0.1+0.4*(n_std+1-i)/(n_std+1)
        ax.fill_between(df.index, (mean_val-i*std).squeeze(), (mean_val+i*std).squeeze(), alpha=alpha, color='b',
                        label=rf"Total {i}$\sigma$")
    if show_legend:
        ax.legend()
        
    return fig, ax


def plot_calibration_curve(dataframe, ground_truth, model_names, quantiles=np.arange(0.0, 1.05, 0.05), show_legend=False):
    
    """ Runs the model prediction on track and plots the calibration at different quantile levels
        Assumes that Quantile Regression models have QR in their name and that all other models predict mean and variance

    Parameters
    ----------
    models: list[uncertainty models]
        Models to use for prediction
    track: torch.Dataloader
        Single track
    track_number: int
        Number of track in the test set
    out_sens: list[string]
        Names of output sensors
    output: int, optional
        Output sensor to plot. The default is 0.
    quantiles: list[x], x in (0,1)
        Quantile levels to analyze for calibration curve. The default is np.arange(0.0, 1.05, 0.05)
    fig_path : string, optional   
        Path to save fig. The default is None

    Returns
    -------
    None.
    """
    fig, ax = plt.subplots(1, 1)
    ax.plot(quantiles, quantiles, c="gray")


    for name in model_names:
        mean_val = torch.tensor(np.array(dataframe[[f'{ground_truth}_{name}']]))
        var_val = torch.tensor(np.array(dataframe[[f'{ground_truth}_{name}_var']]))
        targets = torch.tensor(np.array(dataframe[[ground_truth]]))

        z_scores = [st.norm.ppf(1-(1-p)/2) for p in quantiles]
        picp_scores = np.array([picp(mean_val, targets, var_val, z) for z in z_scores])
        
        ax.plot(quantiles, picp_scores, marker='o', label=name)
        
    ax.set_xlabel("Expected Confidence Level")
    ax.set_ylabel("Observed Confidence Level")
    
    if show_legend:
        ax.legend()
    return fig, ax


def _convert_to_latex(ch_names, ch_str, use_case):
    """ Convert the channel names to LaTeX scientific notation if applicable."""
    ch_dict = {
        'Duffing_MVE': ['$F(t)$', '$x$'],
        'Duffing_Enhanced': ['$F(t)$', '$x_{\\text{lin}}$', '$x$'],
        'Duffing_Delta': ['$F(t)$', '$x_{\\text{lin}}$', '$\\Delta x$'],
        'Duffing_Comparison': ['$F(t)$', '$x_{\\text{lin}}$', '$(\\Delta) x$'],

        'Steering_MVE': ['$M2_x$', '$M2_y$', '$M2_z$', '$M3_x$', '$M3_y$', '$M3_z$', '$M8_x$', '$M8_y$', '$M8_z$'],
        'Steering_Enhanced': ['$M2_x$', '$M2_y$', '$M2_z$', '$M3_x$', '$M3_y$', '$M3_z$', '$M8_{x,\\text{lin}}$',
                              '$M8_{y,\\text{lin}}$', '$M8_{z,\\text{lin}}$', '$M8_x$', '$M8_y$', '$M8_z$'],
        'Steering_Delta': ['$M2_x$', '$M2_y$', '$M2_z$', '$M3_x$', '$M3_y$', '$M3_z$', '$M8_{x,\\text{lin}}$',
                            '$M8_{y,\\text{lin}}$', '$M8_{z,\\text{lin}}$', '$\\Delta M8_{x}$', '$\\Delta M8_{y}$',
                            '$\\Delta M8_{z}$'],
        'Steering_Comparison': ['$M2_x$', '$M2_y$', '$M2_z$', '$M3_x$', '$M3_y$', '$M3_z$', '$M8_{x,\\text{lin}}$',
                                '$M8_{y,\\text{lin}}$', '$M8_{z,\\text{lin}}$', '$(\\Delta) M8_{x}$', '$(\\Delta) M8_{y}$',
                            '$(\\Delta) M8_{z}$'],
    }
    if ch_names:
        ch_str = ch_dict[use_case] if use_case in ch_dict else ch_str
    return ch_str


def plot_sens_analysis(model, sensitivity, ch_names=None, title=None, use_case=None, annotations=False, orig_length=None, save_path=None):
    """
    Visualize different metrics/insights of the post-processed arrays,
    coming from differently averaged sensitivities.

    Parameters
    ----------
    model : Model consisting of nn.Modules
    sensitivity : torch.Tensor
        Sensitivity tensor that contains the gradients of the output with respect to the inputs
    ch_names : list of strings, optional
        List of channel names. The default is None, i.e. the channel names are not provided and
        will be generated in the style of 'Inp_ch_i' and 'Rec_ch_i'.
    title : string, optional
        Title for the plot series. The default is None, i.e. no title is displayed.
    annotations : bool, optional
        If the plot should contain the percentages of the channel-wise mean sensitivities.
        The default is False.
    orig_length : int, optional
        Original length of the time series. Only needed when sensitivity analysis is performed on
        subset of the data. The default is None, i.e. the original length is not provided.
    save_path : string, optional
        Path to save the plots. The default is None, i.e. the plots are not saved.
    """
    # Postprocess the sensitivity tensor
    sum_mean_std_feature, sum_mean_std_inp_channels, out_ch_sens = _postprocess_sens(model, sensitivity)
    rms_out_ch_sens = out_ch_sens[0]

    if title is not None:
        print(title)
    else:
        print(f'### Sensitivity analysis results for {model.__class__.__name__} model ###')

    if save_path:
        base_path = os.path.abspath(BASE_PATH)
        use_case_str = use_case if use_case else ''
        save_path = os.path.join(base_path, 'Evaluation_plots', use_case_str, save_path)
        os.makedirs(save_path, exist_ok=True)

    m_type = model.Type
    input_channels = model.input_channels
    pred_size = model.pred_size
    num_timesteps, num_features = rms_out_ch_sens.shape

    if m_type in ['AR', 'AR_RNN']:
        win_size = max(model.window_size, model.rnn_window)
        ch_size = input_channels + pred_size
        rec_start_idx = input_channels*model.window_size
        flatten_size = rec_start_idx + pred_size*model.rnn_window
        ch_str = [f'Inp_ch_{i}' for i in range(input_channels)] + \
            [f'Rec_ch_{i}' for i in range(pred_size)] if ch_names is None else ch_names
        x_tks_feat = [i for i in np.arange(0, rec_start_idx, model.window_size)] + \
            [i for i in np.arange(rec_start_idx, num_features+1, model.rnn_window)]
        x_center = [i for i in np.arange(model.window_size//2, rec_start_idx, model.window_size)] + \
            [i for i in np.arange(rec_start_idx+model.rnn_window//2, num_features+1, model.rnn_window)]
    else:
        win_size = model.window_size
        ch_size = input_channels
        rec_start_idx = input_channels*model.window_size
        flatten_size = rec_start_idx
        ch_str = [f'Inp_ch_{i}' for i in range(input_channels)] if ch_names is None else ch_names
        x_tks_feat = [i for i in np.arange(0, num_features+1, model.window_size)]
    
    # LaTeX scientific notation for channel names
    new_ticks_pos = x_tks_feat[:-1] + (x_tks_feat[1] - x_tks_feat[0]) / 2
    ch_str = _convert_to_latex(ch_names, ch_str, use_case)
    out_str = [f'Out_ch_{i}' for i in range(pred_size)] if ch_names is None else ch_names[-pred_size:]
    out_str = _convert_to_latex(ch_names, out_str, use_case)[-pred_size:]
    angle = 65 if ch_size > 5 else 0
    halign = 'left' # if ch_size < 5 else 'center'
    font_size = 10 if ch_size < 5 else 8
    ax_zero_label = r'RMS of Mean Sensitivities  $[-]$'
    ax_one_label = r'RMS of Std. Dev. of Sensitivities  $[-]$'
    
    # Plot the sum_mean_inp_channels, sum_std_inp_channels as heatmaps
    sum_inp_channels, std_inp_channels = sum_mean_std_inp_channels # unpack tuple
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    if pred_size == 1:
        categories = ch_str
        mean_values, std_values = sum_inp_channels[0], std_inp_channels[0]
        mean_norm = plt.Normalize(0, max(mean_values))
        std_norm = plt.Normalize(0, max(std_values))
        color_scheme = plt.get_cmap('Reds')
        mean_colors = color_scheme(mean_norm(mean_values))
        std_colors = color_scheme(std_norm(std_values))
        axs[0].bar(categories, mean_values, color=mean_colors, width=0.5, alpha=0.8, edgecolor='black', linewidth=.75)
        axs[1].bar(categories, std_values, color=std_colors, width=0.5, alpha=0.8, edgecolor='black', linewidth=.75)
        axs[0].set_ylabel(ax_zero_label, labelpad=10)
        axs[1].set_ylabel(ax_one_label, labelpad=10)
    else:
        pos0 = axs[0].imshow(sum_inp_channels, cmap='Reds', interpolation='none',
                            aspect='auto', extent=[0, ch_size, pred_size, 0], vmin=0)
        cbar0 = fig.colorbar(pos0, ax=axs[0])
        cbar0.formatter.set_powerlimits((0, 0))
        cbar0.formatter.set_useMathText(True)
        pos1 = axs[1].imshow(std_inp_channels, cmap='Reds', interpolation='none',
                            aspect='auto', extent=[0, ch_size, pred_size, 0], vmin=0)
        cbar1 = fig.colorbar(pos1, ax=axs[1])
        cbar1.formatter.set_powerlimits((0, 0))
        cbar1.formatter.set_useMathText(True)

    for ax in axs:
        ax.set_xlabel('Input Channels', labelpad=10, fontsize=font_size)
        ax.set_xticks(np.arange(ch_size))
        ax.set_xticklabels(ch_str)
        if pred_size > 1:
            ax.set_ylabel('Output Channels', labelpad=10)
            ax.set_xticklabels(ch_str, rotation=angle, ha=halign)
            ax.set_yticks(np.arange(pred_size))
            ax.set_yticklabels(out_str, rotation=angle, va='top', fontsize=font_size)
            ax.grid(which='major', color='black', linestyle='-', linewidth=.5)
            if m_type in ['AR', 'AR_RNN']:
                ax.axvline(input_channels, color='black', linewidth=2)
        else:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
            ax.set_xlim(-0.5, len(categories)-0.5)
    
    if save_path:
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.25)
        plt.savefig(os.path.join(save_path, 'sens_imshow.pdf'), bbox_inches='tight')
    else:
        plt.suptitle('Summed Sensitivities across Input & Output Channels', fontsize=13)
        axs[0].set_title('RMS of Mean Sensitivities', pad=10)
        axs[1].set_title('RMS of Std. Dev. of Sensitivities', pad=10)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.25)
    plt.show()


    # Plot the sum_mean_feature, sum_std_feature as bar plots
    plt.rcParams.update({"axes.xmargin": 0})
    sum_mean_feature, sum_std_feature = sum_mean_std_feature # unpack tuple
    ch_mean = _reshape_array(model, sum_mean_feature, aggregation='mean', repeat=True)
    ch_std_mean = _reshape_array(model, sum_std_feature, aggregation='mean', repeat=True)
    means = _reshape_array(model, sum_mean_feature, aggregation='mean')
    ch_median = _reshape_array(model, sum_mean_feature, aggregation='median', repeat=True)
    ch_std_median = _reshape_array(model, sum_std_feature, aggregation='median', repeat=True)

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    edgecolor = 'black' if ch_size < 5 else 'None'
    axs[0].bar(np.arange(num_features)+0.5, sum_mean_feature, width=0.9, alpha=0.8, edgecolor=edgecolor, linewidth=.5)
    axs[1].bar(np.arange(num_features)+0.5, sum_std_feature, width=0.9, alpha=0.8, edgecolor=edgecolor, linewidth=.5)
    # add a step plot for the mean value of each input/recurrent channel
    axs[0].step(np.arange(num_features+1), ch_mean, lw=2, color='red')
    axs[1].step(np.arange(num_features+1), ch_std_mean, lw=2, color='red')
    axs[0].step(np.arange(num_features+1), ch_median, lw=2, color='green')
    axs[1].step(np.arange(num_features+1), ch_std_median, lw=2, color='green')

    # create manual legend
    legend_entries = [Patch(edgecolor='black', label='Local' + '\n' + r'Sensitivity $s_{ik}$'),
                      Line2D([0], [0], color='red', label='Channel' + '\n' + r'Mean $\mu_k$'),
                      Line2D([0], [0], color='green', label='Channel' + '\n' + r'Median $P_{50, k}$')]

    # plot annotations for the channel-wise percentages if flag is set
    if annotations:
        perc_mean = means / np.sum(means)
        x_positions = [i+3 for i in x_tks_feat] if ch_size < 5 else x_center
        y_positions = means*1.25 if ch_size < 5 else means*2.5
        for x_pos, y_pos, perc in zip(x_positions, y_positions, perc_mean):
            axs[0].text(x_pos, y_pos, f'{perc:.2f}', fontsize=11, ha=halign, va='bottom', color='red')

    axs[0].set_ylabel(ax_zero_label, labelpad=10)
    axs[1].set_ylabel(ax_one_label, labelpad=10)
    for ax in axs:
        ax.legend(handles=legend_entries, loc='upper left')
        ax.set_xlabel('Input Channels', labelpad=15)
        ax.set_xticks(x_tks_feat)
        if ch_size < 5:
            ax.set_xticklabels([])
            for pos, label in zip(new_ticks_pos, ch_str):
                ax.text(pos, ax.get_ylim()[0] - 0.06*ax.get_ylim()[1], label, ha='center', va='bottom')
        else:
            ax.set_xticklabels(ch_str + [''], rotation=angle, ha=halign, fontsize=font_size)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        ax.xaxis.grid(True)
        if m_type in ['AR', 'AR_RNN']:
            ax.axvline(rec_start_idx, color='black', linewidth=2)
    
    if save_path:
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.25)
        plt.savefig(os.path.join(save_path, 'sens_barplot.pdf'), bbox_inches='tight')
    else:
        plt.suptitle('Sensitivities across all Channels, RMS over Output Channels', fontsize=13)
        axs[0].set_title('RMS of Mean Sensitivities', pad=10)
        axs[1].set_title('RMS of Std. Dev. of Sensitivities', pad=10)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.25)
    plt.show()
    plt.rcParams.update({"axes.xmargin": 0.01})


    # Plot the averages across the time windows for each channel as line plots
    rms_sum_out = _reshape_array(model, sum_mean_feature)
    std_sum_out = _reshape_array(model, sum_std_feature)

    rms_sum_inp, rms_sum_rec = np.split(rms_sum_out, [input_channels], axis=0)
    std_sum_inp, std_sum_rec = np.split(std_sum_out, [input_channels], axis=0)
    if use_case and ('Enhanced' in use_case or 'Delta' in use_case): # in case a linear pre-computed solution exists
        rms_sum_inp, rms_sum_lin = np.split(rms_sum_inp, [input_channels-pred_size], axis=0)
        std_sum_inp, std_sum_lin = np.split(std_sum_inp, [input_channels-pred_size], axis=0)
        mean_rms_lin = np.mean(rms_sum_lin, axis=0)
        mean_std_lin = np.mean(std_sum_lin, axis=0)
        std_dev_rms_lin = np.std(rms_sum_lin, axis=0)
        std_dev_std_lin = np.std(std_sum_lin, axis=0)

    mean_rms_inp = np.mean(rms_sum_inp, axis=0)
    mean_std_inp = np.mean(std_sum_inp, axis=0)
    std_dev_rms_inp = np.std(rms_sum_inp, axis=0)
    std_dev_std_inp = np.std(std_sum_inp, axis=0)
    if m_type in ['AR', 'AR_RNN']:
        mean_rms_rec = np.mean(rms_sum_rec, axis=0)
        mean_std_rec = np.mean(std_sum_rec, axis=0)
        std_dev_rms_rec = np.std(rms_sum_rec, axis=0)
        std_dev_std_rec = np.std(std_sum_rec, axis=0)
    
    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    alpha_val = 0.5 if (input_channels > 2 or pred_size > 2) else 1
    lw = 1.5 if (input_channels > 2 or pred_size > 2) else 1.8
    if ch_size <= 3:
        for i in range(ch_size):
            if i < input_channels:
                axs[0].plot(rms_sum_out[i], lw=lw, alpha=alpha_val, label=ch_str[i])
                axs[1].plot(std_sum_out[i], lw=lw, alpha=alpha_val, label=ch_str[i])
            else:
                axs[0].plot(rms_sum_out[i], linestyle='--', lw=lw, alpha=alpha_val, label=ch_str[i])
                axs[1].plot(std_sum_out[i], linestyle='--', lw=lw, alpha=alpha_val, label=ch_str[i])
    else:
        axs[0].plot(mean_rms_inp, lw=2, color='darkred', label=r'$\mu_\text{inp}$')
        axs[1].plot(mean_std_inp, lw=2, color='darkred', label=r'$\mu_\text{inp}$')
        axs[0].fill_between(np.arange(mean_rms_inp.shape[0]), mean_rms_inp-std_dev_rms_inp,
                            mean_rms_inp+std_dev_rms_inp, color='red', alpha=0.3, label=r'$\sigma_\text{inp}$')
        axs[1].fill_between(np.arange(mean_std_inp.shape[0]), mean_std_inp-std_dev_std_inp,
                            mean_std_inp+std_dev_std_inp, color='red', alpha=0.3, label=r'$\sigma_\text{inp}$')
        
        if use_case and ('Enhanced' in use_case or 'Delta' in use_case):
            axs[0].plot(mean_rms_lin, lw=2, color='green', label=r'$\mu_\text{lin}$')
            axs[1].plot(mean_std_lin, lw=2, color='green', label=r'$\mu_\text{lin}$')
            axs[0].fill_between(np.arange(mean_rms_lin.shape[0]), mean_rms_lin-std_dev_rms_lin,
                                mean_rms_lin+std_dev_rms_lin, color='palegreen', alpha=0.4, label=r'$\sigma_\text{lin}$')
            axs[1].fill_between(np.arange(mean_std_lin.shape[0]), mean_std_lin-std_dev_std_lin,
                                mean_std_lin+std_dev_std_lin, color='palegreen', alpha=0.4, label=r'$\sigma_\text{lin}$')
        
        if m_type in ['AR', 'AR_RNN']:
            axs[0].plot(mean_rms_rec, lw=2, color='blue', linestyle='--', label=r'$\mu_\text{rec}$')
            axs[1].plot(mean_std_rec, lw=2, color='blue', linestyle='--', label=r'$\mu_\text{rec}$')
            axs[0].fill_between(np.arange(mean_rms_rec.shape[0]), mean_rms_rec-std_dev_rms_rec,
                                mean_rms_rec+std_dev_rms_rec, color='cornflowerblue', alpha=0.4, label=r'$\sigma_\text{rec}$')
            axs[1].fill_between(np.arange(mean_std_rec.shape[0]), mean_std_rec-std_dev_std_rec,
                                mean_std_rec+std_dev_std_rec, color='cornflowerblue', alpha=0.4, label=r'$\sigma_\text{rec}$')
    
    time_deltas = {
        10: 2,
        20: 5,
        25: 5,
        30: 5,
        40: 10,
        50: 10,
        75: 15,
        100: 20,
        125: 25,
        150: 30,
        175: 35,
        200: 40
    }
    time_delta = time_deltas[win_size]
    time_str = [f't-{i}' for i in np.arange(win_size, -1, -time_delta)][:-1]
    time_str.append('t')

    axs[0].set_ylabel(ax_zero_label, labelpad=10)
    axs[1].set_ylabel(ax_one_label, labelpad=10)
    for ax in axs:
        ax.set_xlabel(r'Sliding Time Window  $[-]$', labelpad=10)
        ax.set_xticks(np.arange(win_size+1, step=time_delta))
        ax.set_xticklabels(time_str)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        if use_case and (use_case == 'Duffing_Enhanced' or use_case == 'Duffing_Delta'):
            ax.get_lines()[0].set_color('blue')
            ax.get_lines()[1].set_color('green')
            ax.get_lines()[2].set_color('purple')
        elif use_case and use_case == 'Duffing_MVE':
            ax.get_lines()[0].set_color('blue')
            ax.get_lines()[1].set_color('purple')
        if ch_size < 5:
            ax.legend(title='Channels', ncol=1)
        else:
            handles, labels = ax.get_legend_handles_labels()
            new_handles = [handles[i] for i in range(0, len(handles), 2)] + [handles[i] for i in range(1, len(handles), 2)]
            new_labels = [labels[i] for i in range(0, len(labels), 2)] + [labels[i] for i in range(1, len(labels), 2)]
            ax.legend(new_handles, new_labels, ncol=2, loc='upper left')
    
    if save_path:
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.25)
        plt.savefig(os.path.join(save_path, 'sens_sliding_time_window.pdf'), bbox_inches='tight')
    else:
        plt.suptitle('Sensitivities across Time Window for all Channels, RMS over Output Channels', fontsize=13)
        axs[0].set_title('RMS of Mean Sensitivities', pad=10)
        axs[1].set_title('RMS of Std of Sensitivities', pad=10)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.25)
    plt.show()


    # Plot the temporal development for one selected channel as line plots
    ch_idx = ch_size-1 # to be chosen between 0 and ch_size-1
    fig, ax = plt.subplots(figsize=(9,5))
    # Define a custom colormap from light blue to light green
    cmap = LinearSegmentedColormap.from_list('blue_to_green', ['blue', 'green'])
    # Randomly select 50 time steps (if applicable) and sort them to avoid aliasing effects
    if num_timesteps > 50:
        samples = LatinHypercube(d=1).random(n=50).flatten()
        random_indices = np.sort(np.floor(samples * num_timesteps)).astype(int)
    else:
        random_indices = np.arange(num_timesteps)
    temp_dev_ch = _reshape_array(model, rms_out_ch_sens, remove_nans=True)
    temp_dev_ch = np.stack([np.append(np.nan, x[ch_idx]) for x in temp_dev_ch])
    mean = np.mean(temp_dev_ch[random_indices], axis=0)
    std_dev = np.std(temp_dev_ch[random_indices], axis=0)

    for i in random_indices:
        color = cmap(i / num_timesteps)
        if i == random_indices[0]:
            ax.plot(temp_dev_ch[i], alpha=0.5, color=color, label='Time Values')
        else:
            ax.plot(temp_dev_ch[i], alpha=0.2, color=color)
    ax.plot(mean, color='black', lw=2, label=r'Mean $\mu$')
    ax.fill_between(np.arange(mean.shape[0]), mean-std_dev, mean+std_dev,
                    color='red', alpha=0.5, label=r'Std. Dev. $\pm\sigma$')

    wind_size = model.window_size if ch_idx < input_channels else model.rnn_window
    time_delta = time_deltas[wind_size]
    time_str = [f't-{i}' for i in np.arange(wind_size, -1, -time_delta)][:-1]
    time_str.append('t')
    ax.set_xlabel(r'Sliding Time Window  $[-]$', labelpad=10)
    ax.set_ylabel(ax_zero_label, labelpad=10)
    ax.set_xticks(np.arange(wind_size+1, step=time_delta))
    ax.set_xticklabels(time_str)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    ax.legend(loc='upper left')
    if save_path:
        fig.tight_layout()
        plt.savefig(os.path.join(save_path, f'sens_window_ch_{ch_idx+1}.pdf'), bbox_inches='tight')
    else:
        ch_name = 'Input' if ch_idx < input_channels else 'Recurrent'
        ax.set_title(f'Temporal Development across Time Window for {ch_name} Channel "{ch_str[ch_idx]}"', pad=10)
        fig.tight_layout()
    plt.show()


    # Plot the temporal development for all channels as line plots
    # 'mean' for correlation with output signal, 'rms' for effective-valued aggregation
    temp_seasonal = _reshape_array(model, rms_out_ch_sens, aggregation='rms').transpose()
    x_length = num_timesteps/10 if not orig_length else orig_length
    fig, ax = plt.subplots(figsize=(9,5))
    filter_size = win_size//2 if flatten_size < 5*win_size else win_size
    for i in range(ch_size):
        # compute the running average for each channel
        if flatten_size < 4*win_size:
            temp_season = temp_seasonal[i]
        else:
            temp_season = np.convolve(temp_seasonal[i], np.ones(filter_size)/filter_size, mode='valid')
            temp_season = np.concatenate([np.full((filter_size-1)//2, np.nan), temp_season])
        if i < input_channels:
            if i == 0:
                ax.plot(temp_season, alpha=0.7, label=ch_str[i])
            else:
                ax.plot(temp_season, lw=2, alpha=alpha_val, label=ch_str[i])
        else:
            ax.plot(temp_season, linestyle='--', lw=2, alpha=alpha_val, label=ch_str[i])
    
    ax.set_xlabel(r'Time  $[s]$', labelpad=10)
    ax.set_ylabel(ax_zero_label, labelpad=10)
    ax.set_xticks(np.arange(num_timesteps+1, step=num_timesteps//5))
    ax.set_xticklabels(np.arange(x_length+1, step=x_length//5, dtype=int))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    if use_case and (use_case == 'Duffing_Enhanced' or use_case == 'Duffing_Delta'):
        ax.get_lines()[0].set_color('blue')
        ax.get_lines()[1].set_color('green')
        ax.get_lines()[2].set_color('purple')
    elif use_case and use_case == 'Duffing_MVE':
        ax.get_lines()[0].set_color('blue')
        ax.get_lines()[1].set_color('purple')
    if ch_size < 5:
        ax.legend(title='Channels', loc='upper left')
    else:
        ax.legend(title='Channels', loc='upper left', bbox_to_anchor=(1,1))
    if save_path:
        plt.savefig(os.path.join(save_path, 'sens_seasonalities.pdf'), bbox_inches='tight')
        fig.tight_layout()
    else:
        ax.set_title('Global Development of Channel Sensitivities, RMS over Output Channels', pad=10)
        fig.tight_layout()
    plt.show()


    def _fft_signal(signal, sampling_rate=10):
        N = len(signal)
        magnitudes = np.abs(np.fft.fft(signal))
        frequencies = np.fft.fftfreq(N, 1/sampling_rate)
        dominant_freq = frequencies[np.argmax(magnitudes[:N//2])]
        return dominant_freq, frequencies[:N//2], magnitudes[:N//2]

    # Plot the frequency spectra of the sensitivities for each channel (DUFFING oscillator only)
    if use_case and 'Duffing' in use_case:
        fig, ax = plt.subplots(figsize=(9,5))
        for i in range(ch_size):
            _, freqs, mags = _fft_signal(temp_seasonal[i])
            ax.plot(freqs, mags, alpha=0.8, label=f'{ch_str[i]}') #: {dominant_freq:.1f} Hz')
        ax.get_lines()[0].set_alpha(0.5)
        if use_case == 'Duffing_Enhanced' or use_case == 'Duffing_Delta':
            ax.get_lines()[0].set_color('blue')
            ax.get_lines()[1].set_color('green')
            ax.get_lines()[2].set_color('purple')
        elif use_case == 'Duffing_MVE':
            ax.get_lines()[0].set_color('blue')
            ax.get_lines()[1].set_color('purple')
        ax.set_yscale('log')
        ax.set_xlabel(r'Frequency  $[Hz]$', labelpad=10)
        ax.set_ylabel(r'Magnitude  $[-]$', labelpad=10)
        ax.legend(title='Channels', loc='upper right')
        if save_path:
            plt.savefig(os.path.join(save_path, 'sens_ffts.pdf'), bbox_inches='tight')
            fig.tight_layout()
        else:
            ax.set_title('Frequency Spectra of Channel Sensitivities', pad=10)
            fig.tight_layout()
        plt.show()


    # Plot the temporal fluctuations of the sensitivities
    plt.rcParams.update({"axes.xmargin": 0})
    fig, ax = plt.subplots(figsize=(9,5))
    sum_out_ch_sensi = np.concatenate((np.full((num_timesteps, 1), np.nan), rms_out_ch_sens), axis=1)
    step_size = 2 if flatten_size > 100 else 1
    for i in random_indices:
        color = cmap(i / num_timesteps)
        if i == random_indices[0]:
            ax.plot(sum_out_ch_sensi[i, ::step_size], alpha=0.5, color=color, label='Time Values')
        else:
            ax.plot(sum_out_ch_sensi[i, ::step_size], alpha=0.2, color=color)
    mean = np.mean(sum_out_ch_sensi[random_indices, ::step_size], axis=0)
    std = np.std(sum_out_ch_sensi[random_indices, ::step_size], axis=0)
    ax.plot(mean, lw=2, color='black', label=r'Mean $\mu$')
    ax.fill_between(np.arange(mean.shape[0]), mean-std, mean+std, color='red', alpha=0.5, label=r'Std. Dev. $\pm\sigma$')

    temp_dev_rms_out = _reshape_array(model, rms_out_ch_sens, aggregation='mean')
    means = np.mean(temp_dev_rms_out[random_indices], axis=0)
    percentages = means / np.sum(means)
    temp1 = means[:model.input_channels].repeat(model.window_size)
    if m_type in ['AR', 'AR_RNN']:
        temp2 = means[model.input_channels:].repeat(model.rnn_window)
        mean_steps = np.hstack((np.append(np.nan, temp1), temp2))[::step_size]
    else:
        mean_steps = np.append(np.nan, temp1)[::step_size]
    ax.step(np.arange(len(mean_steps)), mean_steps, lw=2, color='red', label='Channel Mean')

    # plot annotations for the channel-wise percentages if flag is set
    if annotations:
        x_positions = [i/step_size+5 for i in x_tks_feat] if ch_size < 5 else [i/step_size for i in x_center]
        y_positions = means*1.25 if ch_size < 5 else means*2.5
        for x_pos, y_pos, perc in zip(x_positions, y_positions, percentages):
            ax.text(x_pos, y_pos, f'{perc:.2f}', fontsize=11, ha=halign, va='bottom', color='red')

    if m_type in ['AR', 'AR_RNN']:
        ax.axvline(rec_start_idx/step_size, color='black', linewidth=2)
    ax.set_xlabel('Input Channels', labelpad=18)
    ax.set_ylabel(r'RMS of Sensitivities  $[-]$', labelpad=10)
    ax.set_xticks([i/step_size for i in x_tks_feat])
    if ch_size < 5:
        ax.set_xticklabels([])
        for pos, label in zip(new_ticks_pos, ch_str):
            ax.text(pos, ax.get_ylim()[0] - 0.06*ax.get_ylim()[1], label, ha='center', va='bottom')
    else:
        ax.set_xticklabels(ch_str + [''], rotation=angle, ha=halign, fontsize=font_size)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    ax.xaxis.grid(True)
    ax.legend(loc='upper left')
    if save_path:
        fig.tight_layout()
        plt.savefig(os.path.join(save_path, 'sens_temp_dev.pdf'), bbox_inches='tight')
    else:
        ax.set_title('Temporal Fluctuations of Sensitivities, RMS over Output Channels', pad=10)
        fig.tight_layout()
    plt.show()
    plt.rcParams.update({"axes.xmargin": 0.01})


def plot_feature_importance(model, sensitivity, time_step, ch_names=None, use_case=None, save_path=None):
    """
    Visualize the feature importance of the sensitivity tensor for a given model, at a specific time step.
    """
    if save_path:
        base_path = os.path.abspath(BASE_PATH)
        use_case_str = use_case if use_case else ''
        save_path = os.path.join(base_path, 'Evaluation_plots', use_case_str, save_path)
        os.makedirs(save_path, exist_ok=True)

    # Postprocess the sensitivity tensor
    sens_str = 'Mean' if (model.Pred_Type == 'Mean_Var' or model.Ensemble) else 'Point'
    if isinstance(sensitivity[sens_str], list):
        sensitivity = torch.stack(sensitivity[sens_str]).mean(dim=0)
    else:
        sensitivity = sensitivity[sens_str]
    mean_out_ch_sens = _postprocess_sens(model, sensitivity)[2][1]

    m_type = model.Type
    input_channels = model.input_channels
    pred_size = model.pred_size
    num_features = mean_out_ch_sens.shape[1]

    if m_type in ['AR', 'AR_RNN']:
        ch_size = input_channels + pred_size
        rec_start_idx = input_channels*model.window_size
        ch_str = [f'Inp_ch_{i}' for i in range(input_channels)] + \
            [f'Rec_ch_{i}' for i in range(pred_size)] if ch_names is None else ch_names
        x_tks_feat = [i for i in np.arange(0, rec_start_idx, model.window_size)] + \
            [i for i in np.arange(rec_start_idx, num_features+1, model.rnn_window)]
    else:
        ch_size = input_channels
        ch_str = [f'Inp_ch_{i}' for i in range(input_channels)] if ch_names is None else ch_names
        x_tks_feat = [i for i in np.arange(0, num_features+1, model.window_size)]
    
    # LaTeX scientific notation for channel names
    ch_str = _convert_to_latex(ch_names, ch_str, use_case)
    out_str = [f'Out_ch_{i}' for i in range(pred_size)] if ch_names is None else ch_names[-pred_size:]
    out_str = _convert_to_latex(ch_names, out_str, use_case)[-pred_size:]
    angle = 65 if ch_size > 5 else 0
    halign = 'left' # if ch_size < 5 else 'center'
    font_size = 10 if ch_size < 5 else 8

    # Plot the mean_feature as bar plots for one specific time step
    plt.rcParams.update({"axes.xmargin": 0})
    time_steps = [time_step] if isinstance(time_step, int) else time_step

    fig, ax = plt.subplots(figsize=(9,5))
    # alpha = 0.8 if len(time_steps) == 1 else 0.5
    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b'] # i.e. blue, green, purple, brown
    alphas = [0.9, 0.65, 0.5, 0.4] if len(time_steps) != 1 else [0.8]
    for step, color, alpha in (zip(time_steps, colors, alphas)):
        feat_imp = mean_out_ch_sens[step]
        ch_mean = _reshape_array(model, feat_imp, aggregation='mean', repeat=True)
        ax.bar(np.arange(num_features)+0.5, feat_imp, width=1, alpha=alpha, color=color, edgecolor='black', linewidth=.5)
        if len(time_steps) == 1:
            ax.step(np.arange(num_features+1), ch_mean, lw=2, color='red')
    
    # create manual legend
    legend_entries = []
    for step, color in zip(time_steps, colors):
        legend_entries.append(Patch(facecolor=color, edgecolor='black', label='Local Sensitivities' + '\n' + rf' $s_{{ik}}\vert_{{t={step}}}$'))
    if len(time_steps) == 1:
        legend_entries.append(Line2D([0], [0], color='red', label='Channel Mean' + '\n' + rf'Sensitivities $\mu_k\vert_{{t={time_steps[0]}}}$'))

    ax.set_ylabel(r'Mean Sensitivities  $[-]$', labelpad=10)
    ax.legend(handles=legend_entries, loc='best')
    ax.set_xlabel('Input Channels', labelpad=10)
    ax.set_xticks(x_tks_feat)
    ax.set_xticklabels(ch_str + [''], rotation=angle, ha=halign, fontsize=font_size)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    ax.xaxis.grid(True)
    if m_type in ['AR', 'AR_RNN']:
        ax.axvline(rec_start_idx, color='black', linewidth=2)
    
    if save_path:
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
        plt.savefig(os.path.join(save_path, f'sens_barplot_timestep_{time_step}.pdf'), bbox_inches='tight')
    else:
        ax.set_title(f'Sensitivities across all Channels for Time Step {time_step}', pad=10)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
    plt.show()
    plt.rcParams.update({"axes.xmargin": 0.01})


def plot_uncertainty_sens(model, sens_uq, eps, ch_names, use_case=None, amplification=1, save_path=None):
    """
    Visualize the uncertainty of the sensitivity tensor for a given model.

    Parameters
    ----------
    model : Model consisting of nn.Modules
    sens_uq : torch.Tensor
        Sensitivity tensor that contains the gradients of the output with respect to the inputs
    eps : float
        Epsilon value for the uncertainty quantification
    ch_names : list of strings
        List of channel names
    use_case : string, optional
        Use case for the channel names. The default is use_case.
    amplification : int, optional
        Amplification factor for the uncertainty quantification of sensitivities of MVE models. The default is 1.
    save_path : string, optional
        Path to save the plots. The default is None, i.e. the plots are not saved.
    """
    if save_path:
        base_path = os.path.abspath(BASE_PATH)
        use_case_str = use_case if use_case else ''
        save_path = os.path.join(base_path, 'Evaluation_plots', use_case_str, save_path)
        os.makedirs(save_path, exist_ok=True)
    
    # Postprocess the sensitivity tensor
    sens_uq, eps = sens_uq.numpy(), eps.numpy()
    sum_win_size_sens = _reshape_array(model, sens_uq, aggregation='rms') # shape [sens_length*random_samples, pred_size, input_channels+pred_size]

    pred_size = model.pred_size
    ch_size = model.input_channels + pred_size
    ch_str = [f'Inp_ch_{i}' for i in range(model.input_channels)] + \
        [f'Rec_ch_{i}' for i in range(pred_size)] if ch_names is None else ch_names
    ch_str = _convert_to_latex(ch_names, ch_str, use_case)
    out_str = [f'Out_ch_{i}' for i in range(pred_size)] if ch_names is None else ch_names[-pred_size:]
    out_str = _convert_to_latex(ch_names, out_str, use_case)[-pred_size:]
    
    colors = ['blue', 'purple'] if (use_case and use_case == 'Duffing_MVE') else ['blue', 'green', 'purple']
    # scale the step_size appropriately, such that always 1000 data points are plotted
    step_size = max(1, eps.shape[0]//1000)
    print('Shapes:', eps[::step_size,:].shape, sum_win_size_sens[::step_size,...].shape)

    # Formulate fitting function for parabola
    def fit_func(x, a, b, c):
        return a*x**2 + b*x + c

    def compute_uncertainty(x, y, num_sects):
        x_sections = np.array_split(np.sort(x), num_sects)
        stds = []
        for section in x_sections:
            sect_idc = np.where((x >= section.min()) & (x < section.max()))
            stds.append(np.std(y[sect_idc]))
        return np.array(stds)

    # Plot the uncertainty of the sensitivity tensor
    x_lim = 3 * amplification
    fig, axs = plt.subplots(1, pred_size, figsize=(12,6))
    axs = [axs] if pred_size == 1 else axs
    for i, ax in enumerate(axs):
        x_ = eps[::step_size,i]
        x = x_[(x_ > -x_lim) & (x_ < x_lim)] # remove outliers outside the 3-sigma confidence interval
        x_fit = np.linspace(x.min(), x.max(), 20)

        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes("top", size=1.05, pad=0.2, sharex=ax)
        ax_histy = divider.append_axes("right", size=1.05, pad=0.2, sharey=ax)
        ax_histx.hist(x, bins=30, density=True, alpha=0.5, color='red', orientation='vertical')
        if amplification > 1:
            ax.axvspan(-3, 3, color='gray', alpha=0.2)
        
        for j, color in zip(range(ch_size), colors):
            y = sum_win_size_sens[::step_size,i,j]
            y = y[(x_ > -x_lim) & (x_ < x_lim)]
            ax.scatter(x, y, s=15, alpha=0.3, edgecolors='black', linewidth=1, color=color, label=f'{ch_str[j]}')
            ax_histy.hist(y, bins=10, density=True, weights=np.sqrt(y), alpha=0.5, color=color, orientation='horizontal')
            # fit a quadratic function to the data, weighted by the square root of the output sensitivity
            popt, _ = curve_fit(fit_func, x, y, sigma=1.0/np.sqrt(y))
            print(f'Fitted parabola for Channel {j+1}: {popt[0]:.3f}*x^2 + {popt[1]:.3f}*x + {popt[2]:.3f}')
            ax.plot(x_fit, fit_func(x_fit, *popt), color=color, linestyle='-', lw=2)
            stds = compute_uncertainty(x, y, len(x_fit))
            ax.fill_between(x_fit, fit_func(x_fit, *popt)-stds, fit_func(x_fit, *popt)+stds, color=color, alpha=0.4)
        
        ax_histx.xaxis.set_tick_params(labelbottom=False)
        ax_histy.yaxis.set_tick_params(labelleft=False)
        ax.set_xlim(-x_lim, x_lim)
        ax.set_xlabel(r'$\frac{x-\mu}{\sigma}$  $[-]$', labelpad=5)
        ax.set_ylabel(r'Output Sensitivity  $[-]$', labelpad=10)
        ax.legend(loc='upper left', bbox_to_anchor=(1.0125,0,1,1.29))
    if save_path:
        fig.tight_layout()
        plt.savefig(os.path.join(save_path, 'uncertainty_sens.pdf'), bbox_inches='tight')
    else:
        plt.suptitle('Sensitivity Range Across the Aleatoric Uncertainty in the Predicted Output', fontsize=13)
        for i, ax in enumerate(axs):
            ax_histx.set_title(f'Recurrent Channel {out_str[i]}', pad=10)
        fig.tight_layout()
    plt.show()


def plot_grad_sens_comparison(vals_dict, models, sum_mean_std_list, ch_names=None, use_case=None, scientific_notation=False, save_path=None):
    """
    Visualize the comparison of the sensitivity tensors of different models
    for the same prediction type.

    Parameters
    ----------
    vals_dict: dict
        Dict, containing the mode to compare, e.g. alpha, excitation amplitude/frequency;
        as well as the values to compare, e.g. [0.1, 0.2, 0.3]
    models : list[Model consisting of nn.Modules]
        Models consisting of nn.Modules
    sum_mean_std_list : list of torch.Tensor
        List of sensitivity tensors that contain the gradients of the output with respect to the inputs
    ch_names : list of strings, optional
        List of channel names. The default is None, i.e. the channel names are not provided and
        will be generated in the style of 'Inp_ch_i' and 'Rec_ch_i'.
    scientific_notation : bool, optional
        If the plot legends should be displayed in LaTeX scientific notation. The default is False.
    save_path : string, optional
        Path to save the plots. The default is None, i.e. the plots are not saved.
    """
    if save_path:
        folder, file = os.path.split(save_path)
        save_path = os.path.join(os.getcwd(), 'Evaluation_plots', folder)
        os.makedirs(save_path, exist_ok=True)

    win_size = min([max(model.window_size, model.rnn_window) for model in models])
    model = models[0]
    same_ch_sizes = all([(model.input_channels == m.input_channels) for m in models])
    # find the model with the maximum number of input channels
    for m in models:
        if m.input_channels > model.input_channels:
            diff = abs(m.input_channels - model.input_channels)
            model = m

    max_inp_chs = model.input_channels
    num_chs = model.input_channels + model.pred_size
    num_features = win_size * num_chs
    ch_str = [f'Inp_ch_{i}' for i in range(model.input_channels)] + \
        [f'Rec_ch_{i}' for i in range(model.pred_size)] if ch_names is None else ch_names
    ch_str = _convert_to_latex(ch_names, ch_str, use_case)
    ax_zero_label = r'RMS of Sensitivities  $[-]$'
    ax_one_label = r'RMS of Std. Dev. of Sensitivities  $[-]$'

    x_tks_feat = [i for i in np.arange(num_features+1, step=win_size)]
    if scientific_notation:
        colors = ['blue', 'green', 'red', 'purple', 'purple', 'cyan']
        alphas = [1.0 for _ in range(len(models))]
        patterns = ['' for _ in range(len(models))]
    else:
        colors = ['blue', 'green', 'red', 'blue', 'green', 'red']
        alphas = [1.0 if i < 3 else 0.5 for i in range(len(models))]
        patterns = ['', '', '', '//', '\\\\', '//']

    # Plot the sum_mean_feature, sum_std_feature as bar plots
    fig, axs = plt.subplots(1, 2, figsize=(15,6))
    bar_width = 0.1 if len(models) > 3 else 0.25
    bar1 = np.arange(num_chs)
    bars = [[x + i*bar_width for x in bar1] for i in range(1, len(models)+1)]
    bars.insert(0, bar1)

    for model, x, br, color, alpha, pat, value in zip(models, sum_mean_std_list, bars, colors, alphas, patterns, vals_dict['values']):
        sum_mean_feature, sum_std_feature = x # unpack tuple

        if num_chs <= 3:
            rms_sum_out_inp_ch = _reshape_array(model, sum_mean_feature, aggregation='mean')
            std_sum_out_inp_ch = _reshape_array(model, sum_std_feature, aggregation='mean')
            if not same_ch_sizes and (model.input_channels < max_inp_chs):
                insert = np.full(diff, np.nan)
                rms_sum_out_inp_ch = np.insert(rms_sum_out_inp_ch, model.input_channels, insert)
                std_sum_out_inp_ch = np.insert(std_sum_out_inp_ch, model.input_channels, insert)
            axs[0].bar(br, rms_sum_out_inp_ch, width=bar_width, color=color, edgecolor='black', alpha=alpha, hatch=pat, label=f'{value}')
            axs[1].bar(br, std_sum_out_inp_ch, width=bar_width, color=color, edgecolor='black', alpha=alpha, hatch=pat, label=f'{value}')

        else:
            rms_sum_out_inp_ch = _reshape_array(model, sum_mean_feature, aggregation='mean',
                                               repeat=True, repeat_size=win_size)
            std_sum_out_inp_ch = _reshape_array(model, sum_std_feature, aggregation='mean',
                                               repeat=True, repeat_size=win_size)
            if not same_ch_sizes and (model.input_channels < max_inp_chs):
                rec_start_idx = model.input_channels*win_size + 1
                insert = np.full(diff, np.nan).repeat(win_size)
                rms_sum_out_inp_ch = np.insert(rms_sum_out_inp_ch, rec_start_idx, insert)
                std_sum_out_inp_ch = np.insert(std_sum_out_inp_ch, rec_start_idx, insert)
            axs[0].step(np.arange(num_features+1), rms_sum_out_inp_ch, alpha=0.8, color=color, label=f'{value}')
            axs[1].step(np.arange(num_features+1), std_sum_out_inp_ch, alpha=0.8, color=color, label=f'{value}')
    
    angle = 65 if num_chs > 3 else 0
    halign = 'center' if num_chs < 5 else 'left'
    axs[0].set_ylabel(ax_zero_label, labelpad=10)
    axs[1].set_ylabel(ax_one_label, labelpad=10)
    for ax in axs:
        ax.set_xlabel('Channels', labelpad=10)
        if num_chs <= 3:
            ax.set_xticks([x + (len(models)/2 - 0.5) * bar_width for x in bar1])
            ax.set_xticklabels(ch_str, rotation=angle, ha=halign)
        else:
            ax.xaxis.grid(True)
            ax.set_xticks(x_tks_feat)
            ax.set_xticklabels(ch_str + [''], rotation=angle, ha=halign)
            ax.axvline(model.input_channels*win_size, color='black', linewidth=2)
        if scientific_notation:
            leg = ax.legend(title=rf'$\{vals_dict["mode"]}$', loc='upper left')
        else:
            leg = ax.legend(title=f'{vals_dict["mode"]}', loc='upper left')
        leg.get_title().set_bbox(dict(facecolor='none', edgecolor='none', pad=20))

    if save_path:
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
        plt.savefig(os.path.join(save_path, f'{file}_comp_bar_plot.pdf'), bbox_inches='tight')
    else:
        plt.suptitle('Sensitivities across Input Features, RMS over Output Channels', fontsize=13)
        axs[0].set_title('RMS of Mean Sensitivities', pad=10)
        axs[1].set_title('RMS of Std of Sensitivities', pad=10)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
    plt.show()


    # Plot the averages across the time windows for each channel as line plots
    fig, axs = plt.subplots(1, 2, figsize=(15,6))
    if scientific_notation:
        alphas = [1.0 if (i+1) % 2 == 0 else 0.3 for i in range(len(models))]
    else:
        alphas = [1.0 for _ in range(len(models))]
    for model, x, color, alpha, value in zip(models, sum_mean_std_list, colors, alphas, vals_dict['values']):
        sum_mean_feature, sum_std_feature = x # unpack tuple
        rms_sum_out = _reshape_array(model, sum_mean_feature)
        std_sum_out = _reshape_array(model, sum_std_feature)
        mean_rms_inp = np.mean(rms_sum_out[:model.input_channels,:], axis=0)
        mean_rms_rec = np.mean(rms_sum_out[model.input_channels:,:], axis=0)
        mean_std_inp = np.mean(std_sum_out[:model.input_channels,:], axis=0)
        mean_std_rec = np.mean(std_sum_out[model.input_channels:,:], axis=0)

        if scientific_notation:
            axs[0].plot(np.concatenate([[np.nan], mean_rms_inp[-win_size:]]), color=color,
                        alpha=alpha, lw=1.75)
            axs[0].plot(np.concatenate([[np.nan], mean_rms_rec[-win_size:]]), color=color,
                        alpha=alpha, lw=1.75, linestyle='-.')
            axs[1].plot(np.concatenate([[np.nan], mean_std_inp[-win_size:]]), color=color,
                        alpha=alpha, lw=1.75)
            axs[1].plot(np.concatenate([[np.nan], mean_std_rec[-win_size:]]), color=color,
                        alpha=alpha, lw=1.75, linestyle='-.')
        else:
            axs[0].plot(np.concatenate([[np.nan], mean_rms_inp[-win_size:]]), color=color,
                        alpha=0.5, lw=1.75)
            axs[0].plot(np.concatenate([[np.nan], mean_rms_rec[-win_size:]]), color=color,
                        alpha=1.0, lw=1.75, linestyle='-.')
            axs[1].plot(np.concatenate([[np.nan], mean_std_inp[-win_size:]]), color=color,
                        alpha=0.5, lw=1.75)
            axs[1].plot(np.concatenate([[np.nan], mean_std_rec[-win_size:]]), color=color,
                        alpha=1.0, lw=1.75, linestyle='-.')
    
    time_deltas = {
        10: 2,
        20: 5,
        25: 5,
        30: 5,
        40: 10,
        50: 10,
        75: 15,
        100: 20,
        125: 25,
        150: 30,
        175: 35,
        200: 40
    }
    time_delta = time_deltas[win_size]
    time_str = [f't-{i}' for i in np.arange(win_size, -1, -time_delta)][:-1]
    time_str.append('t')

    if scientific_notation:
        legend_entries = [Line2D([0], [0], color=color, label=value) \
                          for color, value in zip(colors, vals_dict['values'])]
    else:
        legend_entries = [Line2D([0], [0], color=color, label=value, linestyle='-') \
                          for color, value in zip(colors, vals_dict['values'])]
        if 'Comparison' in use_case:
            legend_anchor = (0.275, 1.0)
        elif 'Steering' in use_case:
            legend_anchor = (0.21, 1.0)
        else:
            legend_anchor = (0.213, 1.0)

    axs[0].set_ylabel(ax_zero_label, labelpad=10)
    axs[1].set_ylabel(ax_one_label, labelpad=10)
    for ax in axs:
        ax.set_xlabel(r'Sliding Time Window  $[-]$', labelpad=10)
        ax.set_xticks(np.arange(win_size+1, step=time_delta))
        ax.set_xticklabels(time_str)
        if scientific_notation:
            leg_one = ax.legend(handles=legend_entries, title=rf'$\{vals_dict["mode"]}$', loc='upper left')
            leg_two = ax.legend(handles=[Line2D([0], [0], color='gray', label='Input'),
                                Line2D([0], [0], color='gray', linestyle='-.', label='Recurrent')],
                                title='Channels', loc='upper left', bbox_to_anchor=(0.15, 1.0))
        else:
            leg_one = ax.legend(handles=legend_entries, title=vals_dict['mode'], loc='upper left')
            leg_two = ax.legend(handles=[Line2D([0], [0], color='gray', label='Input'),
                                Line2D([0], [0], color='gray', linestyle='-.', label='Recurrent')],
                                title='Channels', loc='upper left', bbox_to_anchor=legend_anchor)
        leg_one.get_title().set_bbox(dict(facecolor='none', edgecolor='none', pad=20))
        leg_two.get_title().set_bbox(dict(facecolor='none', edgecolor='none', pad=20))
        ax.add_artist(leg_one)
        ax.add_artist(leg_two)
    if save_path:
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
        plt.savefig(os.path.join(save_path, f'{file}_comp_sliding_time_window.pdf'), bbox_inches='tight')
    else:
        plt.suptitle('Sensitivities across Time Window for all Input Channels, RMS over Output Channels', fontsize=13)
        axs[0].set_title('RMS of Mean Sensitivities', pad=10)
        axs[1].set_title('Sum of Std of Sensitivities', pad=10)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)
    plt.show()


def plot_first_weight_matrix(model, abs_values=False, ch_names=None, use_case=None, save_path=None):
    """
    Visualize the model's first weight matrix and subsequent slices of it
    as heatmaps, to compare with the sensitivity plots.

    Parameters
    ----------
    model : nn.module
        Model consisting of nn.Modules
    abs_values : bool, optional
        If the absolute values of the weight matrix entries should be plotted.
        The default is False.
    save_path : string, optional
        Path to save the plots. The default is None, i.e. the plots are not saved.
    """
    if save_path:
        base_path = os.path.abspath(BASE_PATH)
        use_case_str = use_case if use_case else ''
        save_path = os.path.join(base_path, 'Evaluation_plots', use_case_str, save_path)
        os.makedirs(save_path, exist_ok=True)
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            # print(name)
            # print(param.shape)
            weight_matrix = param.detach().numpy()
            break

    input_channels, pred_size = model.input_channels, model.pred_size
    out_features, in_features = weight_matrix.shape
    rec_start_idx = input_channels*model.window_size
    x_tks_feat = list(range(0, rec_start_idx, model.window_size)) + \
                    list(range(rec_start_idx, in_features+1, model.rnn_window))
    y_tks_feat = list(range(0, out_features+1, out_features//4))

    ch_size = input_channels + pred_size
    ch_str = [f'Inp_ch_{i}' for i in range(input_channels)] + \
        [f'Rec_ch_{i}' for i in range(pred_size)] if ch_names is None else ch_names
    inp_str = _convert_to_latex(ch_names, ch_str, use_case)[:input_channels]
    out_str = [f'Out_ch_{i}' for i in range(pred_size)] if ch_names is None else ch_names[-pred_size:]
    out_str = _convert_to_latex(ch_names, out_str, use_case)[-pred_size:]
    angle = 65 if ch_size > 3 else 0
    fig_size = (15, 9) if ch_size < 5 else (15, 11)

    colormap = 'coolwarm'
    abs_str = 'signed values'
    if abs_values:
        weight_matrix = np.abs(weight_matrix)
        colormap = 'jet'
        abs_str = 'absolute values'
    input_ch_matrix = weight_matrix[:, :rec_start_idx]
    rec_ch_matrix = weight_matrix[:, rec_start_idx:]

    fig, axs = plt.subplots(2,2, figsize=fig_size)
    pos0 = axs[0,0].imshow(weight_matrix, cmap=colormap, interpolation='none',
                        aspect='auto', extent=[0, in_features, 0, out_features])
    fig.colorbar(pos0, ax=axs[0,0])
    axs[0,0].axvline(rec_start_idx, color='black', linewidth=2)
    axs[0,0].set_xticks(x_tks_feat)
    axs[0,0].set_yticks(y_tks_feat)
    axs[0,0].grid(axis='x', color='black', lw=0.5)

    pos1 = axs[0,1].imshow(input_ch_matrix, cmap=colormap, interpolation='none',
                        aspect='auto', extent=[0, rec_start_idx, 0, out_features])
    fig.colorbar(pos1, ax=axs[0,1])
    axs[0,1].set_xticks(range(0, rec_start_idx+1, model.window_size))
    if ch_names:
        axs[0,1].set_xticklabels(inp_str + [''], rotation=angle, ha='center')
    axs[0,1].set_yticks(y_tks_feat)
    axs[0,1].grid(axis='x', color='black', lw=0.5)

    pos2 = axs[1,0].imshow(rec_ch_matrix, cmap=colormap, interpolation='none',
                        aspect='auto', extent=[0, in_features-rec_start_idx, 0, out_features])
    fig.colorbar(pos2, ax=axs[1,0])
    axs[1,0].set_xticks(range(0, in_features-rec_start_idx+1, model.rnn_window))
    if ch_names:
        axs[1,0].set_xticklabels(out_str + [''], ha='left')
    axs[1,0].set_yticks(y_tks_feat)
    axs[1,0].grid(axis='x', color='black', lw=0.5)

    # delete empty subplot
    fig.delaxes(axs[1,1])
    for ax in axs.flatten():
        ax.set_xlabel(r'Input Features  $[-]$', labelpad=5)
        ax.set_ylabel(r'Output Features  $[-]$', labelpad=10)

    if save_path:
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2, hspace=0.35)
        plt.savefig(os.path.join(save_path, 'weight_matrix_one.pdf'), bbox_inches='tight')
    else:
        plt.suptitle(f'1st Weight Matrix and Slices of Input & Recurrent Channels ({abs_str})', fontsize=13)
        axs[0,0].set_title(r'$1^{st}$ Weight Matrix (full)')
        axs[0,1].set_title('Weight Matrix slice of Input Channels')
        axs[1,0].set_title('Weight Matrix slice of Recurrent Channels')
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2, hspace=0.35)
    plt.show()


def plot_all_weight_matrices(model, abs_values=False, rel_threshold=0.1, use_case=None, save_path=None):
    """
    Visualize all weight matrices of the model as heatmaps and their distributions
    as histograms, highlighting the percentage of weights that fall within a
    certain threshold range.

    Parameters
    ----------
    model : nn.module
        Model consisting of nn.Modules
    abs_values : bool, optional
        If the absolute values of the weight matrix entries should be plotted.
        The default is False.
    rel_threshold : float, optional
        Threshold value for the percentage of weights that fall within the range [-threshold, threshold].
        The default is 0.1, i.e. 10 % of the maximum absolute value of each weight matrix.
    use_case : string, optional
        Use case for the channel names. The default is None.
    save_path : string, optional
        Path to save the plots. The default is None, i.e. the plots are not saved.
    """
    if save_path:
        base_path = os.path.abspath(BASE_PATH)
        use_case_str = use_case if use_case else ''
        save_path = os.path.join(base_path, 'Evaluation_plots', use_case_str, save_path)
        os.makedirs(save_path, exist_ok=True)
    
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.detach().numpy())
    
    if abs_values:
        weights = [np.abs(w) for w in weights]
        colormap = 'jet'
        abs_str = 'absolute values'
    else:
        colormap = 'coolwarm'
        abs_str = 'signed values'

    w_len = len(weights)
    num_cols = 2 if w_len <= 4 else 3
    num_rows = int(np.ceil(w_len / num_cols))
    fig_size = (12, 8) if num_cols == 2 else (15, 8)

    # plot the values of the weight matrices as imshow plots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size)
    rec_start_idx = model.input_channels * model.window_size
    win_size = max(model.window_size, model.rnn_window)
    for i, (w, ax) in enumerate(zip(weights, axs.flatten()[:w_len])):
        out_features, in_features = w.shape
        x_step = in_features//4 if in_features % 4 == 0 else in_features//5
        if i == 0: # first matrix only
            x_tks_feat = list(range(0, rec_start_idx, model.window_size)) + \
                            list(range(rec_start_idx, in_features+1, model.rnn_window))
        else:
            x_tks_feat = range(0, in_features+1, x_step)
        y_step = out_features//4 if out_features > 4 else 1
        y_tks_feat = range(0, out_features+1, y_step)

        pos = ax.imshow(w, aspect='auto', cmap=colormap, interpolation='none', extent=[0, in_features, 0, out_features])
        fig.colorbar(pos, ax=ax)
        threshold = rel_threshold * max(abs(w.min()), abs(w.max()))
        ax.imshow(np.ma.masked_outside(w, -threshold, threshold), cmap='Greens', alpha=0.2,
                    aspect='auto', interpolation='none', extent=[0, in_features, 0, out_features])
        ax.set_xticks(x_tks_feat)
        if i == 0 and in_features//win_size > 5:
            for label in ax.get_xticklabels()[1:model.input_channels+1:2]:
                label.set_visible(False)
            for label in ax.get_xticklabels()[-model.pred_size:-1]:
                label.set_visible(False)
        ax.set_yticks(y_tks_feat)
        ax.set_xlabel(r'Input features  $[-]$', labelpad=5)
        ax.set_ylabel(r'Output features  $[-]$', labelpad=5)
        ax.set_title(f'Weight Matrix {i+1}')
    
    # delete empty subplots
    num_del_axes = num_rows*num_cols - w_len
    for i in range(1, num_del_axes+1):
        fig.delaxes(axs.flatten()[-i])
    
    if save_path:
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.2)
        plt.savefig(os.path.join(save_path, 'weight_matrices.pdf'), bbox_inches='tight')
    else:
        plt.suptitle(f'Weight matrices ({abs_str})', fontsize=13)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.show()


    # create own legend entries for the histograms
    # legend_entries = [Patch(facecolor='skyblue', edgecolor='black', label='Inside'),
    #                   Patch(facecolor='mediumaquamarine', edgecolor='black', label='Outside')]

    # plot the distributions of the weight matrices as histograms
    fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size)
    for i, (w, ax) in enumerate(zip(weights, axs.flatten()[:w_len])):
        counts, bins, patches = ax.hist(w.flatten(), bins=40, density=True, color='skyblue', edgecolor='black')
        max_value = max(abs(w.min()), abs(w.max()))
        threshold = rel_threshold * max_value
        # color all those bins that fall in the range [-threshold, threshold] in green
        for j in range(len(bins)-1):
            if bins[j] >= -threshold and bins[j+1] <= threshold:
                patches[j].set_facecolor('mediumaquamarine')
        # compute how many percent of the weights fall in the range [-threshold, threshold]
        percentage = np.sum(counts[np.logical_and(bins[:-1] >= -threshold, bins[1:] <= threshold)])
        print(f'Weight Matrix {i+1}')
        print(f'Max. value: {max_value:.3f}, 10% Threshold: +-{threshold:.3f}')
        print(f'Percentage of weights within threshold range: {percentage:.1f}%\n')

        ax.set_xlabel(r'Weight value  $[-]$', labelpad=5)
        ax.set_ylabel(r'Density $[\%]$', labelpad=5)
        ax.set_title(f'Weight Matrix {i+1}')
        # ax.legend(handles=legend_entries, title=rf'$\mathcal{{T}}$ $\in$ [{-threshold:.3f}, {threshold:.3f}]', loc='upper left')
    
    # delete empty subplots
    num_del_axes = num_rows*num_cols - w_len
    for i in range(1, num_del_axes+1):
        fig.delaxes(axs.flatten()[-i])
    
    if save_path:
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        plt.savefig(os.path.join(save_path, 'weight_distributions.pdf'), bbox_inches='tight')
    else:
        plt.suptitle('Weight distributions of the matrices', fontsize=13)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()


'''
def plot_quantiles(model, track, t_start, t_end, output=0, fs=10, show_legend=False, title=None, title_info=None, fig_path=None, is_duffing=True, plot_all=False, show=True):
    """ Runs the QR prediction on track and plots the quantiles

    Parameters
    ----------
    model: QuantileNARX
        QR model to use for prediction
    track: torch.Dataloader
        Single track
    t_start: int, optional
        Start of plotting window. The default is None.
    t_end: int, optional
        End of plotting window. The default is None.
    output: int, optional
        Output sensor to plot. The default is 0.
    fs: int, optional
        Sampling rate of dataset to rescale x axis. The default is 10.
    show_legend : bool, optional   
        Show legend in the plot. The default is False
    title: string, optional
        Title for the plot. The default is None
    title_info: dict[string,string], optional
        Must contain keys ['dataset', 'model', 'track', 'sensor']. The default is None.
    fig_path : string, optional   
        Path to save fig. The default is None
    is_duffing: bool, optional
        Whether it is position prediction (like Duffing dataset) or acceleration prediction. The default is True.
    plot_all: bool, optional
        Whether to plot all quantiles or only the 67.5% and 95% PI. The default is False.
    show: bool, optional
        If the plot should be displayed. The default is True.

    Returns
    -------
    None.

    """
    sns.set(style="white")

    pred = model.prediction(track)
    predicted_quantiles = [x[output] for x in pred]
    ground_truth = torch.tensor([data[1][0][output][0] for data in track])
    median = predicted_quantiles[0]
    
    mse = nn.MSELoss()(median, ground_truth)
    print(f"MSE: {mse}")

    window = slice(t_start, t_end)
    length = len(median[window])
    x = np.arange(0, length) + (0 if not t_start else t_start)
    x = x/fs
    
    sns.lineplot(x=x, y=ground_truth[window], color=sns.color_palette()[1], label="Ground Truth", legend=show_legend)
    sns.lineplot(x=x, y=median[window], color=sns.color_palette()[0], label="Prediction µ", legend=show_legend)
    
    if plot_all:
        for lb, ub in zip(predicted_quantiles[1::2], predicted_quantiles[2::2]):
            plt.fill_between(x, lb[window], ub[window], color=sns.color_palette()[0], alpha=0.2)
    else:
        # Only plot the 67.5% and 95% PI (similar to first and second std of Gaussian)
        plt.fill_between(x, predicted_quantiles[1][window], predicted_quantiles[2][window], color=sns.color_palette()[0], alpha=0.36666)
        plt.fill_between(x, predicted_quantiles[13][window], predicted_quantiles[14][window], color=sns.color_palette()[0], alpha=0.5)
    
    export_plot(show_legend, title, title_info, fig_path, is_duffing, show)
 '''