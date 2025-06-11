# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:41:37 2022

@author: WET2RNG
"""
import os
import pickle
import math

import numpy as np
import pandas as pd
import scipy.signal as sig
import sklearn.metrics as metr
import torch


from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from softsensor.autoreg_models import ARNN, _pred_ARNN_batch

from softsensor.datasets import batch_rec_SW
from softsensor.metrics import (
    crps,
    ece,
    heteroscedasticity,
    mpiw,
    nll,
    pearson,
    picp,
    r2,
    rmse,
    rmv,
    log_area_error,
)
from softsensor.recurrent_models import AR_RNN, RNN_DNN
from softsensor.visualization import plot_sens_analysis, plot_uncertainty_sens


def load_model(Type, path):
    """
    load function of models
    Parameters
    ----------
    Type : str
        Type of Network ['CNN_DNN', 'NARX', 'CNN_NARX', 'AR_CNN', 'RNN'].
    path : str
        path of the Model

    Returns
    -------
    model : nn.Model with nn.Modules
    result_df : pandas.DataFrame
        Results of the hyperparameter optimization.

    """
    result_df = pd.read_csv(os.path.join(path, "result_df.csv"))

    state_dict = os.path.join(path, "Best_model.pt")

    file = open(os.path.join(path, "model_parameters.pkl"), "rb")
    model_params = pickle.load(file)
    file.close()

    model = _get_model(Type, model_params, state_dict)

    return model, result_df


def _get_model(Type, model_params, state_dict):
    """
    internal function to load the model by Type

    Parameters
    ----------
    Type : str
        Type of Network ['CNN_DNN', 'NARX', 'CNN_NARX', 'AR_CNN', 'RNN'].
    model_params : dict
        Model hyper parameters.
    state_dict : str
        path of the state_dict of the model (weights)

    Returns
    -------
    model : nn.Model with nn.Modules

    """
    if Type == "ARNN":
        model = ARNN(**model_params)
    elif Type == "AR_RNN":
        model = AR_RNN(**model_params)
    elif Type == "RNN":
        model = RNN_DNN(**model_params)
    else:
        raise KeyError("No valid model Type given")
    model.load_state_dict(torch.load(state_dict))
    return model


def comp_pred(
    models,
    data_handle,
    track,
    names=None,
    batch_size=256,
    reduce=False,
    sens_analysis=None,
):
    """
    Computes the predictions for given a DataFrame and a Track for the specific
    models

    Parameters
    ----------
    models : list of Models
        DESCRIPTION.
    data_handle : data handle of the Meas Handling class
        data handle that should include the track.
    track : str
        name of the track to observe.
    names : list of str.
        str should include 'NARX' if its a NARX model, 'AR' if its an
        autoregressive model and 'RNN' if is an Recurrent Network
    batch_size : int, optional
        Batch size for models where batching is possible. The default is 256.
    reduce : bool, optional
        If True, reduce the epistemic and aleatoric uncertainty to a total one.
        Only relevant for ensembles and Evidence Estimation ARNN. The default is False.
    sens_analysis : dict, optional
        Dictionary that defines if and how a sensitivity analysis is computed for the prediction.
        If key 'method' is valid, the sensitivity analysis is computed either 'gradient' or 'perturbation'-based.
        If key 'comp' is given & True, gradients of the prediction w.r.t. inputs are computed.
        If key 'plot' is given & True, postprocessing results of the gradients are visualized.
        If key 'sens_length' is given, the prediction is only computed for the n 'sens_length'
            samples in the time series.
        The default is None, i.e. no sensitivity analysis is computed.

    Returns
    -------
    pred_df : pd.DataFrame
        New columns are Added for each Model prediction.
    if comp_sens:
        sens_dict : dict
            Additional list of dictionaries with the sensitivity analysis results for each model.

    Examples
    --------
    >>> t = np.linspace(0, 1.0, 101)
    >>> xlow = np.sin(2 * np.pi * 100 * t)       # 100Hz Signal
    >>> xhigh = np.sin(2 * np.pi * 3000 * t)     # 3000Hz Signal
    >>> d = {'sine_inp': xlow + xhigh,
    >>>      'cos_inp': np.cos(2 * np.pi * 50 * t),
    >>>      'out': np.linspace(0, 1.0, 101)}
    >>> list_of_df = [pd.DataFrame(d), pd.DataFrame(d)]
    >>> test_df = {'sine_inp': 10*xlow + xhigh,
    >>>            'cos_inp': np.cos(2 * np.pi * 50 * t),
    >>>            'out': np.linspace(0, 1.0, 101)}
    >>> test_df = [pd.DataFrame(test_df)]
    >>> data_handle = Meas_handling(list_of_df, train_names=['sine1', 'sine2'],
    >>>                         input_sensors=['sine_inp', 'cos_inp'],
    >>>                         output_sensors=['out'], fs=100,
    >>>                         test_dfs=test_df, test_names=['test'])
    >>> model = ARNN(input_channels=2, pred_size=1, window_size=10, rnn_window=10)
    >>> pred_df = comp_pred([model], data_handle, track, names=['ARNN']) # without sensitivity analysis
    >>> pred_df, sens_dict = comp_pred([model], data_handle, track, names=['ARNN'], sens_analysis={'method': 'gradient', 'params': {'comp': True, 'plot': True}}) # with sensitivity analysis
    """
    if names is None:
        names = [m.Type for m in models]

    pred_df = data_handle.give_dataframe(track)
    sens_list = []

    for m, n in zip(models, names):
        if sens_analysis:
            pred_df, sens = _model_pred(
                m,
                data_handle,
                pred_df,
                n,
                track,
                batch_size,
                reduce=reduce,
                sens_analysis=sens_analysis,
            )
            sens_list.append(sens)
        else:
            pred_df = _model_pred(
                m, data_handle, pred_df, n, track, batch_size, reduce=reduce
            )

    if sens_analysis:
        # return directly the dict itself if only one model is computed
        sens_list = sens_list[0] if len(sens_list) == 1 else sens_list
        return pred_df, sens_list
    else:
        return pred_df


def _model_pred(
    m, data_handle, pred_df, n, track, batch_size=256, reduce=False, sens_analysis=None
):
    """
    Internal Method for the prediction of time series

    Parameters
    ----------
    m : model
        model to predict.
    data_handle : Meas_handling class
        Meas handling class that has internal functions to get Dataloader and
        Dataset.
    pred_df : pd.Dataframe
        Dataframe where the prediction is appended as additional column.
    n : str
        Name of the model. New columns is dataframe will be named as:
        f'{output sensor}_{n}'
    track : str
        name of the track to predict.
    batch_size : int, optional
        Batch size for models where batching is possible. The default is 256.
    reduce : bool, optional
        If True, reduce the epistemic and aleatoric uncertainty to a total one.
        Only relevant for ensembles and Evidence Estimation ARNN. The default is False.
    sens_analysis : dict, optional
        Dictionary that defines if and how a sensitivity analysis is computed for the prediction.
        If key 'method' is valid, the sensitivity analysis is computed either 'gradient' or 'perturbation'-based.
        If key 'comp' is given & True, gradients of the prediction w.r.t. inputs are computed.
        If key 'plot' is given & True, postprocessing results of the gradients are visualized.
        If key 'sens_length' is given, the prediction is only computed for the n 'sens_length'
            samples in the time series.
        The default is None, i.e. no sensitivity analysis is computed.

    Raises
    ------
    AttributeError
        if Modeltype is not implemented.

    Returns
    -------
    pred_df : pd.Dataframe
        prd_df with additional columns for the prediction.
    if comp_sens:
        sens_dict : dict
            Dictionary with the sensitivity analysis results for each model.
    """
    checks = _check_sens_analysis(sens_analysis)
    if sens_analysis:
        sens_params = checks
        method, comp_sens, plot_sens = list(sens_params.values())[:3]
        use_case = sens_params.get("use_case", None)
        orig_length = sens_params.get("orig_length", None)
        random_samples = sens_params.get("random_samples", False)
        amplification = sens_params.get("amplification", 1)
        save_path = sens_params.get("save_path", None)
        channel_names = data_handle.input_sensors + data_handle.output_sensors
        sens_dict = None  # default None type for models without sensitivities
        sens_uq, eps = (
            None,
            None,
        )  # default None type for _predict_ARNN method and ensembles
    else:
        sens_params, method, comp_sens = checks

    # define some generic functions for prediction and sensitivity analysis to avoid code duplication
    def get_prediction(m, dataloader, reduce=False, sens_params=None):
        """Generic function to get the prediction of a model"""
        if sens_params:
            if m.Pred_Type == "Mean_Var" and m.Ensemble:
                return m.prediction(dataloader, reduce=reduce, sens_params=sens_params)
            else:
                return m.prediction(dataloader, sens_params=sens_params)
        else:
            if m.Pred_Type == "Mean_Var" and m.Ensemble:
                return m.prediction(dataloader, reduce=reduce)
            else:
                return m.prediction(dataloader)

    def write_predictions_to_df(pred, pred_df, labels, reduce=False):
        """Update the DataFrame with uncertainty labels, coming from the prediction results."""
        if m.Pred_Type == "Mean_Var" and m.Ensemble and not reduce:
            # Uncertainty labels for Mean_Var and Ensemble without reduction
            uncertainty_labels = [
                f"{out_name}_{n}_ep_var" for out_name in data_handle.output_sensors
            ]
            pred_df.loc[:, uncertainty_labels] = np.array(pred[1].transpose(1, 0))

            uncertainty_labels = [
                f"{out_name}_{n}_al_var" for out_name in data_handle.output_sensors
            ]
            pred_df.loc[:, uncertainty_labels] = np.array(pred[2].transpose(1, 0))
            pred = pred[0]

        elif m.Pred_Type == "Mean_Var" and m.Ensemble and reduce:
            # Uncertainty labels for Mean_Var and Ensemble with reduction
            uncertainty_labels = [
                f"{out_name}_{n}_var" for out_name in data_handle.output_sensors
            ]
            pred_df.loc[:, uncertainty_labels] = np.array(pred[1].transpose(1, 0))
            pred = pred[0]

        elif m.Pred_Type == "Mean_Var" or m.Ensemble:
            # Uncertainty labels for either Mean_Var or Ensemble
            uncertainty_labels = [
                f"{out_name}_{n}_var" for out_name in data_handle.output_sensors
            ]
            pred_df.loc[:, uncertainty_labels] = np.array(pred[1].transpose(1, 0))
            pred = pred[0]

        elif m.Pred_Type == "Quantile":
            # quantiles = np.linspace(0, 1, m.n_quantiles+2)[1:m.n_quantiles+1]*100
            # quantiles = np.rint(quantiles).astype(int)
            labels = ["median"]
            for i in range(math.floor(m.n_quantiles / 2)):
                labels = labels + [f"{n}_lb{i}", f"{n}_ub{i}"]
            labels = [
                f"{out_name}_{l}"
                for out_name in data_handle.output_sensors
                for l in labels
            ]
            pred = torch.reshape(pred, (m.pred_size * m.n_quantiles, -1))
        return pred, labels

    labels = [f"{out_name}_{n}" for out_name in data_handle.output_sensors]
    Type = m.Type
    if Type in ["TF", "ARX"]:
        pred_mimo = m.prediction(pred_df)
        pred_df.loc[:, labels] = np.asarray(pred_mimo)[: len(pred_df)]

    elif Type in ["AR", "AR_RNN"]:
        loader = data_handle.give_list(
            window_size=m.window_size,
            keyword=[track],
            batch_size=1,
            rnn_window=m.rnn_window,
            forecast=m.forecast,
            full_ds=False,
        )

        pred_result = get_prediction(
            m, loader[0], reduce=reduce, sens_params=sens_params
        )
        if method and comp_sens:
            if random_samples:
                pred, sens_dict, sens_uq, eps = pred_result
            else:
                pred, sens_dict = pred_result
            if plot_sens:
                _plot_sensitivities(
                    m,
                    sens_dict,
                    plot_sens,
                    method,
                    channel_names,
                    use_case,
                    orig_length,
                    sens_uq,
                    eps,
                    amplification,
                    save_path=save_path,
                )
        else:
            pred = pred_result

        pred, labels = write_predictions_to_df(pred, pred_df, labels, reduce=reduce)
        pred = pred.transpose(1, 0)
        pred_df.loc[:, labels] = (
            pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.array(pred)
        )

    elif Type == "RNN":
        loader = data_handle.give_list(
            window_size=m.window_size,
            keyword=[track],
            Add_zeros=True,
            batch_size=batch_size,
            forecast=m.forecast,
            full_ds=False,
        )

        pred_result = get_prediction(
            m, loader[0], reduce=reduce, sens_params=sens_params
        )
        if method and comp_sens:
            pred, sens_dict = pred_result
            if plot_sens:
                _plot_sensitivities(
                    m,
                    sens_dict,
                    plot_sens,
                    method,
                    channel_names,
                    use_case,
                    orig_length,
                    sens_uq,
                    eps,
                    save_path=save_path,
                )
        else:
            # pred = m.prediction(loader[0])
            if m.Pred_Type == "Mean_Var" and m.Ensemble:
                pred = m.prediction(loader[0], reduce=reduce)
            else:
                pred = m.prediction(loader[0])

        if m.Pred_Type == "Mean_Var" and m.Ensemble and not reduce:
            uncertainty_labels = [
                f"{out_name}_{n}_ep_var" for out_name in data_handle.output_sensors
            ]
            # pred_df[uncertainty_labels] = pred[1].transpose(1, 0)
            pred_df.loc[:, uncertainty_labels] = pred[1].transpose(1, 0)

            uncertainty_labels = [
                f"{out_name}_{n}_al_var" for out_name in data_handle.output_sensors
            ]
            # pred_df[uncertainty_labels] = pred[2].transpose(1, 0)
            pred_df.loc[:, uncertainty_labels] = pred[2].transpose(1, 0)
            pred = pred[0]

        elif m.Pred_Type == "Mean_Var" and m.Ensemble and reduce:
            uncertainty_labels = [
                f"{out_name}_{n}_var" for out_name in data_handle.output_sensors
            ]
            # pred_df[uncertainty_labels] = pred[1].transpose(1, 0)
            pred_df.loc[:, uncertainty_labels] = pred[1].transpose(1, 0)
            pred = pred[0]

        elif m.Pred_Type == "Mean_Var" or m.Ensemble:
            uncertainty_labels = [
                f"{out_name}_{n}_var" for out_name in data_handle.output_sensors
            ]
            # pred_df[uncertainty_labels] = pred[1].transpose(1, 0)
            pred_df.loc[:, uncertainty_labels] = np.array(pred[1].transpose(1, 0))
            pred = pred[0]

        pred = pred.transpose(1, 0)
        pred_df.loc[:, labels] = (
            pred.detach().cpu().numpy() if torch.is_tensor(pred) else np.array(pred)
        )

    else:
        raise AttributeError(
            f"Modeltype for model {m.__class__.__name__} not implemented"
        )

    if sens_analysis:
        return pred_df, sens_dict
    else:
        return pred_df


def comp_batch(
    models,
    data_handle,
    tracks,
    names,
    device="cpu",
    batch_size=256,
    n_samples=5,
    reduce=False,
    sens_analysis=None,
):
    """
    Compute the prediction for a list of models and tracks

    Parameters
    ----------
    models : list of models
        list of models for computation. MOdels need to be defined in _model_pred.
    data_handle : Meas_handling class
        Meas handling class that has internal functions to get Dataloader and
        Dataset.
    tracks : list of str
        Names of the tracks to predict.
    names : list of str
        Names that are added to the column name.
    device : str, optional
        device for computation. The default is 'cpu'.
    batch_size : int, optional
        Batch size for models where batching is possible. The default is 256.
    reduce : bool, optional
        If True, reduce the epistemic and aleatoric uncertainty to a total one.
        Only relevant for ensembles and Evidence Estimation ARNN. The default is False.
    sens_analysis : dict, optional
        Dictionary that defines if and how a sensitivity analysis is computed for the prediction.
        If key 'method' is valid, the sensitivity analysis is computed either 'gradient' or 'perturbation'-based.
        If key 'comp' is given & True, gradients of the prediction w.r.t. inputs are computed.
        If key 'plot' is given & True, postprocessing results of the gradients are visualized.
        If key 'sens_length' is given, the prediction is only computed for the n 'sens_length'
            samples in the time series.
        The default is None, i.e. no sensitivity analysis is computed.

    Returns
    -------
    pred_df : list of pd.Dataframe
        pred_dfs with additional columns for the predictions.

    Examples
    --------

    Define Data

    >>> import softsensor.meas_handling as ms
    >>> import numpy as np
    >>> import pandas as pd
    >>> t = np.linspace(0, 1.0, 101)
    >>> d = {'inp1': np.random.randn(101),
             'inp2': np.random.randn(101),
             'out': np.random.randn(101)}
    >>> handler = ms.Meas_handling([pd.DataFrame(d, index=t)], ['train'],
                                   ['inp1', 'inp2'], ['out'], fs=100)

    Compute Prediction

    >>> from softsensor.eval_tools import comp_batch
    >>> import softsensor.autoreg_models
    >>> params = {'input_channels': 2,
                  'pred_size': 1,
                  'window_size': 10,
                  'rnn_window': 10}
    >>> m = softsensor.autoreg_models.ARNN(**params, hidden_size=[16, 8])
    >>> dataframes = comp_batch([m], handler, handler.train_names,
                                ['ARNN'], device='cpu')
    >>> list(dataframes[0].columns)
    ['inp1', 'inp2', 'out', 'out_ARNN']


    Compute Prediciton wth uncertainty

    >>> import softsensor.homoscedastic_model as hm
    >>> vars = hm.fit_homoscedastic_var(dataframes, ['out'], ['out_ARNN'])
    >>> homosc_m = hm.HomoscedasticModel(m, vars)
    >>> sepmve = softsensor.autoreg_models.SeparateMVEARNN(**params,mean_model=m,
                                                           var_hidden_size=[16, 8])
    >>> dataframes = comp_batch([m, homosc_m, sepmve], handler, handler.train_names,
                                ['ARNN', 'Homosc_ARNN', 'SepMVE'], device='cpu')
    >>> list(dataframes[0].columns)
    ['inp1','inp2','out','out_ARNN','out_Homosc_ARNN','out_Homosc_ARNN_var','out_SepMVE','out_SepMVE_var']
    """
    checks = _check_sens_analysis(sens_analysis)
    if sens_analysis:
        sens_params = checks
        method, comp_sens, plot_sens = list(sens_params.values())[:3]
        use_case = sens_params.get("use_case", None)
        orig_length = sens_params.get("orig_length", None)
        amplification = sens_params.get("amplification", 1)
        save_path = sens_params.get("save_path", None)
        channel_names = data_handle.input_sensors + data_handle.output_sensors
        sens_list = []
    else:
        method, comp_sens = checks[1:]

    dataframes = data_handle.give_dataframes(tracks)
    for m, n in zip(models, names):
        if sens_analysis:
            sens_dict = None  # default None type for models without sensitivities
            sens_uq, eps = (
                None,
                None,
            )  # default None type for sens analysis without amplification

        if m.Type == "AR" and (m.Ensemble is False) and (m.Pred_Type != "Quantile"):
            if method and comp_sens:
                pred, sens_dict, sens_uq, eps = _comp_ARNN_batch(
                    m, data_handle, tracks, device, sens_params=sens_params
                )
                if plot_sens:
                    _plot_sensitivities(
                        m,
                        sens_dict,
                        plot_sens,
                        method,
                        channel_names,
                        use_case,
                        orig_length,
                        sens_uq,
                        eps,
                        amplification,
                        batched=True,
                        save_path=save_path,
                    )
            else:
                pred = _comp_ARNN_batch(m, data_handle, tracks, device)
            if m.Pred_Type == "Point":
                labels = [f"{out_name}_{n}" for out_name in data_handle.output_sensors]
                dataframes = _ARNN_dataframe_pred(dataframes, pred, labels)
            if m.Pred_Type == "Mean_Var":
                labels = [f"{out_name}_{n}" for out_name in data_handle.output_sensors]
                dataframes = _ARNN_dataframe_pred(dataframes, pred[0], labels)
                labels = [
                    f"{out_name}_{n}_var" for out_name in data_handle.output_sensors
                ]
                dataframes = _ARNN_dataframe_pred(dataframes, pred[1], labels)
            if sens_analysis:
                sens_list.append(sens_dict)

        else:
            if sens_analysis:
                dict_list = []
                sens_analysis["params"]["plot"] = False
                for track, df in zip(tracks, dataframes):
                    df, sens_dict = _model_pred(
                        m,
                        data_handle,
                        df,
                        n,
                        track,
                        batch_size,
                        reduce=reduce,
                        sens_analysis=sens_analysis,
                    )
                    dict_list.append(sens_dict)
                if sens_dict:
                    sens_dict = {
                        k: [] for k in dict_list[0].keys()
                    }  # same list structure as with batched approach
                    for d in dict_list:
                        for k, v in d.items():
                            sens_dict[k].append(v)
                    if plot_sens:
                        _plot_sensitivities(
                            m,
                            sens_dict,
                            plot_sens,
                            method,
                            channel_names,
                            use_case,
                            orig_length,
                            sens_uq,
                            eps,
                            batched=True,
                            save_path=save_path,
                        )
                sens_analysis["params"]["plot"] = True
                sens_list.append(sens_dict)
            else:
                for track, df in zip(tracks, dataframes):
                    df = _model_pred(
                        m, data_handle, df, n, track, batch_size, reduce=reduce
                    )

    if sens_analysis:
        sens_list = sens_list[0] if len(sens_list) == 1 else sens_list
        return dataframes, sens_list
    else:
        return dataframes


def _ARNN_dataframe_pred(dataframes, pred, labels):
    """
    internal method to add predictions to dataframes

    Parameters
    ----------
    dataframes : list of df
        dataframes where the predction is added as columns.
    pred : list of arrays
        prediction to be added to the dataframes.
    labels : list of str
        column names.

    Raises
    ------
    ValueError
        Raised igf lengths of dataframes and pred does not match.

    Returns
    -------
    dataframes : dataframes : list of df
        dataframes with added predictions

    """
    if len(dataframes) != len(pred):
        raise ValueError("length of lists does not match")
    for i, p in enumerate(pred):
        dataframes[i][labels] = p.transpose(1, 0).cpu()
    return dataframes


def _comp_ARNN_batch(model, data_handle, tracks, device="cpu", sens_params=None):
    """
    Internal Method to compute the batch of AR Models for faster computation

    Parameters
    ----------
    model : Autoregressive model
        Model for the prediction.
    data_handle : Meas_handling class
        Measurement handling that contains tracks.
    tracks : list of str
        list of tracks to compute prediction.
    device : str, optional
        device for computation. The default is 'cpu'.
    loss_fkt : nn.Loss, optional
        Loss function for training. The default is None.
    comp_gradients : bool, optional
        If True, gradients of the prediction w.r.t. inputs are computed.
        The default is False.

    Returns
    -------
    if loss_ft=None:
        list of torch.Tensor
            Torch Tensor of same length as tracks
    if loss_ft=torch loss function:
        list of (torch.Tensor, loss)
            list of tuple of Torch Tensor of same length as input and loss

    """
    sws = data_handle.give_Datasets(
        model.window_size,
        keyword=tracks,
        rnn_window=model.rnn_window,
        full_ds=False,
        forecast=model.forecast,
    )

    batch_sw = batch_rec_SW(sws)
    forecast = batch_sw.forecast

    if sens_params:
        method = sens_params.get("method", "")
        comp_sens = sens_params.get("comp", False)
        sens_length = sens_params.get("sens_length", None)
    else:
        comp_sens = False

    if comp_sens:
        if sens_length:
            # commonly shared sens_length cannot be larger than smallest length in the batch
            add_zeros = [sw.add_zero for sw in batch_sw.sws]
            min_length = min(
                [
                    le * forecast - zeros
                    for le, zeros in zip(batch_sw.__lengths__(), add_zeros)
                ]
            )
            if sens_length > min_length:
                sens_params["sens_length"] = (
                    min_length  # update sensitivity length in params dict
                )
                print(
                    f"INFO: Given sensitivity length was changed to {min_length} as the smallest length in the batch."
                )

        results = _pred_ARNN_batch(model, batch_sw, device, sens_params=sens_params)
        prediction_sh, sensitivities, sens_uq, eps = results
    else:
        prediction_sh = _pred_ARNN_batch(model, batch_sw, device)

    # Make List of prediction
    if model.Pred_Type == "Point":
        pred_list = list(prediction_sh)
        if comp_sens:
            sens_mean_list = list(sensitivities)
    elif model.Pred_Type == "Mean_Var":
        pred_list = list(prediction_sh[0])
        var_pred_list = list(prediction_sh[1])
        if comp_sens:
            sens_mean_list = list(sensitivities[0])
            sens_var_list = list(sensitivities[1])

    if len(pred_list) > 1:
        orig_indizes = batch_sw.permutation()
        add_zeros = [sw.add_zero for sw in batch_sw.sws]

        # cut the predictions and gradient matrices to their original length
        for i, (p, le, zeros) in enumerate(
            zip(pred_list, batch_sw.__lengths__(), add_zeros)
        ):
            pred_list[i] = p[:, : le * forecast - zeros]
            if comp_sens and sens_length is None:
                sens_mean_list[i] = sens_mean_list[i][: le * forecast - zeros, :, :]
        if model.Pred_Type == "Mean_Var":
            for i, (p, le, zeros) in enumerate(
                zip(var_pred_list, batch_sw.__lengths__(), add_zeros)
            ):
                var_pred_list[i] = p[:, : le * forecast - zeros]
                if comp_sens and sens_length is None:
                    sens_var_list[i] = sens_var_list[i][: le * forecast - zeros, :, :]

        # sort the predictions and gradient matrices back to their original order
        pred = [x for _, x in sorted(zip(orig_indizes, pred_list))]
        if model.Pred_Type == "Mean_Var":
            var_pred = [x for _, x in sorted(zip(orig_indizes, var_pred_list))]

        if comp_sens:
            sens_mean = [x for _, x in sorted(zip(orig_indizes, sens_mean_list))]
            if model.Pred_Type == "Mean_Var":
                sens_var = [x for _, x in sorted(zip(orig_indizes, sens_var_list))]
    else:
        le = batch_sw.__lengths__()[0]
        pred = [pred_list[0][:, : le * forecast - batch_sw.sws[0].add_zero]]
        if model.Pred_Type == "Mean_Var":
            var_pred = [var_pred_list[0][:, : le * forecast - batch_sw.sws[0].add_zero]]

        if comp_sens:
            sens_mean = (
                [sens_mean_list[0][: le * forecast - batch_sw.sws[0].add_zero, :, :]]
                if sens_length is None
                else [sens_mean_list[0]]
            )
            if model.Pred_Type == "Mean_Var":
                sens_var = (
                    [sens_var_list[0][: le * forecast - batch_sw.sws[0].add_zero, :, :]]
                    if sens_length is None
                    else [sens_var_list[0]]
                )

    for i, p in enumerate(pred):
        if len(p.shape) == 1:
            pred[i] = p.unsqueeze(dim=0)

    if model.Pred_Type == "Mean_Var":
        for i, p in enumerate(var_pred):
            if len(p.shape) == 1:
                var_pred[i] = p.unsqueeze(dim=0)
        pred = (pred, var_pred)

    # print(f'Shapes of predictions: {[p.shape for p in pred]}')
    if comp_sens:
        # print(f'Shapes of {method} matrices: {[s.shape for s in sens_mean]}\n')
        if model.Pred_Type == "Mean_Var":
            sensitivity_dict = {"Mean": sens_mean, "Aleatoric_UQ": sens_var}
        else:
            sensitivity_dict = {"Point": sens_mean}
        return pred, sensitivity_dict, sens_uq, eps
    else:
        return pred


def comp_error(
    test_df,
    out_sens,
    fs=None,
    names=["pred"],
    metrics=["MSE"],
    freq_metrics=None,
    freq_range=None,
    bins=20,
):
    """
    Computes the Error from a df with specific Names in the column

    Parameters
    ----------
    test_df : pandas DataFrame
        DataFrame that must include the original output and the prediction
        column name of the prediction must look like: 'out_sens_name'.
    out_sens : list of str
        column names to observe
    fs : float
        sampling rate of the df for psd error computation.
    names : list of str, optional
        list of names that are appended to the original column.
        The default is ['pred'].
    metrics : list of str, optional
        Metrics to Evaluate in Time domain ['MSE', 'MAE', 'MAPE'].
        The default is ['MSE'].
    freq_range : tuple of float, optional
        range in which the psd error is computed. The default is None.

    Returns
    -------
    result_df : pandas DataFrame
        DataFrame with errors as index and names as columns.

    Examples
    --------

    Based on the examples in comp_batch. Prediction of point Metrics

    >>> from softsensor.eval_tools import comp_error
    >>> comp_error(dataframes[0], out_sens=['out'], names=['ARNN', 'Homosc_ARNN', 'SepMVE'],
                   metrics=['MSE', 'MAE'], freq_range=None)
             ARNN  Homosc_ARNN    SepMVE
    out_MSE  1.297152     1.297152  1.297152
    out_MAE  0.924926     0.924926  0.924926


    Prediction of Distributional Metrics

    >>> comp_error(dataframes[0], out_sens=['out'], names=['Homosc_ARNN', 'SepMVE'],
                   metrics=['NLL', 'ECE'], freq_range=None)
             Homosc_ARNN    SepMVE
    out_NLL     0.630110  0.678553
    out_ECE     0.023137  0.083637

    Prediction of Statistical Metrics

    >>> comp_error(dataframes[0], out_sens=['out'], names=['ARNN', 'Homosc_ARNN', 'SepMVE'],
                   metrics=['JSD', 'Wasserstein'], freq_range=None)
                         ARNN  Homosc_ARNN    SepMVE
    out_JSD          0.680494     0.680494  0.680494
    out_Wasserstein  0.073267     0.073267  0.073267


    Prediction of Metrics in frequency domain (PSD)

    >>> comp_error(dataframes[0], out_sens=['out'], names=['ARNN', 'Homosc_ARNN', 'SepMVE'],
                   metrics=None, freq_metrics=['MSLE'], fs=100, freq_range=(5, 25))
                      ARNN  Homosc_ARNN    SepMVE
    out_PSD_MSLE  0.000864     0.000864  0.000864

    """
    if metrics is not None:
        metr_err = []
        for m in metrics:
            err = comp_metrics(test_df, out_sens, names, metric=m, bins=bins)
            metr_err.append(
                pd.DataFrame(err, columns=names, index=[f"{s}_{m}" for s in out_sens])
            )
        metr_err = pd.concat(metr_err)

    if freq_metrics is not None:
        psd_err = []
        for m in freq_metrics:
            psd = comp_psd(test_df, out_sens, fs, names, freq_range)
            err = comp_metrics(psd, out_sens, names, metric=m)
            psd_err.append(
                pd.DataFrame(
                    err, columns=names, index=[f"{s}_PSD_{m}" for s in out_sens]
                )
            )
        psd_err = pd.concat(psd_err)

    if (metrics is not None) and (freq_metrics is not None):
        result_df = pd.concat((metr_err, psd_err))
    elif (metrics is None) and (freq_metrics is not None):
        result_df = psd_err
    elif (metrics is not None) and (freq_metrics is None):
        result_df = metr_err
    else:
        return None

    return result_df


def comp_psd(test_df, out_sens, fs, names=["pred"], freq_range=None):
    """
    Compute the MLPE of the PSD's

    Parameters
    ----------
    test_df : pandas DataFrame
        DataFrame that must include the original output and the prediction
        column name of the prediction must look like: 'out_sens_name'.
    out_sens : list of str
        column names to observe
    fs : float
        sampling rate of the df for psd error computation.
    names : list of str, optional
        list of names that are appended to the original column.
        The default is ['pred'].
    freq_range : tuple of float, optional
        range in which the psd error is computed. The default is None.

    Returns
    -------
    psd_error : matrix of shape [len(out_sens), len(names)]

    """
    psd_list = []
    temp_n = []
    for i, s in enumerate(out_sens):
        f, psd_original = sig.welch(test_df[s], fs=fs)
        if freq_range is not None:
            psd_original = psd_original[(freq_range[0] < f) & (f < freq_range[1])]
            f = f[(freq_range[0] < f) & (f < freq_range[1])]

        temp_n.append(f"{s}")
        psd_list.append(psd_original.reshape((len(psd_original), 1)))

        for ii, n in enumerate(names):
            f, psd_mod = sig.welch(test_df[f"{s}_{n}"].fillna(0), fs=fs)
            if freq_range is not None:
                psd_mod = psd_mod[(freq_range[0] < f) & (f < freq_range[1])]
                f = f[(freq_range[0] < f) & (f < freq_range[1])]

            temp_n.append(f"{s}_{n}")
            psd_list.append(psd_mod.reshape((len(psd_mod), 1)))

    psd_array = np.concatenate(psd_list, axis=1)
    psd_df = pd.DataFrame(psd_array, index=f, columns=temp_n)
    return psd_df


def comp_metrics(test_df, out_sens, names=["pred"], metric="MSE", bins=20):
    """
    Compute the Errors in Time domain

    Parameters
    ----------
    test_df : pandas DataFrame
        DataFrame that must include the original output and the prediction
        column name of the prediction must look like: 'out_sens_name'.

        if metric is uncertainty metric:
            column name of the uncertainty must look like: 'uncertainty_{out_sens}_{name}'.

    out_sens : list of str
        column names to observe
    names : list of str, optional
        list of names that are appended to the original column.
        The default is ['pred'].
    metrics : list of str, optional
        Metrics to Evaluate in Time domain ['MSE', 'MAE', 'MAPE'].
        The default is ['MSE'].

    Returns
    -------
    error : matrix of shape [len(out_sens), len(names)]

    """
    point_metrics = {
        "MSE": metr.mean_squared_error,
        "MAE": metr.mean_absolute_error,
        "MAPE": metr.mean_absolute_percentage_error,
        "MSLE": metr.mean_squared_log_error,
    }

    uncertainty_metrics = {
        "RMSE": rmse,
        "R2": r2,
        "Corr": pearson,
        "NLL": nll,
        "RMV": rmv,
        "CRPS": crps,
        "Het": heteroscedasticity,
        "PICP": picp,
        "MPIW": mpiw,
        "ECE": ece,
    }

    error = np.zeros((len(out_sens), len(names)))

    if metric in ["KLD", "JSD", "Wasserstein"]:
        nam = [f"{s}" for s in out_sens] + [f"{s}_{n}" for n in names for s in out_sens]
        bins = np.histogram(np.hstack((np.array(test_df[nam]))), bins=bins)[1]

    for i, s in enumerate(out_sens):
        for ii, n in enumerate(names):
            if metric in point_metrics:
                error[i, ii] = point_metrics[metric](test_df[s], test_df[f"{s}_{n}"])
            elif metric in uncertainty_metrics:
                target, mean, variance = [
                    torch.tensor(test_df[x].values)
                    for x in [s, f"{s}_{n}", f"{s}_{n}_var"]
                ]
                error[i, ii] = float(
                    uncertainty_metrics[metric](mean, target, variance)
                )

            elif metric == "KLD":
                p_x = np.histogram(test_df[f"{s}"], bins=bins)[0] / len(test_df[f"{s}"])
                p_m = np.histogram(test_df[f"{s}_{n}"], bins=bins)[0] / len(
                    test_df[f"{s}_{n}"]
                )
                if not np.all(p_m):
                    print(
                        "WARNING: Histogram contains zero elements."
                        + "KL Divergence will result in infinite values."
                        + "Consider using the Wasserstein Distance or Jensen Shannon Divergence instead"
                    )
                    error[i, ii] = np.inf
                else:
                    error[i, ii] = np.sum(kl_div(p_x, p_m))

            elif metric == "JSD":
                p_x = np.histogram(test_df[f"{s}"], bins=bins)[0] / len(test_df[f"{s}"])
                p_m = np.histogram(test_df[f"{s}_{n}"], bins=bins)[0] / len(
                    test_df[f"{s}_{n}"]
                )
                error[i, ii] = jensenshannon(p_x, p_m)

            elif metric == "Wasserstein":
                p_x = np.histogram(test_df[f"{s}"], bins=bins)[0] / len(test_df[f"{s}"])
                p_m = np.histogram(test_df[f"{s}_{n}"], bins=bins)[0] / len(
                    test_df[f"{s}_{n}"]
                )
                error[i, ii] = wasserstein_distance(p_x, p_m)

            elif metric == "log_area":
                # err = log_area_error(test_df[f'{s}_{n}'], test_df[s], test_df.index)
                error[i, ii] = log_area_error(
                    test_df[f"{s}_{n}"], test_df[s], test_df.index
                )

    return error


def _comp_mean_metrics_single_track(
    models, data_handle, track, fs, model_names, metrics, freq_range=None
):
    """Compute the metric scores of models on a single track

    Parameters
    ----------
    models: list[nn.Module]
        Torch models to evaluate
    data_handle: Datahandle
        Datahandle that contains track
    track: String
        Name of the track to evaluate
    fs : float
        Sampling rate of the df for psd error computation.
    model_names: list[string]
        Names of the models
    metrics: list[string]
        Names of the metrics to evaluate
    freq_range : tuple of float, optional
        range in which the psd error is computed. The default is None.

    Returns
    -------
    scores_df: dict[str, float]
    """

    pred_df = comp_pred(models, data_handle, track, model_names)
    scores_df = comp_error(
        pred_df, data_handle.output_sensors, fs, model_names, metrics, freq_range
    )
    return scores_df


def comp_mean_metrics(models, data_handle, fs, model_names, metrics, freq_range=None):
    """Compute the mean metric scores of models on the test tracks

    Parameters
    ----------
    models: list[nn.Module]
        Torch models to evaluate
    data_handle: Datahandle
        Datahandle that contains track
    fs : float
        Sampling rate of the df for psd error computation.
    model_names: list[string]
        Names of the models
    metrics: list[string]
        Names of the metrics to evaluate
    freq_range : tuple of float, optional
        range in which the psd error is computed. The default is None.
    reduce : bool, optional
        If True we compute the mean scores over all output variables. The default is False.

    Returns
    -------
    scores_df: dict[str, float]
    """
    track_scores = [
        _comp_mean_metrics_single_track(
            models, data_handle, track, fs, model_names, metrics, freq_range
        )
        for track in data_handle.test_names
    ]
    mean_scores = pd.concat(track_scores).groupby(level=0).mean()

    return mean_scores[::-1]


def _check_sens_analysis(sens_analysis):
    """
    Check the validity of the sensitivity analysis dictionary.

    Parameters
    ----------
    sens_analysis : dict
        Dictionary that defines if and how a sensitivity analysis is computed for the prediction.

    Returns
    -------
    if sens_analysis is None:
        Flattened dict (None), method (empty string), comp_sens (False).
    else:
        Flattened dict (from input argument)
    """
    if sens_analysis:
        method = sens_analysis.get("method", "")
        params = sens_analysis.get("params", {})
        comp_sens = params.get("comp", False)
        plot_sens = params.get("plot", False)
        sens_params = {"method": method, "comp": comp_sens, "plot": plot_sens}
        sens_params.update(params)
        return sens_params
    else:
        sens_params = None
        method = ""
        comp_sens = False
        return sens_params, method, comp_sens


def _plot_sensitivities(
    model,
    sens_dict,
    plot_sens,
    method,
    ch_names,
    use_case=None,
    orig_length=None,
    sens_uq=None,
    eps=None,
    amplification=1,
    batched=False,
    save_path=None,
):
    """
    Plot the results of the sensitivity analysis for every item in the sensitivity dictionary.
    """
    plot_sens = plot_sens.lower() if isinstance(plot_sens, str) else plot_sens
    if plot_sens not in [True, False, "all"]:
        raise ValueError(
            (
                f'Invalid value given for key "plot"! '
                f'Expected True, False or "all", got "{plot_sens}".'
            )
        )

    sens_dict = sens_dict if plot_sens == "all" else dict(list(sens_dict.items())[:1])
    for pred_result, sens_item in sens_dict.items():
        if batched:  # only for comp_batch method
            sens = torch.stack(sens_item).mean(dim=0)
            title = (
                f"\n### {method.upper()}-based Sensitivity analysis results for "
                f"{model.__class__.__name__} model, avg. over {len(sens_item)} test tracks, {pred_result.upper()} prediction ###"
            )
        else:
            sens = sens_item
            title = (
                f"\n### {method.upper()}-based Sensitivity analysis results for "
                f"{model.__class__.__name__} model, {pred_result.upper()} prediction ###"
            )
        plot_sens_analysis(
            model,
            sens,
            ch_names,
            title=title,
            use_case=use_case,
            orig_length=orig_length,
            save_path=save_path,
        )

    if (sens_uq is not None) and (eps is not None):
        plot_uncertainty_sens(
            model, sens_uq, eps, ch_names, use_case, amplification, save_path
        )
