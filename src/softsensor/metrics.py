# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from scipy.stats import rv_histogram, rv_continuous
from scipy.interpolate import CubicSpline
from pylife.stress.timesignal import psd_df
from softsensor.frequency_methods import psd_moment

"""
Methods to compute metrics with the same argument structure as nn.GaussianNLLLoss()

For a description of the provided uncertainty metrics refer to
"Uncertainty Quantification for Traffic Forecasting: A Unified Approach"
[Qian et al. 2022 https://arxiv.org/abs/2208.05875]
"""


def rmse(mu, targets, var):
    """Root Mean Square Error (RMSE)

    Parameters
    ----------
    mu: torch.Tensor
        Predicted means
    targets: torch.Tensor
        Target values
    var: torch.Tensor
        Predicted variances

    Returns
    -------
    rmse: torch.Tensor
    """
    return nn.MSELoss()(mu, targets).sqrt()


def mae(mu, targets, var):
    """Mean Absolute Error (MAE)

    Parameters
    ----------
    mu: torch.Tensor
        Predicted means
    targets: torch.Tensor
        Target values
    var: torch.Tensor
        Predicted variances

    Returns
    -------
    mae: torch.Tensor
    """
    return nn.L1Loss()(mu, targets)


def r2(mu, targets, var):
    """R2 score

    Parameters
    ----------
    mu: torch.Tensor
        Predicted means
    targets: torch.Tensor
        Target values
    var: torch.Tensor
        Predicted variances

    Returns
    -------
    r2: float
    """
    y_true = targets.squeeze(0).numpy()
    y_pred = mu.squeeze(0).numpy()
    return r2_score(y_true, y_pred)


def pearson(mu, targets, var):
    """Pearson correlation coefficient

    Parameters
    ----------
    mu: torch.Tensor
        Predicted means
    targets: torch.Tensor
        Target values
    var: torch.Tensor
        Predicted variances

    Returns
    -------
    pearson: float
    """
    y_true = targets.numpy()
    y_pred = mu.numpy()
    return np.corrcoef(y_true, y_pred)[0, 1]


def nll(mu, targets, var):
    """Gaussian Negative Log Likelihood Loss (NLL)

    Parameters
    ----------
    mu: torch.Tensor
        Predicted means
    targets: torch.Tensor
        Target values
    var: torch.Tensor
        Predicted variances

    Returns
    -------
    nll: torch.Tensor
    """
    return nn.GaussianNLLLoss()(mu, targets, var)


def nll_statistic(mu, targets, var):
    """Gaussian Negative Log Likelihood (rather than the score of the optimization objective)
       We mainly report the NLL loss as its minimization is equivalent to NLL minimization but add this metric for comparability

    Parameters
    ----------
    mu: torch.Tensor
        Predicted means
    targets: torch.Tensor
        Target values
    var: torch.Tensor
        Predicted variances

    Returns
    -------
    nll: torch.Tensor
    """
    gaussians = torch.distributions.Normal(mu, var.sqrt())
    return -gaussians.log_prob(targets).mean()


def rmv(mu, targets, var):
    """Root Mean Variance (RMV) measures the sharpness of the uncertainty distributions

    Parameters
    ----------
    mu: torch.Tensor
        Predicted means
    targets: torch.Tensor
        Target values
    var: torch.Tensor
        Predicted variances

    Returns
    -------
    sharpness: torch.Tensor
    """
    return var.mean().sqrt()


def heteroscedasticity(mu, targets, var):
    """Heteroscedasticity of uncertainty estimate as std of std

    Parameters
    ----------
    mu: torch.Tensor
        Predicted means
    targets: torch.Tensor
        Target values
    var: torch.Tensor
        Predicted variances

    Returns
    -------
    heteroscedasticity: torch.Tensor
    """
    return torch.round(var.sqrt().std(dim=-1).mean(), decimals=6)


def crps(mu, targets, var) -> float:
    """The negatively oriented continuous ranked probability score for Gaussians.

    Computes CRPS for held out data (y_true) given predictive uncertainty with mean
    (y_pred) and standard-deviation (y_std). Each test point is given equal weight
    in the overall score over the test set.

    Negatively oriented means a smaller value is more desirable.

    Adapted from https://github.com/uncertainty-toolbox/uncertainty-toolbox

    Parameters
    ----------
    mu: torch.Tensor
        Predicted means
    targets: torch.Tensor
        Target values
    var: torch.Tensor
        Predicted variances

    Returns
    -------
    crps: float
    """
    mu = mu.numpy()
    targets = targets.numpy()
    std = var.sqrt().numpy()

    y_standardized = (targets - mu) / std
    term_1 = 1 / np.sqrt(np.pi)
    term_2 = 2 * st.norm.pdf(y_standardized, loc=0, scale=1)
    term_3 = y_standardized * (2 * st.norm.cdf(y_standardized, loc=0, scale=1) - 1)

    crps_list = -1 * std * (term_1 - term_2 - term_3)
    crps = np.mean(crps_list)
    return crps


def picp(mu, targets, var, z=1.96):
    """Prediction Interval Coverage Probability (PICP) measures the coverage of a specific PI

    Parameters
    ----------
    mu: torch.Tensor
        Predicted means
    targets: torch.Tensor
        Target values
    var: torch.Tensor
        Predicted variances
    z: float, optional
        Z-score for the specific quantile, the default is 1.96 (95% interval)

    Returns
    -------
    pi_coverage: torch.Tensor
    """

    std = torch.sqrt(var)
    lower_bound = mu - z * std
    upper_bound = mu + z * std

    pi_coverage = (
        torch.logical_and(lower_bound <= targets, targets <= upper_bound)
        .int()
        .mean(dtype=float)
    )
    return pi_coverage


def mpiw(mu, targets, var, z=1.96):
    """Mean Prediction Interval Width (MPIW) measures the width of a specific PI

    Parameters
    ----------
    mu: torch.Tensor
        Predicted means
    targets: torch.Tensor
        Target values
    var: torch.Tensor
        Predicted variances
    z: float, optional
        Z-score for the specific quantile, the default is 1.96 (95% interval)

    Returns
    -------
    pi_width: torch.Tensor
    """

    std = torch.sqrt(var)
    lower_bound = mu - z * std
    upper_bound = mu + z * std

    pi_width = (upper_bound - lower_bound).mean()
    return pi_width


def ece(mu, targets, var, quantiles=np.arange(0.05, 1.0, 0.05)):
    """
    Expected Calibration Error (ECE) measures the mean absolute calibration error of multiple PICPs

    See "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
    [Kuleshov et al. 2018 https://arxiv.org/abs/1807.00263]

    See https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa

    Parameters
    ----------
    mu: torch.Tensor
        Predicted means
    targets: torch.Tensor
        Target values
    var: torch.Tensor
        Predicted variances
    quantiles: list[x], x in (0,1)
        Quantiles to evaluate

    Returns
    -------
    pi_width: torch.Tensor
    """
    z_scores = [st.norm.ppf(1 - (1 - p) / 2) for p in quantiles]
    picp_scores = torch.tensor([picp(mu, targets, var, z) for z in z_scores])
    return nn.L1Loss()(picp_scores, torch.tensor(quantiles))


def log_area_error(psd_original, psd_targets, f):

    if f[0] == 0:
        f = f[1:]
        psd_original = psd_original[1:]
        psd_targets = psd_targets[1:]

    log_original = np.log(psd_original)
    log_targets = np.log(psd_targets)
    diff = np.abs(log_original - log_targets)

    log_f = np.log(f)
    area = np.trapezoid(diff, log_f)

    return torch.mean(torch.tensor(area))


'''
def compute_metrics(model, test_loader, metric_names=None):
    """Compute the mean metric scores of a model on the test set

    Note:
        This function reduces scores across test tracks and variables.
        Use eval_tools.comp_metrics if you want to compute scores for individual output sensors.

    Parameters
    ----------
    model: Uncertainty model
        Model to evaluate
    test_loader: list[Dataloader]
        Test dataset
    metric_names: list[string], optional
        Must be a subset of uncertainty metrics:
        ["RMSE", "MAE", "R2", "Corr", "NLL", "RMV", "CRPS", "Het", "PICP", "MPIW", "ECE"]

    Returns
    -------
    mean_scores: dict[str, float]
    """ 
    uncertainty_metrics = {
        "RMSE": rmse,
        "MAE": mae,
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
    
    if metric_names:
        uncertainty_metrics = {k: v for k, v in uncertainty_metrics.items() if k in metric_names}
    
    scores = defaultdict(list)
    for track in test_loader:
        mean, var = model.prediction(track)
        targets = torch.tensor([[data[1][0][i][0] for data in track] for i in range(mean.shape[0])])
    
        for name, metric in uncertainty_metrics.items():
            scores[name].append(metric(mean, targets, var))
            
    mean_scores = {k: np.mean(v)  for k, v in scores.items()}
    return mean_scores

def compute_eval_metrics(model, test_loader):
    """Wrapper of compute metrics that focuses on the most important metrics
    
    Parameters
    ----------
    model: Uncertainty model
        Model to evaluate
    test_loader: list[Dataloader]
        Test dataset

    Returns
    -------
    mean_scores: dict[str, float]
    """
    metric_names = ["RMSE", "NLL", "ECE", "Het"]
    return compute_metrics(model, test_loader, metric_names)

def compare_models(models, model_names, test_loader):
    """Compare multiple models on the same test set

    Parameters
    ----------
    models: list[Uncertainty model]
        Models to evaluate
    model_names: list[string]
        Identifier of the models
    test_loader: list[Dataloader]
        Test dataset

    Returns
    -------
    df: pd.Dataframe
        Scores as rows and model names as columns
    """
    ds = [compute_metrics(model, test_loader) for model in models]
    df = pd.DataFrame(ds, index=model_names).T    
    return df
'''


def quantile_ece(
    predicted_quantiles, targets, expected_quantiles=np.arange(0.05, 1.0, 0.05)
):
    """
    Expected Calibration Error (ECE) measures the mean absolute calibration error of multiple PICPs

    See "Accurate Uncertainties for Deep Learning Using Calibrated Regression"
    [Kuleshov et al. 2018 https://arxiv.org/abs/1807.00263]

    Parameters
    ----------
    predicted_quantiles: list[torch.Tensor]
        Expected to be of the form [median, lb0, ub0, lb1, ub1, ...]
    targets: torch.Tensor
        Ground truth for median
    expected_quantiles: Quantiles to evaluate
        Expected to be of the form [lb0, lb1, ..., median, ..., ub1, ub0]

    Returns
    -------
    pi_width: torch.Tensor
    """
    num_quantiles = len(expected_quantiles)

    assert num_quantiles % 2 == 1
    assert len(predicted_quantiles) // 2 == num_quantiles
    picp_scores = []
    for lb, ub in zip(predicted_quantiles[1::2], predicted_quantiles[2::2]):
        picp_scores.append(
            float(
                torch.logical_and(lb <= targets, targets <= ub).int().mean(dtype=float)
            )
        )
    picp_scores = picp_scores[::-1]

    return nn.L1Loss()(torch.tensor(picp_scores), torch.tensor(expected_quantiles))


def compute_quantile_metrics(
    model,
    test_loader,
    output_names=["x"],
    expected_quantiles=np.arange(0.05, 1.0, 0.05),
):
    """Compute metrics that are compatible with quantile models

    Parameters
    ----------
    model: QuantileNARX
        Quantile model to evaluate
    test_loader: list[Dataloader]
        Test dataset
    output_names: list[str], optional
        Output sensors to consider. The default is ["x"]
    expected_quantiles: list[float]
        Quantiles to evaluate
        Expected to be of the form [lb0, lb1, ..., median, ..., ub1, ub0]

    Returns
    -------
    mean_scores: dict[str, float]
    """
    scores = defaultdict(list)

    for track in test_loader:
        pred = model.prediction(track)
        var = None

        for i, output in enumerate(output_names):
            predidcted_quantiles = [x[i] for x in pred]
            median, lb, ub = predidcted_quantiles[:3]

            targets = torch.tensor([data[1][0][i][0] for data in track])

            for metric_name, metric in zip(
                ["RMSE", "MAE", "R2", "Corr"], [rmse, mae, r2, pearson]
            ):
                scores[f"{output}_{metric_name}"].append(metric(median, targets, var))

            scores[f"{output}_PICP"].append(
                torch.logical_and(lb <= targets, targets <= ub).int().mean(dtype=float)
            )
            scores[f"{output}_MPIW"].append((ub - lb).mean())
            scores[f"{output}_ECE"].append(
                quantile_ece(predidcted_quantiles, targets, expected_quantiles)
            )

    mean_scores = {k: np.mean(v) for k, v in scores.items()}
    return mean_scores


class wf_distribution(rv_continuous):

    def __init__(self, frequency, norm_psd_cumsum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cdf_interpolator = CubicSpline(frequency, norm_psd_cumsum)

    def _cdf(self, x):
        return self.cdf_interpolator(x)


def _calc_pdf(x, distribution):
    return distribution.pdf(x)


def _calc_cdf(x, distribution):
    return distribution.cdf(x)


def _calc_ppf(x, distribution):
    return distribution.ppf(x)


def distance_calc(inverse_cdf, p):
    w_distance = pd.DataFrame(index=inverse_cdf.columns, columns=inverse_cdf.columns)
    for col1 in inverse_cdf.columns:
        for col2 in inverse_cdf.columns:
            # if col1 != col2:
            wstf_act = np.trapezoid(
                ((inverse_cdf[col1] - inverse_cdf[col2]).abs().dropna().values.ravel())
                ** p,
                x=inverse_cdf.index,
            ) ** (1 / p)
            w_distance.loc[col1, col2] = wstf_act
    return w_distance


class WassersteinDistance:
    """
    WassersteinDistance class to calculate the Wasserstein distance and the wasserstein Fourier distance.

    Attributes:
        data (pd.DataFrame): The input data as a pandas DataFrame.
        weights_u (pd.DataFrame): The weights associated with the data as a pandas DataFrame.

    Methods:
        sort_data: Sort the data and weights in ascending order.
        weighted_hist: Calculate the weighted histogram of the data.
        pdf: Calculate the probability density function of the data.
        cdf: Calculate the cumulative density function of the data.
        inverse_cdf: Calculate the inverse cumulative density function of the data.
        wasserstein_distance_p: Calculate the p-th Wasserstein distance.
    """

    def __init__(self, data, weights=None):
        self.data = data
        if weights is None:
            self.weights = pd.DataFrame(
                np.ones_like(data.values), columns=data.columns, index=data.index
            )
        else:
            self.weights = weights
        self.columns = data.columns
        self.sorted_indices, self.sorted_data, self.sorted_weights = self.sort_data()

        nfft = min(2048, len(self.data))
        nperseg = min(2048, len(self.data))
        self.weighted_hist(nfft=nfft, nperseg=nperseg)

    def sort_data(self):
        """
        Sorts the data and weights based on the values in the data.

        This method sorts the data and weights along the first axis (rows) based on the values in the data.
        It returns the sorted indices, sorted data, and sorted weights.

        Returns:
            tuple: A tuple containing:
                - sorted_indices (numpy.ndarray): The indices that would sort the data.
                - sorted_data (pandas.DataFrame): The data sorted according to the sorted indices.
                - sorted_weights (pandas.DataFrame): The weights sorted according to the sorted indices.
        """
        sorted_indices = np.argsort(self.data.values, axis=0)
        sorted_data = pd.DataFrame(
            np.take_along_axis(self.data.values, sorted_indices, axis=0),
            columns=self.columns,
        )
        sorted_weights = pd.DataFrame(
            np.take_along_axis(self.weights.values, sorted_indices, axis=0),
            columns=self.columns,
        )
        return sorted_indices, sorted_data, sorted_weights

    def weighted_hist(self, bins=100, nfft=2048, nperseg=2048):
        """
        Compute weighted histograms for each column in the data.

        This method calculates the weighted histogram for each column in the
        dataset using the specified number of bins. The weights for each column
        are used to compute the histogram. The resulting histograms are stored
        in the `self.hist` attribute, and the corresponding probability
        distributions are stored in the `self.distribution` attribute.

        Parameters:
        bins (int): The number of bins to use for the histogram. Default is 100.

        Attributes:
        self.hist (dict): A dictionary where keys are column names and values
                          are tuples containing the histogram values and bin edges.
        self.distribution (dict): A dictionary where keys are column names and
                                  values are `rv_histogram` objects representing
                                  the probability distributions of the histograms.
        """
        self.hist = {
            col: np.histogram(
                self.data[col].dropna().to_numpy(),
                bins=bins,
                weights=self.weights[col][self.data[col].dropna().index].to_numpy(),
                density=True,
            )
            for col in self.columns
        }
        self.distribution = {col: rv_histogram(self.hist[col]) for col in self.columns}

        psd_dic = {}
        for col in self.data.columns:
            psd = psd_df(
                self.data[col].dropna().to_frame(), nfft=nfft, nperseg=nperseg
            ).squeeze()

            psd.index = psd.index.round(
                4 - int(np.floor(np.log10(abs(psd.index[1])))) - 1
            )

            psd_dic[col] = psd
        psd = pd.DataFrame(psd_dic)

        self.psd = psd[psd.index > 0]

        moment_0 = psd_moment(self.psd, n_moment=0)
        self.NPSD = (
            np.mean(self.psd.index.diff().dropna()) * self.psd / moment_0.values
        )  # scaled to frequency delta and area

        ind = self.NPSD.index.values
        self.wsfourier_distribution = {
            col: wf_distribution(
                ind, self.NPSD[col].cumsum().values, a=ind.min(), b=ind.max()
            )
            for col in self.columns
        }

    def pdf(self, n_points=1000):
        x = np.linspace(self.data.min().min(), self.data.max().max(), n_points)
        return pd.DataFrame(
            {col: _calc_pdf(x, self.distribution[col]) for col in self.columns},
            index=x,
        )

    def cdf(self, n_points=1000):
        x = np.linspace(self.data.min().min(), self.data.max().max(), n_points)
        return pd.DataFrame(
            {col: _calc_cdf(x, self.distribution[col]) for col in self.columns},
            x,
        )

    def inverse_cdf(self, n_points=1000):
        x = np.linspace(0, 1, n_points)
        return pd.DataFrame(
            {col: _calc_ppf(x, self.distribution[col]) for col in self.columns}, index=x
        )

    def wsf_pdf(self, n_points=1000):
        x = np.linspace(self.NPSD.index.min(), self.NPSD.index.max(), n_points)
        return pd.DataFrame(
            {
                col: _calc_pdf(x, self.wsfourier_distribution[col])
                for col in self.columns
            },
            # index=pd.Index(np.exp(x), name="freq"),
            index=pd.Index(x, name="freq"),
        )

    def wsf_cdf(self, n_points=1000):
        x = np.linspace(self.NPSD.index.min(), self.NPSD.index.max(), n_points)
        # x = np.log(
        #     np.logspace(
        #         np.log(self.NPSD.index.min()),
        #         np.log(self.NPSD.index.max()),
        #         base=np.e,
        #         num=n_points,
        #     )
        # )
        return pd.DataFrame(
            {
                col: _calc_cdf(x, self.wsfourier_distribution[col])
                for col in self.columns
            },
            # index=pd.Index(np.exp(x), name="freq"),
            index=pd.Index(x, name="freq"),
        )

    def wsf_inverse_cdf(self, n_points=1000):
        x = np.linspace(0, 1, n_points)
        return pd.DataFrame(
            {
                col: _calc_ppf(x, self.wsfourier_distribution[col])
                for col in self.columns
            },
            index=x,
        )  # .apply(np.exp)

    def wasserstein_distance_p(self, p=1):
        """
        Compute the Wasserstein distance (p-th order) based on the inverse cumulative distribution function (CDF).

        This method calculates the Wasserstein distance, a measure of the distance between two probability distributions,
        using the inverse CDF of the distribution. The distance is computed for a specified order `p`.

        Parameters:
            p (int, optional): The order of the Wasserstein distance. Defaults to 1.

        Returns:
            float: The computed Wasserstein distance of order `p`.
        """
        inverse_cdf = self.inverse_cdf(n_points=1000).dropna()
        w_distance = distance_calc(inverse_cdf, p)
        return w_distance

    def wasserstein_fourier_distance(self, p=2):
        """
        Compute the Wasserstein Fourier distance for the given data.

        This method calculates the Wasserstein Fourier distance by first obtaining
        the inverse cumulative distribution function (CDF) using the normalized power spectral density and then computing the distance using the specified metric.

        Parameters:
            p (int, optional): The order of the norm used in the distance calculation.
                Defaults to 2.

        Returns:
            float: The computed Wasserstein Fourier distance.
        """
        inverse_cdf = self.wsf_inverse_cdf(n_points=1000).dropna()
        wsf_distance = distance_calc(inverse_cdf, p)
        return wsf_distance
