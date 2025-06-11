# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from softsensor.arx import ARX
from typing import List, Tuple


class ARMAX(ARX):
    """
    Represents an AutoRegressive Moving Average with eXogenous input model.

    Parameters
    ----------
    order: Tuple[int, int, int]
        Parameter for the order of the outputs and inputs in den equation (na, nb, nc)
    delta1: float
        Threshold for parameters.
    delta2: float
        Threshold for prediction error.
    num_iteration: int
        Number of iteration for the parameter estimation with the extended least squares algorithm.

    Returns
    -------
    None.

    Example
    -------
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> import softsensor.armax as armax
    >>> d = {'in_col': np.random.rand(101),
             'out_col': np.random.rand(101)}
    
    >>> df = pd.DataFrame(d)
    >>> armax = armax.ARMAX(order=(4, 4, 4), num_iteration=1)
    >>> armax.fit(data_train=[df], input_sensors=['in_col'], output_sensors=['out_col'])
    >>> print(len(armax.parameters))
    12

    """
    
    
    nc: int
    delta1: float
    delta2: float
    num_iteration: int
    prediction_error: np.ndarray = np.zeros(0)
    input_matrix: np.ndarray = np.zeros(0)

    def __init__(self, order: Tuple[int, int, int], delta1: float = 0.03,
                 delta2: float = 0.01, num_iteration: int = 10):
        super(ARMAX, self).__init__(order[:2])
        self.nc = order[2]
        self.delta1 = delta1
        self.delta2 = delta2
        self.num_iteration = num_iteration

    def fit(self, data_train: List[pd.DataFrame], input_sensors: List[str], output_sensors: List[str],
            windows: List[Tuple[int, int]] = None, verbose: bool = False) -> None:
        """
        Fit the ARMAX model

        Parameters
        ----------
        data_train: list[pd.DataFrame]
            Training data
        input_sensors: list[str]
            Name of the input sensors.
        output_sensors: list[str]
            Name of the output sensors.
        windows: Optional[list[tuple[int, int]]]
            Set the windows for fitting the ARMAX parameters.
        verbose: bool
            Verbose flag for additional information.

        Returns
        -------
        None
        """
        super(ARMAX, self).fit(data_train, input_sensors, output_sensors, windows, verbose=verbose)

        self.input_matrix = self.build_data_matrix(data_train, data_type='inputs')

        output_predicted = np.zeros_like(self.output_matrix)
        for i in range(self.num.output_sensors):
            output_predicted[:, i] = self._predict_single_output(self.input_matrix, i)
        self.prediction_error = self.output_matrix - output_predicted

        # extend parameters and regression matrix
        self.parameters = np.row_stack((self.parameters, np.zeros((self.nc, self.num.output_sensors))))
        self.regression_matrix = self._extended_regression_matrix()

        deltas = [self._fit_single_output(i) for i in range(self.num.output_sensors)]
        if verbose:
            model_name = 'ARMAX'
            print(f'\n{model_name}-Model')
            for index, sensor_name in enumerate(self.output_sensors):
                ARX.print_parameters(model_name, sensor_name, self.parameters[:, index])
                for i, (delta1, delta2) in enumerate(deltas[index]):
                    print(f'\t\t Iteration {i}: Delta1: {delta1} - Delta2: {delta2}')

    def _fit_single_output(self, index: int) -> List[Tuple[int, int]]:
        """
        Fit the ARMAX model for a single output.

        Parameters
        ----------
        index: int
            The index determining the number of the output as id.

        Returns
        -------
        outputs: list[tuple[int, int]]
            Return the termination criterion of each loop.
        """
        deltas: list[tuple[int, int]] = []
        delta1 = delta2 = np.inf
        output_data = self.output_matrix[:, index]
        prediction_error = self.prediction_error[:, index]

        indices = list(range(self.na*index, self.na * (index+1)))
        indices_inputs = list(range(self.num.outputs, self.num.outputs_inputs + self.nc))
        indices.extend(indices_inputs)

        for num_iter in range(self.num_iteration):
            self._update_noise_regression_matrix(prediction_error)
            last_parameters = np.copy(self.parameters[:, index])
            self.parameters[:, index] = np.dot(np.linalg.pinv(self.regression_matrix[:, indices]),
                                               output_data[self.relevant_indices])

            prediction_output = self._predict_single_output(self.input_matrix, index, prediction_error)
            last_prediction_error = np.copy(prediction_error)
            prediction_error = output_data - prediction_output

            delta1, delta2 = self._calculate_termination_criterion(self.parameters[:, index], last_parameters,
                                                                   prediction_error, last_prediction_error)

            deltas.append((delta1, delta2))
            if (delta1 <= self.delta1 or delta2 <= self.delta2) and num_iter > 2:
                break

        return deltas

    def prediction(self, data_test: pd.DataFrame, noise: np.ndarray = None) -> pd.DataFrame:
        """
        Predict the outputs to the input data

        Parameters
        ----------
        data_test: pd.DataFrame
            Input to the system
        noise: np.ndarray
            Noise sequence

        Returns
        -------
        output: pd.DataFrame
            Predicted output of the system
        """
        input_data = data_test[self.input_sensors].values
        output_df = pd.DataFrame(index=data_test.index)
        if noise is None:
            # noise = np.random.normal(0, 0.01, input_data.shape[0])
            noise = np.zeros(data_test.shape[0])

        for i, sensor in enumerate(self.output_sensors):
            output = self._predict_single_output(input_data, i, noise)
            output_df[sensor] = output
        return output_df

    def _extended_regression_matrix(self) -> np.ndarray:
        """
        Extended the regression matrix with zeros for the noise term.

        Parameters
        ----------
        data_train: list[pd.DataFrame]
                Training data

        Returns
        -------
        output: np.ndarray
                Returns the extended regression matrix.
        """
        shape = self.regression_matrix.shape
        extended_regression_matrix = np.zeros(shape=(shape[0], shape[1] + self.nc))
        extended_regression_matrix[:, :shape[1]] = self.regression_matrix
        return extended_regression_matrix

    def _update_noise_regression_matrix(self, noise_data: np.ndarray) -> np.ndarray:
        """
        Update the noise in the extended regression matrix.

        Parameters
        ----------
        noise_data: np.ndarray
            Noise values
        relevant_indices: list[int]
            List of relevant indices of the noise vector

        Returns
        -------
        output: np.ndarray
            Returns the updated extended regression matrix
        """
        relevant_indices = np.array(self.relevant_indices)
        for i in range(self.nc):
            self.regression_matrix[:, -self.nc + i] = noise_data[relevant_indices - i]
        return self.regression_matrix

    def _calculate_termination_criterion(self, parameters, last_parameters,
                                         prediction_error, last_prediction_error) -> Tuple[float, float]:
        """
        Calculate the termination criterions for the extended least squares algorithm.

        Parameters
        ----------
        parameters: np.ndarray
            Current parameters
        last_parameters: np.ndarray
            Parameters of the iteration before
        prediction_error: np.ndarray
            Current prediction error
        last_prediction_error: np.ndarray
            Prediction error of the iteration before

        Returns
        -------
        output: tuple[float, float]
            Returns the values for the termination criterions
        """
        try:
            delta1 = np.mean(np.abs(parameters - last_parameters) / np.abs(parameters))
            diff_error = prediction_error[self.relevant_indices] - last_prediction_error[self.relevant_indices]
            delta2 = np.mean(np.power(diff_error, 2))
        except RuntimeWarning:
            delta1 = delta2 = np.inf

        return delta1, delta2
