# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List, Tuple
import warnings
from typing import Optional
import numpy as np
import pandas as pd
from scipy.signal import tf2zpk


@dataclass
class _Window:
    """Represents the sliding window"""
    start: int
    end: int
    length: int = field(init=False)
    window: range = field(init=False)

    def __post_init__(self):
        self.window = range(self.start, self.end)
        self.length = len(self.window)

    def slide(self, step: int) -> range:
        """
        Slide the window to the right for n steps.

        Parameters
        ----------
        step: int
            Integer for determining the sliding step.

        Returns
        -------
        output: range
            Returns the shifted window.
        """
        return range(self.start + step, self.end + step)

    def reverse(self) -> range:
        """
        Reverse the window.

        Parameters
        ----------

        Returns
        -------
        output: range
            Returns the reversed window.
        """
        return self.window[::-1]

    def slide_rev(self, step: int) -> range:
        """
        Slide and reverse the window for n steps.

        Parameters
        ----------
        step: int
            Integer for determining the sliding step.

        Returns
        -------
        output: range
            Returns the shifted and reversed window.
        """
        return self.slide(step)[::-1]

    def slide_periodic(self, step: int, times: int) -> List[int]:
        """
        Slide the window to the right for n steps and m times.

        Parameters
        ----------
        step: int
            Integer for determining the sliding step.
        times: int
            Integer for determining the repetitions.

        Returns
        -------
        output: list[int]
            Returns the shifted and periodic window.
        """
        indices = []
        for i in range(times):
            indices.extend(list(self.slide(i * step)))
        return indices

    def slide_periodic_rev(self, step: int, times: int) -> List[int]:
        """
        Slide the window to the right starting in the back for n steps and m times.

        Parameters
        ----------
        step: int
            Integer for determining the sliding step.
        times: int
            Integer for determining the repetitions.

        Returns
        -------
        output: list[int]
            Returns the shifted and periodic window.
        """
        indices = []
        for i in range(times)[::-1]:
            indices.extend(list(self.slide(i * step)))
        return indices

    def slide_rev_periodic(self, step: int, times: int) -> List[int]:
        """
        Slide the window to the right for n steps and m times and reverse the window range.

        Parameters
        ----------
        step: int
            Integer for determining the sliding step.
        times: int
            Integer for determining the repetitions.

        Returns
        -------
        output: list[int]
            Returns the shifted, reversed and periodic window.
        """
        indices = []
        for i in range(times):
            indices.extend(list(self.slide_rev(i * step)))
        return indices

    def slide_rev_periodic_rev(self, step: int, times: int) -> List[int]:
        """
        Slide the window to the right starting in the back for n steps and m times and reverse the window range.

        Parameters
        ----------
        step: int
            Integer for determining the sliding step.
        times: int
            Integer for determining the repetitions.

        Returns
        -------
        output: list[int]
            Returns the shifted, reversed and periodic window.
        """
        indices = []
        for i in range(times)[::-1]:
            indices.extend(list(self.slide_rev(i * step)))
        return indices


@dataclass
class _StaticNumbers:
    """Represents all static numbers and length in an ARX model."""
    na: int
    nb: int
    input_sensors: int
    output_sensors: int
    data_frames: int
    data_lengths: List[int]
    inputs: int = field(init=False)
    outputs: int = field(init=False)
    output_inputs: int = field(init=False)
    outputs_inputs: int = field(init=False)

    def __post_init__(self):
        self.inputs = self.nb * self.input_sensors
        self.outputs = self.na * self.output_sensors
        self.output_inputs = self.na + self.inputs
        self.outputs_inputs = self.outputs + self.inputs


class ARX:
    """
    Represents an AutoRegressive with eXogenous input model.

    Parameters
    ----------
    order: Tuple[int, int]
        Parameter for the order of the outputs and inputs in den equation (na, nb)#

    Returns
    -------
    None.

    Example
    -------
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> import softsensor.arx as arx
    >>> d = {'in_col': np.linspace(0, 100, 101),
             'out_col': np.linspace(100, 0, 101)}
    
    >>> df = pd.DataFrame(d)
    >>> arx = arx.ARX(order=(2, 2))
    >>> arx.fit(data_train=[df], input_sensors=['in_col'], output_sensors=['out_col'])
    >>> print(len(arx.parameters))
    4

    """
    na: int
    nb: int
    input_sensors: List[str] = []
    output_sensors: List[str] = []
    windows: List[_Window] = None

    num: _StaticNumbers = None
    parameters: np.ndarray = np.zeros(0)
    regression_matrix: np.ndarray = np.zeros(0)
    output_matrix: np.ndarray = np.zeros(0)
    relevant_indices: List[int] = []

    def __init__(self, order: Tuple[int, int]):
        self.na, self.nb = order
        self.Type = 'ARX'

    def fit(self, data_train: List[pd.DataFrame], input_sensors: List[str], output_sensors: List[str],
            windows: List[Tuple[int, int]] = None, verbose: bool = False) -> None:
        """
        Fit the ARX model

        Parameters
        ----------
        data_train: list[pd.DataFrame]
            Training data
        input_sensors: list[str]
            Name of the input sensors.
        output_sensors: list[str]
            Name of the output sensors.
        windows: Optional[list[tuple[int, int]]]
            Set the windows for fitting the ARX parameters.
        verbose: bool
            Verbose flag for additional information.

        Returns
        -------
        None
        """
        len_data_frames = len(data_train)
        data_lengths = [data_train[i].shape[0] for i in range(len_data_frames)]
        self._initialize(input_sensors, output_sensors, len_data_frames, data_lengths, windows)

        self.regression_matrix = self._build_regression_matrix(data_train)
        self.output_matrix = self.build_data_matrix(data_train, data_type='outputs')

        self.relevant_indices = []
        for i, window in enumerate(self.windows):
            data_length = self.num.data_lengths[i-1] if i > 0 else 0
            self.relevant_indices.extend(list(window.slide_rev(data_length)))
        relevant_outputs = self.output_matrix[self.relevant_indices, :]

        indices_inputs = list(range(self.num.outputs, self.num.outputs_inputs))
        for i in range(self.num.output_sensors):
            indices = list(range(self.na*i, self.na * (i+1)))
            indices.extend(indices_inputs)
            self.parameters[:, i] = np.dot(np.linalg.pinv(self.regression_matrix[:, indices]),
                                           relevant_outputs[:, i])

        if verbose:
            model_name = 'ARX'
            print(f'\n{model_name}-Model')
            for index, sensor_name in enumerate(self.output_sensors):
                ARX.print_parameters(model_name, sensor_name, self.parameters[:, index])

            stable, poles = self._is_stable()
            if not stable:
                warnings.warn(f'The {type(self).__name__} model is unstable with poles at {poles}!', UserWarning)

    def prediction(self, data_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the outputs to the input data

        Parameters
        ----------
        data_test: pd.DataFrame
                Input to the system

        Returns
        -------
        output: pd.DataFrame
                Predicted output of the system
        """
        input_data = data_test[self.input_sensors].values
        output_df = pd.DataFrame(index=data_test.index)

        for i, sensor in enumerate(self.output_sensors):
            output = self._predict_single_output(input_data, index=i)
            output_df[sensor] = output
        return output_df

    def _predict_single_output(self, input_data: np.ndarray, index: int, noise: np.ndarray = None) -> np.ndarray:
        """
        Predict the outputs to the input data

        Parameters
        ----------
        input_data: np.ndarray
                Input to the system
        index: int
            Index of the output
        noise: np.ndarray
            Noise sequence

        Returns
        -------
        output: np.ndarray
                Predicted output of the system
        """
        output = np.zeros(input_data.shape[0])
        latest_regression_vec = np.zeros_like(self.parameters[:, index].T)

        for i, input_values in enumerate(input_data):
            latest_regression_vec[self.na:self.num.output_inputs:self.nb] = input_values

            output[i] = np.dot(latest_regression_vec, self.parameters[:, index])
            latest_regression_vec = np.roll(latest_regression_vec, 1)
            latest_regression_vec[0] = output[i]
            if noise is not None:
                latest_regression_vec[self.num.output_inputs] = noise[i]

        return output

    def _build_regression_matrix(self, data_train: List[pd.DataFrame]) -> np.ndarray:
        """
        Build the regression matrix.

        Parameters
        ----------
        data_train: list[pd.DataFrame]
                Training data

        Returns
        -------
        output: np.ndarray
                Returns the regression matrix.
        """
        lengths = [0] + [window.length for window in self.windows]

        regression_matrix = np.zeros((np.sum(lengths), self.num.outputs_inputs))
        for index, data in enumerate(data_train):
            inputs = data[self.input_sensors].values
            outputs = data[self.output_sensors].values

            row_index = lengths[index]
            regression_matrix = self._update_regression_matrix(regression_matrix, row_index,
                                                              self.windows[index], inputs, outputs)

        return regression_matrix

    def _update_regression_matrix(self, regression_matrix: np.ndarray, row_index: int, window: _Window,
                                 input_data: Optional[np.ndarray] = None,
                                 output_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Update the regression matrix.

        Parameters
        ----------
        regression_matrix: np.ndarray
            Regression matrix to modify
        row_index: int
            Start position in the regression matrix
        window: Window
            Relevant window
        input_data: Optional[np.ndarray]
            Input to the system
        output_data: Optional[np.ndarray]
            Output from the system

        Returns
        -------
        output: np.ndarray
            Updated regression matrix
        """
        rows = range(row_index, row_index + window.length)
        temporary_matrix = regression_matrix[rows, :]

        if output_data is not None:
            for i in range(self.na):
                columns = range(i, self.num.outputs, self.na)
                temporary_matrix[:, columns] = output_data[window.slide_rev(-i - 1), :]

        if input_data is not None:
            for i in range(self.nb):
                columns = range(self.num.outputs + i, self.num.outputs_inputs, self.nb)
                temporary_matrix[:, columns] = input_data[window.slide_rev(-i), :]

        regression_matrix[rows, :] = temporary_matrix

        return regression_matrix

    def build_data_matrix(self, data_train: List[pd.DataFrame], data_type: str) -> np.ndarray:
        """
        Build the output matrix or input matrix with all data points from list of DataFrames.

        Parameters
        ----------
        data_train: list[pd.DataFrame]
                Training data
        data_type: str
                Selector to create the outputs or inputs

        Returns
        -------
        output: tuple[np.ndarray, np.ndarray]
                Output matrix
        """
        create_inputs = data_type == 'inputs'
        num_columns = self.num.input_sensors if create_inputs else self.num.output_sensors
        sensor_names = self.input_sensors if create_inputs else self.output_sensors

        data_matrix = np.zeros((np.sum(self.num.data_lengths), num_columns))
        start = 0
        for index, data in enumerate(data_train):
            data_matrix[start:start + self.num.data_lengths[index], :] = data[sensor_names].values
            start += self.num.data_lengths[index]
        return data_matrix

    def _initialize(self, input_sensors: List[str], output_sensors: List[str], len_data_frames: int,
                    data_lengths: List[int], windows: List[Tuple[int, int]] = None) -> None:
        """
        Initialize the class member variables.

        Parameters
        ----------
        input_sensors: list[str]
                Name of the input sensors.
        output_sensors: list[str]
                Name of the output sensors.
        len_data_frames: int
            Length of the different datasets.
        data_lengths: int
                Lengths of the data.
        windows: Optional[list[tuple[int, int]]]
                Set the windows for fitting the ARX parameters.

        Returns
        -------
        None
        """
        self.input_sensors = input_sensors
        self.output_sensors = output_sensors
        self.num = _StaticNumbers(self.na, self.nb, len(self.input_sensors),
                                 len(self.output_sensors), len_data_frames, data_lengths)
        shape = (self.num.output_inputs, self.num.output_sensors)
        self.parameters = np.ones(shape=shape)

        if windows is None or len(windows) != len_data_frames:
            windows = [(0, length) for length in data_lengths]

        self.windows = []
        for window, data_len in zip(windows, data_lengths):
            if (window[1] - window[0]) > data_len:
                window = (0, data_len)
            length = min(data_len - self.na, data_len - self.nb, window[1] - window[0])
            self.windows.append(_Window(window[1] - length, window[1]))

    def _is_stable(self) -> Tuple[bool, np.ndarray]:
        """
        Determine the stability by calculating the poles

        Parameters
        ----------

        Returns
        -------
        output: tuple[bool, np.ndarray]
                Returns the flag for the stability and all poles
        """
        all_poles = np.zeros(0)

        for i in range(self.num.output_sensors):
            parameters = self.parameters[:, i]
            denominator_polynomial = parameters[:self.na]
            denominator_polynomial = np.insert(denominator_polynomial, 0, 1)

            _, poles, _ = tf2zpk([1], denominator_polynomial)
            if poles.dtype == complex:
                poles = np.abs(poles)

            all_poles = np.append(all_poles, poles)

        all_poles = np.unique(all_poles)
        unstable_poles = all_poles[(all_poles <= -1) | (all_poles >= 1)]
        return len(unstable_poles) == 0, unstable_poles

    @staticmethod
    def print_parameters(model_name: str, sensor_name: str, parameters: np.ndarray):
        """
        Prints the parameters formatted according to the sensor name.

        Parameters
        ----------
        model_name: str
            The model name as string.
        sensor_name: str
            The sensor name as string.
        parameters: np.ndarray
            The parameters according to the sensor name.

        Returns
        -------
        """
        print(f'Output sensor: {sensor_name}')
        filtered_parameters = str(parameters).replace("\n", "")
        print(f'\t {model_name}-Parameters: {filtered_parameters}')