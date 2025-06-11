# vibration-soft-sensor   <!-- omit in toc -->

A general toolbox for the development of vibrational softsensors. You can find a lot of functionality for data  preprocessing, model training and evaluation for linear and non linear methods. 
We have also adapt the general approach to a variety of signal types.

The main paper `Stabilised Auto-Regressive Neural Networks (S-Arnns) for Data Driven Prediction of Forced Nonlinear Systems` can be found on http://dx.doi.org/10.2139/ssrn.4720155.

## Table of Contents  <!-- omit in toc -->

- [Content](#content)
- [Maintainers](#maintainers)  
- [License](#license)
- [3rd Party Licenses](#3rd-party-licenses)
  

<a name="content"></a>

## Content

The vibration-soft-sensor library provides comprehensive tools for analyzing vibration data and building soft sensors. Key components include:

### Core Components
- **Model Types**
  - `autoreg_models`: Autoregressive models including ARNN
  - `recurrent_models`: Recurrent neural networks for time series prediction
  - `linear_methods`: Transfer function and other linear methods
  - `model`: Base model implementations (LSTM, CNN)
  - `arx`: Tools for ARX modeling with sliding window approaches

### Analysis Tools
- `frequency_methods`: Frequency domain analysis including FDS (Fatigue Damage Spectrum)
- `visualization`: Extensive plotting utilities for sensitivity analysis and model evaluation
- `eval_tools`: Model evaluation and error computation functionalities

### Data Handling
- `data_gen`: Simulated data generation with various excitation signals (sine, sweep, white noise)
- `meas_handling`: Measurement data preprocessing, including filtering

### Advanced Features
- `stab_scheduler`: Stability scoring and schedulers for neural networks
- `losses`: Custom loss functions including PSD-based losses
- `ensemble_wrappers`: Ensemble model implementations (Sync/AsyncEnsemble)
- `hyperparameter_optimization`: Tools for optimizing model parameters

### Sensitivity Analysis
The library offers multiple sensitivity analysis methods:
- Gradient-based methods
- SmoothGrad
- Integrated gradients
- Perturbation analysis
- Uncertainty quantification


<a name="maintainers"></a>

### Maintainers
[Tobias Westmeier](tobias.westmeier@iee.fraunhofer.de)

[Daniel Kreuter](danielchristopher.kreuter@de.bosch.com)

<a name="license"></a>

### License
`softsensor` is open-sourced under the Apache-2.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in pyLife, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

