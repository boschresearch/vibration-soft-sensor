# -*- coding: utf-8
"""
Created on Tue Mar  1 11:01:26 2022

@author: Tobias Westmeier CR/AMP4

The described scripts are used for the synthetic generation of time data for
testing models. The class: 'get_academic_data' is used for the numerical
integration of the described differential equations.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import chirp, butter, filtfilt
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

def get_academic_data(time, Model, params, F=None, x0=None, rtol=1e-3):
    """
    Define Dataframe with the response of academic examples equation to given
    force

    Parameters
    ----------
    time : array
        A sequence of time points for which to solve for y. The initial value
        point should be the first element of this sequence. This sequence must
        be monotonically increasing or monotonically decreasing; repeated
        values are allowed.
    Model : str
        either 'Duffing' 'Duffing_fp' or 'vd_pol' to define academic
        model.
    params : dict
        Define Parameters for the models.
        if Model = 'Duffing' dict must contain c_nlin and D.
        if Model = 'vd_Pol' dict must contain epsilon.
    F : Force, optional
        Force to apply to the ems. Must be callable by F.comp and F.comp_dt and
        must include self.F and self.dF with the same time steps as time
        The default is None.
    x0 : array, optional
        initial conditions. The default is [0, 0].
    rtol : float, optional
        relative tolerance for the numerical integration

    Returns
    -------
    df : pd.DataFrame
        Dataframe of response with columns ['F(t)', 'x', 'v'].
        
    Examples
    --------
    >>> import softsensor.data_gen as dg
    >>> import numpy as np
    >>> time = np.linspace(0, 100, 1000)
    >>> params = {'D': 0.05, 'c_nlin': 0.1}
    >>> F = dg.Zero(time)
    >>> df = dg.get_academic_data(time, Model='Duffing', F=F, params=params)
    >>> print(df.shape)
    (1000, 3)

    """

    if F is None:
        F = Zero(time)

    if Model == 'Duffing':
        sol = solve_ivp(duffing, t_span=[time[0], time[-1]], y0=x0 if x0 is not None else [0, 0],
                        t_eval=time, rtol=rtol, args=(params['D'], F, params['c_nlin']))
    elif Model == 'Duffing_fp':
        sol = solve_ivp(duffing_fp, t_span=[time[0], time[-1]], y0=x0 if x0 is not None else [0, 0],
                        t_eval=time, rtol=rtol, args=(params['D'], params['c_nlin'], F))
    elif Model == 'vd_Pol':
        if 'offset' not in params:
            params['offset'] = [0, 0]
        sol = solve_ivp(vd_Pol, t_span=[time[0], time[-1]], y0=x0 if x0 is not None else [0, 0], t_eval=time,
                        rtol=rtol, args=(params['epsilon'], params['offset'], F))
    elif Model == 'Pendulum':
        sol = solve_ivp(pendulum, t_span=[time[0], time[-1]], y0=x0 if x0 is not None else [0, 0],
                        t_eval=time, rtol=rtol, args=(params['D'], F))
    elif Model == 'Two_Mass_System':
        sol = solve_ivp(two_mass_oscillator, t_span=[time[0], time[-1]], y0=x0 if x0 is not None else [0, 0, 0, 0],
                        t_eval=time, rtol=rtol, args=(params['D'], 
                                                      params['mue'],
                                                      params['kappa'],
                                                      params['delta'], F))
    else:
        raise ValueError("Invalid model given: Implemented are" +
                         "['Duffing, Duffing_fp, 'vd_Pol', 'Pendulum'']")

    if Model == 'Duffing_fp':
        df = {'z(t)':   F.F,
              'dz(t)':   F.dF,
              'x':      sol.y[0],
              'v':      sol.y[1]
              }
    elif Model == 'Two_Mass_System':
        df = {'F(t)':   F.F,
              'x1':      sol.y[0],
              'v1':      sol.y[1],
              'x2':      sol.y[2],
              'v2':      sol.y[3]
              }
    else:
        df = {'F(t)':   F.F,
              'x':      sol.y[0],
              'v':      sol.y[1]
              }
    try:
        df['Frequency'] = F.freq
    except AttributeError:
        pass

    df = pd.DataFrame(df).set_index(pd.Index(time, name="time"))
    return df

def duffing(t, y, D, F, c_nlin=0):
    r"""
    Duffing oscillator with external forcing as described in
    https://en.wikipedia.org/wiki/Duffing_equation

    .. math:: dz1 = z2
    .. math:: dz2 = - 2 \cdot D \cdot z2 - z1 - c_{nlin} \cdot z1^{3} + F.comp(t)


    Parameters
    ----------
    y : array of size 2
        initial conditions. System state (z1, z2)
    t : float()
        time step at which the force F is computed.
    D : float()
        damping of the system.
    F : Force class
        Abstract class for the force excitation at the mass.
    c_nlin : float(), optional
        indicating the nonlinearity of the system. The default is 0.

    Returns
    -------
    (dz1, dz2) : Derivatives of the duffing equation.

    Examples
    --------
    >>> import softsensor.data_gen as dg
    >>> import numpy as np
    >>> time = np.linspace(0, 100, 1000)
    >>> F = dg.Zero(time)
    >>> (dz1, dz2) = dg.duffing(t=0, y=(0, 0), D=0.05, F=F, c_nlin=.2)
    >>> print((dz1, dz2))
    (0, 0.0)

    """
    z1, z2 = y
    dz1dt = z2
    dz2dt = - 2 * D * z2 - z1 - c_nlin * z1**3 + F.comp(t)
    return(dz1dt, dz2dt)


def pendulum(t, y, D, F):
    r"""
    Pendulum equation with external forcing and damping
    https://en.wikipedia.org/wiki/Pendulum_(mechanics)

    .. math:: dz1 = z2
    .. math:: dz2 = - 2 \cdot D \cdot z2 - sine(z1) + F.comp(t)


    Parameters
    ----------
    y : array of size 2
        initial conditions. System state (z1, z2)
    t : float()
        time step at which the force F is computed.
    D : float()
        damping of the system.
    F : Force class
        Abstract class for the force excitation at the mass.

    Returns
    -------
    (dz1, dz2) : Derivatives of the pendulum equation.

    Examples
    --------
    >>> import softsensor.data_gen as dg
    >>> import numpy as np
    >>> time = np.linspace(0, 100, 1000)
    >>> F = dg.Zero(time)
    >>> (dz1, dz2) = dg.pendulum(t=0, y=(0, 0), D=0.05, F=F, c_nlin=.2)
    >>> print((dz1, dz2))
    (0, 0.0)

    """
    z1, z2 = y
    dz1dt = z2
    dz2dt = - 2 * D * z2 - np.sin(z1) + F.comp(t)
    return(dz1dt, dz2dt)


def duffing_fp(t, y, D, c_nlin, z):
    """
    Duffing oscillator with excitation at the base point as described in
    https://en.wikipedia.org/wiki/Duffing_equation
    
    .. math:: dz1 = z2
    .. math:: dz2 = 2 \cdot D \cdot (z.comp_dt(t) - z2) - z1 + c_{nlin} \cdot (z.comp(t) - z1)^3

    Parameters
    ----------
    y : array of size 2
        initial conditions. System state (z1, z2)
    t : float()
        time step at which the Force is computed.
    D : float()
        damping of the system.
    c_nlin : float()
        indicating the nonlinearity of the system.
    F : Force class
        Abstract class for the force excitation at the mass.

    Returns
    -------
    (dz1, dz2) : Derivatives of the duffing equation.

    Examples
    --------
    >>> import softsensor.data_gen as dg
    >>> import numpy as np
    >>> time = np.linspace(0, 10, 100)
    >>> F = dg.Zero(time)
    >>> (dz1, dz2) = dg.duffing_fp(t=0, y=(0, 0), D=0.05, c_nlin=.2, z=F)
    >>> print((dz1, dz2))
    (0, 0.0)

    """
    z1, z2 = y
    dz1dt = z2
    dz2dt = 2 * D * (z.comp_dt(t) - z2) - z1 + c_nlin * (z.comp(t) - z1)**3
    return(dz1dt, dz2dt)


def two_mass_oscillator(t, y, D, mue, kappa, delta, F):
    r"""
    Two Mass oscillator with damping and Forcing on the first mass
    Coverning Equation:
    
    .. math::

        \begin{bmatrix} 1 & 0 \\ 0 & \mu \end{bmatrix}
        \begin{bmatrix} q_1^{''}\\ q_2^{''} \end{bmatrix}
        + 
        2D
        \begin{bmatrix} 1 + \delta & -\delta \\ -\delta & \delta \end{bmatrix}
        \begin{bmatrix} q_1^{'}\\ q_2^{'} \end{bmatrix}
        + 
        \begin{bmatrix} 1 + \kappa & -\kappa \\ -\kappa & \kappa \end{bmatrix}
        \begin{bmatrix} q_1\\ q_2 \end{bmatrix}
        =
        \begin{bmatrix} F.comp(t)\\ 0 \end{bmatrix}
    
    with
        
    .. math::
        
        z_{11} = q_{1}, z_{12} = q_{1}^{'}
        
        z_{21} = q_{2}, z_{22} = q_{2}^{'}
    
    
    Parameters
    ----------
    y : matrix of size 4
        initial conditions. System state [z11, z12, z21, z22]
    t : float()
        time step at which the Force is computed.
    D : float()
        damping of the system.
    mue : float()
        relative mass of the system m2/m1
    kappa : float()
        relative restoring force of the system c2/c1
    delta : float()
        relative damping of the system d2/d1
    F : Force class
        Abstract class for the force excitation at the mass.
        
    Returns
    -------
    ([dz11, dz12], [dz21, dz22]) : Derivatives of the duffing equation.

    Examples
    --------
    >>> import softsensor.data_gen as dg
    >>> import numpy as np
    >>> time = np.linspace(0, 10, 100)
    >>> F = dg.Zero(time)
    >>> (dz1, dz2) = dg.duffing_fp(t=0, y=(0, 0), D=0.05, c_nlin=.2, z=F)
    >>> print((dz1, dz2))
    (0, 0.0)

    """
    z11, z12 = y[0], y[1]
    z21, z22 = y[2], y[3]
    
    dz11 = z12
    dz21 = z22
    
    dz12 = -2*D*((1+delta)*z12 - delta*z22) - (1+kappa)*z11 + kappa*z21 + F.comp(t)
    dz22 = 1/mue * (2*D*delta*(z12 - z22) + kappa*(z11 - z21))
    
    return (dz11, dz12, dz21, dz22)

def vd_Pol(t, y, epsilon, offset, F):
    """
    van der Pol oscillation as described in
    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator

    .. math:: dz1 = z2
    .. math:: dz2 = epsilon \cdot (1 - z1^2) \cdot z2 - z1 + F.comp(t)

    Parameters
    ----------
    y : array of size 2
        initial conditions.
    t : float()
        time step at which the Force is computed.
    epsilon : float()
        indicating the nonlinearity and the strength of the damping.
    F : Force class
        forcing term of van der Pol equation.

    Returns
    -------
    (dz1, dz2) : Derivatives of the duffing equation.


    Examples
    --------
    >>> import softsensor.data_gen as dg
    >>> import numpy as np
    >>> time = np.linspace(0, 10, 100)
    >>> F = dg.Zero(time)
    >>> (dz1, dz2) = dg.vd_Pol(t=0, y=(0, 0), epsilon=1, F=F)
    >>> print((dz1, dz2))
    (0, 0)
    
    """
    
    z1, z2 = y
    z1 = z1 + offset[0]
    z2 = z2 + offset[1]
    dz1dt = z2
    dz2dt = epsilon * (1 - z1**2) * z2 - z1 + F.comp(t)
    return(dz1dt, dz2dt)


class sine():
    """
    Generate Sine Wave
    
    .. math:: gamma \cdot sin(w_0 \cdot time)

    Parameters
    ----------
    time : array
        time array at which the sines are computed.
    gamma : float(), optional
        amplification sine wave. The default is 1.
    w0 : float(), optional
        frequency of sine wave. The default is 1.

    Returns
    -------
    None.

    """
    def __init__(self, time, gamma=1, w0=1):

        self.gamma = gamma
        self.w0 = w0
        self.time = time
        self.F = self.gamma * np.sin(self.w0*self.time)
        self.dF = self.gamma * self.w0 * np.cos(self.w0*self.time)

    def comp(self, t):
        """
        Compute sine at specific timestep
        
        .. math:: gamma \cdot sin(w_0 \cdot t)

        Parameters
        ----------
        t : float()
            time step to compute sine at.

        Returns
        -------
        w : float()
            sine at the specific time step.

        """
        return self.gamma * np.sin(self.w0*t)

    def comp_dt(self, t):
        """
        Compute derivative of the sine wave at specific timestep
        
        .. math:: gamma \cdot w_0 \cdot cos(w_0 \cdot t)

        Parameters
        ----------
        t : float()
            time step to compute derivative at.

        Returns
        -------
        w : float()
            derivative at the time step.

        """
        return self.gamma * self.w0 * np.cos(self.w0*t)


class sine_2():
    """
    Generate the addition of two sine oscillation
    
    .. math:: gamma1 \cdot sin(w_{01} \cdot time) + gamma2 \cdot sin(w_{02} \cdot time)
    
    Parameters
    ----------
    time : array
        time array at which the sines are computed.
    gamma1 : float(), optional
        amplification of first sine wave. The default is 1.
    w01 : float(), optional
        frequency of first sine wave. The default is 1.
    gamma2 : float(), optional
        amplification of second sine wave. The default is 0.5.
    w02 : float(), optional
        frequency of second sine wave. The default is 26.4.

    Returns
    -------
    None.

    """
    def __init__(self, time, gamma1=1, w01=1, gamma2=0.5, w02=26.4):

        self.gamma1 = gamma1
        self.w01 = w01
        self.gamma2 = gamma2
        self.w02 = w02
        self.time = time
        F1 = self.gamma1 * np.sin(self.w01*self.time)
        F2 = self.gamma2 * np.sin(self.w02*self.time)

        dF1 = self.gamma1 * self.w01 * np.cos(self.w01*self.time)
        dF2 = self.gamma2 * self.w02 * np.cos(self.w02*self.time)

        self.F = (F1 + F2)
        self.dF = (dF1 + dF2)

    def comp(self, t):
        """
        Compute sines at specific timestep
        
        .. math:: gamma1 \cdot sin(w_{01} \cdot t) + gamma2 \cdot sin(w_{02} \cdot t)
        
        Parameters
        ----------
        t : float()
            time step to compute sines at.

        Returns
        -------
        w : float()
            sines at the time step.

        """
        return (self.gamma1 * np.sin(self.w01*t) +
                self.gamma2 * np.sin(self.w02*t))

    def comp_dt(self, t):
        """
        Compute derivative of sine waves at specific timestep
        
        .. math:: gamma1 \cdot w_{01} \cdot cos(w_{01} \cdot t) + gamma2 \cdot w_{02} \cdot cos(w_{02} \cdot t)

        Parameters
        ----------
        t : float()
            time step to compute derivative at.

        Returns
        -------
        w : float()
            derivative at the time step.

        """
        return (self.gamma1 * self.w01 * np.cos(self.w01*t) +
                self.gamma2 * self.w02 * np.cos(self.w02*t))


class sweep():
    """
    Define sweep signal and derivative based on the scipy implementation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html

    Parameters
    ----------
    time : array
        time array at which the sweep is computed.
    f0 : float()
        start frequency.
    f1 : float()
        stop frequency.
    t1 : float()
        time at which end frequency is met.
    method : str, optional
        Kind of frequency sweep. The default is 'linear'. options are:
        [‘linear’, ‘quadratic’, ‘logarithmic’, ‘hyperbolic’]
    direction : str, optional
        either 'up' or 'down'. The default is 'up'.
    f : float(), optional
        Amplification factor of the sweep. The default is 1.

    Returns
    -------
    None.

    """
    def __init__(self, time, f0, f1, t1, method='linear', direction='up', f=1):
        self.f0 = f0
        self.f1 = f1
        self.t1 = t1
        self.method = method
        self.direction = direction
        self.f = f

        self.time = time

        if self.direction == 'up':
            self.F = self.f*chirp(self.time, f0=self.f0, f1=self.f1,
                                  t1=self.t1, method=self.method)
            self.freq = np.linspace(self.f0, self.f1,
                                    self.time.shape[0])*2*np.pi
        elif self.direction == 'down':
            self.F = self.f*chirp(self.time, f0=self.f1, f1=self.f0,
                                  t1=self.t1, method=self.method)
            self.freq = np.linspace(self.f1, self.f0,
                                    self.time.shape[0])*2*np.pi

        self.derivative = IUS(self.time, self.F, k=3).derivative(n=1)
        self.dF = self.derivative(time)

    def comp(self, t):
        """
        Compute sweep at specific timestep

        Parameters
        ----------
        t : float()
            time step to compute sweep at.

        Returns
        -------
        w : float()
            sweep at the time step.

        """
        if self.direction == 'up':
            w = self.f * chirp(t, f0=self.f0, f1=self.f1, t1=self.t1,
                               method=self.method)
        elif self.direction == 'down':
            w = self.f*chirp(t, f0=self.f1, f1=self.f0, t1=self.t1,
                             method=self.method)
        return w

    def comp_dt(self, t):
        """
        Compute derivative of the sweep at specific timestep

        Parameters
        ----------
        t : float()
            time step to compute derivative at.

        Returns
        -------
        w : float()
            derivative at the time step.

        """
        return self.derivative(t)

class white_noise():
    """
    Generate White Noise signal for a defined time with numpy.random.randn
    method: 
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html

    Parameters
    ----------
    time : array
        time array at which the white noise is computed.
    f : float(), optional
        Amplification factor of the white noise. The default is 1.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
        The default is None, i.e. no seed is provided.

    Returns
    -------
    None.

    """
    def __init__(self, time, f=1, seed=None):
        temp_t = np.append(time, time[-1] + time[-1] - time[-2])
        if seed:
            np.random.seed(seed)
        temp_F = f * np.random.randn(time.shape[0] + 1)

        b, a = butter(5, [0.01, 0.99], btype='bandpass')
        temp_dF = filtfilt(b, a, temp_F)

        self.interpol_func = IUS(temp_t, temp_F, k=1)
        self.interpol_func_dt = IUS(temp_t, temp_dF, k=3).derivative(n=1)
        self.time = time

        self.F = temp_F[0:-1]
        self.dF = self.interpol_func_dt(self.time)

    def comp(self, t):
        """
        Compute white noise at specific timestep

        Parameters
        ----------
        t : float()
            time step to compute white noise at.

        Returns
        -------
        w : float()
            white noise at the time step.

        """
        return self.interpol_func(t)

    def comp_dt(self, t):
        """
        Compute derivative of white noise at specific timestep. Derivation is
        realised via interpolatable splines

        Parameters
        ----------
        t : float()
            time step to compute derivative at.

        Returns
        -------
        w : float()
            derivative at the time step.

        """
        return self.interpol_func_dt(t)

class Zero():
    """
    Generate an array of Zeros as input Force

    Parameters
    ----------
    time : array
        an array of the same length as the array of zeros

    Returns
    -------
    None.

    """
    def __init__(self, time):
        self.time = time
        self.F = np.zeros(self.time.shape)
        self.dF = np.zeros(self.time.shape)

    def comp(self, t):
        """
        Give back zero

        Parameters
        ----------
        t : float()
            time step.

        Returns
        -------
        int
            zero.

        """
        return 0

    def comp_dt(self, t):
        """
        Give back zero

        Parameters
        ----------
        t : float()
            time step.

        Returns
        -------
        int
            zero.

        """
        return 0
