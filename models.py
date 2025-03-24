"""
This file contains the python implementations for various models discussed in this class to prevent the need for copying code.
"""

#Imports
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



def simulate_ddm(v, a, beta, tau, dt=1e-3, scale=1.0, max_time=10.):
    """
    Simulates one realization of the diffusion process given
    a set of parameters and a step size `dt`.

    Parameters:
    -----------
    v     : float
        The drift rate (rate of information uptake)
    a     : float
        The boundary separation (decision threshold).
    beta  : float in [0, 1]
        Relative starting point (prior option preferences)
    tau   : float
        Non-decision time (additive constant)
    dt    : float, optional (default: 1e-3 = 0.001)
        The step size for the Euler algorithm.
    scale : float, optional (default: 1.0)
        The scale (sqrt(var)) of the Wiener process. Not considered
        a parameter and typically fixed to either 1.0 or 0.1.
    max_time: float, optional (default: .10)
        The maximum number of seconds before forced termination.

    Returns:
    --------
    (x, c) - a tuple of response time (y - float) and a 
        binary decision (c - int) 
    """

    # Inits (process starts at relative starting point)
    y = beta * a
    num_steps = tau
    const = scale*np.sqrt(dt)

    # Loop through process and check boundary conditions
    while (y <= a and y >= 0) and num_steps <= max_time:

        # Perform diffusion equation
        z = np.random.randn()
        y += v*dt + const*z

        # Increment step counter
        num_steps += dt

    if y >= a:
        c = 1.
    else:
        c = 0.
    return (round(num_steps, 3), c)


def simulate_diffusion_n(num_sims, v, a, beta, tau, dt=1e-3, scale=1.0, max_time=10.):
    """
    Simulates num_sims realizations of the diffusion process given
    a set of parameters and a step size `dt`.

    Parameters:
    -----------
    num_sims: float
        The number of realzations to generate of the diffusion model
    v     : float
        The drift rate (rate of information uptake) for the diffusion model.
    a     : float
        The boundary separation (decision threshold) for the diffusion model.
    beta  : float in [0, 1]
        Relative starting point (prior option preferences) for the diffusion model
    tau   :
        Non-decision time (additive constant) for the diffusion model
    dt    : float, optional (default: 1e-3 = 0.001)
        The step size for the Euler algorithm.
    scale : float, optional (default: 1.0)
        The scale (sqrt(var)) of the Wiener process. Not considered
        a parameter and typically fixed to either 1.0 or 0.1.
    max_time: float, optional (default: 10.)
        The maximum number of seconds before forced termination.

    Returns:
    --------
    np.array - An array of tuples (x, c) where x is the response time and
        c is a binary decision
    """

    data = np.zeros((num_sims, 2))
    for n in range(num_sims):
        data[n, :] = simulate_ddm(v, a, beta, tau, dt, scale, max_time)
    return data


def visualize_ddm(data, figsize=(8, 6), hist_1_label='Correct responses', hist_2_label='Incorrect responses'):
    """"
    Simulates num_sims realizations of the diffusion process given
    a set of parameters and a step size `dt`.

    Parameters:
    -----------
    data        : np.ndarray
        The data to be visualized
    figsize     : float, optional (default: (8,6)
        The size for the figure to be displayed
    hist_1_label: str, optional (default: 'Correct responses')
        The label for the first histogram to be displayed with
    hist_2_label: str, optional (default: 'Incorrect responses')
        The label for the second histogram to be displayed with
    
    Returns:
    --------
    Figure - A figure with both displayed histograms
    """
    f, ax = plt.subplots(1, 1, figsize=figsize)
    sns.histplot(data[:, 0][data[:, 1] == 1], color='maroon', alpha=0.7, ax=ax, label=hist_1_label)
    sns.histplot(data[:, 0][data[:, 1] == 0], color='gray', ax=ax, label=hist_2_label)
    sns.despine(ax=ax)
    ax.set_xlabel('Response time (s)', fontsize=18)
    ax.set_ylabel('')
    ax.legend(fontsize=18)
    ax.set_yticks([])
    return f