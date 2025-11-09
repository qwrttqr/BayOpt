from typing import Callable
import numpy as np


class BayesianOptimizer:
    """
    Class for performing Bayesian optimization.

    Args:
        surrogate_function (Callable): The mean function.
        acquisition_function (Callable): Kernel function used to compute correlations.
        target_function (Callable): Kernel function used to compute correlations.
        max_iter (int): Optional. Max iteration count for optimization.
        stop_epsilon (float): Optional. Difference between y_i and y_i+1 iteration of evaluating target. If
        difference is smaller than this quantity - optimization is stopped.
    """
    def __init__(self, surrogate_function: Callable, acquisition_function: Callable, target_function: Callable,
                 max_iter=100, stop_epsilon=0.01):
        self.mean_function = surrogate_function
        self.corr_func = acquisition_function
        self.target_function = target_function
        self.max_iter = max_iter
        self.stop_epsilon = 0.01


    def optimize(self, x: np.ndarray, y: np.ndarray):
        """
        Function for performing optimization algorithm.
        Args:
            x (numpy.ndarray): Values on which target function already been evaluated.
            y (numpy.ndarray): Values of target function gotten from evaluating target function on x values.

        Returns:

        """

def priormean(xin, **kwargs):
    c2 = kwargs['c2']
    return c2 * xin ** 2


def corr_func(a: np.ndarray, b: np.ndarray, sigmaf2=1.0, ell=1.0):
    diff = a[:, None] - b[None, :]
    return sigmaf2 * np.exp(-diff ** 2 / (2 * ell ** 2))
