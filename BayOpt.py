from typing import Callable
import numpy as np


class BayesianOptimizer:
    """
    Class for performing Bayesian optimization.

    Args:
        surrogate_kernel: Kernel of optimization. Should be instantiable and should implement fit predict methods
        target_function (Callable): Kernel function used to compute correlations.
        max_iter (int): Optional. Max iteration count for optimization.
        stop_epsilon (float): Optional. Difference between y_i and y_i+1 iteration of evaluating target. If
        difference is smaller than this quantity - optimization is stopped.
    """

    def __init__(self, target_function: Callable, surrogate_kernel: Callable,
                 max_iter: int = 100, stop_epsilon: float = 0.01):
        if not(callable(getattr(surrogate_kernel, 'fit', None))) or not(callable(getattr(surrogate_kernel, 'predict', None))):
            raise AttributeError('Surrogate kernel does not has desired methods')
        self.surrogate_kernel = surrogate_kernel
        self.target_function = target_function
        self.max_iter = max_iter
        self.stop_epsilon = stop_epsilon


    def optimize(self, x: np.ndarray, y: np.ndarray):
        """
        Function for performing optimization algorithm.
        Args:
            x (numpy.ndarray): Values on which target function already been evaluated.
            y (numpy.ndarray): Values of target function gotten from evaluating target function on x values.
        Returns:
        """

        predictions, stds = self.surrogate_kernel.fit(x,y)
        current_val = max(y)

        for i in range(self.max_iter):
            Xsamples = random(100)
            Xsamples = Xsamples.reshape(len(Xsamples), 1)


def priormean(xin, **kwargs):
    c2 = kwargs['c2']
    return c2 * xin ** 2


def corr_func(a: np.ndarray, b: np.ndarray, sigmaf2=1.0, ell=1.0):
    diff = a[:, None] - b[None, :]
    return sigmaf2 * np.exp(-diff ** 2 / (2 * ell ** 2))
