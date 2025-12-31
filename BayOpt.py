import numpy as np
import scipy

from typing import Callable, Any
from GP import GPRegressor


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

    def __init__(self, target_function: Callable, surrogate_kernel: Any,
                 max_iter: int = 100, stop_epsilon: float = 0.01, optimization_type: str = 'max'):
        if not (callable(getattr(surrogate_kernel, 'fit', None))) or not (
                callable(getattr(surrogate_kernel, 'predict', None))):
            raise AttributeError('Surrogate kernel does not has desired methods')
        if optimization_type not in ['max', 'min']:
            raise Exception('Unsupported optimization type')
        self.optimization_type = optimization_type
        self.surrogate_kernel = surrogate_kernel
        self.target_function = target_function
        self.max_iter = max_iter
        self.stop_epsilon = stop_epsilon

    def optimize(self, x: np.ndarray, y: np.ndarray, optimization_borders: np.ndarray, n_samples: int = 1000):
        """
        Perform Bayesian optimization loop.
        """
        self.surrogate_kernel.fit(x, y)
        ybest = np.max(y) if self.optimization_type == 'max' else np.min(y)

        for i in range(self.max_iter):
            # Generate candidate samples
            d = optimization_borders.shape[0]

            Xsamples = np.random.uniform(
                low = optimization_borders[:, 0],
                high = optimization_borders[:, 1],
                size = (n_samples, d)
            )
            # Predict mean and std for each candidate
            new_preds, new_stds = self.surrogate_kernel.predict(Xsamples, return_stds=True)
            # Compute acquisition: probability of improvement
            probs = scipy.stats.norm.cdf((new_preds - ybest) / (new_stds + 1e-9))
            # Select best candidate
            ix = np.argmax(probs)
            best_x = Xsamples[ix]
            # Evaluate real objective
            new_y = self.target_function(best_x)
            # Add to training data
            x = np.vstack((x, [[best_x]]))
            y = np.vstack((y, [[new_y]]))
            # Update GP
            self.surrogate_kernel.fit(x, y)
            # Update current best
            ybest = np.max(y) if self.optimization_type == 'max' else np.min(y)

        return x, y


def objective(x, noise=0.1):
    noise = np.random.normal(loc=0, scale=noise)
    return -x**2 + 10234 + noise




gpReg = GPRegressor({'c2': 0.25}, {'ell': 1, 'sigmaf2': 2}, priormean, corr_func, 0.005)

BayOpt = BayesianOptimizer(objective, gpReg)

x = np.arange(0, 1, 0.01).reshape(-1, 1)
y = np.array([objective(i, 0) for i in x])

