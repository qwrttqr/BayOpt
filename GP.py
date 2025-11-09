from typing import Callable
import numpy as np


class GPRegressor:
    """
    Class for performing Gaussian Process regression.

    Args:
        mean_args (dict): Arguments for the mean function.
        correlation_args (dict): Arguments for the correlation function.
        mean_function (Callable): The mean function.
        correlation_func (Callable): Kernel function used to compute correlations.
        noise (float): Noise term, optional.
    """

    def __init__(self, mean_args: dict, correlation_args: dict, mean_function: Callable, correlation_func: Callable,
                 noise: float = 0, return_stds=False):
        self.mean_args = mean_args
        self.corr_args = correlation_args
        self.mean_function = mean_function
        self.corr_func = correlation_func
        self.noise = noise
        self.mean_x_training = None
        self.cov_training_training = None
        self.cov_training_predict = None
        self.cov_predict_predict = None
        self.inv_cov_training_training = None
        self.x_training = None
        self.y_training = None
        self.return_stds = return_stds

    def calc_corr_matr(self, a: np.ndarray, b: np.ndarray = None) -> np.ndarray:
        b = a if b is None else b
        K = self.corr_func(a, b, **self.corr_args)

        if b is a:
            K = K + np.eye(len(a)) * self.noise
        return K

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.x_training = x
        self.y_training = y
        self.mean_x_training = self.mean_function(x, **self.mean_args)
        self.cov_training_training = self.calc_corr_matr(x, x)
        self.inv_cov_training_training = np.linalg.inv(self.cov_training_training)

    def predict(self, x: np.ndarray):
        cov_training_predict = self.calc_corr_matr(x, self.x_training)
        cov_predict_predict = self.calc_corr_matr(x, x)
        mean_x_predict = self.mean_function(x)
        predictions = mean_x_predict + np.dot(np.dot(cov_training_predict, self.inv_cov_training_training),
                                              (self.y_training - self.mean_x_training))
        prediction_stds = None
        if self.return_stds:
            prediction_stds = np.diag(
                cov_predict_predict - np.dot(self.inv_cov_training_training,
                                             np.transpose(cov_training_predict)))
            return predictions, prediction_stds
        else:
            return predictions


def priormean(xin, **kwargs):
    c2 = kwargs['c2']
    return c2 * xin ** 2

def corr_func(a: np.ndarray, b: np.ndarray, sigmaf2=1.0, ell=1.0):
    diff = a[:, None] - b[None, :]
    return sigmaf2 * np.exp(-diff ** 2 / (2 * ell ** 2))
