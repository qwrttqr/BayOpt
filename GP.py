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
                 noise: float = 0):
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

    def predict(self, x: np.ndarray, return_stds: bool = False):
        # Covariance between training and new points
        cov_training_predict = self.calc_corr_matr(self.x_training, x)
        cov_predict_predict = self.calc_corr_matr(x, x)
        mean_x_predict = self.mean_function(x, **self.mean_args)

        # GP mean
        predictions = mean_x_predict + cov_training_predict.T @ self.inv_cov_training_training @ (
                self.y_training - self.mean_x_training
        )

        if return_stds:
            # Compute the covariance of the predictions
            cov = cov_predict_predict - cov_training_predict.T @ self.inv_cov_training_training @ cov_training_predict
            prediction_stds = np.sqrt(np.maximum(np.diag(cov), 0))
            return predictions, prediction_stds.reshape(-1, 1)
        else:
            return predictions

