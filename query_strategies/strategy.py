import jax.numpy as jnp
from jax import jit, grad
from abc import ABC, abstractmethod
import time


# Estimate variance of the estimated parameters analytically
# ASSUMPTION: Unbiased Estimator:
# b^2(x) = MSE
#        = B^2 + Var
#        = 0 + Var(theta-hat)
@jit
def estimate_variance_meas_err(err, residual):
    # Compute the weights
    weights = 1 / err
    # 2nd moment = E[(x - mu)^2] = sigma^2 + sigma_e^2
    # sigma^2 = 1/n sum ((x - x-bar)^2) - sigma_e^2
    A = jnp.sum(weights) / (jnp.sum(weights) ** 2 - jnp.sum(weights**2))

    # Compute the weighted measurements
    error_diff = (residual**2) - (err**2)
    weighted_measurements = weights * (error_diff)

    # Compute the variance estimator
    variance_estimator = A * jnp.sum(weighted_measurements)

    return variance_estimator


@jit
def estimate_variance(residual):
    # unweighted estimator of variance
    return jnp.sum(residual**2) / (len(residual) - 1)  # SSE


class Strategy(ABC):
    """
    This is the base class for different strategies we use in our experiments
    """

    def __init__(
        self,
        model_inference_fn,
        model_training_fn,
        generate_data,
        name="Interface",
        pool_sz=100,
        initial_sample_sz=20,
        budget=1,
        iter=100,
        true_coeff=None,
        given_key=None,
        measurement_error=False,
    ):
        # Model and Grad
        self.model_inference_fn = model_inference_fn
        self.model_training_fn = model_training_fn
        self.grad_f = grad(model_inference_fn)
        self.generate_data = generate_data
        self.current_params = None

        # Active Learning Strategy Properties
        self.name = name
        self.initial_sample_sz = max(1, initial_sample_sz)
        self.pool_sz = pool_sz
        self.budget = budget
        self.iter = iter
        self.true_coeff = true_coeff
        self.measurement_error = measurement_error

        # Simulation Data
        self.labeled_X = None
        self.labeled_y = None
        self.error = None
        self.given_key = given_key

    @abstractmethod
    def update_sample(self, key, X, y, error):
        return

    def choose_sample(self, key, X=None, y=None, error=None):
        if self.labeled_X is None:
            # Init self.labeled: (initial sample)
            self.labeled_X = X
            self.labeled_y = y
            self.error = error
        else:
            self.update_sample(key, X, y, error)

    def estimate_variance(self, params, y, X, err):
        residual = y - self.model_inference_fn(params, X)
        return (
            estimate_variance_meas_err(err, residual)
            if self.measurement_error
            else estimate_variance(residual)
        )
