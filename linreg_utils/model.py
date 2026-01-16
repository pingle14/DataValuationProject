import jax.numpy as jnp
from jax import jit


# functional form
@jit
def linear_model(params, X):
    return jnp.matmul(X, params)


@jit
def linear_regression(X, y):
    X_transpose = jnp.transpose(X)
    X_transpose_X_inv = jnp.linalg.inv(jnp.matmul(X_transpose, X))
    coeff = jnp.matmul(jnp.matmul(X_transpose_X_inv, X_transpose), y)
    return coeff
