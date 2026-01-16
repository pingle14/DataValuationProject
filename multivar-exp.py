import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from jax import random
import jax.numpy as jnp
from query_strategies.adjusted_fisher import AdjustedFisher
from query_strategies.random_sampling import RandomSampling
from linreg_utils.data_gen import generate_data
from linreg_utils.model import (
    linear_model,
    linear_regression,
)


def multivar_experiment(
    num_coeffs=5,
    measurement_error=False,
):
    initial_sample_sz = 10
    pool_sz = 100
    budget = 1
    iter_per_algo = 1000

    true_coeff = np.asarray([0 if i == 0 else 1 for i in range(num_coeffs)])
    step_keys = random.split(random.PRNGKey(0), 1)

    model_inference_fn = linear_model
    model_training_fn = linear_regression
    kwargs = {
        "model_inference_fn": model_inference_fn,
        "model_training_fn": model_training_fn,
        "generate_data": generate_data,
        "initial_sample_sz": initial_sample_sz,
        "pool_sz": pool_sz,
        "budget": budget,
        "iter": iter_per_algo,
        "true_coeff": true_coeff,
        "given_key": step_keys[0][0],
        "measurement_error": measurement_error,
    }

    rand_model = RandomSampling(**kwargs)

    adj_fisher_model = AdjustedFisher(**kwargs)
    adj_fisher_model.num_params = 2 if num_coeffs > 2 else 1
    adj_fisher_model.param_start = 1

    big_budget = AdjustedFisher(**kwargs)
    big_budget.num_params = 2 if num_coeffs > 2 else 1
    big_budget.param_start = 1
    big_budget.budget = budget * 10

    big_pool = AdjustedFisher(**kwargs)
    big_pool.num_params = 2 if num_coeffs > 2 else 1
    big_pool.param_start = 1
    big_pool.pool_sz = pool_sz * 10

    models = {
        f"Our Approach (Pool Size = {pool_sz}, Budget = {budget})": adj_fisher_model,
        f"Our Approach (Pool Size = {pool_sz}, Budget = {budget * 10})": big_budget,
        "Random": rand_model,
    }

    iter_step_keys = random.split(random.PRNGKey(step_keys[0][0]), iter_per_algo)
    for iter in tqdm(range(iter_per_algo)):
        "Generate Data"
        X, y, error, _ = generate_data(
            sample_size=initial_sample_sz if iter == 0 else pool_sz,
            coeff=true_coeff,
            key=iter_step_keys[iter],
            measurement_error=measurement_error,
        )

        # else:
        "Simulate model"
        for algo, model in models.items():
            X_cp = jnp.array(X)

            "Decorrelation"
            if model.labeled_X is not None:
                labeled_meanX = jnp.mean(model.labeled_X, axis=0)
                X_cp -= labeled_meanX

            model.choose_sample(iter_step_keys[iter], X_cp, y, error)
            estimated_coeffs = model.model_training_fn(model.labeled_X, model.labeled_y)
            model.current_params = estimated_coeffs

    # Larger pool size
    print("Now model with larger pool size")
    models[f"Our Approach (Pool Size = {pool_sz * 10}, Budget = {budget})"] = big_pool
    for iter in tqdm(range(iter_per_algo)):
        "Generate Data"
        X, y, error, _ = generate_data(
            sample_size=initial_sample_sz if iter == 0 else big_pool.pool_sz,
            coeff=true_coeff,
            key=iter_step_keys[iter],
            measurement_error=measurement_error,
        )

        # else:
        "Simulate model"
        X_cp = jnp.array(X)

        "Decorrelation"
        if big_pool.labeled_X is not None:
            labeled_meanX = jnp.mean(big_pool.labeled_X, axis=0)
            X_cp -= labeled_meanX

        big_pool.choose_sample(iter_step_keys[iter], X_cp, y, error)
        estimated_coeffs = big_pool.model_training_fn(
            big_pool.labeled_X, big_pool.labeled_y
        )
        big_pool.current_params = estimated_coeffs

    df = pd.DataFrame()
    for algo, model in models.items():
        mini_df = pd.DataFrame()
        print(f"{algo}: labeledX: {model.labeled_X.shape}")
        mini_df["X1"] = pd.Series(model.labeled_X[:, 1])
        if num_coeffs > 2:
            mini_df["X2"] = pd.Series(model.labeled_X[:, 2])
        mini_df["Algorithm"] = algo
        mini_df.reset_index(inplace=True)
        mini_df.rename(columns={"index": "Iteration"}, inplace=True)
        df = pd.concat([df, mini_df])
    df.to_csv(
        f"data/multiVar_s{initial_sample_sz}_b{budget}_p{pool_sz}_n1_i{iter_per_algo}_c{num_coeffs}_m{measurement_error}.csv",
        index=False,
    )


# ------------------- RUN ---------------------
# ###############################
# num_coeffs=5,
# ###############################


def percentage_type(value):
    ivalue = float(value)
    if ivalue < 0.0 or ivalue > 1.0:
        raise argparse.ArgumentTypeError("Percentage must be between 0 and 1")
    return ivalue


parser = argparse.ArgumentParser(prog="BenchMark", description="Benchamarks stuff")
parser.add_argument(
    "-c",
    "--numCoeffs",
    action="store",
    help="Enter number of params/coefficients in linreg model (default=5)",
    type=int,
    required=False,
    default=5,
)
parser.add_argument(
    "-v",
    "--verbose",
    help="Bool to print stuff or not",
    action="store_true",
    required=False,
    default=False,
)
parser.add_argument(
    "-m",
    "--measurement_error",
    help="Bool to include measurement_error",
    action="store_true",
    required=False,
    default=False,
)

args = vars(parser.parse_args())

num_coeffs = int(args["numCoeffs"])
verbose = bool(args["verbose"])
measurement_err = bool(args["measurement_error"])


if verbose:
    print("*" * 42)
    print("*" + " " * 10 + f"Benching with args: {args}")
    print("*" * 42)

# ------------ EXPERIMENT TO RUN MULTI-VAR -------------
multivar_experiment(num_coeffs=num_coeffs, measurement_error=measurement_err)

if verbose:
    print("DONE")
