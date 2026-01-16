import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
from jax import random
import jax.numpy as jnp
from query_strategies.core_set import CoreSet
from query_strategies.adjusted_fisher import AdjustedFisher
from query_strategies.bait import BAIT
from query_strategies.random_sampling import RandomSampling
from linreg_utils.data_gen import generate_data
from linreg_utils.model import (
    linear_model,
    linear_regression,
)


def experiment(
    num_rounds=10,
    num_coeffs=5,
    initial_sample_sz=20,
    pool_sz=100,
    budget=10,
    iter_per_algo=10,
    measurement_error=False,
):
    true_coeff = np.asarray([0 if i == 0 else 1 for i in range(num_coeffs)])
    step_keys = random.split(random.PRNGKey(0), num_rounds)
    param_diffs = defaultdict(list)

    for realization in tqdm(range(num_rounds)):
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
            "given_key": step_keys[realization][0],
            "measurement_error": measurement_error,
        }

        core_set_model = CoreSet(**kwargs)
        bait_model = BAIT(**kwargs)
        adj_fisher_model = AdjustedFisher(**kwargs)
        if num_coeffs > 2:
            adj_fisher_model.num_params = 2
            adj_fisher_model.param_start = 1

        rand_model = RandomSampling(**kwargs)

        models = {
            "Fisher": adj_fisher_model,
            "BAIT": bait_model,
            "CoreSet": core_set_model,
            "Random": rand_model,
        }

        iter_step_keys = random.split(
            random.PRNGKey(step_keys[realization][0]), iter_per_algo
        )

        current_param_diffs = defaultdict(list)
        for iter in range(iter_per_algo):
            "Generate Data"
            X, y, error, _ = generate_data(
                sample_size=initial_sample_sz if iter == 0 else pool_sz,
                coeff=true_coeff,
                key=iter_step_keys[iter],
                measurement_error=measurement_error,
            )

            "Simulate model"
            for algo, model in models.items():
                X_cp = jnp.array(X)

                "Decorrelation"
                if model.labeled_X is not None:
                    labeled_meanX = jnp.mean(model.labeled_X, axis=0)
                    X_cp -= labeled_meanX

                model.choose_sample(iter_step_keys[iter], X_cp, y, error)
                estimated_coeffs = model.model_training_fn(
                    model.labeled_X, model.labeled_y
                )

                model.current_params = estimated_coeffs

                "Tracks the literall difference between estimated and true coefficients. Ex. [0.5 0.8] - [0 1]"
                current_param_diffs[algo].append(estimated_coeffs - true_coeff)

        "data"
        for algo in models:
            per_realization_param_diff = pd.DataFrame()
            diffs = current_param_diffs[algo]
            per_realization_param_diff["param_diffs"] = [np.array(_) for _ in diffs]
            per_realization_param_diff.reset_index(inplace=True)
            per_realization_param_diff.rename(
                columns={"index": "Iteration"}, inplace=True
            )
            param_diffs[algo].append(per_realization_param_diff)

    for algo in param_diffs:
        param_diffs_df = pd.concat(param_diffs[algo], axis=0)
        param_diffs_df.to_csv(
            f"data/{algo}_param_diff_s{initial_sample_sz}_b{budget}_p{pool_sz}_n{num_rounds}_i{iter_per_algo}_c{num_coeffs}_m{measurement_error}.csv",
            index=False,
        )


# ------------------- RUN ---------------------
# ###############################
# num_rounds=10,
# num_coeffs=5,
# initial_sample_sz=20,
# pool_sz=1000,
# budget=10,
# iter_per_algo=10,
# ###############################


def percentage_type(value):
    ivalue = float(value)
    if ivalue < 0.0 or ivalue > 1.0:
        raise argparse.ArgumentTypeError("Percentage must be between 0 and 1")
    return ivalue


parser = argparse.ArgumentParser(prog="BenchMark", description="Benchamarks stuff")
parser.add_argument(
    "-n",
    "--numRounds",
    action="store",
    help="Enter number of rounds to build distribution of vars (default=10)",
    type=int,
    required=False,
    default=10,
)
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
    "-s",
    "--initSampleSz",
    action="store",
    help="Enter initial_sample_sz (num points to sample in 0th round) (default=20)",
    type=int,
    required=False,
    default=20,
)
parser.add_argument(
    "-p",
    "--poolSz",
    action="store",
    help="Enter num total points collected each night (default=1000)",
    type=int,
    required=False,
    default=1000,
)
parser.add_argument(
    "-b",
    "--budget",
    action="store",
    help="Enter num points can sample each subsequent night (default=10)",
    type=int,
    required=False,
    default=10,
)
parser.add_argument(
    "-i",
    "--itersPerRound",
    action="store",
    help="Enter num iterations each algo should take to converge each round (default=10)",
    type=int,
    required=False,
    default=10,
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

num_rounds = int(args["numRounds"])
num_coeffs = int(args["numCoeffs"])
initial_sample_sz = int(args["initSampleSz"])
pool_sz = int(args["poolSz"])
budget = int(args["budget"])
iter_per_algo = int(args["itersPerRound"])
verbose = bool(args["verbose"])
measurement_err = bool(args["measurement_error"])


if verbose:
    print("*" * 42)
    print("*" + " " * 10 + f"Benching with args: {args}")
    print("*" * 42)

# ------------ EXPERIMENT TO RUN MULTIPLE REALIZATIONS -------------
experiment(
    num_rounds=num_rounds,
    num_coeffs=num_coeffs,
    initial_sample_sz=initial_sample_sz,
    pool_sz=pool_sz,
    budget=budget,
    iter_per_algo=iter_per_algo,
    measurement_error=measurement_err,
)

if verbose:
    print("DONE")
