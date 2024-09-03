"""
Copyright (c) 2024 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from collections import defaultdict

import numpy as np
from numpy.random import RandomState
from rliable import library as rly, metrics
from scipy.stats import sem

RS = RandomState(0)


def rolling_average(w: int, arr: np.ndarray) -> np.ndarray:
    """
    Expects an array of size (num_points, num_seeds) where we want to smoothen out each
    individual curve by a moving average
    """
    if arr.size == 0:
        return arr
    one_array = np.ones(w) / w
    padded_arr = np.pad(arr, [(w - 1, 0), (0, 0)], mode="edge")
    return np.apply_along_axis(
        lambda m: np.convolve(m, one_array, mode="valid"), axis=0, arr=padded_arr
    )


def process_raw_results(data: dict):
    """
    Grabs raw returns, passes a window average filter and computes mean and standard error

    """
    raw_returns = data["raw_returns"]
    steps = data["steps"]
    env_names = data["env_names"]
    smoothened_returns = defaultdict(dict)
    mean_returns = defaultdict(dict)
    ci_returns = defaultdict(dict)
    for env_name in env_names:
        for idx in raw_returns[env_name].keys():
            smoothened_returns[env_name][idx] = rolling_average(
                10, raw_returns[env_name][idx]
            )
            mean_returns[env_name][idx] = np.mean(
                smoothened_returns[env_name][idx], axis=-1
            )
            sem_return = sem(smoothened_returns[env_name][idx], axis=-1)
            ci_returns[env_name][idx] = 1.00 * sem_return

    return {
        "env_names": env_names,
        "steps": steps,
        "smoothened_returns": smoothened_returns,
        "mean_returns": mean_returns,
        "ci_returns": ci_returns,
    }


def get_mean_median_returns(data: dict):
    """
    Compute mean and median returns across environments
    """

    env_names = data["env_names"]
    smoothened_returns = data["smoothened_returns"]
    steps = data["steps"]

    # Compute mean / median across environments
    all_envs_mean_returns = {}
    all_envs_mean_ci_returns = {}
    all_envs_median_returns = {}
    all_envs_median_ci_returns = {}
    all_envs_steps = {}
    expected_num_episodes = 250
    for idx in data["smoothened_returns"][env_names[0]].keys():
        l, s = [], []
        for env in env_names:
            r = smoothened_returns[env][idx]
            t = steps[env][idx]
            pad_length = expected_num_episodes - r.shape[0]
            l.append(
                np.pad(r, ((0, pad_length), (0, 0)), "constant", constant_values=np.nan)
            )
            s.append(np.pad(t, (0, pad_length), "constant", constant_values=0))
        arr = np.array(l)
        mean = np.mean(arr, axis=0)
        median = np.median(arr, axis=0)
        arr = np.array(s)
        all_envs_mean_returns[idx] = np.mean(mean, axis=-1)
        all_envs_mean_ci_returns[idx] = sem(mean, axis=-1)
        all_envs_median_returns[idx] = np.mean(median, axis=-1)
        all_envs_median_ci_returns[idx] = sem(median, axis=-1)
        all_envs_steps[idx] = np.mean(arr, axis=0)

    return {
        "mean_returns": all_envs_mean_returns,
        "mean_ci_returns": all_envs_mean_ci_returns,
        "median_returns": all_envs_median_returns,
        "median_ci_returns": all_envs_median_ci_returns,
        "steps": all_envs_steps,
    }


def get_bootstrap_intervals(data: dict):
    """
    Compute mean and median returns across environments
    """

    env_names = data["env_names"].tolist()
    smoothened_returns = data["smoothened_returns"]
    steps = data["steps"]

    # For rliable we need to transform our data to a dictionary mapping algorithm to
    # tensor of scores in the format [envs x num_seeds x episodes]
    scores_dict = defaultdict()
    algos = list(smoothened_returns[env_names[0]].keys())

    num_episodes, num_runs = smoothened_returns[env_names[0]][algos[0]].shape
    num_episodes = 250
    for algo in algos:
        scores_dict[algo] = np.zeros((len(env_names), num_runs, num_episodes))
        for i, env in enumerate(env_names):
            scores_dict[algo][i, :, :] = (
                smoothened_returns[env][algo][:num_episodes].transpose() / 1000.0
            )

    agg_steps = steps[env_names[0]][algo]
    mean = lambda scores: np.array(
        [metrics.aggregate_mean(scores[..., frame]) for frame in range(num_episodes)]
    )
    median = lambda scores: np.array(
        [metrics.aggregate_median(scores[..., frame]) for frame in range(num_episodes)]
    )
    iqm = lambda scores: np.array(
        [metrics.aggregate_iqm(scores[..., frame]) for frame in range(num_episodes)]
    )

    mean_scores, mean_cis = rly.get_interval_estimates(
        scores_dict, mean, reps=500, random_state=RS, confidence_interval_size=0.95
    )
    median_scores, median_cis = rly.get_interval_estimates(
        scores_dict, median, reps=500, random_state=RS, confidence_interval_size=0.95
    )
    iqm_scores, iqm_cis = rly.get_interval_estimates(
        scores_dict, iqm, reps=500, random_state=RS, confidence_interval_size=0.95
    )
    return {
        "mean_returns": mean_scores,
        "mean_ci_returns": mean_cis,
        "median_returns": median_scores,
        "median_ci_returns": median_cis,
        "iqm_returns": iqm_scores,
        "iqm_ci_returns": iqm_cis,
        "steps": agg_steps,
    }


def get_performance_profile(data: dict, episode=-1):
    """
    Compute the performance profile from rlibale library
    """

    env_names = data["env_names"].tolist()
    smoothened_returns = data["smoothened_returns"]

    # For rliable we need to transform our data to a dictionary mapping algorithm to
    # tensor of normalized scores in the format [num_seeds x envs]
    scores_dict = defaultdict()
    algos = list(smoothened_returns[env_names[0]].keys())
    _, num_runs = smoothened_returns[env_names[0]][algos[0]].shape
    for algo in algos:
        scores_dict[algo] = np.zeros((num_runs, len(env_names)))
        for i, env in enumerate(env_names):
            scores_dict[algo][:, i] = (smoothened_returns[env][algo][episode]) / 1000.0

    dmc_thresholds = np.linspace(0.0, 1.0, 301)
    score_distributions, score_cis = rly.create_performance_profile(
        scores_dict, dmc_thresholds, reps=2000
    )

    return {
        "score_distributions": score_distributions,
        "score_cis": score_cis,
        "steps": dmc_thresholds,
    }
