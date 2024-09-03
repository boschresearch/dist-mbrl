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

import random
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


def fix_rng(
    env: Optional[gym.Env] = None,
    seed: Optional[int] = 0,
):
    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed=seed)
    random.seed(seed)
    return rng


def solve_bellman_eq(
    dynamics: np.ndarray, rewards: np.ndarray, policy: np.ndarray, discount: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a Bellman equation in closed form for tabular RL problems. Allows to solve
    both the classical Bellman equation for value functions, as well as UBEs for the
    uncertainty of values
    """
    num_states = dynamics.shape[0]
    r_pi = np.einsum("ij, ij -> i", rewards, policy)
    p_pi = np.einsum("ijk, ij -> ik", dynamics, policy)
    vf = np.linalg.solve(np.eye(num_states) - discount * p_pi, r_pi)
    qf = rewards + discount * np.einsum("ijk, k -> ij", dynamics, vf)
    return vf, qf
