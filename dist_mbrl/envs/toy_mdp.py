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

import numpy as np


class ToyMDP:
    """
    A toy MRP example parameterized by two random variables [alpha, beta]
    """

    def __init__(self, alpha: float, beta: float, gamma: float = 0):
        self.alpha = alpha
        self.beta = beta
        self.p = self.build_transition_matrix(alpha, beta)
        self.r = self.build_reward_vector(gamma)

    def build_transition_matrix(
        self, alpha: float, beta: float, correlated: bool = False
    ) -> np.ndarray:
        return np.array(
            [
                [0, 1 - alpha, alpha, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, beta, 1 - beta],
                [0.1, 0, 0, 0, 0.9],
                [0, 0, 0, 0, 1],
            ]
        )

    def build_reward_vector(self, gamma) -> np.ndarray:
        return np.array([0, 0, gamma, 100, 0])


class ToyMDP1:
    """
    A toy MRP example parameterized by one random variable
    """

    def __init__(self, alpha: float, beta: float = 0.1):
        self.alpha = alpha
        self.p = self.build_transition_matrix(alpha, beta)
        self.r = self.build_reward_vector()

    def build_transition_matrix(self, alpha: float, beta: float) -> np.ndarray:
        return np.array(
            [
                [0, 1 - alpha, alpha, 0],
                [0.0, 0, 0, 1.0],
                [beta, 0, 0.0, 1 - beta],
                [0, 0, 0, 1],
            ]
        )

    def build_reward_vector(self) -> np.ndarray:
        return np.array([0, 1, -1, 0])
