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

import dist_mbrl.tabular.util as utils

from .tabular_agent import TabularAgent


class PSRLAgent(TabularAgent):
    def __init__(self, num_states: int, num_actions: int, params: dict = None):
        super().__init__(num_states, num_actions, params)

    def update_policy(self):
        # Sample from posterior
        p, r = self.sample_mdp_from_posterior()

        # Solve MDP to update policy
        self.pi = self.solve_mdp(p, r)

    def solve_mdp(self, p, r) -> np.ndarray:
        """
        Solves MDP specified by transition function P and reward function r by policy
        iteration.

        Returns the optimal policy found by PI
        """
        # Init policy
        random_actions = np.random.choice(self.num_actions, self.num_states)
        pi = self.actions_to_policy_matrix(random_actions)

        for _ in range(self.max_pi_steps):
            # Policy evaluation
            v, q = utils.solve_bellman_eq(p, r, pi, discount=self.gamma)

            # Policy improvement
            new_pi = self.policy_improvement(q)

            # If policy doesn't change, algorithm has converged
            if np.prod(new_pi == pi) == 1:
                break
            else:
                pi = new_pi

        return pi
