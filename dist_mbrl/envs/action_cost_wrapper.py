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
from gymnasium import Env, Wrapper


class ActionCostWrapper(Wrapper):
    def __init__(self, env: Env, action_cost_weight):
        """Constructor for the Reward wrapper."""
        super().__init__(env)
        self.action_cost_weight = action_cost_weight

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(reward, action), terminated, truncated, info

    def reward(self, reward, action):
        return reward - self.action_cost(action)

    def action_cost(self, action):
        control_cost = self.action_cost_weight * np.sum(np.square(action))
        return control_cost
