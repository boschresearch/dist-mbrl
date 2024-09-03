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

import gymnasium as gym
import torch

from dist_mbrl.agents.base import BaseQRSAC


class QRSAC(BaseQRSAC):
    """
    Model-free Quantile-regression Soft-actor Critic
    """

    def __init__(self, env: gym.Env, device: torch.device, params: dict, **kwargs):
        super().__init__(env, device, params, **kwargs)

    def get_q_target(
        self,
        state: torch.Tensor,
        act: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the standard target Q-values. We unsqueeze the middle dimension for
        computing the quantile-regression Huber loss.
        """
        next_act, next_act_log_pi, _ = self.actor.sample(next_state)
        q_next_values = self.get_min_q(next_state, next_act, self.critic_target)
        qf_next_target = q_next_values - self.alpha * next_act_log_pi
        target_q_values = reward + done_mask * self.gamma * qf_next_target
        return target_q_values.unsqueeze(dim=1)
