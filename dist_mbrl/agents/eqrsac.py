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

import dist_mbrl.mbrl.types
import dist_mbrl.mbrl.util.math
from dist_mbrl.agents.base import BaseQRSAC
from dist_mbrl.mbrl.models import Ensemble


class EQRSAC(BaseQRSAC):
    """
    Epistemic Quantile-regression Soft-Actor Critic
    """

    def __init__(
        self,
        env: gym.Env,
        device: torch.device,
        params: dict,
        dynamics_model: Ensemble,
        termination_fn: dist_mbrl.mbrl.types.TermFnType,
        reward_fn: dist_mbrl.mbrl.types.RewardFnType = None,
        **kwargs,
    ):
        super().__init__(env, device, params, **kwargs)
        self.reward_fn = reward_fn
        self.term_fn = termination_fn
        self.dynamics_model = dynamics_model
        self.ensemble_size = len(dynamics_model)
        self.num_state_samples = params["num_state_samples"]
        self.num_act_samples = params["num_act_samples"]

    def get_q_target(
        self,
        state: torch.Tensor,
        act: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the target quantiles for model-based distributional Q-function. Note
        that the target distribution has KxM quantiles, while the predicted distribution
        only has M quantiles (forward pass through the network). This mismatch is not an
        issue since we can still minimize the projection from one onto the other via the
        Huber quantile regression loss
        """
        state = state.repeat(self.num_state_samples, 1)
        act = act.repeat(self.num_state_samples, 1)
        model_input = self.dynamics_model._get_model_input(state, act)

        # Use the model to generate new samples s' given (s,a)
        # Randomly choose one model
        pred_means, pred_log_vars = self.dynamics_model.forward(model_input)
        vars = pred_log_vars.exp()
        stds = torch.sqrt(vars)
        pred_samples = torch.normal(pred_means, stds)

        next_state = pred_samples[:, :, :-1]
        reward = torch.unsqueeze(pred_samples[:, :, -1], dim=-1)

        # If we are predicting deltas, then we need to add it to the pred samples
        if self.dynamics_model.target_is_delta:
            next_state += state.unsqueeze(dim=0)

        # Use the policy to generate new samples a' given the new s'
        next_state = next_state.repeat(1, self.num_act_samples, 1)
        next_act, next_act_log_pi, _ = self.actor.sample(next_state)
        q_next_values = self.get_min_q(next_state, next_act, self.critic_target)
        q_next_values = q_next_values.view(
            self.ensemble_size, next_state.size(dim=1), self.critic.num_quantiles
        )

        # Use (known) terminal function to get the done mask for the new s'
        act = act.repeat(self.ensemble_size, self.num_act_samples, 1)
        done = self.term_fn(act, next_state.view(-1, self.obs_dim)).view(
            self.ensemble_size, -1, 1
        )
        done_mask = (~done).to(self.device)

        qf_next_target = (
            done_mask * self.gamma * (q_next_values - self.alpha * next_act_log_pi)
        )

        # Average out the aleatoric uncertainty from the rewards and the predicted next
        # values rewards     --> \E[r(s,a)]
        # next values --> \sum_{s',a'}pi(a'|s')P(s'|s,a) Q(s', a')
        reward = torch.mean(
            reward.view(self.ensemble_size, self.num_state_samples, self.batch_size, 1),
            axis=1,
        )
        qf_next_target = torch.mean(
            qf_next_target.view(
                self.ensemble_size,
                self.num_state_samples * self.num_act_samples,
                self.batch_size,
                self.critic.num_quantiles,
            ),
            axis=1,
        )
        # Group the quantile and ensemble dimensions to compose target quantiles
        target_q = (reward + qf_next_target).permute(1, 0, 2)
        target_q = torch.reshape(
            target_q, (self.batch_size, self.ensemble_size * self.critic.num_quantiles)
        ).unsqueeze(dim=1)
        return target_q
