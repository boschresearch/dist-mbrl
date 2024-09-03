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

from typing import List, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import dist_mbrl.agents.util as agent_util
import dist_mbrl.mbrl.models.util as models_util
import dist_mbrl.mbrl.types
import dist_mbrl.mbrl.util.math
from dist_mbrl.agents.base import BaseSAC, BufferType
from dist_mbrl.mbrl.models import Ensemble
from dist_mbrl.mbrl.types import TransitionBatch


def add_ensemble_dim(batch_list: List[TransitionBatch]) -> TransitionBatch:
    """
    Takes a TransitionBatch list of size ensemble_size and returns a new TransitionBatch
    that stacks the list into a new numpy array dimension.
    """
    obs = np.stack([batch.obs for batch in batch_list], axis=0)
    act = np.stack([batch.act for batch in batch_list], axis=0)
    next_obs = np.stack([batch.next_obs for batch in batch_list], axis=0)
    rewards = np.stack([batch.rewards for batch in batch_list], axis=0)
    dones = np.stack([batch.dones for batch in batch_list], axis=0)
    return TransitionBatch(obs, act, next_obs, rewards, dones)


class QUSAC(BaseSAC):
    """
    Q-uncertainty SAC. A model-based version of SAC that trains an ensemble of
    Q-functions in independent buffers, each one filled by a different predictive model
    of the environment's dynamics. As such, SAC uses an average of Q-values that
    accounts for epistemic uncertainty.
    """

    def __init__(
        self,
        env: gym.Env,
        device: torch.device,
        params: dict,
        dynamics_model: Ensemble,
        reward_fn: dist_mbrl.mbrl.types.RewardFnType = None,
        **kwargs,
    ):
        """
        The reward fn is optional. In case the rewards are assumed to be known, we use
        the reward function during uncertainty reward estimation.
        """
        super().__init__(env, device, params, **kwargs)
        self.dynamics_model = dynamics_model
        self.reward_fn = reward_fn

    def update_params(self, data: BufferType, step: int):
        # Sample batches from the model buffers
        batch_list = [buffer.sample(self.batch_size) for buffer in data]
        batch = [cast(TransitionBatch, batch) for batch in batch_list]

        # Split the list of transition batch into the inidividual model batches (used
        # for training the critic), and the mean model batch, used to train the UBE and
        # actor
        model_batch = batch[: len(self.dynamics_model)]
        model_batch = add_ensemble_dim(model_batch)
        mean_model_batch = batch[-1]

        # Critic update
        self.update_critic(model_batch, step)

        # Target updates
        if step % self.target_update_freq == 0:
            agent_util.soft_update(self.critic_target, self.critic, self.tau)

        # Actor update
        if step % self.actor_update_freq == 0:
            self.update_actor(mean_model_batch, step)
            self.actor_updates += 1

    def update_actor(self, batch: TransitionBatch, step: int):
        # Get actions/log_probs from batch
        obs, *_ = batch.astuple()
        state = models_util.to_tensor(obs).to(self.device)
        pi, log_pi, _ = self.actor.sample(state)

        # Compute Q-values
        q_values = self.get_min_q(state, pi, self.critic)
        mean_q_values = torch.mean(q_values, dim=0)

        # Actor loss: maximize values and entropy of policy
        actor_loss = (self.alpha * log_pi - mean_q_values).mean()

        # Reset gradients, do backward pass, clip gradients (if enabled) and take a
        # gradient step
        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.clip_grad_norm:
            nn.utils.clip_grad_norm_(list(self.actor.parameters()), max_norm=0.5)
        self.actor_optim.step()

        # If we are using auto-entropy tuning, then we update the alpha gain
        self.maybe_update_entropy_gain(log_pi)
        return actor_loss.item()

    def get_mean_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Pass the state-action batch through the ensemble of critics and aggregate the
        predictions using the mean.
        """
        return self.critic.mean_forward(state, action)
