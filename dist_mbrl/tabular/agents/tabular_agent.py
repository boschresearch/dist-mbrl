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

from abc import ABC
from typing import Tuple

import numpy as np

from dist_mbrl.mbrl.util import ReplayBuffer
from dist_mbrl.tabular.posterior import (
    DirichletPosterior,
    NormalGammaPosterior,
    NormalPosterior,
)


class TabularAgent(ABC):
    def __init__(self, num_states: int, num_actions: int, params: dict = None):
        self.num_states = num_states + 1  # We add a terminal state
        self.num_actions = num_actions

        # Common params for all agent types
        self.gamma = params["gamma"]
        self.model_type = params["model_type"]
        self.transition_repeat = params["transition_repeat"]
        self.max_pi_steps = params["max_pi_steps"]

        # Add terminal state abstraction to the agent, not the environment
        self.terminal = self.num_states - 1

        # Initialize MDP posterior
        self.counts = np.zeros((self.num_states, self.num_actions, self.num_states))
        dirichlet_prior = num_states ** (-0.5) * np.ones(
            (self.num_states, self.num_actions, self.num_states)
        )

        # Force terminal state transition onto itself
        epsilon = 1e-8
        dirichlet_prior[self.terminal, :, :] = epsilon * np.ones(
            (self.num_actions, self.num_states)
        )
        dirichlet_prior[self.terminal, :, self.terminal] = (1000) * np.ones(
            (self.num_actions)
        )
        self.p_posterior = DirichletPosterior(dirichlet_prior)

        if self.model_type == "normal_gamma":
            # Params for the normal-gamma model of the reward function r(s,a)
            model_prior = params["model_prior"]
            mu = model_prior["mu"]
            kappa = model_prior["kappa"]
            alpha = model_prior["alpha"]
            beta = model_prior["beta"]
            normalgamma_prior = np.array(
                [[[mu, kappa, alpha, beta]] * self.num_actions] * self.num_states
            )
            # Set specific params for terminal state, which should always be close to
            # zero when sampling
            normalgamma_prior[self.terminal, :] = [0.0, 1e9, 1e12, 1e9]
            self.r_posterior = NormalGammaPosterior(normalgamma_prior)
        elif self.model_type == "normal":
            mu = 0
            tau = 1
            normal_prior = np.array([[[mu, tau]] * self.num_actions] * self.num_states)
            # Force terminal state to be close-to-deterministic with zero reward
            normal_prior[self.terminal, :] = [0.0, 1e9]
            self.r_posterior = NormalPosterior(normal_prior)

        # Policy is represented as a matrix (table) of size SxA. We initialize it with
        # some random deterministic action selection
        random_actions = np.random.choice(self.num_actions, self.num_states)
        self.pi = self.actions_to_policy_matrix(random_actions)

    def update_policy(self) -> np.ndarray:
        pass

    def update(self, ep_buffer: ReplayBuffer):
        """
        Update the agents internal state and relevant parameters based on environment
        data.
        """
        self.update_posterior_mdp(ep_buffer)
        self.update_policy()

    def update_posterior_mdp(self, ep_buffer: ReplayBuffer):
        """
        Updates the posterior distribution over MDPs:
        1) Update the Dirichlet distribution for the transition probabilities p(s'|s,a)
        2) Update the Normal-Gamma (or Normal) model of the mean reward function r(s,a)
        """
        counts = np.zeros((self.num_states, self.num_actions, self.num_states))
        r_sums = np.zeros((self.num_states, self.num_actions))
        rsquared_sums = np.zeros((self.num_states, self.num_actions))
        data = ep_buffer.get_all()

        for transition in data:
            s, a, s_prime, r, done = transition.astuple()
            if done:
                # artificially transition from s_prime to terminal
                counts[s, a, self.terminal] += self.transition_repeat
            else:
                counts[s, a, s_prime] += self.transition_repeat

            r_sums[s, a] += r * self.transition_repeat
            rsquared_sums[s, a] += (r**2) * self.transition_repeat

        # Update MDP posterior
        self.p_posterior.update_params(counts)
        self.r_posterior.update_params(counts, r_sums, rsquared_sums)

    def sample_ensemble_from_posterior(
        self, num_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Composes an ensemble of MDPs of size num_samples.
        The resulting model has dimensions: (ensemble x S x A x S)
        """
        # TODO: could probably be replaced by a list comprehension
        p_ensemble = []
        r_ensemble = []
        for _ in range(num_samples):
            p, r = self.sample_mdp_from_posterior()
            p_ensemble.append(p)
            r_ensemble.append(r)
        return np.stack(p_ensemble, axis=0), np.stack(r_ensemble, axis=0)

    def sample_mdp_from_posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples a dynamics model p(s' | s,a) and reward function r(s,a) from their
        respective distributions
        """
        p = np.zeros((self.num_states, self.num_actions, self.num_states))
        r = np.zeros((self.num_states, self.num_actions))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                p[s, a, :] = self.p_posterior.sample(s, a)
                r[s, a] = self.r_posterior.sample(s, a)

        return p, r

    def get_mean_mdp(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the mean MDP from the transition and reward distributions.

        1) For a Dirichlet distribution, the mean is the alpha counts divided by their
           sum
        2) For the Normal-Gamma, we want the mean, which is the "mu" parameter.
        """
        P_bar = self.p_posterior.get_mean()
        r_bar = self.r_posterior.get_mean()
        return P_bar, r_bar

    def policy_improvement(self, qf: np.ndarray) -> np.ndarray:
        """
        Compute the greedy policy according to some utility function qf
        """
        greedy_actions = np.array(
            [np.random.choice(np.argwhere(q == np.amax(q))[0]) for q in qf]
        )
        return self.actions_to_policy_matrix(greedy_actions)

    def actions_to_policy_matrix(self, actions: np.ndarray) -> np.ndarray:
        """
        Transforms deterministic action selection into the proper (probabilistic) policy
        matrix form
        """
        pi = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            pi[s, actions[s]] = 1
        return pi

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Given an observation, returns an action sampled from the policy. In general, we
        assume the policy to be stochastic
        """
        return np.random.choice(self.num_actions, p=self.pi[obs])
