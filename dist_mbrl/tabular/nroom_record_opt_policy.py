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

from typing import Dict

import gymnasium as gym
import numpy as np
import util
from agents import PSRLAgent, TabularAgent
from gymnasium.wrappers.compatibility import EnvCompatibility
from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
from rlberry.seeding import Seeder, safe_reseed

from dist_mbrl.mbrl.util import ReplayBuffer
from dist_mbrl.tabular.config import (
    psrl_default_parameters,
)

CONFIG_TO_AGENT: Dict[str, TabularAgent] = {
    "psrl": PSRLAgent,
}


def train(parameters):
    env = EnvCompatibility(
        NRoom(
            nrooms=3,
            room_size=5,
            success_probability=parameters["transition_probability"],
            initial_state_distribution="center",
        )
    )
    obs_shape = (1,)
    act_shape = (1,)

    # Fix RNG
    seed = parameters["seed"]
    rng = util.fix_rng(seed=seed)
    safe_reseed(env.env, Seeder(seed))

    # Task horizon
    steps_per_episode = 15

    # Aprox. optimal value of initial state
    opt_value = parameters["opt_value"]

    agent = CONFIG_TO_AGENT[parameters["agent_type"]](
        env.observation_space.n, env.action_space.n, params=parameters["agent"]
    )

    num_episodes = parameters["num_episodes"]
    avg_returns = []

    total_regret = np.zeros(num_episodes)
    for i in range(num_episodes):
        # Execute policy and record data
        ep_buffer = ReplayBuffer(
            steps_per_episode,
            obs_shape,
            act_shape,
            obs_type=np.int32,
            action_type=np.int32,
            rng=rng,
        )
        obs, _ = env.reset()
        ep_return = 0
        ep_step = 0
        terminated = False
        while (not terminated) and (ep_step < steps_per_episode):
            action = agent.act(obs)
            next_obs, reward, terminated, *_ = env.step(action)
            # We are forcing the left-most room prize become a terminal state as well.
            # Since this is not implemented in the environment itself, we simulate it
            # here.
            if reward == 0.1:
                terminated = True
            ep_buffer.add(obs, action, next_obs, reward, terminated)
            ep_return += reward
            obs = next_obs
            ep_step += 1

        # Evaluate the agent
        avg_return = evaluate(agent, env, num_episodes=100, max_steps=steps_per_episode)
        total_regret[i] = total_regret[i - 1] + max(opt_value - avg_return, 0.0)
        print(f"Episode #{i}: avg_return = {avg_return}")
        avg_returns.append(avg_return)

        # Update agent with new data
        agent.update(ep_buffer)


def evaluate(
    agent: TabularAgent, env: gym.Env, num_episodes: int, max_steps: int
) -> float:
    episode_returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_return = 0
        step = 0
        terminated = False
        while (not terminated) and (step < max_steps):
            action = agent.act(obs)
            obs, reward, terminated, *_ = env.step(action)
            if reward == 0.1:
                terminated = True
            ep_return += reward
            step += 1
        episode_returns.append(ep_return)

    return np.mean(episode_returns)


if __name__ == "__main__":
    train(psrl_default_parameters)
