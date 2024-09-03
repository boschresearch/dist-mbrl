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

import pathlib
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import (
    FlattenObservation,
    TransformReward,
)

import dist_mbrl.envs
import dist_mbrl.envs.term_fns as custom_term_fns
import dist_mbrl.mbrl.env.termination_fns as term_fns
import dist_mbrl.mbrl.planning
import dist_mbrl.mbrl.types
import dist_mbrl.mbrl.util.common as mbrl_utils
from dist_mbrl.agents import EQRSAC, QRSAC, QUSAC, SAC, BaseAgent
from dist_mbrl.config import AgentTypes
from dist_mbrl.envs.action_cost_wrapper import ActionCostWrapper
from dist_mbrl.mbrl.models import (
    Ensemble,
    Model,
    ModelEnv,
    ModelTrainer,
    OneDTransitionRewardModel,
)
from dist_mbrl.mbrl.planning import Agent
from dist_mbrl.mbrl.util import ReplayBuffer
from dist_mbrl.utils.data_collection import ModelBufferType
from dist_mbrl.utils.video import VideoRecorder

PathType = Union[str, pathlib.Path]

ENV_NAME_TO_TERM_FN = {
    "InvertedPendulum-v2": "inverted_pendulum",
    "InvertedPendulumBulletEnv-v0": "inverted_pendulum",
    "InvertedPendulumSwingupBulletEnv-v0": "no_termination",
    "Pendulum-v1": "no_termination",
    "Hopper-v4": "hopper",
    "HalfCheetah-v4": "no_termination",
    "Walker2d-v4": "walker2d",
    "HalfCheetahBulletEnv-v0": "no_termination",
}

AGENT_DICT: Dict[str, BaseAgent] = {
    AgentTypes.SAC: SAC,
    AgentTypes.MBPO: SAC,
    AgentTypes.QRSAC: QRSAC,
    AgentTypes.QRMBPO: QRSAC,
    AgentTypes.QRMBPO_BIG: QRSAC,
    AgentTypes.QRMBPO_CONSISTENT: QRSAC,
    AgentTypes.QRMBPO_LOSS: EQRSAC,
    AgentTypes.EQRSAC: EQRSAC,
    AgentTypes.QUSAC: QUSAC,
}


def fix_rng(
    env: gym.Env,
    agent: Agent = None,
    seed: Optional[int] = 0,
    eval_env: Optional[gym.Env] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.random.Generator, torch.Generator]:
    """Fix the seed for all sources of randomness"""
    env.reset(seed=seed)
    if eval_env is not None:
        eval_env.reset(seed=seed)
    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    generator = torch.Generator(device)
    generator.manual_seed(seed)
    return rng, generator


def get_env(params: dict) -> Tuple[str, gym.Env, gym.Env]:
    """
    Returns independent training and evaluation environments, along with the name of the
    env
    """
    env_name = params["name"]
    kwargs = {"render_mode": "rgb_array"}
    if params.get("kwargs"):
        kwargs = {**kwargs, **params["kwargs"]}

    if "dm_control" in env_name:
        env = make_dmc_env(env_name, params["action_cost"], **kwargs)
        eval_env = make_dmc_env(env_name, params["action_cost"], **kwargs)
    elif "MountainCar" in env_name:
        # We normalize rewards for mountain car env
        reward_scale = params["reward_scale"]
        f = lambda r: reward_scale * r
        env = TransformReward(gym.make(env_name, **kwargs), f)
        eval_env = TransformReward(gym.make(env_name, **kwargs), f)
    else:
        env = gym.make(env_name, **kwargs)
        eval_env = gym.make(env_name, **kwargs)

    return env, eval_env


def make_dmc_env(env_name, action_cost, **kwargs):
    env = gym.make(env_name, **kwargs)
    env = ActionCostWrapper(FlattenObservation(env), action_cost)
    return env


def get_term_fn(env_name: str):
    if env_name == "MountainCarContinuous-v0":
        return custom_term_fns.continuous_mountain_car
    elif "dm_control" in env_name:
        return term_fns.no_termination
    else:
        return getattr(term_fns, ENV_NAME_TO_TERM_FN[env_name], term_fns.no_termination)


def get_reward_fn(env_name: str):
    raise ValueError(f"No reward function defined for {env_name}")


def set_device_in_hydra_cfg(device: str, cfg: dict):
    cfg["dynamics_model"]["device"] = str(device)
    cfg["dynamics_model"]["member_cfg"]["device"] = str(device)


def ensemble_to_envs(
    ensemble: Ensemble,
    env: gym.Env,
    rng: torch.Generator,
    agent_type: str,
    reward_fn: dist_mbrl.mbrl.types.RewardFnType = None,
    termination_fn: Optional[dist_mbrl.mbrl.types.TermFnType] = term_fns.no_termination,
) -> List[ModelEnv]:
    """
    Take an ensemble of transition models and return a list of gym-like environments
    that we use to collect data. Internally, the gym environment will use the model
    dynamics to take steps, and use the reward function provided to calculate rewards.
    """
    # Unpack model config to replicate it for each member of the ensemble
    target_is_delta = ensemble.target_is_delta
    normalize = ensemble.input_normalizer is not None
    learned_rewards = ensemble.learned_rewards
    num_elites = ensemble.num_elites

    env_list = []

    if agent_type not in (
        AgentTypes.MBPO,
        AgentTypes.QRMBPO,
        AgentTypes.QRMBPO_BIG,
        AgentTypes.QRMBPO_LOSS,
    ):
        for member in ensemble.model:
            member.device = ensemble.device
            wrapped_model = OneDTransitionRewardModel(
                member,
                target_is_delta=target_is_delta,
                normalize=normalize,
                learned_rewards=learned_rewards,
                num_elites=num_elites,
            )
            wrapped_model.input_normalizer = ensemble.input_normalizer
            env_list.append(
                ModelEnv(env, wrapped_model, termination_fn, reward_fn, rng)
            )

    # Add an additional environment that randomly selects a model to pick the next state
    # prediction
    if agent_type not in (AgentTypes.EQRSAC, AgentTypes.QRMBPO_CONSISTENT):
        env_list.append(ModelEnv(env, ensemble, termination_fn, reward_fn, rng))
    return env_list


def create_model_buffers(
    agent_type: str,
    ensemble_size: int,
    base_capacity: int,
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    rng: Optional[np.random.Generator] = None,
) -> ModelBufferType:
    if agent_type in (
        AgentTypes.MBPO,
        AgentTypes.QRMBPO,
        AgentTypes.QRMBPO_LOSS,
    ):
        buffer = ReplayBuffer(base_capacity, obs_shape, act_shape, rng=rng)
    elif agent_type in (
        AgentTypes.EQRSAC,
        AgentTypes.QRMBPO_BIG,
        AgentTypes.QRMBPO_CONSISTENT,
    ):
        buffer = ReplayBuffer(
            base_capacity * ensemble_size, obs_shape, act_shape, rng=rng
        )
    elif agent_type == AgentTypes.QUSAC:
        buffer = [
            ReplayBuffer(base_capacity, obs_shape, act_shape, rng=rng)
            for _ in range(ensemble_size + 1)
        ]
    return buffer


def train_model(
    model: Model,
    model_trainer: ModelTrainer,
    replay_buffer: ReplayBuffer,
    train_params: dict,
):
    dataset_train, dataset_val = mbrl_utils.get_basic_buffer_iterators(
        replay_buffer,
        batch_size=train_params["batch_size"],
        val_ratio=train_params["validation_ratio"],
        ensemble_size=len(model),
        shuffle_each_epoch=True,
        bootstrap_permutes=False,
    )
    model.update_normalizer(replay_buffer.get_all())
    model_trainer.train(
        dataset_train,
        dataset_val,
        patience=train_params["patience"],
    )


def step_env_and_add_to_buffer(
    env: gym.Env,
    obs: np.ndarray,
    agent: dist_mbrl.mbrl.planning.Agent,
    agent_kwargs: Dict,
    replay_buffer: ReplayBuffer,
) -> Tuple[np.ndarray, float, bool, Dict]:
    action = agent.act(obs, **agent_kwargs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    # We ignore the time termination signal for the Buffer transitions
    replay_buffer.add(obs, action, next_obs, reward, terminated)
    # We terminate episodes if we reach the time limit or the terminated signal is True
    returned_termination = terminated or truncated
    return next_obs, reward, returned_termination, info


def evaluate(
    agent: Agent, env: gym.Env, dir: str, num_episodes: int, max_steps: int
) -> dict:
    """
    Evaluate agent in a given RL environment for a number of episodes. In the first
    evaluation episode, the method saves a video of the agent's performance in the given
    directory.
    """
    video_recorder = VideoRecorder(dir)
    video_recorder.init(enabled=True)
    episode_returns = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        agent.reset()
        video_recorder.init(enabled=(episode == 0))
        terminated = False
        episode_return = 0
        step = 0
        while not (terminated or step >= max_steps):
            action = agent.act(obs, sample=False)
            obs, reward, terminated, *_ = env.step(action)
            video_recorder.record_default(env)
            episode_return += reward
            step += 1

        episode_returns.append(episode_return)
        video_recorder.save("agent.mp4")

    return {
        "avg_return": np.mean(episode_returns),
    }
