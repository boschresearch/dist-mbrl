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

import os

import torch
from hydra import compose, initialize

import dist_mbrl.mbrl.models as models
import dist_mbrl.mbrl.util.common as mbrl_utils
import dist_mbrl.utils.common as utils_common
import dist_mbrl.utils.data_collection as utils_data
from dist_mbrl.config import MODEL_BASED_TYPE, DefaultExperiment
from dist_mbrl.mbrl.util import ReplayBuffer
from dist_mbrl.utils.video import close_virtual_display


def train(parameters: dict):
    env, eval_env = utils_common.get_env(parameters["env"])
    env_name = env.unwrapped.spec.id
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Fix RNG
    seed = parameters["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng, generator = utils_common.fix_rng(
        env, eval_env=eval_env, seed=seed, device=device
    )

    env_buffer_capacity = int(parameters["env_buffer_capacity"])
    device = torch.device(device)

    # Environment replay buffer
    env_replay_buffer = ReplayBuffer(env_buffer_capacity, obs_shape, act_shape, rng=rng)

    # Instantiate dynamics model for model-based approaches
    agent_params = parameters["agent"]
    agent_type = agent_params["type"]
    agent_kwargs = {}
    if agent_type in MODEL_BASED_TYPE:
        model_params = parameters["dynamics_model"]
        initialize(version_base=None, config_path="../config")
        cfg = compose(config_name="mbrl_lib.yaml")
        cfg["algorithm"]["learned_rewards"] = model_params["learned_rewards"]
        cfg["dynamics_model"]["ensemble_size"] = model_params["ensemble_size"]
        utils_common.set_device_in_hydra_cfg(device, cfg)
        dynamics_model = mbrl_utils.create_one_dim_tr_model(cfg, obs_shape, act_shape)
        model_lr = model_params["trainer"]["learning_rate"]
        model_wd = model_params["trainer"]["weight_decay"]
        model_trainer = models.ModelTrainer(
            dynamics_model, optim_lr=model_lr, weight_decay=model_wd
        )

        # Create gym-like wrapper to rollout the model
        term_fn = utils_common.get_term_fn(env_name)
        reward_fn = (
            None
            if model_params["learned_rewards"]
            else utils_common.get_reward_fn(env_name)
        )
        ensemble_envs = utils_common.ensemble_to_envs(
            dynamics_model,
            env,
            generator,
            agent_type,
            reward_fn=reward_fn,
            termination_fn=term_fn,
        )

        # Define buffer to store model-based rollouts
        num_model_rollouts_per_step = model_params["num_rollouts_per_step"]
        model_rollout_length = model_params["rollout_length"]
        freq_model_retrain = model_params["freq_retrain"]
        rollout_batch_size = num_model_rollouts_per_step * freq_model_retrain
        num_updates_to_retain_buffer = model_params["num_updates_to_retain_buffer"]
        base_capacity = (
            model_rollout_length * rollout_batch_size * num_updates_to_retain_buffer
        )
        model_buffers = utils_common.create_model_buffers(
            agent_type, len(dynamics_model), base_capacity, obs_shape, act_shape, rng
        )

        # Extra arguments to instantiate model-based agents
        agent_kwargs = {
            "dynamics_model": dynamics_model,
            "reward_fn": reward_fn,
            "termination_fn": term_fn,
            "ensemble_size": len(dynamics_model),
        }

    # Instantiate the agent that will be trained
    agent = utils_common.AGENT_DICT[agent_type](
        env, device, agent_params, **agent_kwargs
    )
    # Define which buffer the agent uses for training
    buffer_train_data = (
        model_buffers if agent_type in MODEL_BASED_TYPE else env_replay_buffer
    )

    # Pre-fill the env replay buffer with data from a random agent
    buffer_init_steps = parameters["buffer_init_steps"]
    mbrl_utils.rollout_agent_trajectories(
        env, buffer_init_steps, agent, {}, replay_buffer=env_replay_buffer
    )

    # Directory for creating agent checkpoints
    artifact_dir = os.getcwd()
    checkpoints_dir = os.path.join(artifact_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Agent training loop
    num_steps = int(parameters["num_steps"])
    steps_per_epoch = int(parameters["steps_per_epoch"])
    global_step = 0
    agent_updates = 1

    while global_step <= num_steps:
        obs, _ = env.reset()
        terminated = False
        ep_step = 0
        while not (terminated or ep_step >= steps_per_epoch):
            # Take an env step
            next_obs, reward, terminated, _ = utils_common.step_env_and_add_to_buffer(
                env, obs, agent, {}, env_replay_buffer
            )

            # Retrain model after desired number of steps
            if agent_type in MODEL_BASED_TYPE:
                if global_step % model_params["freq_retrain"] == 0:
                    utils_common.train_model(
                        dynamics_model,
                        model_trainer,
                        env_replay_buffer,
                        model_params["trainer"],
                    )
                    utils_data.collect_ensemble_model_transitions(
                        ensemble_envs,
                        agent,
                        model_buffers,
                        env_replay_buffer,
                        model_rollout_length,
                        rollout_batch_size,
                    )

            for _ in range(parameters["agent_updates_per_step"]):
                agent.update_params(buffer_train_data, agent_updates)
                agent_updates += 1

            # Evaluate agent
            if (global_step + 1) % parameters["freq_agent_eval"] == 0:
                epoch_dir = os.path.join(checkpoints_dir, f"step_{global_step+1}")
                os.makedirs(epoch_dir, exist_ok=True)
                eval_dict = utils_common.evaluate(
                    agent,
                    eval_env,
                    epoch_dir,
                    num_episodes=parameters["eval_episodes"],
                    max_steps=steps_per_epoch,
                )
                print(
                    f'Return = {eval_dict["avg_return"]:.3f} @ Env step {global_step + 1}'
                )

            obs = next_obs
            ep_step += 1
            global_step += 1
            if global_step > num_steps:
                break

    # Close env and pyvirtual display before exiting
    env.close()
    eval_env.close()
    close_virtual_display()


if __name__ == "__main__":
    train(DefaultExperiment.EQRSAC)
