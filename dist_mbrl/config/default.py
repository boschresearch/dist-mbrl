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

from copy import deepcopy
from enum import Enum


# AGENT TYPES
class AgentTypes(str, Enum):
    SAC = "sac"
    QRSAC = "qrsac"
    MBPO = "mbpo"
    QRMBPO = "qrmbpo"
    QUSAC = "qusac"
    EQRSAC = "eqrsac"
    QRMBPO_BIG = "qrmbpo-big"
    QRMBPO_CONSISTENT = "qrmbpo-consistent"
    QRMBPO_LOSS = "qrmbpo-loss"

    def __str__(self) -> str:
        return self.name


MODEL_BASED_TYPE = [
    AgentTypes.MBPO,
    AgentTypes.QRMBPO,
    AgentTypes.QUSAC,
    AgentTypes.EQRSAC,
    AgentTypes.QRMBPO_BIG,
    AgentTypes.QRMBPO_CONSISTENT,
    AgentTypes.QRMBPO_LOSS,
]

# BASE FOR ALL ALGORITHMS
BASE_EXPERIMENT_CONFIG = dict(
    seed=0,
    env=None,
    num_steps=100,
    env_buffer_capacity=8000,
    agent_updates_per_step=1,
    buffer_init_steps=1000,
    freq_agent_eval=10,
    steps_per_epoch=400,
    eval_episodes=1,
    agent=None,
    dynamics_model=None,
)

# ENV BASE CONFIGS
gym_baseconfig = dict(name="MountainCarContinuous-v0", reward_scale=1.0)
mujoco_baseconfig = dict(
    name="HalfCheetah-v4",
    kwargs=dict(
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
    ),
)
dm_control_baseconfig = dict(
    name="dm_control/cartpole-swingup_sparse-v0",
    kwargs=dict(render_height=480, render_width=480),
    action_cost=0.0,
)

# AGENT CONFIGS

# SAC BASE
sac_baseconfig = dict(
    type=AgentTypes.SAC.value,
    gamma=0.99,
    alpha_temp=0.2,
    smoothness_coef=0.005,
    auto_entropy_tuning=True,
    clip_grad_norm=False,
    target_entropy=-1,
    batch_size=256,
    target_update_freq=1,
    actor_update_freq=1,
    critics_per_model=1,
    actor=dict(
        num_layers=2,
        hid_size=64,
        activation_fn=dict(_target_="torch.nn.Tanh"),
        learning_rate=3e-4,
    ),
    critic=dict(
        num_layers=2,
        hid_size=256,
        activation_fn=dict(_target_="torch.nn.Tanh"),
        learning_rate=3e-4,
    ),
)

# QRSAC BASE
default_quantile_critic = dict(sac_baseconfig["critic"], **dict(num_quantiles=3))
qrsac_baseconfig = {
    **sac_baseconfig,
    **dict(
        type=AgentTypes.QRSAC.value,
        utility_type="quantile",
        opt_quantile=50,
        critic=default_quantile_critic,
    ),
}

# DEFAULT DYNAMIC MODEL CONFIG
DYNAMICS_BASE = dict(
    learned_rewards=True,
    ensemble_size=5,
    trainer=dict(
        learning_rate=3e-4,
        weight_decay=5e-5,
        batch_size=256,
        validation_ratio=0.2,
        patience=5,
        num_epochs=10,
    ),
    num_updates_to_retain_buffer=1,
    freq_retrain=200,
    num_rollouts_per_step=100,
    rollout_length=1,
    rollout_mode="random_model",
)

# DEFAULT QUSAC
qusac_baseconfig = {
    **sac_baseconfig,
    **dict(
        type=AgentTypes.QUSAC.value,
        critics_per_model=1,
    ),
}

# DEFAULT MBPO
mbpo_baseconfig = {
    **sac_baseconfig,
    **dict(type=AgentTypes.MBPO.value, critics_per_model=2),
}

# DEFAULT QR-MBPO
qrmbpo_baseconfig = {
    **qrsac_baseconfig,
    **dict(type=AgentTypes.QRMBPO.value),
}

# QR-MBPO VARIANTS
qrmbpo_big_baseconfig = deepcopy(qrmbpo_baseconfig)
qrmbpo_big_baseconfig["type"] = AgentTypes.QRMBPO_BIG.value

qrmbpo_consistent_baseconfig = deepcopy(qrmbpo_baseconfig)
qrmbpo_consistent_baseconfig["type"] = AgentTypes.QRMBPO_CONSISTENT.value

# DEFAULT EQRSAC
eqrsac_baseconfig = {
    **qrsac_baseconfig,
    **dict(type=AgentTypes.EQRSAC.value, num_state_samples=5, num_act_samples=5),
}

qrmbpo_loss_baseconfig = deepcopy(eqrsac_baseconfig)
qrmbpo_loss_baseconfig["type"] = AgentTypes.QRMBPO_LOSS.value


class DefaultEnvParams(dict, Enum):
    GYM = gym_baseconfig
    MUJOCO = mujoco_baseconfig
    DM_CONTROL = dm_control_baseconfig


class DefaultAgentParams(dict, Enum):
    SAC = sac_baseconfig
    QRSAC = qrsac_baseconfig
    QUSAC = qusac_baseconfig
    MBPO = mbpo_baseconfig
    QRMBPO = qrmbpo_baseconfig
    QRMBPO_BIG = qrmbpo_big_baseconfig
    QRMBPO_CONSISTENT = qrmbpo_consistent_baseconfig
    QRMBPO_LOSS = qrmbpo_loss_baseconfig
    EQRSAC = eqrsac_baseconfig


# Default env
env_config = DefaultEnvParams.DM_CONTROL


class DefaultExperiment(dict, Enum):
    SAC = dict(BASE_EXPERIMENT_CONFIG, env=env_config, agent=sac_baseconfig)
    QRSAC = dict(BASE_EXPERIMENT_CONFIG, env=env_config, agent=qrsac_baseconfig)
    QUSAC = dict(
        BASE_EXPERIMENT_CONFIG,
        env=env_config,
        dynamics_model=DYNAMICS_BASE,
        agent=qusac_baseconfig,
    )
    MBPO = dict(
        BASE_EXPERIMENT_CONFIG,
        env=env_config,
        dynamics_model=DYNAMICS_BASE,
        agent=mbpo_baseconfig,
    )
    QRMBPO = dict(
        BASE_EXPERIMENT_CONFIG,
        env=env_config,
        dynamics_model=DYNAMICS_BASE,
        agent=qrmbpo_baseconfig,
    )
    QRMBPO_BIG = dict(
        BASE_EXPERIMENT_CONFIG,
        env=env_config,
        dynamics_model=DYNAMICS_BASE,
        agent=qrmbpo_big_baseconfig,
    )
    QRMBPO_CONSISTENT = dict(
        BASE_EXPERIMENT_CONFIG,
        env=env_config,
        dynamics_model=DYNAMICS_BASE,
        agent=qrmbpo_consistent_baseconfig,
    )
    QRMBPO_LOSS = dict(
        BASE_EXPERIMENT_CONFIG,
        env=env_config,
        dynamics_model=DYNAMICS_BASE,
        agent=qrmbpo_loss_baseconfig,
    )
    EQRSAC = dict(
        BASE_EXPERIMENT_CONFIG,
        env=env_config,
        dynamics_model=DYNAMICS_BASE,
        agent=eqrsac_baseconfig,
    )
