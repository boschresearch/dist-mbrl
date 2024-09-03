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

base_default_parameters = dict(
    seed=0,
    num_episodes=1000,
    transition_probability=1.0,
    opt_value=19.0,
    agent=dict(
        gamma=0.99,
        model_type="normal",
        max_pi_steps=40,
        transition_repeat=1,
        model_prior=dict(mu=2.0, kappa=1.0, alpha=10.0, beta=100.0),
    ),
)

psrl_default_parameters = {**base_default_parameters, **dict(agent_type="psrl")}
