# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Single-stream environment."""

from typing import Any

import chex
import dm_env
import numpy as np

from disco_rl import types
from disco_rl.environments import base


def _to_env_timestep(timestep: dm_env.TimeStep) -> types.EnvironmentTimestep:
  return types.EnvironmentTimestep(
      step_type=np.array(timestep.step_type, dtype=np.int32),
      reward=np.array(timestep.reward or 0.0),
      observation=timestep.observation,
  )


class UnusedEnvState:
  pass


class SingleStreamEnv(base.Environment):
  """Wrapper for a single-stream environment."""

  def __init__(self, env: Any):
    self._env = env

  def reset(
      self, rng_key: chex.PRNGKey
  ) -> tuple[UnusedEnvState | None, types.EnvironmentTimestep]:
    """Resets the environment."""
    del rng_key
    return UnusedEnvState(), _to_env_timestep(self._env.reset())

  def step(
      self, state: UnusedEnvState | None, action: int
  ) -> tuple[UnusedEnvState | None, types.EnvironmentTimestep]:
    """Steps the environment."""
    del state
    ts = _to_env_timestep(self._env.step(action))

    # Reset terminal episodes.
    if ts.step_type == dm_env.StepType.LAST:
      # Step the terminated episodes once again to switch to the next episodes.
      ts_start = _to_env_timestep(self._env.step(action))

      # Recover step_type and rewards from the terminal states.
      ts_start.step_type = ts.step_type
      ts_start.reward = ts.reward

      return UnusedEnvState(), ts_start
    else:
      return UnusedEnvState(), ts

  def single_observation_spec(self) -> types.Specs:
    """Returns the observation spec."""
    return self._env.observation_spec()

  def single_action_spec(self) -> types.ActionSpec:
    """Returns the action spec."""
    return self._env.action_spec()
