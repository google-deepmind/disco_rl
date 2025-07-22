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

"""Base interface for environments."""

import abc
from typing import Any, TypeVar

import chex

from disco_rl import types
from disco_rl import utils

_EnvState = TypeVar('_EnvState')


class Environment(abc.ABC):
  """Interface for environments.

  All environments are supposed to be batched.
  """

  batch_size: int

  @abc.abstractmethod
  def step(
      self, states: _EnvState, actions: chex.ArrayTree
  ) -> tuple[_EnvState, types.EnvironmentTimestep]:
    pass

  @abc.abstractmethod
  def reset(
      self, rng_key: chex.PRNGKey
  ) -> tuple[Any, types.EnvironmentTimestep]:
    """Resets episodes."""
    pass

  @abc.abstractmethod
  def single_observation_spec(self) -> types.Specs:
    pass

  @abc.abstractmethod
  def single_action_spec(self) -> types.ActionSpec:
    pass

  def batched_action_spec(self) -> types.ActionSpec:
    return utils.broadcast_specs(self.single_action_spec(), self.batch_size)

  def batched_observation_spec(self) -> types.Specs:
    return utils.broadcast_specs(
        self.single_observation_spec(), self.batch_size
    )
