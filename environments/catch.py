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

"""Catch environment."""

import dm_env
from dm_env import specs
from ml_collections import config_dict as configdict
import numpy as np

from disco_rl.environments.wrappers import batched_env
from disco_rl.environments.wrappers import single_stream_env

_ACTIONS = (-1, 0, 1)  # Left, no-op, right.


class SingleStreamCatch:
  """A Catch environment built on the `dm_env.Environment` class."""

  def __init__(self, env_settings: configdict.ConfigDict):
    """Initializes a new Catch environment."""
    self._rows = env_settings.rows
    self._columns = env_settings.columns
    self._rng = np.random.RandomState(env_settings.random_seed)
    self._board = np.zeros((self._rows, self._columns), dtype=np.float32)
    self._ball_x = None
    self._ball_y = None
    self._paddle_x = None
    self._paddle_y = self._rows - 1
    self._reset_next_step = True

  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""
    self._reset_next_step = False
    self._ball_x = self._rng.randint(self._columns)
    self._ball_y = 0
    self._paddle_x = self._columns // 2
    return dm_env.restart(self._observation())

  def step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    if self._reset_next_step:
      return self.reset()

    # Move the paddle.
    dx = _ACTIONS[action]
    self._paddle_x = np.clip(self._paddle_x + dx, 0, self._columns - 1)

    # Drop the ball.
    self._ball_y += 1

    # Check for termination.
    if self._ball_y == self._paddle_y:
      reward = 1.0 if self._paddle_x == self._ball_x else -1.0
      self._reset_next_step = True
      return dm_env.termination(reward=reward, observation=self._observation())
    else:
      return dm_env.transition(reward=0.0, observation=self._observation())

  def observation_spec(self) -> specs.Array:
    """Returns the observation spec."""
    return specs.Array(
        shape=self._board.shape,
        dtype=self._board.dtype,
        name="board",
    )

  def action_spec(self) -> specs.BoundedArray:
    """Returns the action spec."""
    return specs.BoundedArray((), np.int32, 0, len(_ACTIONS) - 1)

  def _observation(self) -> np.ndarray:
    self._board.fill(0.0)
    self._board[self._ball_y, self._ball_x] = 1.0
    self._board[self._paddle_y, self._paddle_x] = 1.0
    return self._board.copy()


class CatchEnvironment(batched_env.BatchedSingleStreamEnvironment):
  """A batched Catch environment."""

  def __init__(
      self,
      batch_size: int,
      env_settings: configdict.ConfigDict,
  ) -> None:

    def _single_stream_catch(
        env_settings: configdict.ConfigDict,
    ) -> single_stream_env.SingleStreamEnv:
      return single_stream_env.SingleStreamEnv(
          env=SingleStreamCatch(env_settings)
      )

    super().__init__(
        _single_stream_catch,
        batch_size,
        env_settings,
    )


def get_config() -> configdict.ConfigDict:
  """Returns default config for CatchEnvironment."""
  return configdict.ConfigDict(
      dict(
          rows=8,
          columns=8,
          random_seed=1,
      )
  )
