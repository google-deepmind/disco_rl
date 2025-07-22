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

"""Jittable environments."""

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from disco_rl.environments.wrappers import batched_jittable_env


class _SingleStreamCatch:
  """Catch with lifetime reset."""

  def __init__(self, rows: int = 10, columns: int = 5):
    self._rows = rows
    self._columns = columns

  @property
  def num_actions(self) -> int:
    return 3

  def initial_state(self, rng):
    ball_y = 0
    ball_x = jax.random.randint(rng, (), 0, self._columns)
    paddle_y = self._rows - 1
    paddle_x = self._columns // 2
    return jnp.array((ball_y, ball_x, paddle_y, paddle_x), dtype=jnp.int32)

  def episode_reset(self, rng, state):
    del state
    return self.initial_state(rng)

  def step(self, rng, state, action):
    del rng
    paddle_x = jnp.clip(state[3] + action - 1, 0, self._columns - 1)
    return jnp.array([state[0] + 1, state[1], state[2], paddle_x])

  def is_terminal(self, state):
    return state[0] == self._rows - 1

  def render(self, state):
    """Render the screen."""

    def f(y, x):
      return jax.lax.select(
          jnp.bitwise_or(
              jnp.bitwise_and(y == state[0], x == state[1]),
              jnp.bitwise_and(y == state[2], x == state[3]),
          ),
          1.0,
          0.0,
      )

    y_board = jnp.repeat(jnp.arange(self._rows), self._columns)
    x_board = jnp.tile(jnp.arange(self._columns), self._rows)
    return jax.vmap(f)(y_board, x_board).reshape((self._rows, self._columns, 1))

  def reward(self, state):
    return jax.lax.select(
        state[0] == self._rows - 1,
        jax.lax.select(state[1] == state[3], 1.0, -1.0),
        0.0,
    )


class CatchJittableEnvironment(batched_jittable_env.BatchedJittableEnvironment):
  """Catch environment."""

  def __init__(
      self,
      batch_size: int,
      env_settings: config_dict.ConfigDict,
  ) -> None:

    super().__init__(
        _SingleStreamCatch,
        batch_size,
        env_settings,
    )


def get_config_catch() -> config_dict.ConfigDict:
  """Returns default config for CatchEnvironment."""
  return config_dict.ConfigDict(
      dict(
          rows=8,
          columns=8,
      )
  )
