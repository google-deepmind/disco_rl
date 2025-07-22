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

"""Models for conditioning on actions."""

import chex
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np

from disco_rl import types
from disco_rl import utils


def get_action_model(name: str, *args, **kwargs):
  if name == 'lstm':
    net = LSTMModel(*args, **kwargs)
  else:
    raise ValueError(f'Invalid model network name {name}.')
  return net


class LSTMModel:
  """LSTM-based action-conditional model inspried by Muesli/MuZero."""

  def __init__(
      self,
      action_spec: types.ActionSpec,
      out_spec: types.Specs,
      head_mlp_hiddens: tuple[int, ...],
      lstm_size: int,
  ) -> None:
    self._out_spec = out_spec
    self._action_spec = action_spec
    self._head_mlp_hiddens = head_mlp_hiddens
    self._lstm_size = lstm_size

  def _model_transition_all_actions(
      self, embedding: hk.LSTMState
  ) -> chex.Array:
    """Performs a model transition pass for all actions."""
    num_actions = utils.get_num_actions_from_spec(self._action_spec)
    batch_size = embedding.cell.shape[0]

    # Enumerate all action embeddings.
    one_hot_actions = jnp.eye(num_actions).astype(
        embedding.cell.dtype
    )  # [A, A]
    batched_one_hot_actions = jnp.tile(
        one_hot_actions, [batch_size, 1]
    )  # [BA, A]

    all_actions_embed = jax.tree.map(
        lambda x: jnp.repeat(x, repeats=num_actions, axis=0), embedding
    )  # [BA, *H]

    lstm_output, _ = hk.LSTM(self._lstm_size, name='action_cond')(
        batched_one_hot_actions, all_actions_embed
    )
    return lstm_output

  def _model_head_pass(
      self, transition_output: chex.Array
  ) -> dict[str, chex.Array]:
    """Gets outputs from MLPs with shapes according to out spec."""
    # transition_output has shape [BA, ...]
    num_actions = utils.get_num_actions_from_spec(self._action_spec)
    batch_size = transition_output.shape[0] // num_actions

    model_outputs = dict()
    for key, pred_spec in self._out_spec.items():
      pred = hk.nets.MLP(self._head_mlp_hiddens + (np.prod(pred_spec.shape),))(
          transition_output
      )
      model_outputs[key] = pred.reshape(
          (batch_size, num_actions, *pred_spec.shape)
      )

    return model_outputs

  def model_step(self, embedding: hk.LSTMState) -> dict[str, chex.Array]:
    """Steps model."""
    transition_output = self._model_transition_all_actions(embedding)
    model_outputs = self._model_head_pass(transition_output)
    return model_outputs

  def root_embedding(self, state: chex.Array) -> hk.LSTMState:
    """Constructs a root node from agent's state."""
    flat_state = hk.Flatten()(state)
    cell = hk.Linear(self._lstm_size)(flat_state)
    return hk.LSTMState(hidden=jnp.tanh(cell), cell=cell)
