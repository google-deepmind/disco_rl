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

"""Meta networks used by update rules."""

import functools
from typing import Any, Callable, Mapping, Sequence

import chex
import haiku as hk
from haiku import initializers as hk_init
import jax
from jax import lax
from jax import numpy as jnp

from disco_rl import types
from disco_rl import utils
from disco_rl.update_rules import input_transforms


class MetaNet(hk.Module):
  """Meta Network base class."""

  def __call__(
      self,
      inputs: types.UpdateRuleInputs,
      axis_name: str | None,
  ) -> types.UpdateRuleOuts:
    """Produces outputs needed for update rule's agent loss."""
    raise NotImplementedError


class LSTM(MetaNet):
  """Meta network with LSTMs."""

  def __init__(
      self,
      hidden_size: int,
      embedding_size: Sequence[int],
      prediction_size: int,
      meta_rnn_kwargs: Mapping[str, Any],
      input_option: types.MetaNetInputOption,
      policy_channels: Sequence[int] = (16, 2),
      policy_target_channels: Sequence[int] = (128,),
      policy_target_stddev: float | None = None,
      output_stddev: float | None = None,
      aux_stddev: float | None = None,
      state_stddev: float | None = None,
      name: str | None = None,
  ) -> None:
    super().__init__(name=name)
    self._hidden_size = hidden_size
    self._embedding_size = embedding_size
    self._prediction_size = prediction_size
    self._input_option = input_option
    self._output_init = _maybe_get_initializer(output_stddev)
    self._aux_init = _maybe_get_initializer(aux_stddev)
    self._state_init = _maybe_get_initializer(state_stddev)
    self._policy_target_init = _maybe_get_initializer(policy_target_stddev)
    self._policy_channels = policy_channels
    self._policy_target_channels = policy_target_channels
    self._meta_rnn_core: 'MetaLSTM' = MetaLSTM(
        **meta_rnn_kwargs, input_option=input_option
    )

  def __call__(
      self, inputs: types.UpdateRuleInputs, axis_name: str | None
  ) -> types.UpdateRuleOuts:
    # Initialize or extract the meta RNN core state.
    initial_meta_rnn_state = self._meta_rnn_core.initial_state()
    meta_rnn_state = hk.get_state(
        'meta_rnn_state',
        shape=jax.tree.map(lambda t: t.shape, initial_meta_rnn_state),
        dtype=jax.tree.map(lambda t: t.dtype, initial_meta_rnn_state),
        init=lambda *_: initial_meta_rnn_state,
    )
    assert isinstance(meta_rnn_state, hk.LSTMState)

    # Inputs have shapes of [T+1, B, ...]
    logits = inputs.agent_out['logits']
    assert isinstance(logits, chex.Array)
    _, batch_size, num_actions = logits.shape

    # Construct inputs for the meta network.
    y_net = _batch_mlp(self._embedding_size, num_dims=2)
    z_net = _batch_mlp(self._embedding_size, num_dims=3)
    policy_net = _conv1d_net(self._policy_channels)
    x, policy_emb = _construct_input(
        inputs,
        y_net=y_net,
        z_net=z_net,
        policy_net=policy_net,
        input_option=self._input_option,
        axis_name=axis_name,
    )

    # Unroll the per-trajectory RNN core in reverse direction for bootstrapping.
    per_trajectory_rnn_core = hk.ResetCore(hk.LSTM(self._hidden_size))
    should_reset_bwd = inputs.should_reset_mask_bwd[:-1]  # [T, B]
    x, _ = hk.dynamic_unroll(  # [T, B, H]
        per_trajectory_rnn_core,
        (x, should_reset_bwd),
        per_trajectory_rnn_core.initial_state(batch_size=batch_size),
        reverse=True,
    )

    # Perform multipl-ve interaction with the (per-lifetime) meta RNN's outputs.
    x = _multiplicative_interaction(
        x=x,
        y=self._meta_rnn_core.output(meta_rnn_state),
        initializer=self._state_init,
    )

    # Compute an additional input embedding for the meta network unrolling.
    meta_input_emb = hk.BatchApply(hk.Linear(1, w_init=self._output_init))(
        x
    )  # [T, B, 1]

    # Compute the y, z targets.
    y_hat = hk.BatchApply(
        hk.Linear(self._prediction_size, w_init=self._aux_init)
    )(x)
    z_hat = hk.BatchApply(
        hk.Linear(self._prediction_size, w_init=self._aux_init)
    )(x)

    # Compute the policy target (pi).
    w = jnp.repeat(jnp.expand_dims(x, 2), num_actions, axis=2)  # [T, B, A, H]
    w = jnp.concatenate([w, policy_emb], axis=-1)  # [T, B, A, H+C]
    w = _conv1d_net(self._policy_target_channels)(w)  # [T, B, A, O]
    w = hk.BatchApply(hk.Linear(1, w_init=self._policy_target_init))(
        w
    )  # [T, B, A, 1]
    pi_hat = jnp.squeeze(w, -1)  # [T, B, A]

    # Set the meta network outputs.
    meta_out = dict(pi=pi_hat, y=y_hat, z=z_hat, meta_input_emb=meta_input_emb)

    # Unroll the meta RNN core and update its state.
    new_meta_rnn_state = self._meta_rnn_core.unroll(
        inputs, meta_out, meta_rnn_state, axis_name=axis_name
    )
    hk.set_state('meta_rnn_state', new_meta_rnn_state)

    return meta_out


class MetaLSTM(hk.Module):
  """A meta LSTM that processes trajectories and meta targets throughout the agent's lifetime."""

  def __init__(
      self,
      input_option: types.MetaNetInputOption,
      policy_channels: Sequence[int],
      embedding_size: Sequence[int],
      pred_embedding_size: Sequence[int],
      hidden_size: int,
  ):
    super().__init__()
    self._input_option = input_option
    self._hidden_size = hidden_size
    self._embedding_size = embedding_size
    self._policy_channels = policy_channels
    self._pred_embedding_size = pred_embedding_size
    self._core_constructor = lambda: hk.LSTM(self._hidden_size)

  def unroll(
      self,
      inputs: types.UpdateRuleInputs,
      meta_out: types.UpdateRuleOuts,
      state: hk.LSTMState,
      axis_name: str | None,
  ) -> hk.LSTMState:
    """Updates meta_state given a rollout and a rnn_state."""

    # Get meta inputs.
    y_net = _batch_mlp(self._pred_embedding_size, num_dims=2)
    z_net = _batch_mlp(self._pred_embedding_size, num_dims=3)
    policy_net = _conv1d_net(self._policy_channels)
    meta_inputs, _ = _construct_input(  # [T, B, ...]
        inputs,
        y_net=y_net,
        z_net=z_net,
        policy_net=policy_net,
        input_option=self._input_option,
        axis_name=axis_name,
    )
    input_list = [
        meta_inputs,
        meta_out['meta_input_emb'],
        y_net(jax.nn.softmax(meta_out['y'])),
    ]

    # Concatenate & embed all inputs.
    x = jnp.concatenate(input_list, axis=-1)  # [T, B, ...]
    x = _batch_mlp(self._embedding_size)(x)  # [T, B, E]

    # Apply average pooling over batch-time dimensions.
    x_avg = x.mean(axis=(0, 1))  # [E]
    if axis_name is not None:
      x_avg = jax.lax.pmean(x_avg, axis_name=axis_name)

    # Unroll the meta RNN core and update its state.
    core = self._core_constructor()
    _, new_state = core(x_avg, state)
    return new_state

  def initial_state(self) -> hk.LSTMState:
    """Returns an initial a rnn_state."""
    return self._core_constructor().initial_state(batch_size=None)

  def output(self, state: hk.LSTMState) -> chex.Array:
    """Extracts an output vector from a rnn_state."""
    return state.hidden  # pytype: disable=attribute-error  # numpy-scalars


def _multi_level_extract_by_attr_or_key(x: Any, keys: str) -> Any:
  """Returns `x[k0][k1]...[kn]` where `keys` is of the form `k0[/k1]`."""

  # Note that the keys can also be attributes of `x`.
  # A simple usage example: assert extract({'a': {'b': {'c': 3}}}, 'a/b/c') == 3

  def _get_attr_or_key(x: Any, key: str, keys: str) -> Any:
    if hasattr(x, key):
      return getattr(x, key)
    else:
      try:
        return x[key]
      except:
        raise KeyError(f'Input {x} has no attr or key {key}. {keys}') from None

  processed_keys = []
  for key in keys.split('/'):
    if x is None:
      raise ValueError(
          f'x/{"/".join(processed_keys)} is `None`, cannot recurse up to'
          f' x/{keys}'
      )
    x = _get_attr_or_key(x, key, keys)
    processed_keys.append(key)
  return x


def _construct_input(
    inputs: types.UpdateRuleInputs,
    input_option: types.MetaNetInputOption,
    y_net: Callable[[chex.Array], chex.Array],
    z_net: Callable[[chex.Array], chex.Array],
    policy_net: Callable[[chex.Array], chex.Array],
    axis_name: str | None = None,
) -> tuple[chex.Array, chex.Array | None]:
  """Maps update rule inputs to a single vector."""
  unroll_len, batch_size = inputs.is_terminal.shape

  actions = jax.tree.map(lambda x: x[:-1], inputs.actions)  # [T, B]
  policy = lax.stop_gradient(
      jax.nn.softmax(inputs.agent_out['logits'])
  )  # [T+1, B, A]
  num_actions = policy.shape[2]

  def preprocess_from_config(inputs, preproc_config, prefix_shape):
    inputs_t = []
    for input_config in preproc_config:
      # Extract inputs according to the config.
      x = _multi_level_extract_by_attr_or_key(inputs, input_config.source)

      # Align extra input dims.
      if (
          input_config.source.startswith('extra_from_rule')
          and 'target_out' not in input_config.source
      ) or input_config.source == 'extra_from_rule/target_out/q':
        x = jnp.expand_dims(x, axis=-1)

      # Apply transforms.
      for tx in input_config.transforms:
        if tx == 'y_net':
          x = y_net(x)
        elif tx == 'z_net':
          x = z_net(x)
        else:
          if tx not in input_transforms.TRANSFORMS:
            raise KeyError(
                f'Transform {tx} was not found in {input_config.transforms}.'
            )
          transform_fn = input_transforms.TRANSFORMS[tx]()
          x = transform_fn(x, actions, policy, axis_name)

      # Flatten to a vector.
      x = jnp.reshape(x, (*prefix_shape, -1))
      inputs_t.append(x)
    return inputs_t

  inputs_t = preprocess_from_config(
      inputs, input_option.base, prefix_shape=(unroll_len, batch_size)
  )

  # Get action-conditional inputs, if required.
  if input_option.action_conditional:
    act_cond_inputs = preprocess_from_config(
        inputs,
        input_option.action_conditional,
        prefix_shape=(unroll_len, batch_size, num_actions),
    )
    act_cond_inputs.append(
        jnp.expand_dims(
            jax.nn.one_hot(actions, num_actions, dtype=jnp.float32), axis=-1
        )
    )
    act_cond_inputs = jnp.concatenate(act_cond_inputs, axis=-1)
    act_cond_emb = policy_net(act_cond_inputs)  # [T, B, A, C]
    act_cond_emb_avg = jnp.mean(act_cond_emb, axis=2)  # [T, B, C]
    act_cond_emb_a = utils.batch_lookup(act_cond_emb, actions)  # [T, B, C]
    inputs_t += [act_cond_emb_avg, act_cond_emb_a]
  else:
    act_cond_emb = None

  chex.assert_rank(inputs_t, 3)
  chex.assert_tree_shape_prefix(inputs_t, (unroll_len, batch_size))
  return jnp.concatenate(inputs_t, axis=-1), act_cond_emb


def _maybe_get_initializer(
    stddev: float | None,
) -> hk.initializers.Initializer | None:
  return hk_init.TruncatedNormal(stddev=stddev) if stddev is not None else None


def _multiplicative_interaction(
    x: chex.Array, y: chex.Array, initializer: hk.initializers.Initializer
) -> chex.Array:
  """Returns out = x * Linear(y) if y is not None. Otherwise, returns x."""
  if isinstance(y, chex.Array) and y.shape:  # not scalar
    # Condition on rnn_state via multiplicative interaction.
    y_embed = hk.Linear(x.shape[-1], w_init=initializer)(y)
    return jnp.multiply(x, y_embed)
  else:
    return x


def _batch_mlp(
    hiddens: Sequence[int], num_dims: int = 2
) -> Callable[[chex.Array], chex.Array]:
  return hk.BatchApply(hk.nets.MLP(hiddens), num_dims=num_dims)


def _conv1d_block(x: chex.Array, n_channels: int) -> chex.Array:
  x_avg = jnp.mean(x, axis=2, keepdims=True)  # [T, B, 1, C]
  x_avg = jnp.repeat(x_avg, x.shape[2], axis=2)  # [T, B, A, C]
  x = jnp.concatenate([x, x_avg], axis=-1)  # [T, B, A, 2C]
  x = hk.BatchApply(hk.Conv1D(output_channels=n_channels, kernel_shape=1))(x)
  x = jax.nn.relu(x)  # [T, B, A, C_new]
  return x


def _conv1d_net(channels: Sequence[int]) -> Callable[[chex.Array], chex.Array]:
  return hk.Sequential(
      [functools.partial(_conv1d_block, n_channels=c) for c in channels]
  )
