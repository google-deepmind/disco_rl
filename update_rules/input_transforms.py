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

"""Transformations of meta-network inputs."""

import functools
from typing import Callable, Sequence

import chex
import haiku as hk
import immutabledict
import jax
import jax.numpy as jnp
import rlax

from disco_rl import utils


class InputTransform:

  def __call__(
      self, x, actions: chex.Array, policy: chex.Array, axis: str | None
  ) -> chex.Array:
    raise NotImplementedError


class SelectByAction(InputTransform):

  def __call__(self, x, actions, policy, axis):
    del policy, axis
    chex.assert_rank(actions, 2)
    chex.assert_tree_shape_prefix([x, actions], actions.shape)
    return utils.batch_lookup(x, actions)


class PiWeightedAvg(InputTransform):

  def __call__(self, x, actions, policy, axis):
    del actions, axis
    chex.assert_rank(x, 4)
    chex.assert_rank(policy, 3)
    chex.assert_tree_shape_prefix([x, policy], policy.shape)
    return jnp.sum(x * jnp.expand_dims(policy, -1), axis=2)


class Normalize(InputTransform, hk.Module):

  def __call__(self, x, actions, policy, axis):
    del actions, policy
    assert x.ndim >= 2  # Average over B, T
    return EmaNorm(
        decay_rate=0.99, eps=1e-6, axis=(0, 1), cross_replica_axis=axis
    )(x)


class EmaNorm(hk.Module):
  """Normalize by EMA estimate of mean and variance."""

  def __init__(
      self,
      decay_rate: float,
      eps: float = 1e-6,
      eps_root: float = 1e-12,
      axis: Sequence[int] | None = None,
      cross_replica_axis: str | Sequence[str] | None = None,
      cross_replica_axis_index_groups: Sequence[Sequence[int]] | None = None,
      data_format: str = 'channels_last',
      name: str | None = None,
  ):
    """Constructs an EmaNorm module. Based on hk.BatchNorm.

    Args:
      decay_rate: Decay rate for EMA.
      eps: Small epsilon to avoid division by zero variance. Defaults ``1e-6``.
      eps_root: Small epsilon to assist metagrad stability. Defaults ``1e-12``.
      axis: Which axes to reduce over. The default (``None``) signifies that all
        but the channel axis should be normalized. Otherwise this is a list of
        axis indices which will have normalization statistics calculated.
      cross_replica_axis: If not ``None``, it should be a string (or sequence of
        strings) representing the axis name(s) over which this module is being
        run within a jax map (e.g. ``jax.pmap`` or ``jax.vmap``). Supplying this
        argument means that batch statistics are calculated across all replicas
        on the named axes.
      cross_replica_axis_index_groups: Specifies how devices are grouped. Valid
        only within ``jax.pmap`` collectives.
      data_format: The data format of the input. Can be either
        ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
        default it is ``channels_last``. See :func:`get_channel_index`.
      name: The module name.
    """
    super().__init__(name=name)

    self.eps = eps
    self.eps_root = eps_root
    self.axis = axis
    self.cross_replica_axis = cross_replica_axis
    self.cross_replica_axis_index_groups = cross_replica_axis_index_groups
    self.channel_index = hk.get_channel_index(data_format)
    self.m1_ema = hk.ExponentialMovingAverage(decay_rate, name='m1_ema')
    self.m2_ema = hk.ExponentialMovingAverage(decay_rate, name='m2_ema')

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Updates EMA state (always in `train` mode), returns normalized input."""

    channel_index = self.channel_index
    if channel_index < 0:
      channel_index += inputs.ndim

    if self.axis is not None:
      axis = self.axis
    else:
      axis = [i for i in range(inputs.ndim) if i != channel_index]

    mean = jnp.mean(inputs, axis, keepdims=True)
    mean_of_squares = jnp.mean(jnp.square(inputs), axis, keepdims=True)

    if self.cross_replica_axis:
      mean = jax.lax.pmean(
          mean,
          axis_name=self.cross_replica_axis,
          axis_index_groups=self.cross_replica_axis_index_groups,
      )
      mean_of_squares = jax.lax.pmean(
          mean_of_squares,
          axis_name=self.cross_replica_axis,
          axis_index_groups=self.cross_replica_axis_index_groups,
      )
    self.m1_ema(mean)
    self.m2_ema(mean_of_squares)

    ema_m1 = self.m1_ema.average.astype(inputs.dtype)
    ema_m2 = self.m2_ema.average.astype(inputs.dtype)

    ema_var = jnp.maximum(ema_m2 - jnp.square(ema_m1), 0.0)

    eps = jax.lax.convert_element_type(self.eps, ema_var.dtype)
    eps_root = jax.lax.convert_element_type(self.eps_root, ema_var.dtype)
    return (inputs - ema_m1) / (jnp.sqrt(ema_var + eps_root) + eps)


def td_pair(x):
  # Concat inputs at t and t+1 to ease calculation of TD-error-like quantities
  return jnp.concatenate([x[:-1], x[1:]], axis=-1)


def tx_factory(tx_call: Callable[[chex.Array], chex.Array]):
  """Wraps single-arg transforms so they all have the same interface."""

  # The dummy and wrap allows the fn to be called with the same interface as the
  # modules: tx()(x, actions, policy, axis)
  def dummy_builder():
    def _wrap_transform(x, actions, policy, axis):
      del actions, policy, axis
      return tx_call(x)

    return _wrap_transform

  return dummy_builder


_TRANSFORM_FNS = immutabledict.immutabledict({
    'identity': lambda x: x,
    'softmax': jax.nn.softmax,
    'max_a': functools.partial(jnp.max, axis=2),
    'stop_grad': jax.lax.stop_gradient,
    'clip': functools.partial(jnp.clip, a_min=-2.0, a_max=2.0),
    'sign': jnp.sign,
    'drop_last': lambda x: x[:-1],
    'td_pair': td_pair,
    'sign_log': rlax.signed_logp1,
    'sign_hyp': rlax.signed_hyperbolic,
    'masks_to_discounts': lambda x: 1.0 - x,
})

TRANSFORMS = immutabledict.immutabledict({
    'select_a': SelectByAction,
    'pi_weighted_avg': PiWeightedAvg,
    'normalize': Normalize,
    **{name: tx_factory(tx) for name, tx in _TRANSFORM_FNS.items()},
})
