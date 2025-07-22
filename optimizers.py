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

"""Optimizers."""

import jax
from jax import numpy as jnp
import optax


def scale_by_adam_sg_denom(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
  """Adam grad rescaling; but denominator does not receive (meta-)gradients.

  References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    mu = jax.tree.map(jnp.zeros_like, params)  # First moment.
    nu = jax.tree.map(jnp.zeros_like, params)  # Second moment.
    return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = optax.update_moment(updates, state.mu, b1, 1)
    nu = optax.update_moment(updates, state.nu, b2, 2)
    count_inc = optax.safe_int32_increment(state.count)
    mu_hat = optax.bias_correction(mu, b1, count_inc)
    nu_hat = optax.bias_correction(nu, b2, count_inc)
    updates = jax.tree.map(
        lambda m, v: m / (jnp.sqrt(v) + eps),
        mu_hat,
        jax.lax.stop_gradient(nu_hat),  # NOTE: stop_gradient on nu_hat here
    )
    return updates, optax.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)
