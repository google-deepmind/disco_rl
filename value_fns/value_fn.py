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

"""Value function for meta-learning."""

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from disco_rl import optimizers
from disco_rl import types
from disco_rl import utils
from disco_rl.networks import nets
from disco_rl.value_fns import value_utils


class ValueFunction:
  """Domain value function approximator.

  Used only in meta-training.

  Attributes:
    config: a value function's config.
    axis_name: (optional) an axis name to average over, if runs under `pmap`.
  """

  _value_fn: types.PolicyNetwork
  _value_opt: optax.GradientTransformation

  def __init__(
      self,
      config: types.ValueFnConfig,
      axis_name: str | None,
  ) -> None:
    self.config = config
    self.axis_name = axis_name

    self._discount_factor = config.discount_factor
    self._td_lambda = config.td_lambda
    self._outer_value_cost = config.outer_value_cost

    # Build value function network.
    self._value_fn = nets.get_network(
        config.net,
        out_spec={'value': types.ArraySpec([1], jnp.float32)},
        module_name='value_fn',
        **config.net_args,
    )
    self._value_opt = optax.chain(
        optimizers.scale_by_adam_sg_denom(),
        optax.clip(max_delta=config.max_abs_update),
        optax.scale(-config.learning_rate),
    )

    # Build exponential moving average of advantages.
    self._adv_ema = utils.MovingAverage(
        jnp.zeros(()), decay=config.ema_decay, eps=self.config.ema_eps
    )
    self._td_ema = utils.MovingAverage(
        jnp.zeros(()), decay=config.ema_decay, eps=self.config.ema_eps
    )

  def initial_state(
      self, rng: chex.PRNGKey, dummy_observation: chex.ArrayTree
  ) -> types.ValueState:
    """Initializes a value function state: parameters and optimizer state.

    Args:
      rng: a JAX random number key.
      dummy_observation: dummy observations to use for the networks init.

    Returns:
      A value function state.
    """
    params, state = self._value_fn.init(rng, dummy_observation, None)
    if state:
      raise ValueError(
          'Value functions do not support stateful networks, but the state is'
          ' not empty: {state}.'
      )

    return types.ValueState(
        params=params,
        state=state,
        opt_state=self._value_opt.init(params),
        adv_ema_state=self._adv_ema.init_state(),
        td_ema_state=self._td_ema.init_state(),
    )

  def get_value_outs(
      self,
      value_state: types.ValueState,
      rollout: types.ActorRollout,
      target_logits: chex.Array,
  ) -> tuple[
      types.ValueOuts,
      chex.Array,
      types.EmaState | None,
      types.EmaState | None,
  ]:
    """Runs forward pass of value network and gets value estimates."""
    value_net_outputs, _ = hk.BatchApply(
        lambda x: self._value_fn.one_step(
            value_state.params, value_state.state, x, None
        )
    )(rollout.observations)
    value_net_outputs = value_net_outputs['value']
    value_outs, adv_ema_state, td_ema_state = value_utils.get_value_outs(
        value_net_out=value_net_outputs,
        target_value_net_out=None,
        q_net_out=None,
        target_q_net_out=None,
        rollout=rollout,
        pi_logits=target_logits,
        discount=self._discount_factor,
        lambda_=self._td_lambda,
        nonlinear_transform=True,
        categorical_value=False,
        adv_ema_state=value_state.adv_ema_state,
        adv_ema_fn=self._adv_ema,
        td_ema_state=value_state.td_ema_state,
        td_ema_fn=self._td_ema,
        axis_name=self.axis_name,
    )
    return value_outs, value_net_outputs, adv_ema_state, td_ema_state

  def update(
      self,
      value_state: types.ValueState,
      rollout: types.ActorRollout,
      target_logits: chex.Array,
  ) -> tuple[types.ValueState, types.ValueOuts, types.LogDict]:
    """Updates value function state.

    Args:
      value_state: a value state to update.
      rollout: a rollout to use in the update.
      target_logits: target logits for the given rollout.

    Returns:
      A tuple of an updated state, value outputs, and log.
    """

    def value_loss_fn(v_params, value_state, rollout, target_logits):
      """Compute value functions losses."""
      v_params_no_tracer = value_state.params
      value_state.params = v_params
      value_outs, net_out, adv_ema_state, td_ema_state = self.get_value_outs(
          value_state, rollout, target_logits
      )
      value_losses = value_utils.value_loss_from_td(
          net_out[:-1], jax.lax.stop_gradient(value_outs.normalized_td)
      )
      value_loss = (self._outer_value_cost * value_losses).mean()
      value_state.params = v_params_no_tracer
      return value_loss, (value_outs, adv_ema_state, td_ema_state)

    (value_loss, (value_outs, adv_ema_state, td_ema_state)), dv_dparams = (
        jax.value_and_grad(value_loss_fn, has_aux=True)(
            value_state.params, value_state, rollout, target_logits
        )
    )
    if self.axis_name is not None:
      dv_dparams = jax.lax.pmean(dv_dparams, axis_name=self.axis_name)
    update, new_opt_state = self._value_opt.update(
        dv_dparams, value_state.opt_state, value_state.params
    )

    new_params = optax.apply_updates(value_state.params, update)

    new_state = types.ValueState(
        params=new_params,
        state=value_state.state,
        opt_state=new_opt_state,
        adv_ema_state=adv_ema_state,
        td_ema_state=td_ema_state,
    )

    log = dict(
        value_loss=value_loss,
        value_td=value_outs.td,
        value_normalized_td=value_outs.normalized_td,
    )
    return new_state, value_outs, log
