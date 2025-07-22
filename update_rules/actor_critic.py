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

"""Actor-critic update rule.

This is an example of the well-known Actor-Critic algorithm implemented as an
update rule.
"""

import chex
import distrax
import jax
from jax import numpy as jnp
import numpy as np

from disco_rl import types
from disco_rl import utils
from disco_rl.update_rules import base
from disco_rl.value_fns import value_utils


class ActorCritic(base.UpdateRule):
  """Various actor-critic baselines."""

  def __init__(
      self,
      categorical_value: bool = False,
      num_bins: int = 601,
      max_abs_value: float = 300.0,
      nonlinear_value_transform: bool = False,
      normalize_adv: bool = False,
      normalize_td: bool = False,
      moving_average_decay: float = 0.99,
      moving_average_eps: float = 1e-6,
  ) -> None:
    """Configure the actor-critic module.

    Args:
      categorical_value: use categorical value function if true.
      num_bins: number of bins for categorical value. This is ignored when
        categorical_value is false.
      max_abs_value: maximum absolute value representable by the categorical
        value. This is ignored when categorical_value is False; must be set when
        categorical_value is True.
      nonlinear_value_transform: apply non-linear transform to value if true.
      normalize_adv: normalize advantange for policy gradient if true.
      normalize_td: normalize TD for value loss if true.
      moving_average_decay: moving average decay for advantage and TD.
      moving_average_eps: moving average epsilon for advantage and TD.
    """
    if normalize_td and categorical_value:
      raise ValueError(
          'normalize_td and categorical_value should not be used together.'
      )
    self._normalize_adv = normalize_adv
    self._normalize_td = normalize_td
    self._num_bins = num_bins
    self._max_abs_value = max_abs_value
    self._categorical_value = categorical_value
    self._nonlinear_value_transform = nonlinear_value_transform

    # Moving average advantage.
    self._adv_ema = utils.MovingAverage(
        np.zeros(()), decay=moving_average_decay, eps=moving_average_eps
    )
    self._td_ema = utils.MovingAverage(
        np.zeros(()), decay=moving_average_decay, eps=moving_average_eps
    )

  def init_params(
      self, rng: chex.PRNGKey
  ) -> tuple[types.MetaParams, chex.ArrayTree]:
    del rng
    return {'dummy': jnp.array(0.0)}, {}

  def flat_output_spec(
      self, single_action_spec: types.ActionSpec
  ) -> types.Specs:
    """Returns the agent's unconditional output specs.

    Args:
      single_action_spec: An action spec.

    Returns:
      A nested dict with tuples specifying output specs.
    """
    return dict(
        logits=utils.get_logits_specs(single_action_spec),
        v=types.ArraySpec(
            (self._num_bins if self._categorical_value else 1,), np.float32
        ),
    )

  def model_output_spec(
      self, single_action_spec: types.ActionSpec
  ) -> types.Specs:
    """Returns the agent's action-conditional output specs.

    Args:
      single_action_spec: An action spec.

    Returns:
      A nested dict with tuples specifying model output specs.
    """
    del single_action_spec
    return dict()

  def init_meta_state(
      self,
      rng: chex.PRNGKey,
      params: types.AgentParams,
  ) -> types.MetaState:
    del rng
    meta_state = dict()
    meta_state['adv_ema_state'] = self._adv_ema.init_state()
    meta_state['td_ema_state'] = self._td_ema.init_state()
    return meta_state

  def unroll_meta_net(
      self,
      meta_params: types.MetaParams,
      params: types.AgentParams,
      state: types.HaikuState,
      meta_state: types.MetaState,
      rollout: types.UpdateRuleInputs,
      hyper_params: types.HyperParams,
      unroll_policy_fn: types.AgentUnrollFn,
      rng: chex.PRNGKey,
      axis_name: str | None = None,
  ) -> tuple[types.UpdateRuleOuts, types.MetaState]:
    """Prepare quantities for the loss (no meta network)."""
    del meta_params

    value_outs, adv_ema_state, td_ema_state = value_utils.get_value_outs(
        value_net_out=rollout.agent_out['v'],
        target_value_net_out=None,
        q_net_out=None,
        target_q_net_out=None,
        rollout=rollout,
        pi_logits=rollout.agent_out['logits'],
        discount=hyper_params['discount_factor'],
        lambda_=hyper_params['vtrace_lambda'],
        nonlinear_transform=self._nonlinear_value_transform,
        categorical_value=self._categorical_value,
        max_abs_value=self._max_abs_value,
        drop_last=False,
        adv_ema_state=meta_state['adv_ema_state'],
        adv_ema_fn=self._adv_ema,
        td_ema_state=meta_state['td_ema_state'],
        td_ema_fn=self._td_ema,
        axis_name=axis_name,
    )

    meta_out = dict(
        raw_advs=value_outs.adv,
        normalized_advs=value_outs.normalized_adv,
        value_target=value_outs.value_target,
        values=value_outs.value,
        normalized_tds=value_outs.normalized_td,
        tds=value_outs.td,
    )
    new_meta_state = meta_state
    meta_state['adv_ema_state'] = adv_ema_state
    meta_state['td_ema_state'] = td_ema_state

    return meta_out, new_meta_state

  def agent_loss(
      self,
      rollout: types.UpdateRuleInputs,
      meta_out: types.UpdateRuleOuts,
      hyper_params: types.HyperParams,
      backprop: bool,
  ) -> tuple[chex.Array, types.UpdateRuleLog]:
    """Construct policy and value loss."""
    del backprop
    actions = rollout.actions[:-1]
    logits = rollout.agent_out['logits'][:-1]

    pg_advs = (
        meta_out['normalized_advs']
        if self._normalize_adv
        else meta_out['raw_advs']
    )
    value_tds = (
        meta_out['normalized_tds'] if self._normalize_td else meta_out['tds']
    )

    value_loss_per_step = value_utils.value_loss_from_td(
        rollout.agent_out['v'][:-1],
        jax.lax.stop_gradient(value_tds),
        nonlinear_transform=self._nonlinear_value_transform,
        categorical_value=self._categorical_value,
        max_abs_value=self._max_abs_value,
    )

    # Entropy loss.
    entropy_loss_per_step = -distrax.Softmax(logits).entropy()

    # PG loss: R * log(prod(p_i)) = R * Sum(log(p_i)).
    pg_loss_per_step = utils.differentiable_policy_gradient_loss(
        logits, actions, adv_t=pg_advs, backprop=False
    )

    # Compute total loss.
    chex.assert_rank(  # [T, B]
        (pg_loss_per_step, value_loss_per_step, entropy_loss_per_step), 2
    )
    total_loss_per_step = (
        hyper_params['pg_cost'] * pg_loss_per_step
        + hyper_params['value_cost'] * value_loss_per_step
        + hyper_params['entropy_cost'] * entropy_loss_per_step
    )

    logs = dict(
        logits=jnp.mean(logits),
        entropy=-jnp.mean(entropy_loss_per_step),
        pg_advs=jnp.mean(pg_advs),
        raw_advs=jnp.mean(meta_out['raw_advs']),
        avg_value=jnp.mean(meta_out['values']),
    )

    return total_loss_per_step, logs
