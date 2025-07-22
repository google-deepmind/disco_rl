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

"""Policy gradient update rule.

This is an example of the well-known Policy Gradient algorithm implemented
as an update rule.
"""

import chex
import distrax
from jax import numpy as jnp

from disco_rl import types
from disco_rl import utils
from disco_rl.update_rules import base


class PolicyGradientUpdate(base.UpdateRule):
  """Policy gradient update rule (uses value functions)."""

  def __init__(
      self,
      entropy_cost: float = 0.01,
      normalize_adv: bool = True,
      pg_cost: float = 1,
      kl_prior_cost: float = 0.5,
      p_actor_prior: float = 0.03,
      p_uniform_prior: float = 0.003,
      target_params_coeff: float = 0.1,
  ) -> None:
    """Init."""
    self._normalize_adv = normalize_adv
    self._target_params_coeff = target_params_coeff
    self._pg_cost = pg_cost
    self._entropy_cost = entropy_cost
    self._kl_prior_cost = kl_prior_cost
    self._p_actor_prior = p_actor_prior
    self._p_uniform_prior = p_uniform_prior

  def init_params(
      self, rng: chex.PRNGKey
  ) -> tuple[types.MetaParams, chex.ArrayTree]:
    del rng
    return {'dummy': jnp.array(0.0)}, {}

  def flat_output_spec(
      self, single_action_spec: types.ActionSpec
  ) -> types.Specs:
    return dict(logits=utils.get_logits_specs(single_action_spec))

  def model_output_spec(
      self, single_action_spec: types.ActionSpec
  ) -> types.Specs:
    del single_action_spec
    return dict()

  def init_meta_state(
      self,
      rng: chex.PRNGKey,
      params: types.AgentParams,
  ) -> types.MetaState:
    del rng
    return dict()

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
    del meta_params, hyper_params, axis_name, rng
    assert rollout.value_out is not None
    pg_adv = (
        rollout.value_out.normalized_adv
        if self._normalize_adv
        else rollout.value_out.adv
    )
    return dict(pi=pg_adv), meta_state

  def agent_loss(
      self,
      rollout: types.UpdateRuleInputs,
      meta_out: types.UpdateRuleOuts,
      hyper_params: types.HyperParams,
      backprop: bool,
  ) -> tuple[chex.Array, types.UpdateRuleLog]:
    """Construct policy and value loss."""
    del backprop, hyper_params
    actions = rollout.actions[:-1]
    logits = rollout.agent_out['logits'][:-1]

    pg_loss_per_step = utils.differentiable_policy_gradient_loss(
        logits, actions, adv_t=meta_out['pi'], backprop=False
    )
    entropy_loss_per_step = -distrax.Softmax(logits).entropy()

    # Compute total loss.
    chex.assert_rank((pg_loss_per_step, entropy_loss_per_step), 2)  # [T, B]
    total_loss_per_step = (
        self._pg_cost * pg_loss_per_step
        + self._entropy_cost * entropy_loss_per_step
    )

    log = dict(
        entropy=-jnp.mean(entropy_loss_per_step),
        pg_advs=jnp.mean(meta_out['pi']),
    )
    return total_loss_per_step, log
