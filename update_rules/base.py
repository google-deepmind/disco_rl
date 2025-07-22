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

"""Base class for update rules. See `UpdateRule`'s docstrings."""

import chex
from dm_env import specs as dm_env_specs
import jax
import jax.numpy as jnp

from disco_rl import types
from disco_rl import utils

ArraySpec = types.ArraySpec


def get_agent_out_spec(
    action_spec: types.ActionSpec,
    flat_out_spec: types.Specs,
    model_out_spec: types.Specs,
) -> types.Specs:
  """Constructs a tree of shapes of outputs for the provided specs.

  Example:
      >> action_spec = specs.BoundedArray((), int, minimum=0, maximum=A-1)
      >> unconditional_output_spec = {'logits': (A,),
                                      'y': (Y,),
                                      }
      >> conditional_output_spec = {'z': (Z,),
                                    'aux_pi': (P,),
                                    }
      >> agent_output_spec = {
            'logits': (A,),
            'y': (Y,),

            'z': (A, Z),
            'aux_pi': (A, P),
         }

  Args:
      action_spec: An action spec.
      flat_out_spec: A nested dict of agent's flat output shapes.
      model_out_spec: A nested dict of agent's model output shapes.

  Returns:
      A nested dict with shapes specifying agent's total output shapes.
  """
  if set(flat_out_spec.keys()).intersection(set(model_out_spec.keys())):
    raise ValueError(
        'Keys overlap between flat_out_spec and model_out_spec.'
        f'Given: {flat_out_spec} and {model_out_spec}'
    )

  num_actions = utils.get_num_actions_from_spec(action_spec)
  agent_out_spec = {key: val for key, val in flat_out_spec.items()}
  for key, val in model_out_spec.items():
    agent_out_spec[key] = ArraySpec((num_actions, *val.shape), val.dtype)
  return agent_out_spec


class UpdateRule:
  """Base class for update rule."""

  def _get_dummy_input(
      self,
      include_behaviour_out: bool = True,
      include_value_out: bool = False,
      include_agent_adv: bool = False,
  ) -> types.UpdateRuleInputs:
    """Generate a dummy meta-network input for initialization."""
    b = 1
    t = 2
    unroll_batch_shape = (t, b)
    bootstrapped_shape = (t + 1, b)
    dummy_action_spec = dm_env_specs.BoundedArray(
        shape=(3,), dtype=int, minimum=0, maximum=3
    )
    num_actions = utils.get_num_actions_from_spec(dummy_action_spec)
    dummy_actions = jnp.zeros(bootstrapped_shape, dtype=jnp.int32)
    agent_out_shapes = self.agent_output_spec(dummy_action_spec)

    agent_out = jax.tree.map(
        lambda s: jnp.zeros(bootstrapped_shape + s.shape), agent_out_shapes
    )

    dummy_input = types.UpdateRuleInputs(
        observations=jnp.zeros(bootstrapped_shape),
        actions=dummy_actions,
        rewards=jnp.zeros(unroll_batch_shape),
        is_terminal=jnp.ones(unroll_batch_shape, dtype=jnp.bool_),
        agent_out=agent_out,
    )

    if include_behaviour_out:
      dummy_input.behaviour_agent_out = {}
      dummy_input.behaviour_agent_out.update(agent_out)

    target_out = agent_out
    dummy_input.extra_from_rule = dict(target_out=target_out)

    if include_agent_adv:
      value_unroll_batch_shape = (t, b, 1)
      value_bootstrapped_shape = (t + 1, b, 1)
      q_bootstrapped_shape = (t + 1, b, num_actions, 1)

      dummy_input.extra_from_rule = dict(
          adv=jnp.zeros(value_unroll_batch_shape),
          normalized_adv=jnp.zeros(value_unroll_batch_shape),
          v_scalar=jnp.zeros(value_bootstrapped_shape),
          q=jnp.zeros(q_bootstrapped_shape),
          qv_adv=jnp.zeros(q_bootstrapped_shape),
          normalized_qv_adv=jnp.zeros(q_bootstrapped_shape),
          target_out=target_out,
      )

    if include_value_out:
      num_discounts = 1
      value_unroll_batch_shape = (t, b, num_discounts)
      q_shape = (t, b, num_actions, num_discounts)
      bootstrapped_q_shape = (t + 1, b, num_actions, num_discounts)
      value_bootstrapped_shape = (t + 1, b, num_discounts)
      dummy_input.value_out = types.ValueOuts(
          value=jnp.ones(value_bootstrapped_shape),
          target_value=jnp.ones(value_bootstrapped_shape),
          rho=jnp.ones(unroll_batch_shape),
          adv=jnp.ones(value_unroll_batch_shape),
          normalized_adv=jnp.ones(value_unroll_batch_shape),
          td=jnp.ones(value_unroll_batch_shape),
          normalized_td=jnp.ones(value_unroll_batch_shape),
          value_target=jnp.ones(value_unroll_batch_shape),
          qv_adv=jax.tree.map(jnp.ones, bootstrapped_q_shape),
          normalized_qv_adv=jax.tree.map(jnp.ones, bootstrapped_q_shape),
          q_target=jax.tree.map(jnp.ones, q_shape),
          q_value=jax.tree.map(jnp.ones, q_shape),
          target_q_value=jax.tree.map(jnp.ones, q_shape),
          q_td=jax.tree.map(jnp.ones, q_shape),
          normalized_q_td=jax.tree.map(jnp.ones, q_shape),
      )

    return dummy_input

  def init_params(
      self, rng: chex.PRNGKey
  ) -> tuple[types.MetaParams, chex.ArrayTree]:
    """Initialize meta-parameters.

    Args:
      rng: random key.

    Returns:
      Meta-parameters and initial meta-network state.
    """
    raise NotImplementedError

  def flat_output_spec(self, action_spec: types.ActionSpec) -> types.Specs:
    """Returns the agent's unconditional output specs.

    Args:
      action_spec: An action spec.

    Returns:
      A nested dict with tuples specifying output specs.
    """
    del action_spec
    return dict()

  def model_output_spec(self, action_spec: types.ActionSpec) -> types.Specs:
    """Returns the agent's action-conditional output specs.

    Args:
      action_spec: An action spec.

    Returns:
      A nested dict with tuples specifying model output specs.
    """
    del action_spec
    return dict()

  def agent_output_spec(self, action_spec: types.ActionSpec) -> types.Specs:
    """Returns the agent total outputs' specs.

    Args:
      action_spec: An action spec.

    Returns:
      A pair of dicts with specs specifying agent's total output specs.
    """
    return get_agent_out_spec(
        action_spec=action_spec,
        flat_out_spec=self.flat_output_spec(action_spec),
        model_out_spec=self.model_output_spec(action_spec),
    )

  def init_meta_state(
      self,
      rng: chex.PRNGKey,
      params: types.AgentParams,
  ) -> types.MetaState:
    """The agent initial meta state.

    Args:
      rng: random key.
      params: agent params.

    Returns:
      An array tree with meta state.
    """
    raise NotImplementedError

  def unroll_meta_net(
      self,
      meta_params: types.MetaParams,
      params: types.AgentParams,
      state: types.HaikuState,
      meta_state: types.MetaState,
      rollout: types.UpdateRuleInputs,
      hyper_params: types.HyperParams,
      unroll_policy_fn: types.AgentStepFn,
      rng: chex.PRNGKey,
      axis_name: str | None,
  ) -> tuple[types.UpdateRuleOuts, types.MetaState]:
    """Unroll the meta network to prepare for the agent's loss.

    Args:
      meta_params: meta parameters.
      params: agent parameters.
      state: state of the agent.
      meta_state: meta state of the agent.
      rollout: rollout. [T, B, ...] and [T+1, B, ...] for `agent_out`
      hyper_params: hyper_params for the agent loss.
      unroll_policy_fn: agent's policy unroll function.
      rng: random key.
      axis_name: an axis name to use in collective ops, if runs under `pmap`.

    Returns:
      The output of meta network. [T, B, ...]
      An updated meta state.
    """
    raise NotImplementedError

  def agent_loss(
      self,
      rollout: types.UpdateRuleInputs,
      meta_out: types.UpdateRuleOuts,
      hyper_params: types.HyperParams,
      backprop: bool,
  ) -> tuple[chex.Array, types.UpdateRuleLog]:
    """The agent loss.

    Args:
      rollout: rollout with reward, discount, etc. [T, B, ...]
      meta_out: meta network output along the rollout. [T, B, ...]
      hyper_params: hyper_params for the agent loss.
      backprop: whether to make the loss differentiable or not.

    Returns:
      Loss per step (a tensor) and logs.
    """
    raise NotImplementedError

  def agent_loss_no_meta(
      self,
      rollout: types.UpdateRuleInputs,
      meta_out: types.UpdateRuleOuts | None,
      hyper_params: types.HyperParams,
  ) -> tuple[chex.Array, types.UpdateRuleLog]:
    """An optional part of the agent loss which shouldn't receive metagradients.

    Args:
      rollout: rollout with reward, discount, etc. [T, B, ...]
      meta_out: meta network output along the rollout. [T, B, ...]
      hyper_params: hyper_params for the agent loss.

    Returns:
      Loss per step (a tensor) and logs.
    """
    del meta_out, hyper_params
    return jnp.zeros_like(rollout.rewards), {}

  def __call__(
      self,
      meta_params: types.MetaParams,
      params: types.AgentParams,
      state: types.HaikuState,
      rollout: types.UpdateRuleInputs,
      hyper_params: types.HyperParams,
      meta_state: types.MetaState,
      unroll_policy_fn: types.AgentUnrollFn,
      rng: chex.PRNGKey,
      axis_name: str | None,
      backprop: bool = False,
  ) -> tuple[chex.Array, types.MetaState, types.UpdateRuleLog]:
    """The agent loss from rollout and agent output.

    Args:
      meta_params: meta parameters.
      params: agent parameters.
      state: state of the agent.
      rollout: rollout with reward, discount, etc. [T+1, B, ...]
      hyper_params: scalar hyper_params for the agent loss.
      meta_state: meta state of the agent.
      unroll_policy_fn: agent's policy unroll function.
      rng: random key.
      axis_name: an axis name to use in collective ops, if runs under `pmap`.
      backprop: whether to make the loss differentiable wrt meta_params or not.

    Returns:
      A tuple (per-step loss, meta_state, log).
    """
    meta_out, new_meta_state = self.unroll_meta_net(
        meta_params=meta_params,
        params=params,
        state=state,
        meta_state=meta_state,
        rollout=rollout,
        hyper_params=hyper_params,
        unroll_policy_fn=unroll_policy_fn,
        rng=rng,
        axis_name=axis_name,
    )
    loss_per_step, log_with_meta = self.agent_loss(
        rollout, meta_out, hyper_params, backprop=backprop
    )
    loss_per_step_no_meta, log_no_meta = self.agent_loss_no_meta(
        rollout, meta_out, hyper_params
    )

    loss_per_step = loss_per_step + loss_per_step_no_meta
    logs = log_with_meta | log_no_meta

    return loss_per_step, new_meta_state, logs
