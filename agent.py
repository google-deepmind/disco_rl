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

"""An agent that uses update rules to learn from the environment."""

from absl import logging
import chex
import distrax
import dm_env
import haiku as hk
import jax
from jax import numpy as jnp
from ml_collections import config_dict
import numpy as np
import optax

from disco_rl import optimizers
from disco_rl import types
from disco_rl.networks import nets
from disco_rl.update_rules import actor_critic
from disco_rl.update_rules import base as update_rules_base
from disco_rl.update_rules import disco
from disco_rl.update_rules import policy_gradient


@chex.dataclass(frozen=True)
class LearnerState:
  """Dataclass for the learner's state: params, and optimizer/update's state."""

  params: hk.Params
  opt_state: optax.OptState
  meta_state: types.MetaState


class Agent:
  """A generic agent with an update rule.

  Its API supports both evaluation and meta-training of the update rule.

  Note that for evaluation/inference only, the update rule's application can be
  simplified and encapsulated by using its __call__ method.
  """

  update_rule: update_rules_base.UpdateRule

  def __init__(
      self,
      *,
      single_observation_spec: types.Specs,
      single_action_spec: types.ActionSpec,
      agent_settings: config_dict.ConfigDict,
      batch_axis_name: str | None,
  ):
    self.settings = agent_settings
    self.single_observation_spec = single_observation_spec
    self.single_action_spec = single_action_spec
    self._batch_axis_name = batch_axis_name

    # This agent only supports scalar discrete action specs.
    assert single_action_spec.dtype == np.int32
    assert not single_action_spec.shape

    # Create the update rule.
    if agent_settings.update_rule_name == 'disco':
      self.update_rule = disco.DiscoUpdateRule(**agent_settings.update_rule)
    elif agent_settings.update_rule_name == 'actor_critic':
      self.update_rule = actor_critic.ActorCritic(**agent_settings.update_rule)
    elif agent_settings.update_rule_name == 'policy_gradient':
      self.update_rule = policy_gradient.PolicyGradientUpdate(
          **agent_settings.update_rule
      )
    else:
      raise ValueError(
          f'Unsupported update rule: {agent_settings.update_rule_name}'
      )
    logging.info('Update rule config %r', agent_settings.update_rule)

    # Define the agent's neural network.
    flat_out_spec = self.update_rule.flat_output_spec(self.single_action_spec)
    model_out_spec = self.update_rule.model_output_spec(self.single_action_spec)
    self._network = nets.get_network(
        name=agent_settings.net_settings.name,
        action_spec=self.single_action_spec,
        out_spec=flat_out_spec,
        model_out_spec=model_out_spec,
        **agent_settings.net_settings.net_args,
    )

    # Define the optimiser.
    self._optimizer = optax.chain(
        optimizers.scale_by_adam_sg_denom(),
        optax.clip(max_delta=self.settings.max_abs_update),
        optax.scale(-self.settings.learning_rate),
    )

  def _dummy_obs(self, batch_size: int) -> chex.ArrayTree:
    """Create dummy observation for params and actor state initialisation."""
    return jax.tree.map(
        lambda v: np.zeros((batch_size,) + v.shape, dtype=v.dtype),
        self.single_observation_spec,
    )

  def _dummy_act(self, batch_size: int) -> chex.Array:
    """Create dummy action for params and actor state initialisation."""
    return np.zeros((batch_size,), dtype=self.single_action_spec.dtype)

  def _dummy_should_reset(self, batch_size: int) -> chex.Array:
    """Create dummy should_reset for params and actor state initialisation."""
    return jnp.zeros([batch_size], dtype=bool)

  def initial_actor_state(self, rng: chex.PRNGKey) -> types.HaikuState:
    """Create (potentially empty) initial actor state."""
    dummy_obs = self._dummy_obs(batch_size=1)
    should_reset = self._dummy_should_reset(batch_size=1)
    _, rnn_state = self._network.init(
        rng,  # params init rng
        dummy_obs,
        should_reset,
    )
    return rnn_state

  def initial_learner_state(self, rng_key: chex.PRNGKey) -> LearnerState:
    """Create the initial learner state."""
    net_rng, state_rng = jax.random.split(rng_key)
    dummy_obs = self._dummy_obs(batch_size=1)
    should_reset = self._dummy_should_reset(batch_size=1)
    # Init params, optimiser state and the discovered update's state.
    params, _ = self._network.init(net_rng, dummy_obs, should_reset)
    opt_state = self._optimizer.init(params)
    meta_state = self.update_rule.init_meta_state(rng=state_rng, params=params)
    return LearnerState(
        params=params, opt_state=opt_state, meta_state=meta_state
    )

  def actor_step(
      self,
      actor_params: hk.Params,
      rng: chex.PRNGKey,
      timestep: types.EnvironmentTimestep,
      actor_state: hk.State,
  ) -> tuple[types.ActorTimestep, hk.State]:
    """Compute actions for the provided timestep."""
    # Perform inference on the agent's network.
    should_reset = timestep.step_type == dm_env.StepType.LAST
    agent_outs, next_actor_state = self._network.one_step(
        actor_params, actor_state, timestep.observation, should_reset
    )
    # Sample actions.
    actions = distrax.Softmax(logits=agent_outs['logits']).sample(seed=rng)

    # Return the actor timestep.
    actor_timestep = types.ActorTimestep(
        observations=timestep.observation,
        actions=actions,
        agent_outs=agent_outs,
        rewards=timestep.reward,
        discounts=jnp.logical_not(should_reset),
        states=actor_state,
        logits=agent_outs['logits'],
    )
    return actor_timestep, next_actor_state

  def unroll_net(
      self,
      agent_net_params: hk.Params,
      agent_net_state: hk.State,
      rollout: types.ActorRollout,
  ) -> tuple[types.AgentOuts, hk.State]:
    """Unroll the agent's network."""
    # Mask, with 0s only on timesteps of type LAST.
    masks = rollout.discounts[:-1] > 0

    # Shifts is_terminal to the right by a step.
    prepend_non_terminal = jax.numpy.zeros_like(masks[:1])
    should_reset = jnp.concatenate(
        (prepend_non_terminal, masks), axis=0, dtype=masks.dtype
    )

    # Unroll the network.
    agent_out, new_agent_net_state = self._network.unroll(
        agent_net_params,
        agent_net_state,
        rollout.observations,
        should_reset,
    )
    return agent_out, new_agent_net_state

  def _loss(
      self,
      agent_net_params: hk.Params,
      agent_net_state: hk.State,
      meta_state: types.MetaState,
      rollout: types.ActorRollout,
      meta_out: types.UpdateRuleOuts,
      is_meta_training: bool,
  ) -> tuple[chex.Array, tuple[types.MetaState, hk.State, types.LogDict]]:
    """Computes the loss according to the update rule."""
    # Extract rewards and discounts.
    reward = rollout.rewards[1:]
    masks = rollout.discounts[:-1] > 0

    # Unroll the network.
    agent_out, new_agent_net_state = self.unroll_net(
        agent_net_params, agent_net_state, rollout
    )

    # Construct inputs for the update rule.
    discount = rollout.discounts[1:]
    eta_inputs = types.UpdateRuleInputs(
        observations=rollout.observations,  # [T, ...]
        actions=rollout.actions,  # [T, ...]
        rewards=reward,  # [T-1]
        is_terminal=discount == 0,  # [T-1]
        behaviour_agent_out=rollout.agent_outs,  # [T, ...]
        agent_out=agent_out,  # [T, ...]
        value_out=None,
    )

    # When not meta-training, this can be simplified by calling the update rule
    # directly to get the combined loss. See the class docstring for details.
    hyper_params = self.settings.hyper_params.to_dict()
    loss_per_step, log = self.update_rule.agent_loss(
        eta_inputs, meta_out, hyper_params, backprop=is_meta_training
    )
    loss_per_step_no_meta, log_no_meta = self.update_rule.agent_loss_no_meta(
        eta_inputs, meta_out, hyper_params
    )
    disco_log = log_no_meta | log
    total_loss_per_step = loss_per_step + loss_per_step_no_meta
    total_loss = (total_loss_per_step * masks).sum() / (masks.sum() + 1e-8)

    # Make logs.
    log_dict = dict(total_loss=total_loss, **jax.tree.map(jnp.mean, disco_log))
    return total_loss, (meta_state, new_agent_net_state, log_dict)

  def learner_step(
      self,
      rng: chex.PRNGKey,
      rollout: types.ActorRollout,
      learner_state: LearnerState,
      agent_net_state: hk.State,
      update_rule_params: types.MetaParams,
      is_meta_training: bool,
  ) -> tuple[LearnerState, hk.State, types.LogDict]:
    """Runs a training step across all learner devices for one rollout."""
    reward = rollout.rewards[1:]
    agent_out, _ = self.unroll_net(
        learner_state.params, agent_net_state, rollout
    )

    # Compute the loss using the discovered update(s).
    # Construct inputs for the discovered update.
    eta_inputs = types.UpdateRuleInputs(
        observations=rollout.observations,  # [T, ...]
        actions=rollout.actions,  # [T, ...]
        rewards=reward,  # [T-1]
        is_terminal=rollout.discounts[1:] == 0,  # [T-1]
        behaviour_agent_out=rollout.agent_outs,  # [T, ...]
        agent_out=agent_out,  # [T, ...]
        value_out=None,
    )

    # Apply the update network.
    meta_out, new_meta_state = self.update_rule.unroll_meta_net(
        meta_params=update_rule_params,
        params=learner_state.params,
        state=agent_net_state,
        meta_state=learner_state.meta_state,
        rollout=eta_inputs,
        hyper_params=self.settings.hyper_params.to_dict(),
        unroll_policy_fn=self._network.unroll,
        rng=rng,
        axis_name=self._batch_axis_name,
    )

    # Get gradient of the loss function using the latest rollout and parameters.
    dloss_dparams = jax.grad(self._loss, has_aux=True)
    grads, (_, last_agent_net_state, logging_dict) = dloss_dparams(
        learner_state.params,
        agent_net_state=agent_net_state,
        meta_state=learner_state.meta_state,
        rollout=rollout,
        meta_out=meta_out,
        is_meta_training=is_meta_training,
    )
    # Average gradients across the other learner devices involved in the `pmap`.
    if self._batch_axis_name is not None:
      grads = jax.lax.pmean(grads, axis_name=self._batch_axis_name)
    # Use the optimizer to apply a suitable gradient transformation.
    updates, new_opt_state = self._optimizer.update(
        grads, learner_state.opt_state, learner_state.params
    )
    # Update parameters.
    new_params = optax.apply_updates(learner_state.params, updates)

    # Log the gradient and update norms.
    logging_dict['global_gradient_norm'] = optax.global_norm(grads)
    logging_dict['global_update_norm'] = optax.global_norm(updates)
    logging_dict['meta_out'] = meta_out
    # Return the updated learner state and logging outputs.
    learner_state = LearnerState(
        params=new_params, opt_state=new_opt_state, meta_state=new_meta_state
    )
    return learner_state, last_agent_net_state, logging_dict


def get_settings_disco():
  """Disco-103 setting."""
  return config_dict.ConfigDict(
      dict(
          # discovered objective
          hyper_params=dict(
              pi_cost=1.0,
              y_cost=1.0,
              z_cost=1.0,
              value_cost=0.2,
              aux_policy_cost=1.0,
              target_params_coeff=0.9,
              value_fn_td_lambda=0.95,
              discount_factor=0.997,
          ),
          update_rule_name='disco',
          update_rule=dict(
              net=config_dict.ConfigDict(
                  dict(
                      name='lstm',
                      prediction_size=600,
                      hidden_size=256,
                      embedding_size=(16, 1),
                      policy_target_channels=(16,),
                      policy_channels=(16, 2),
                      output_stddev=0.3,
                      aux_stddev=0.3,
                      policy_target_stddev=0.3,
                      state_stddev=1.0,
                      meta_rnn_kwargs=dict(
                          policy_channels=(16, 2),
                          embedding_size=(16,),
                          pred_embedding_size=(16, 1),
                          hidden_size=128,
                      ),
                      input_option=disco.get_input_option(),
                  )
              ),
              value_discount=0.997,
              num_bins=601,
              max_abs_value=300.0,
          ),
          # optimizer.
          learning_rate=0.0003,
          max_abs_update=1.0,
          # agent network.
          net_settings=dict(
              name='mlp',
              net_args=dict(
                  dense=(512, 512),
                  model_arch_name='lstm',
                  head_w_init_std=1e-2,
                  model_kwargs=dict(
                      head_mlp_hiddens=(128,),
                      lstm_size=128,
                  ),
              ),
          ),
      )
  )


def get_settings_actor_critic():
  """Actor-Critic setting."""
  return config_dict.ConfigDict(
      dict(
          hyper_params=dict(
              discount_factor=0.997,
              vtrace_lambda=0.95,
              entropy_cost=0.2,
              pg_cost=1.0,
              value_cost=0.5,
          ),
          update_rule_name='actor_critic',
          update_rule=config_dict.ConfigDict(
              dict(
                  categorical_value=True,
                  num_bins=601,
                  max_abs_value=300.0,
                  nonlinear_value_transform=True,
                  normalize_adv=False,
                  normalize_td=False,
              )
          ),
          # optimizer.
          learning_rate=5e-4,
          max_abs_update=1.0,
          # agent network.
          net_settings=dict(
              name='mlp',
              net_args=dict(
                  dense=(64, 32, 32),
                  model_arch_name='lstm',
                  head_w_init_std=1e-2,
                  model_kwargs=dict(
                      head_mlp_hiddens=(64,),
                      lstm_size=64,
                  ),
              ),
          ),
      ),
  )
