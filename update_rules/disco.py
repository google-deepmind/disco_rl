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

"""DiscoRL update rule."""

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import rlax

from disco_rl import types
from disco_rl import utils
from disco_rl.networks import meta_nets
from disco_rl.update_rules import base
from disco_rl.value_fns import value_utils


class DiscoUpdateRule(base.UpdateRule):
  """Discoved update rule, as described in the paper."""

  def __init__(
      self,
      net: config_dict.ConfigDict,
      value_discount: float,
      max_abs_value: float,
      num_bins: int,
      moving_average_decay: float = 0.99,
      moving_average_eps: float = 1e-6,
  ) -> None:
    """The meta-network constructor."""
    super().__init__()
    self._prediction_size = net.prediction_size
    self._value_discount = value_discount
    self._max_abs_value = max_abs_value
    self._num_bins = num_bins

    # Exponential moving averages.
    self._adv_ema = utils.MovingAverage(
        jnp.zeros(()), decay=moving_average_decay, eps=moving_average_eps
    )
    self._td_ema = utils.MovingAverage(
        jnp.zeros(()), decay=moving_average_decay, eps=moving_average_eps
    )

    # Meta-network.
    def meta_net_fn(*args, **kwargs):
      if net['name'] == 'lstm':
        return meta_nets.LSTM(**net)(*args, **kwargs)
      else:
        raise ValueError(f'Invalid model network name: {net["name"]}.')

    self._eta_init_fn, self._eta_apply = hk.transform_with_state(meta_net_fn)

  def init_params(
      self, rng: chex.PRNGKey
  ) -> tuple[types.MetaParams, chex.ArrayTree]:
    dummy_input = self._get_dummy_input(
        include_behaviour_out=True, include_agent_adv=True
    )
    meta_params, meta_rnn_state = self._eta_init_fn(
        rng, dummy_input, axis_name=None
    )
    return meta_params, meta_rnn_state

  def init_meta_state(
      self,
      rng: chex.PRNGKey,
      params: types.AgentParams,
  ) -> types.MetaState:
    """Create meta_state."""
    _, meta_rnn_state = self.init_params(rng)
    meta_state = dict(
        rnn_state=meta_rnn_state,
        adv_ema_state=self._adv_ema.init_state(),
        td_ema_state=self._td_ema.init_state(),
        target_params=params,
    )
    return meta_state

  def flat_output_spec(
      self, single_action_spec: types.ActionSpec
  ) -> types.Specs:
    return dict(
        logits=utils.get_logits_specs(single_action_spec),
        y=types.ArraySpec((self._prediction_size,), jnp.float32),
    )

  def model_output_spec(
      self, single_action_spec: types.ActionSpec
  ) -> types.Specs:
    return dict(
        z=types.ArraySpec((self._prediction_size,), jnp.float32),
        aux_pi=utils.get_logits_specs(single_action_spec),
        q=types.ArraySpec((self._num_bins,), jnp.float32),
    )

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
    del rng
    t, b = rollout.rewards.shape

    chex.assert_shape((rollout.rewards, rollout.is_terminal), (t, b))
    chex.assert_tree_shape_prefix(
        (rollout.agent_out, rollout.actions), (t + 1, b)
    )

    # Unroll the target policy.
    target_out, _ = unroll_policy_fn(
        meta_state['target_params'],
        state,
        rollout.observations,
        rollout.should_reset_mask_fwd,
    )

    # TD-value targets.
    value_outs, adv_ema_state, td_ema_state = value_utils.get_value_outs(
        value_net_out=None,
        target_value_net_out=None,
        q_net_out=rollout.agent_out['q'],
        target_q_net_out=target_out['q'],
        rollout=rollout,
        pi_logits=rollout.agent_out['logits'],
        discount=self._value_discount,
        lambda_=hyper_params['value_fn_td_lambda'],
        nonlinear_transform=True,
        categorical_value=True,
        max_abs_value=self._max_abs_value,
        drop_last=False,
        adv_ema_state=meta_state['adv_ema_state'],
        adv_ema_fn=self._adv_ema,
        td_ema_state=meta_state['td_ema_state'],
        td_ema_fn=self._td_ema,
        axis_name=axis_name,
    )

    # Apply the meta-network.
    rollout.extra_from_rule = dict(
        v_scalar=value_outs.value,
        adv=value_outs.adv,
        normalized_adv=value_outs.normalized_adv,
        q=value_outs.target_q_value,
        qv_adv=value_outs.qv_adv,
        normalized_qv_adv=value_outs.normalized_qv_adv,
        target_out=target_out,
    )
    meta_out, new_rnn_state = self._eta_apply(
        meta_params,
        meta_state['rnn_state'],
        None,  # unused rng
        rollout,
        axis_name=axis_name,
    )
    chex.assert_rank(meta_out['pi'], 3)  # [T, B, A]
    chex.assert_rank(meta_out['y'], 3)  # [T, B, Y]
    chex.assert_rank(meta_out['z'], 3)  # [T, B, Y]

    # Enrich the meta-net's outputs with the value function's outputs.
    meta_out['q_target'] = value_outs.q_target
    meta_out['adv'] = value_outs.adv
    meta_out['normalized_adv'] = value_outs.normalized_adv
    meta_out['qv_adv'] = value_outs.qv_adv
    meta_out['normalized_qv_adv'] = value_outs.normalized_qv_adv
    meta_out['q_value'] = value_outs.q_value
    meta_out['q_td'] = value_outs.q_td
    meta_out['normalized_q_td'] = value_outs.normalized_q_td
    meta_out['target_out'] = target_out

    # Update the meta state.
    new_meta_state = meta_state | dict(
        rnn_state=new_rnn_state,
        adv_ema_state=adv_ema_state,
        td_ema_state=td_ema_state,
    )

    # Update target params.
    coeff = hyper_params['target_params_coeff']
    new_meta_state['target_params'] = jax.tree.map(
        lambda old, new: old * coeff + (1.0 - coeff) * new,
        meta_state['target_params'],
        params,
    )

    return meta_out, new_meta_state

  def agent_loss(
      self,
      rollout: types.UpdateRuleInputs,
      meta_out: types.UpdateRuleOuts,
      hyper_params: types.HyperParams,
      backprop: bool,
  ) -> tuple[chex.Array, types.UpdateRuleLog]:
    """Defines an agent loss."""
    t, b = rollout.rewards.shape

    chex.assert_shape((rollout.rewards, rollout.is_terminal), (t, b))
    chex.assert_tree_shape_prefix(
        (rollout.agent_out, rollout.actions), (t + 1, b)
    )

    # Parse the agent's output.
    agent_out, actions = jax.tree.map(
        lambda x: x[:-1], (rollout.agent_out, rollout.actions)
    )
    logits = agent_out['logits']
    y = agent_out['y']
    z = agent_out['z']
    z_a = utils.batch_lookup(agent_out['z'], actions)

    # Parse the meta-net's output.
    pi_hat = meta_out['pi']
    y_hat = meta_out['y']
    z_hat = meta_out['z']
    if not backprop:
      pi_hat, y_hat, z_hat = jax.lax.stop_gradient((pi_hat, y_hat, z_hat))

    # Compute losses.
    chex.assert_equal_shape([pi_hat, logits])  # [T, B, A]
    chex.assert_equal_shape([y_hat, y])  # [T, B, Y]
    chex.assert_equal_shape([z_hat, z_a])  # [T, B, Z]
    pi_loss_per_step = rlax.categorical_kl_divergence(pi_hat, logits)
    y_loss_per_step = rlax.categorical_kl_divergence(y_hat, y)
    z_loss_per_step = rlax.categorical_kl_divergence(z_hat, z_a)

    # Compute auxiliary 1-step policy prediction loss.
    aux_pi = rollout.agent_out['aux_pi'][:-1]  # [T, B, A, A]
    aux_pi_a = utils.batch_lookup(aux_pi, actions)  # [T, B, A]
    aux_policy_target = rollout.agent_out['logits'][1:]  # [T, B, A]
    aux_policy_loss_per_step = rlax.categorical_kl_divergence(
        jax.lax.stop_gradient(aux_policy_target), aux_pi_a
    )
    # Mask out terminal states.
    aux_policy_loss_per_step *= 1.0 - rollout.is_terminal

    # Compute total loss.
    chex.assert_shape(
        (
            pi_loss_per_step,
            y_loss_per_step,
            z_loss_per_step,
            aux_policy_loss_per_step,
        ),
        (t, b),  # [T, B]
    )
    total_loss_per_step = (
        hyper_params['pi_cost'] * pi_loss_per_step
        + hyper_params['y_cost'] * y_loss_per_step
        + hyper_params['z_cost'] * z_loss_per_step
        + hyper_params['aux_policy_cost'] * aux_policy_loss_per_step
    )

    log = dict(
        logits=jnp.mean(logits),
        y=jnp.mean(y),
        z=jnp.mean(z),
        entropy=jnp.mean(distrax.Softmax(logits).entropy()),
        y_entropy=jnp.mean(distrax.Softmax(y).entropy()),
        z_entropy=jnp.mean(distrax.Softmax(z_a).entropy()),
        pi_loss=jnp.mean(pi_loss_per_step),
        aux_kl_loss=jnp.mean(aux_policy_loss_per_step),
        pi_hat=jnp.mean(pi_hat),
        y_hat=jnp.mean(y_hat),
        z_hat=jnp.mean(z_hat),
        pi_hat_entropy=jnp.mean(distrax.Softmax(pi_hat).entropy()),
        aux_policy_entropy=jnp.mean(distrax.Softmax(aux_pi_a).entropy()),
        aux_target_entropy=jnp.mean(
            distrax.Softmax(aux_policy_target).entropy()
        ),
    )
    return total_loss_per_step, log

  def agent_loss_no_meta(
      self,
      rollout: types.UpdateRuleInputs,
      meta_out: types.UpdateRuleOuts | None,
      hyper_params: types.HyperParams,
  ) -> tuple[chex.Array, types.UpdateRuleLog]:
    """Value losses that do not interfere with meta-gradient."""
    assert meta_out is not None
    td = meta_out['q_td']

    q_a = utils.batch_lookup(rollout.agent_out['q'], rollout.actions)[:-1]
    value_loss_per_step = value_utils.value_loss_from_td(
        value_net_out=q_a,
        td=jax.lax.stop_gradient(td),
        nonlinear_transform=True,
        categorical_value=True,
        max_abs_value=self._max_abs_value,
    )
    loss_per_step = value_loss_per_step * hyper_params['value_cost']

    log = dict(
        q_target=jnp.mean(meta_out['q_target']),
        agent_q_mean=jnp.mean(meta_out['q_value']),
        q_loss=jnp.mean(value_loss_per_step),
        td=jnp.mean(meta_out['q_td']),
        normalized_td=jnp.mean(meta_out['normalized_q_td']),
    )
    return loss_per_step, log


def get_input_option() -> types.MetaNetInputOption:
  """Returns the input option for the meta-network.

  Detailed description can be found in the paper.

  Returns:
    The input option for the meta-network.
  """
  return types.MetaNetInputOption(
      base=(
          types.TransformConfig(
              source='agent_out/logits',
              transforms=('drop_last', 'softmax', 'stop_grad', 'select_a'),
          ),
          types.TransformConfig(
              source='behaviour_agent_out/logits',
              transforms=('drop_last', 'softmax', 'stop_grad', 'select_a'),
          ),
          types.TransformConfig(source='rewards', transforms=('sign_log',)),
          types.TransformConfig(
              source='is_terminal',
              transforms=('masks_to_discounts',),
          ),
          types.TransformConfig(
              source='extra_from_rule/v_scalar',
              transforms=('sign_log', 'td_pair', 'stop_grad'),
          ),
          types.TransformConfig(
              source='extra_from_rule/adv', transforms=('sign_log', 'stop_grad')
          ),
          types.TransformConfig(
              source='extra_from_rule/normalized_adv', transforms=('stop_grad',)
          ),
          types.TransformConfig(
              source='extra_from_rule/target_out/logits',
              transforms=('drop_last', 'softmax', 'stop_grad', 'select_a'),
          ),
          types.TransformConfig(
              source='agent_out/y', transforms=('softmax', 'y_net', 'td_pair')
          ),
          types.TransformConfig(
              source='extra_from_rule/target_out/y',
              transforms=('softmax', 'y_net', 'td_pair'),
          ),
          types.TransformConfig(
              source='agent_out/z',
              transforms=('drop_last', 'softmax', 'z_net', 'select_a'),
          ),
          types.TransformConfig(
              source='agent_out/z',
              transforms=('softmax', 'z_net', 'pi_weighted_avg', 'td_pair'),
          ),
          types.TransformConfig(
              source='agent_out/z',
              transforms=('softmax', 'z_net', 'max_a', 'td_pair'),
          ),
          types.TransformConfig(
              source='extra_from_rule/target_out/z',
              transforms=('drop_last', 'softmax', 'z_net', 'select_a'),
          ),
          types.TransformConfig(
              source='extra_from_rule/target_out/z',
              transforms=('softmax', 'z_net', 'pi_weighted_avg', 'td_pair'),
          ),
          types.TransformConfig(
              source='extra_from_rule/target_out/z',
              transforms=('softmax', 'z_net', 'max_a', 'td_pair'),
          ),
      ),
      action_conditional=(
          types.TransformConfig(
              source='agent_out/logits',
              transforms=('drop_last', 'softmax', 'stop_grad'),
          ),
          types.TransformConfig(
              source='behaviour_agent_out/logits',
              transforms=('drop_last', 'softmax', 'stop_grad'),
          ),
          types.TransformConfig(
              source='extra_from_rule/target_out/logits',
              transforms=('drop_last', 'softmax', 'stop_grad'),
          ),
          types.TransformConfig(
              source='agent_out/z', transforms=('drop_last', 'softmax', 'z_net')
          ),
          types.TransformConfig(
              source='extra_from_rule/target_out/z',
              transforms=('drop_last', 'softmax', 'z_net'),
          ),
          types.TransformConfig(
              source='extra_from_rule/q',
              transforms=('sign_log', 'drop_last', 'stop_grad'),
          ),
          types.TransformConfig(
              source='extra_from_rule/qv_adv',
              transforms=('sign_log', 'drop_last', 'stop_grad'),
          ),
          types.TransformConfig(
              source='extra_from_rule/normalized_qv_adv',
              transforms=('drop_last', 'stop_grad'),
          ),
      ),
  )
