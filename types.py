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

"""Types."""

from typing import Any, Callable, Mapping, Sequence

import chex
import dm_env
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray
ArraySpec = jax.ShapeDtypeStruct
ActionSpec = specs.BoundedArray
Specs = dict[str, specs.Array | ArraySpec]
SpecsTree = ArraySpec | Sequence['SpecsTree'] | dict[str, 'SpecsTree']
MetaState = dict[str, chex.ArrayTree | None]
UpdateRuleLog = dict[str, chex.ArrayTree]

HaikuState = hk.State
AgentOuts = dict[str, chex.ArrayTree]
UpdateRuleOuts = dict[str, chex.ArrayTree]
HyperParams = dict[str, chex.Array | float]

OptState = chex.ArrayTree
AgentParams = chex.ArrayTree
MetaParams = chex.ArrayTree
MetaParamsEMA = dict[float, MetaParams]  # {decay: params}
# State that can be directly updated by update rule.
MetaState = dict[str, chex.ArrayTree | None]
LogDict = dict[str, chex.Array]
RNNState = chex.ArrayTree


# (params, state, observations, should_reset) -> (agent_out, new_state)
AgentStepFn = Callable[
    [AgentParams, HaikuState, chex.ArrayTree, chex.Array | None],
    tuple[AgentOuts, HaikuState],
]
AgentUnrollFn = Callable[
    [
        AgentParams,
        HaikuState,
        chex.ArrayTree,
        chex.Array | None,
    ],
    tuple[AgentOuts, HaikuState],
]
SampleActionFn = Callable[
    [chex.PRNGKey, MetaParams, AgentOuts],
    tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree],
]


@chex.dataclass
class ValueFnConfig:
  """Value function config."""

  net: str
  net_args: dict[str, Any]
  learning_rate: float
  max_abs_update: float
  discount_factor: float
  td_lambda: float
  outer_value_cost: float
  ema_decay: float = 0.99
  ema_eps: float = 1e-6


@chex.dataclass
class ValueState:
  params: AgentParams
  state: HaikuState
  opt_state: OptState
  adv_ema_state: 'EmaState'
  td_ema_state: 'EmaState'


@chex.dataclass
class TransformConfig:
  source: str
  transforms: Sequence[str | Callable[[Any], chex.Array]]


@chex.dataclass
class MetaNetInputOption:
  """Meta network input options."""

  base: Sequence[TransformConfig]
  action_conditional: Sequence[TransformConfig]


@chex.dataclass(mappable_dataclass=False, frozen=True)
class PolicyNetwork:
  """Collects useful callable transformations of underlying agent network."""

  # hk-transformed functions.
  init: Callable[
      [
          chex.PRNGKey,  # rng for params
          chex.ArrayTree,  # obs
          chex.Array | None,  # should_reset
      ],
      tuple[AgentParams, HaikuState],
  ]
  one_step: AgentStepFn
  unroll: AgentUnrollFn


@chex.dataclass
class EmaState:
  # The tree of first moments.
  moment1: chex.ArrayTree
  # The tree of second moments.
  moment2: chex.ArrayTree
  # The product of the all decays from the start of accumulating.
  decay_product: float


@chex.dataclass
class EnvironmentTimestep:
  observation: Mapping[str, chex.ArrayTree]
  step_type: chex.Array
  reward: chex.Array


@chex.dataclass
class ActorTimestep:
  """Actor timestep."""

  observations: chex.ArrayTree
  actions: Any
  rewards: Any
  discounts: Any
  agent_outs: AgentOuts
  states: HaikuState
  logits: Any

  @classmethod
  def from_rollout(cls, rollout: 'ActorRollout') -> 'ActorTimestep':
    return cls(
        observations=rollout.observations,
        actions=rollout.actions,
        rewards=rollout.rewards,
        discounts=rollout.discounts,
        agent_outs=rollout.agent_outs,
        states=rollout.states,
        logits=rollout.logits,
    )

  def to_env_timestep(self) -> 'EnvironmentTimestep':
    return EnvironmentTimestep(
        observation=self.observations,
        step_type=jnp.where(
            self.discounts > 0, dm_env.StepType.MID, dm_env.StepType.LAST
        ),
        reward=self.rewards,
    )


@chex.dataclass
class ActorRollout(ActorTimestep):
  """Stacked actor timesteps.

  Shapes: [D, O, T, B, ...] (by default; can be changed in the code).

  where:
    D: number of learner devices
    O: outer rollout length (aka, meta rollout length)
    T: trajectory length
    B: batch size
  """

  @classmethod
  def from_timestep(cls, timestep: ActorTimestep) -> 'ActorRollout':
    return cls(**timestep)

  def first_state(self, time_axis: int) -> HaikuState:
    index = tuple([np.s_[:]] * (time_axis - 1) + [0])
    return jax.tree.map(lambda x: x[index], self.states)


@chex.dataclass
class ValueOuts:
  """Value function outputs."""

  value: jax.typing.ArrayLike = 0.0  # Scalar value
  target_value: jax.typing.ArrayLike = 0.0  # Scalar target value
  rho: jax.typing.ArrayLike = 0.0  # Importance weight
  adv: jax.typing.ArrayLike = 0.0  # Advantage
  normalized_adv: jax.typing.ArrayLike = 0.0  # Normalised advantage
  value_target: jax.typing.ArrayLike = 0.0  # Value target
  td: jax.typing.ArrayLike = 0.0  # value_target - value
  normalized_td: jax.typing.ArrayLike = 0.0  # Normalised TD
  qv_adv: chex.ArrayTree | None = None  # Q - V
  normalized_qv_adv: chex.ArrayTree | None = None  # Normalised Q - V
  q_value: chex.ArrayTree | None = None  # Scalar Q-value
  target_q_value: chex.ArrayTree | None = None  # Scalar target Q-value
  q_target: chex.ArrayTree | None = None  # Q-value target
  q_td: chex.ArrayTree | None = None  # q_target - q_a
  normalized_q_td: chex.ArrayTree | None = None  # Normalised q_td


@chex.dataclass
class UpdateRuleInputs:
  """Update rule inputs."""

  observations: chex.ArrayTree
  actions: chex.Array
  rewards: chex.Array
  is_terminal: chex.Array  # whether the action was terminal
  agent_out: chex.ArrayTree
  behaviour_agent_out: AgentOuts | None = None
  value_out: ValueOuts | None = None
  # Inputs with pre-processing in update rule before meta-net (e.g. advantages)
  extra_from_rule: chex.ArrayTree | None = None

  @property
  def should_reset_mask_fwd(self) -> chex.Array:
    """Returns `should_reset` mask for forward RNNs."""
    # Shifts is_terminal to the right by a step, mimicking step_type.is_first().
    prepend_non_terminal = jnp.zeros_like(self.is_terminal[:1])
    return jnp.concatenate(
        (prepend_non_terminal, self.is_terminal),
        axis=0,
        dtype=self.is_terminal.dtype,
    )

  @property
  def should_reset_mask_bwd(self) -> chex.Array:
    """Returns `should_reset` mask for backward RNNs."""
    # Appends one non-terminal step, for bootstrapping.
    append_non_terminal = jnp.zeros_like(self.is_terminal[:1])
    return jnp.concatenate(
        (self.is_terminal, append_non_terminal),
        axis=0,
        dtype=self.is_terminal.dtype,
    )
