import copy
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from twm.core.data import save_gif
from twm.core.env import WorldModelEnv
from twm.core.types import TensorDict


class PolicyRepresentation(nn.Module, ABC):
    '''Abstract policy representation for plan-by-backprop.'''

    def __init__(self) -> None:
        super().__init__()
        self.rollout_env = None

    def bind(self, rollout_env: WorldModelEnv) -> None:
        '''Attach world-model context to the policy representation.'''
        self.rollout_env = rollout_env

    def begin_optimization(self, horizon: int) -> None:
        '''Optional hook run before each planning optimization.'''
        del horizon

    def begin_decision_epoch(self, warm_start: bool=True) -> None:
        '''Optional hook run at each MPC decision epoch before optimization.'''
        del warm_start

    def begin_rollout(self, horizon: int) -> None:
        '''Optional hook run before each imagined rollout.'''
        del horizon

    @abstractmethod
    def optim_parameters(self):
        '''Return iterable of parameters that should be optimized by planner.'''
        raise NotImplementedError

    @abstractmethod
    def action(self, obs: TensorDict, step: int) -> TensorDict:
        '''Return differentiable action tensors for the current imagined step.'''
        raise NotImplementedError

    def _decode_action_tensors(self, action_tensors: Dict[str, torch.Tensor]
                               ) -> Dict[str, torch.Tensor]:
        '''Decode raw policy outputs (logits/raw) into executable action tensors.'''
        if self.rollout_env is None:
            raise RuntimeError('Policy is not bound to rollout_env.')

        result = {}
        wm = self.rollout_env.world_model
        for key, param in action_tensors.items():
            spec = wm.env_spec.action_spec[key]

            # convert logits to probabilities over discrete actions
            if spec.prange in ('int', 'bool'):
                low, high = wm.spec_bounds(key)
                n_classes = high - low + 1
                if param.shape[-1] != n_classes:
                    raise ValueError(
                        f'Discrete action {key} must have trailing class dim '
                        f'{n_classes}, got shape {tuple(param.shape)}.')
                result[key] = F.softmax(param, dim=-1)

            # for real-valued actions, optionally apply sigmoid and rescale to bounds
            elif spec.prange == 'real':
                if spec.values is not None:
                    low, high = spec.values
                    if np.isfinite(low) and np.isfinite(high):
                        result[key] = low + (high - low) * torch.sigmoid(param)
                    else:
                        result[key] = param
                else:
                    result[key] = param
            
            else:
                raise ValueError(
                    f'Unsupported action range for plan-by-backprop: {key} ({spec.prange}).')
        return result

    @staticmethod
    def _encode_obs(obs: Dict[str, Any], wm: WorldModelEnv, device: torch.device,
                    add_batch: bool = True) -> TensorDict:
        '''Encode raw observation dict into tensor dict with discrete/continuous handling.'''
        result = {}
        for key, spec in wm.env_spec.state_spec.items():
            value = torch.as_tensor(obs[key], device=device)

            # convert discrete obs to one-hot encoding
            if spec.prange in ('int', 'bool'):
                low, high = wm.spec_bounds(key)
                n_classes = high - low + 1
                if value.dim() == len(spec.shape):
                    value = F.one_hot(value.long() - low, n_classes)
                elif value.shape[-1] != n_classes:
                    raise ValueError(
                        f'Discrete obs {key} must be labels or have trailing class dim '
                        f'{n_classes}, got shape {tuple(value.shape)}.')

            # add batch dim if needed (for policy input consistency)
            expected_dim = len(spec.shape) + int(spec.prange in ('int', 'bool'))
            if add_batch and value.dim() == expected_dim:
                value = value.unsqueeze(0)
            
            # ensure tensor is float for policy input
            result[key] = value.float()
        return result


class SLP(PolicyRepresentation):
    '''Sequence-level plan representation (open-loop): optimize action tensors directly.'''

    def __init__(self) -> None:
        super().__init__()
        self.plan_params = nn.ParameterDict()
        self._decoded_plan = None
        self._decoded_steps = None

    def begin_optimization(self, horizon: int) -> None:
        if self.rollout_env is None:
            raise RuntimeError('Policy is not bound to rollout_env.')

        params = nn.ParameterDict()
        device = self.rollout_env.device
        wm = self.rollout_env.world_model

        for key, spec in wm.env_spec.action_spec.items():
            if spec.prange in ('int', 'bool'):
                low, high = wm.spec_bounds(key)
                n_classes = high - low + 1
                shape = (1, horizon, *spec.shape, n_classes)
            elif spec.prange == 'real':
                shape = (1, horizon, *spec.shape)
            else:
                raise ValueError(
                    f'Unsupported action range for plan-by-backprop: {key} ({spec.prange}).')
            params[key] = nn.Parameter(torch.zeros(shape, device=device))

        self.plan_params = params

    def optim_parameters(self):
        return self.plan_params.parameters()

    def begin_rollout(self, horizon: int) -> None:
        # decode once per rollout to avoid rebuilding a full-horizon graph at every step
        self._decoded_plan = self._decode_action_tensors(dict(self.plan_params))
        self._decoded_steps = [
            {key: tensor[:, step] for key, tensor in self._decoded_plan.items()}
            for step in range(horizon)
        ]

    def action(self, obs: TensorDict, step: int) -> TensorDict:
        del obs
        assert self._decoded_steps is not None, \
            'begin_rollout must be called before action.'
        return self._decoded_steps[step]


class DRP(PolicyRepresentation):
    '''Deep reactive policy sample network: maps current observation to action.'''

    def __init__(self, hidden_dim: int=128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.trunk = None
        self.heads = nn.ModuleDict()
        self.state_keys = []
        self._initial_state_dict = None

    def bind(self, rollout_env: WorldModelEnv) -> None:
        super().bind(rollout_env)
        wm = rollout_env.world_model
        self.state_keys = list(wm.env_spec.state_spec.keys())

        # compute input dimension by flattening and concatenating all state components
        state_dim = 0
        for key, spec in wm.env_spec.state_spec.items():
            if spec.prange in ('int', 'bool'):
                low, high = wm.spec_bounds(key)
                state_dim += spec.size * (high - low + 1)
            else:
                state_dim += spec.size

        # simple MLP with separate heads for each action component
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        ).to(rollout_env.device)

        # compute output dimensions for each action component and create heads
        self.heads = nn.ModuleDict()
        for key, spec in wm.env_spec.action_spec.items():
            if spec.prange in ('int', 'bool'):
                low, high = wm.spec_bounds(key)
                out_dim = spec.size * (high - low + 1)
            elif spec.prange == 'real':
                out_dim = spec.size
            else:
                raise ValueError(
                    f'Unsupported action range for plan-by-backprop: {key} ({spec.prange}).')
            self.heads[key] = nn.Linear(self.hidden_dim, out_dim)

        self.to(rollout_env.device)

        # snapshot initial parameters for optional per-decision reinitialization
        self._initial_state_dict = {
            k: v.detach().clone() for k, v in self.state_dict().items()
        }

    def begin_decision_epoch(self, warm_start: bool=True) -> None:
        if warm_start:
            return
        if self._initial_state_dict is None:
            raise RuntimeError('DRP initial state snapshot is not available.')
        self.load_state_dict(self._initial_state_dict)

    def optim_parameters(self):
        return self.parameters()

    def _flatten_obs(self, obs: TensorDict) -> torch.Tensor:
        if self.rollout_env is None:
            raise RuntimeError('Policy is not bound to rollout_env.')
        wm = self.rollout_env.world_model

        # normalize obs shape so flattening always sees a leading batch dimension
        encoded = self._encode_obs(obs, wm, self.rollout_env.device, add_batch=True)
        flats = []
        for key in self.state_keys:
            x = encoded[key]
            flats.append(x.reshape(x.shape[0], -1))
        return torch.cat(flats, dim=-1)

    def action(self, obs: TensorDict, step: int) -> TensorDict:
        del step
        if self.trunk is None or self.rollout_env is None:
            raise RuntimeError('Policy is not bound to rollout_env.')
        wm = self.rollout_env.world_model

        z = self.trunk(self._flatten_obs(obs))
        raw = {}
        for key, head in self.heads.items():
            spec = wm.env_spec.action_spec[key]
            y = head(z)
            if spec.prange in ('int', 'bool'):
                low, high = wm.spec_bounds(key)
                n_classes = high - low + 1
                y = y.view(z.shape[0], *spec.shape, n_classes)
            else:
                y = y.view(z.shape[0], *spec.shape)
            raw[key] = y

        return self._decode_action_tensors(raw)


class PlanByBackpropMPC:
    '''Model-predictive controller using gradient-based action optimization.''' 

    def __init__(self, rollout_env: WorldModelEnv, real_env: gym.Env,
                 lookahead: int, opt_steps: int=20, lr: float=0.001,
                 policy: Optional[PolicyRepresentation]=None,
                 drp_warm_start: bool=True) -> None:
        self.rollout_env = rollout_env
        self.real_env = real_env
        self.lookahead = lookahead
        self.opt_steps = opt_steps
        self.lr = lr
        self.drp_warm_start = drp_warm_start
        self.policy = policy if policy is not None else DRP()
        self.policy = self.policy.to(self.rollout_env.device)
        self.policy.bind(self.rollout_env)

        self._obs_history = []
        self._action_history = []

    def _batched(self, arr):
        '''Convert an array to a single-batch tensor on the rollout device.'''
        t = torch.as_tensor(arr).float().to(self.rollout_env.device)
        return t[None].clone()

    def _align_world_model(self, grad: bool=False) -> None:
        '''Reset the world model rollout context from the real env history.'''
        wm = self.rollout_env
        seq_len = wm.world_model.seq_len
        env_spec = wm.world_model.env_spec

        # extract the most recent history of states and actions 
        obs_hist = self._obs_history[-seq_len:]
        act_hist = self._action_history[-(seq_len - 1):]

        # stack and batchify state history tensors
        init_states = {
            key: self._batched(np.stack([np.asarray(o[key]) for o in obs_hist], axis=0))
            for key in env_spec.state_spec
        }

        # stack and batchify action history tensors, with zero padding for missing history
        init_actions = {}
        for key, spec in env_spec.action_spec.items():
            if act_hist:
                acts = np.stack([np.asarray(a[key]).reshape(spec.shape) for a in act_hist], axis=0)
            else:
                acts = np.zeros((0, *spec.shape))
            zero_pad = np.zeros((1, *spec.shape))
            init_actions[key] = self._batched(np.concatenate([acts, zero_pad], axis=0))

        # reset world model rollout context with aligned history tensors
        wm.rollout.reset(init_states, init_actions, grad=grad)

    def _rollout_return(self, horizon: int) -> torch.Tensor:
        '''Compute differentiable return for the current policy representation.'''
        rollout = self.rollout_env.rollout

        # align world model context with real environment history before each rollout
        self._align_world_model(grad=True)
        self.policy.begin_rollout(horizon)

        # perform imagined rollout with current policy and accumulate rewards for return
        total_return = torch.zeros((), device=self.rollout_env.device)
        for step in range(horizon):
            obs = rollout.last_states()
            action = self.policy.action(obs, step)
            rollout.step(action, grad=True)
            next_obs = rollout.last_states()
            reward = self.rollout_env.reward_fn(obs, action, next_obs)
            if not torch.is_tensor(reward):
                raise TypeError('reward_fn must return a torch.Tensor for plan-by-backprop.')
            total_return = total_return + reward.reshape(-1).mean()
        return total_return

    def _latest_obs_tensor(self) -> TensorDict:
        '''Convert latest real-env observation into policy input tensor dict.'''
        wm = self.rollout_env.world_model
        obs = self._obs_history[-1]
        # Use shared encoder with batch dim for policy input
        return self.policy._encode_obs(obs, wm, self.rollout_env.device, add_batch=True)

    def _to_env_action(self, action: TensorDict):
        '''Convert policy action tensors into real-env action dict.'''
        wm = self.rollout_env.world_model
        result = {}
        for key, tensor in action.items():
            spec = wm.env_spec.action_spec[key]
            if spec.prange in ('int', 'bool'):
                low, _ = wm.spec_bounds(key)
                value = tensor.argmax(dim=-1) + low
            else:
                value = tensor
            value = value.squeeze(0).detach().cpu().numpy()
            result[key] = value.item() if np.asarray(value).shape == () else value
        return result

    def _to_real_env_action(self, action) -> Dict[str, Any]:
        '''Convert canonical actions to the representation expected by pyRDDLGym.'''
        result = {}

        for key, value in action.items():
            spec = self.rollout_env.world_model.env_spec.action_spec[key]
            array = np.asarray(value)

            # pyRDDLGym expects single real actions as Python floats
            if spec.prange == 'real' and array.size == 1:
                result[key] = float(array.reshape(()))
            else:
                result[key] = value
            
        return result

    def _select_action(self):
        '''Optimize policy representation and return the first executable action.'''
        horizon = min(self.lookahead, self.rollout_env.max_steps)
        warm_start = self.drp_warm_start if isinstance(self.policy, DRP) else False
        self.policy.begin_decision_epoch(warm_start=warm_start)
        self.policy.begin_optimization(horizon)
        optimizer = torch.optim.Adam(self.policy.optim_parameters(), lr=self.lr)

        # disable world model gradients during policy optimization
        model_params = tuple(self.rollout_env.world_model.parameters())
        requires_grad = [p.requires_grad for p in model_params]
        for param in model_params:
            param.requires_grad_(False)

        # optimize policy parameters
        try:
            for _ in (pbar := tqdm(range(self.opt_steps), desc='Optimizing plan')):
                optimizer.zero_grad()
                total_return = self._rollout_return(horizon)
                (-total_return).backward()
                optimizer.step()
                pbar.set_postfix({'Plan Return': f'{float(total_return.detach()):.3f}'})
        finally:
            # restore world model gradient settings after optimization
            for param, flag in zip(model_params, requires_grad):
                param.requires_grad_(flag)

        # select first action from optimized plan
        with torch.no_grad():
            self.policy.begin_rollout(horizon)
            action = self.policy.action(self._latest_obs_tensor(), step=0)
        return self._to_env_action(action)

    @property
    def last_obs(self):
        '''The most recent observation returned by the real environment.'''
        return self._obs_history[-1]

    @property
    def last_action(self):
        '''The most recently executed canonical action.'''
        if not self._action_history:
            raise RuntimeError('No action has been executed since reset.')
        return self._action_history[-1]

    def reset(self) -> None:
        '''Reset the real environment and clear history.'''
        obs, _ = self.real_env.reset()
        self._obs_history = [obs]
        self._action_history = []
        self.frames = []

    def step(self, save_frames: bool=True) -> Tuple:
        '''Take one step in the real environment using the optimized action.'''
        action = self._select_action()
        env_action = self._to_real_env_action(action)
        obs, reward, term, trunc, info = self.real_env.step(env_action)
        if save_frames:
            self.frames.append(self.real_env.render())
        # Store the canonical action, not the converted environment action
        self._action_history.append(action)
        self._obs_history.append(obs)
        return obs, reward, term, trunc, info

    def run(self, plot_name: str, max_steps: int=200, episodes: int=1,
            save_frames: bool=True) -> float:
        '''Run the MPC agent in the real environment for a given number of episodes.'''
        avg = 0.0
        for _ in range(episodes):
            total = 0.0
            self.reset()
            for _ in (pbar := tqdm(range(max_steps), desc='Running PBP MPC')):
                _, reward, term, trunc, _ = self.step(save_frames=save_frames)
                total += reward
                if term or trunc:
                    break
                pbar.set_postfix({'Cuml Return': f'{total:.3f}'})
            avg += total / episodes

        if save_frames:
            save_gif(self.frames, plot_name)

        return avg
