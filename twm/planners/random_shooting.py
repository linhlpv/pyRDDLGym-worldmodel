import gymnasium as gym
import numpy as np
import os
import torch
from tqdm import tqdm
from typing import Tuple

from twm.core.data import PLOTS_PATH
from twm.core.env import DiscreteActionWrapper, WorldModelEnv


class RandomShootingMPC:
    '''Model-predictive controller using random shooting through a world model.'''

    def __init__(self, rollout_env: WorldModelEnv, real_env: gym.Env,
                 lookahead: int, num_parallel_evals: int=32) -> None:
        self.rollout_env = rollout_env
        self.real_env = real_env
        self.lookahead = lookahead
        self.num_parallel_evals = num_parallel_evals

        self._action_lut = DiscreteActionWrapper.build_action_lut(rollout_env)
        self._obs_history = []
        self._action_history = []
    
    def _batched(self, arr):
        '''Convert a numpy array to a batch of tensors on the rollout device.'''
        t = torch.as_tensor(arr).float().to(self.rollout_env.device)
        return t[None].expand(self.num_parallel_evals, *t.shape).clone()

    def _align_world_model(self):
        '''Reset the world model rollout context from the real env history.'''
        wm = self.rollout_env
        seq_len = wm.world_model.seq_len
        env_spec = wm.world_model.env_spec

        obs_hist = self._obs_history[-seq_len:]
        act_hist = self._action_history[-(seq_len - 1):]

        # stack and batch observations, ensuring correct shapes and types
        init_states = {
            key: self._batched(np.stack([np.asarray(o[key]) for o in obs_hist], axis=0))
            for key in env_spec.state_spec
        }

        # stack and batch actions, padding with zeros if history is too short
        init_actions = {}
        for key, spec in env_spec.action_spec.items():
            if act_hist:
                acts = np.stack([np.asarray(a[key]) for a in act_hist], axis=0)
            else:
                acts = np.zeros((0, *spec.shape))
            zero_pad = np.zeros((1, *spec.shape))
            init_actions[key] = self._batched(np.concatenate([acts, zero_pad], axis=0))

        wm.rollout.reset(init_states, init_actions)

    def _make_action_batch(self, action_indices):
        '''LUT indices → (tensor_dict, numpy_dict) with batch dim N.'''
        decoded = [self._action_lut[i] for i in action_indices]
        tensor_dict = {}
        for key in self.rollout_env.world_model.env_spec.action_spec:
            arr = np.stack([np.asarray(a[key]) for a in decoded], axis=0)
            tensor_dict[key] = torch.as_tensor(arr).to(self.rollout_env.device)
        return tensor_dict

    def _estimate_return(self, action_idx):
        '''Estimate return for a fixed first action with random-shooting continuations.'''
        N = self.num_parallel_evals
        rollout = self.rollout_env.rollout
        indices = np.full((N,), action_idx, dtype=np.int64)
        returns = np.zeros(N, dtype=np.float32)
        horizon = min(self.lookahead, self.rollout_env.max_steps)

        self._align_world_model()
        
        for _ in range(horizon):
            obs = rollout.last_states()
            action = self._make_action_batch(indices)
            rollout.step(action)
            next_obs = rollout.last_states()
            rewards = self.rollout_env.reward_fn(obs, action, next_obs)
            returns += rewards.detach().cpu().numpy().ravel()
            indices = np.random.randint(len(self._action_lut), size=N)

        return float(returns.mean())

    def _select_action(self):
        '''Return the LUT index with the highest estimated return.'''
        best_idx, best_return = 0, -np.inf
        for idx in range(len(self._action_lut)):
            ret = self._estimate_return(idx)
            if ret > best_return:
                best_return = ret
                best_idx = idx
        return best_idx

    def reset(self) -> None:
        '''Reset the real environment and clear history.'''
        obs, _ = self.real_env.reset()
        self._obs_history = [obs]
        self._action_history = []
        self.frames = []

    def step(self, save_frames: bool=True) -> Tuple:
        ''''Take one step in the real environment using the MPC-selected action.'''
        lut_idx = self._select_action()
        action = self._action_lut[lut_idx]
        obs, reward, term, trunc, info = self.real_env.step(action)
        if save_frames:
            self.frames.append(self.real_env.render())
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
            for _ in (pbar := tqdm(range(max_steps), desc='Running MPC')):
                _, reward, term, trunc, _ = self.step(save_frames=save_frames)
                total += reward
                if term or trunc:
                    break
                pbar.set_postfix({'Cuml Return': f'{total:.3f}'})
            avg += total / episodes

        if save_frames:
            self.frames[0].save(
                fp=os.path.join(PLOTS_PATH, plot_name),
                format='GIF', append_images=self.frames[1:], save_all=True, duration=100)
            
        return avg
