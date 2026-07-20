import os
from typing import Callable, Dict

import numpy as np
import torch
import pyRDDLGym
from tqdm import tqdm

from twm.core.data import get_dataloader, PLOTS_PATH, \
    plot_data_trajectories, plot_trajectories, save_video
from twm.core.env import WorldModelEnv
from twm.core.spec import EnvSpec, FluentSpec
from twm.core.model import WorldModel, WorldModelEvaluator, EMA
from twm.planners.plan_by_backprop import PlanByBackpropMPC
from twm.planners.random_shooting import RandomShootingMPC
from twm.trainner.offline_trainer import OfflineTrainer

Tensor = torch.Tensor
TensorDict = Dict[str, Tensor]
Array = np.ndarray
ArrayDict = Dict[str, Array]


class OnlineTrainer(OfflineTrainer):
    '''
    Base trainer class for Online Planning with Learned World Model.
    '''
    def __init__(self, world_model: WorldModel, real_env: pyRDDLGym.RDDLEnv,
                 reward_fn: Callable, initial_state: ArrayDict | TensorDict,
                 planner_type: str='random_shooting', offline_data_dir: str='pong_data.pkl',
                 pretrained_wm_epoch: int=50, wm_batch_size: int=1024, wm_lr: float=1e-3, lr_decay: float=0.9, wm_name: str='pong_world_model_8_offline.pth',
                 seq_len: int=8, device: str='cuda') -> None:
        super().__init__(world_model, real_env, reward_fn, initial_state,
                         planner_type, offline_data_dir, pretrained_wm_epoch,
                         wm_batch_size, wm_lr, lr_decay, wm_name,
                         seq_len, device)

        # Create a buffer to store the new data collected during online planning
        self.buffer = SequenceDataset(self.train_loader)
        self.finetune_epochs = 5

    def _update_world_model(self) -> None:
        '''Update the world model with the new data in the buffer.'''
        self.finetune_loader = get_dataloader_from_buffer(self.buffer, seq_len=self.seq_len, batch_size=self.wm_batch_size)
        for epoch in range(self.finetune_epochs):
            self.world_model.train()
            # training loop
            epoch_loss = 0.0
            for batch_data in tqdm(self.finetune_loader, desc=f'Epoch {epoch+1}/{self.finetune_epochs}'):
                loss = self.world_model.batch_loss(batch_data, self.device)
                self.wm_optim.zero_grad()
                loss.backward()
                self.wm_optim.step()
                self.ema.update(self.world_model)
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(self.finetune_loader)
            self.wm_scheduler.step(avg_loss)
    
    def solve(self, plot_name: str, max_steps: int=200, episodes: int=1,
            save_frames: bool=True) -> float:
        '''Solve the planning problem using the planner.'''
        avg = 0.0
        for _ in range(episodes):
            total = 0.0
            self.planner.reset()
            for _ in (pbar := tqdm(range(max_steps), desc='Running MPC')):
                obs, action, reward, term, trunc, info = self.planner.step(save_frames=save_frames)
                total += reward
                if term or trunc:
                    break
                pbar.set_postfix({'Cuml Return': f'{total:.3f}'})
                transition = {'obs': obs, 'action': action, 'reward': reward, 'term': term, 'trunc': trunc, 'info': info}
                self.buffer.add_transition(transition)
            avg += total / episodes

        # Update the world model with the new data in the buffer
        self._update_world_model()

        if not os.path.exists(PLOTS_PATH):
            os.makedirs(PLOTS_PATH)

        if save_frames:
            self.planner.frames[0].save(
                fp=os.path.join(PLOTS_PATH, plot_name),
                format='GIF', append_images=self.planner.frames[1:], save_all=True, duration=100)

        return avg