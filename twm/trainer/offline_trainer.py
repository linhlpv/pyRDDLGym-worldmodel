from typing import Callable

import torch
import pyRDDLGym
from tqdm import tqdm

from twm.core.data import get_dataloader, save_gif
from twm.core.env import WorldModelEnv
from twm.core.model import WorldModel, EMA
from twm.core.types import ArrayDict, TensorDict
from twm.planners.plan_by_backprop import PlanByBackpropMPC
from twm.planners.random_shooting import RandomShootingMPC


class OfflineTrainer:
    '''
    Base trainer class for Offline Planning with Learned World Model.
    '''
    def __init__(self, world_model: WorldModel, real_env: pyRDDLGym.RDDLEnv,
                 reward_fn: Callable, initial_state: ArrayDict | TensorDict,
                 planner_type: str='random_shooting', offline_data_dir: str='pong_data.pkl',
                 pretrained_wm_epoch: int=50, wm_batch_size: int=1024, wm_lr: float=1e-3,
                 lr_decay: float=0.9, wm_name: str='pong_world_model_8_offline.pth',
                 seq_len: int=8, max_steps: int=200, device: str='cuda') -> None:
        self.env_spec = world_model.env_spec
        self.world_model = world_model
        self.real_env = real_env
        self.reward_fn = reward_fn
        self.initial_state = initial_state
        self.planner_type = planner_type

        self.offline_data_dir = offline_data_dir
        self.pretrained_wm_epoch = pretrained_wm_epoch
        self.wm_batch_size = wm_batch_size
        self.wm_lr = wm_lr
        self.lr_decay = lr_decay
        self.wm_name = wm_name
        self.seq_len = seq_len
        self.max_steps = max_steps
        self.device = device

        self.wm_optim = torch.optim.Adam(self.world_model.parameters(), lr=self.wm_lr)
        self.wm_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.wm_optim, factor=self.lr_decay, patience=10, min_lr=1e-5)

        # Pretrain the world model and create the planner
        self._pretrain_world_model()
        self._create_planner(max_steps=self.max_steps)

    def _pretrain_world_model(self) -> None:
        '''Pretrain the world model on the dataset.'''
        self.train_loader, self.test_loader = get_dataloader(
            self.offline_data_dir, seq_len=self.seq_len, batch_size=self.wm_batch_size, augment_starts=False)
        
        self.world_model.set_dataset_stats(self.train_loader.dataset)

        # the EMA must be built after the statistics are registered
        self.ema = EMA(self.world_model)

        for epoch in range(self.pretrained_wm_epoch):
            self.world_model.train()
            # training loop
            epoch_loss = 0.
            for batch_data in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.pretrained_wm_epoch}'):
                loss = self.world_model.batch_loss(batch_data, self.device)
                self.wm_optim.zero_grad()
                loss.backward()
                self.wm_optim.step()
                self.ema.update(self.world_model)
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(self.train_loader)
            self.wm_scheduler.step(avg_loss)

            # evaluation with EMA weights
            train_state = {k: v.clone() for k, v in self.world_model.state_dict().items()}
            self.world_model.load_state_dict(self.ema.state_dict)
            test_loss = self.world_model.evaluate(self.test_loader)
            current_lr = self.wm_optim.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{self.pretrained_wm_epoch}, Train Loss: {avg_loss:.6f}, '
                  f'Test Loss: {test_loss:.6f}, LR: {current_lr:.2e}')
            self.world_model.load_state_dict(train_state)
        
        # save the EMA weights as the final model
        self.world_model.load_state_dict(self.ema.state_dict)
        if self.wm_name:
            self.world_model.save(self.wm_name)
        
    def _create_planner(self, max_steps: int=200) -> None:
        '''Create a planner'''
        self.rollout_env = WorldModelEnv(self.world_model, self.reward_fn,
                                          initial_state=self.initial_state, max_steps=max_steps)
        if self.planner_type == 'plan_by_backprop':
            self.planner = PlanByBackpropMPC(self.rollout_env, self.real_env, lookahead=40, opt_steps=20)
        elif self.planner_type == 'random_shooting':
            self.planner = RandomShootingMPC(self.rollout_env, self.real_env, lookahead=40)
        else:
            raise ValueError(f"Unknown planner type: {self.planner_type}")

    def solve(self, plot_name: str, max_steps: int=200, episodes: int=1,
            save_frames: bool=True) -> float:
        '''Solve the planning problem using the planner.'''
        avg = 0.0
        for _ in range(episodes):
            total = 0.0
            self.planner.reset()
            for _ in (pbar := tqdm(range(max_steps), desc='Running MPC')):
                _, reward, term, trunc, _ = self.planner.step(save_frames=save_frames)
                total += reward
                if term or trunc:
                    break
                pbar.set_postfix({'Cuml Return': f'{total:.3f}'})
            avg += total / episodes

        if save_frames:
            save_gif(self.planner.frames, plot_name)

        return avg
