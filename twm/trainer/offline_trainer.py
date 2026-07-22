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
                 planner_type: str='random_shooting', max_steps: int=200) -> None:
        self.env_spec = world_model.env_spec
        self.world_model = world_model
        self.real_env = real_env
        self.reward_fn = reward_fn
        self.initial_state = initial_state
        self.planner_type = planner_type
        self.seq_len = world_model.seq_len
        self.max_steps = max_steps

        # Training state is initialized explicitly for each training phase.
        self.wm_optim = None
        self.wm_scheduler = None
        self.ema = None
        self._training_phase = None
        self.train_loader = None
        self.test_loader = None

        # Planner construction is cheap and keeps a reference to the same model.
        self._create_planner(max_steps=self.max_steps)

    def _init_training_state(self, lr: float, lr_decay: float) -> None:
        '''Initializes optimizer, scheduler and EMA for a new training phase.'''
        self.wm_optim = torch.optim.Adam(self.world_model.parameters(), lr=lr)
        self.wm_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.wm_optim, factor=lr_decay, patience=10, min_lr=1e-5)
        self.ema = EMA(self.world_model)

    def _train_epoch(self, data_loader, description: str='Training') -> float:
        '''Runs one training epoch and returns the average batch loss.'''
        if self.wm_optim is None or self.ema is None:
            raise RuntimeError('Training state must be initialized before training.')

        self.world_model.train()
        epoch_loss = 0.0
        for batch_data in tqdm(data_loader, desc=description):
            loss = self.world_model.batch_loss(batch_data, self.world_model.device)
            self.wm_optim.zero_grad()
            loss.backward()
            self.wm_optim.step()
            self.ema.update(self.world_model)
            epoch_loss += loss.item()
        return epoch_loss / len(data_loader)

    def _evaluate_with_ema(self, data_loader) -> float:
        '''Evaluates EMA weights without replacing the current training weights.'''
        if self.ema is None:
            raise RuntimeError('Training state must be initialized before evaluation.')

        train_state = {k: v.clone() for k, v in self.world_model.state_dict().items()}
        try:
            self.world_model.load_state_dict(self.ema.state_dict)
            return self.world_model.evaluate(data_loader)
        finally:
            self.world_model.load_state_dict(train_state)

    def pretrain_world_model(self, data_name: str, epochs: int=50,
                             batch_size: int=1024, lr: float=1e-3,
                             lr_decay: float=0.9, model_name: str='') -> None:
        '''Pretrains the world model on an offline dataset.'''
        self.train_loader, self.test_loader = get_dataloader(
            data_name, seq_len=self.seq_len, batch_size=batch_size,
            augment_starts=False)

        self.world_model.set_dataset_stats(self.train_loader.dataset)
        self._init_training_state(lr=lr, lr_decay=lr_decay)
        self._training_phase = 'offline'

        for epoch in range(epochs):
            avg_loss = self._train_epoch(
                self.train_loader, description=f'Epoch {epoch + 1}/{epochs}')
            assert self.wm_scheduler is not None
            self.wm_scheduler.step(avg_loss)

            # evaluation with EMA weights
            test_loss = self._evaluate_with_ema(self.test_loader)
            assert self.wm_optim is not None
            current_lr = self.wm_optim.param_groups[0]['lr']
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, '
                  f'Test Loss: {test_loss:.6f}, LR: {current_lr:.2e}')

        # save the EMA weights as the final model
        assert self.ema is not None
        self.world_model.load_state_dict(self.ema.state_dict)
        self.world_model.eval()
        if model_name:
            self.world_model.save(model_name)

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
