from typing import Callable

import pyRDDLGym
from tqdm import tqdm

from twm.core.data import EpisodeBuffer, make_dataloader, save_gif
from twm.core.model import WorldModel
from twm.core.types import ArrayDict, TensorDict
from twm.trainer.offline_trainer import OfflineTrainer


class OnlineTrainer(OfflineTrainer):
    '''
    Trainer that interleaves planning in the real environment with fine-tuning of
    the world model on the freshly collected transitions.
    '''
    def __init__(self, world_model: WorldModel, real_env: pyRDDLGym.RDDLEnv,
                 reward_fn: Callable, initial_state: ArrayDict | TensorDict,
                 planner_type: str='random_shooting', max_steps: int=200,
                 finetune_epochs: int=5, finetune_batch_size: int=1024,
                 finetune_lr: float=1e-3, finetune_lr_decay: float=0.9,
                 buffer_capacity: int | None=None) -> None:
        super().__init__(world_model, real_env, reward_fn, initial_state,
                         planner_type=planner_type, max_steps=max_steps)

        # buffer for the data collected while planning in the real environment
        self.buffer = EpisodeBuffer(self.env_spec, capacity=buffer_capacity)
        self.finetune_epochs = finetune_epochs
        self.finetune_batch_size = finetune_batch_size
        self.finetune_lr = finetune_lr
        self.finetune_lr_decay = finetune_lr_decay

    def _ensure_online_training_state(self) -> None:
        '''Initializes fresh training state when entering the online phase.'''
        if self._training_phase == 'online':
            return
        self._init_training_state(
            lr=self.finetune_lr, lr_decay=self.finetune_lr_decay)
        self._training_phase = 'online'

    def _update_world_model(self) -> None:
        '''Fine-tune the world model on the episodes collected so far.'''
        if len(self.buffer) == 0:
            return

        self._ensure_online_training_state()
        loader = make_dataloader(self.buffer.episodes, seq_len=self.seq_len,
                                 batch_size=self.finetune_batch_size)
        for epoch in range(self.finetune_epochs):
            avg_loss = self._train_epoch(
                loader, description=f'Epoch {epoch + 1}/{self.finetune_epochs}')
            assert self.wm_scheduler is not None
            self.wm_scheduler.step(avg_loss)

    def solve(self, plot_name: str, max_steps: int=200, episodes: int=20,
              save_frames: bool=True) -> float:
        '''Solve the planning problem, fine-tuning the world model after each episode.'''
        avg = 0.0
        for _ in range(episodes):
            total = 0.0
            self.planner.reset()
            for _ in (pbar := tqdm(range(max_steps), desc='Running MPC')):
                obs = self.planner.last_obs
                next_obs, reward, term, trunc, _ = self.planner.step(
                    save_frames=save_frames)
                action = self.planner.last_action
                total += reward
                self.buffer.add_transition(obs, action, reward, next_obs, term or trunc)
                if term or trunc:
                    break
                pbar.set_postfix({'Cuml Return': f'{total:.3f}'})
            avg += total / episodes

            # fine-tune the world model on everything collected up to now
            self.buffer.end_episode()
            self._update_world_model()

        if save_frames:
            save_gif(self.planner.frames, plot_name)

        return avg
