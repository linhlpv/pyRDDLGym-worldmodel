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
                 planner_type: str='random_shooting', offline_data_dir: str='pong_data.pkl',
                 pretrained_wm_epoch: int=50, wm_batch_size: int=1024, wm_lr: float=1e-3,
                 lr_decay: float=0.9, wm_name: str='pong_world_model_8_offline.pth',
                 seq_len: int=8, max_steps: int=200, finetune_epochs: int=5,
                 buffer_capacity: int | None=None, device: str='cuda') -> None:
        super().__init__(world_model, real_env, reward_fn, initial_state,
                         planner_type=planner_type, offline_data_dir=offline_data_dir,
                         pretrained_wm_epoch=pretrained_wm_epoch,
                         wm_batch_size=wm_batch_size, wm_lr=wm_lr, lr_decay=lr_decay,
                         wm_name=wm_name, seq_len=seq_len, max_steps=max_steps,
                         device=device)

        # buffer for the data collected while planning in the real environment
        self.buffer = EpisodeBuffer(self.env_spec, capacity=buffer_capacity)
        self.finetune_epochs = finetune_epochs

    def _update_world_model(self) -> None:
        '''Fine-tune the world model on the episodes collected so far.'''
        if len(self.buffer) == 0:
            return

        loader = make_dataloader(self.buffer.episodes, seq_len=self.seq_len,
                                 batch_size=self.wm_batch_size)
        for epoch in range(self.finetune_epochs):
            self.world_model.train()
            epoch_loss = 0.0
            for batch_data in tqdm(loader, desc=f'Epoch {epoch + 1}/{self.finetune_epochs}'):
                loss = self.world_model.batch_loss(batch_data, self.device)
                self.wm_optim.zero_grad()
                loss.backward()
                self.wm_optim.step()
                self.ema.update(self.world_model)
                epoch_loss += loss.item()
            self.wm_scheduler.step(epoch_loss / len(loader))

    def solve(self, plot_name: str, max_steps: int=200, episodes: int=20,
              save_frames: bool=True) -> float:
        '''Solve the planning problem, fine-tuning the world model after each episode.'''
        avg = 0.0
        for _ in range(episodes):
            total = 0.0
            self.planner.reset()
            for _ in (pbar := tqdm(range(max_steps), desc='Running MPC')):
                obs = self.planner.last_obs
                next_obs, action, reward, term, trunc, _ = self.planner.step(
                    save_frames=save_frames)
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
