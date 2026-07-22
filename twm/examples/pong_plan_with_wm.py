import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pyRDDLGym 

from twm.core.model import WorldModel
from twm.examples.pong_train import create_pong_data, env_spec
from twm.trainer.offline_trainer import OfflineTrainer


if __name__ == "__main__":
    offline_data_dir = 'pong_data.pkl'
    create_pong_data(save_path=offline_data_dir)

    real_env = pyRDDLGym.make("Pong_arcade", '0', vectorized=True)
    planner_type = 'random_shooting'  # or 'plan_by_backprop'
    seq_len = 8
    device = 'cuda'
    world_model = WorldModel(env_spec=env_spec, seq_len=seq_len).to(device)
    initial_state = {'ball-x': np.array([0.5]), 'ball-y': np.array([0.5]),
                      'paddle-y': np.array([0.4])}
    reward_fn = lambda s, a, ns: -ns['ball-x'][0]
    offline_trainer = OfflineTrainer(world_model=world_model, real_env=real_env,
                                     reward_fn=reward_fn, initial_state=initial_state,
                                     planner_type=planner_type, max_steps=200)
    offline_trainer.pretrain_world_model(
        data_name=offline_data_dir, epochs=50, batch_size=1024, lr=0.001,
        model_name='pong_world_model_8_offline.pth')
    offline_trainer.solve(plot_name='pong_random_shoot_trainer.gif', max_steps=200, episodes=1, save_frames=True)
