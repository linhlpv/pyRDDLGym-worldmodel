import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pyRDDLGym
import torch

from twm.core.env import WorldModelEnv
from twm.core.model import WorldModel
from twm.planners.plan_by_backprop import PlanByBackpropMPC
from twm.planners.random_shooting import ContinuousRandomShootingMPC

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_world_model_env():
    world_model = WorldModel.load(
        'cartpole_world_model_8.pth', device=DEVICE
    ).to(DEVICE)
    init_state = {'pos': np.array([0.0]), 'ang-pos': np.array([0.0]), 
                  'vel': np.array([0.0]), 'ang-vel': np.array([0.0])}
    reward_fn = lambda s, a, ns: (
        1.0
        - 2.0 * torch.abs(ns['ang-pos'][..., 0])
        - 0.2 * torch.abs(ns['pos'][..., 0])
        - 0.05 * torch.abs(ns['ang-vel'][..., 0])
    )
    return WorldModelEnv(world_model, reward_fn, initial_state=init_state, max_steps=200)


def run_random_shooting_agent():
    rollout_env = create_world_model_env()
    eval_env = pyRDDLGym.make("CartPole_Continuous_gym", '0', vectorized=True)
    mpc = ContinuousRandomShootingMPC(
        rollout_env, eval_env, lookahead=40, num_parallel_evals=256
    )
    mpc.run('cartpole_random_shooting.gif', save_frames=True, episodes=1)


def run_backprop_mpc_agent():
    rollout_env = create_world_model_env()
    eval_env = pyRDDLGym.make("CartPole_Continuous_gym", '0', vectorized=True)
    mpc = PlanByBackpropMPC(
        rollout_env, eval_env, lookahead=40,
    )
    mpc.run('cartpole_backprop_mpc.gif', save_frames=True, episodes=1)


if __name__ == "__main__":
    run_random_shooting_agent()
    # run_backprop_mpc_agent()
