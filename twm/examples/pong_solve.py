import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pyRDDLGym

from twm.core.env import WorldModelEnv
from twm.core.model import WorldModel
from twm.planners.plan_by_backprop import PlanByBackpropMPC, SLP
from twm.planners.random_shooting import RandomShootingMPC


def create_world_model_env():
    world_model = WorldModel.load('pong_world_model_8.pth').to('cuda')
    init_state = {'ball-x': np.array([0.5]), 'ball-y': np.array([0.5]), 
                  'paddle-y': np.array([0.4])}
    reward_fn = lambda s, a, ns: -ns['ball-x'][0]
    return WorldModelEnv(world_model, reward_fn, initial_state=init_state, max_steps=200)

def run_random_shooting_agent():
    rollout_env = create_world_model_env()
    eval_env = pyRDDLGym.make("Pong_arcade", '0', vectorized=True)
    mpc = RandomShootingMPC(rollout_env, eval_env, lookahead=40)
    mpc.run('pong_random_shooting.gif', save_frames=True, episodes=1)


def run_plan_by_backprop_mpc():
    rollout_env = create_world_model_env()
    eval_env = pyRDDLGym.make("Pong_arcade", '0', vectorized=True)
    slp_mpc = PlanByBackpropMPC(rollout_env, eval_env, lookahead=40, opt_steps=20, policy=SLP())
    slp_mpc.run('pong_plan_by_backprop_slp.gif', save_frames=True, episodes=1, max_steps=200)


if __name__ == "__main__":
    run_plan_by_backprop_mpc()
