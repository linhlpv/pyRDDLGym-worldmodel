import numpy as np
import pyRDDLGym
from pyRDDLGym.core.policy import BaseAgent
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import torch

from twm.core.data import create_data, get_dataloader, \
    plot_data_trajectories, plot_trajectories, save_video
from twm.core.model import WorldModel, WorldModelEvaluator
from twm.planners.plan_by_backprop import PlanByBackpropMPC
from twm.planners.random_shooting import RandomShootingMPC
from twm.core.spec import EnvSpec, FluentSpec
from twm.trainner.offline_trainer import OfflineTrainer


state_spec = {
    'ball-x': FluentSpec(shape=(1,), prange='real'),
    'ball-y': FluentSpec(shape=(1,), prange='real'),
    'paddle-y': FluentSpec(shape=(1,), prange='real'),
}
action_spec = {
    'move': FluentSpec(shape=(), prange='int', values=(-1, +1)),
}
env_spec = EnvSpec(state_spec=state_spec, action_spec=action_spec)


class PongEnvWithRandomStarts:

    def __init__(self):
        self.env = pyRDDLGym.make("Pong_arcade", '0', vectorized=True)
        self._visualizer = self.env._visualizer
    
    def reset(self):
        vel_x = np.random.choice([-0.03, 0.03], size=(1,))
        vel_y = np.random.choice([-0.01, 0.01], size=(1,))
        state, info = self.env.reset()
        self.env.sampler.subs['vel-x'] = vel_x
        self.env.sampler.subs['vel-y'] = vel_y
        self.env.sampler.states['vel-x'] = vel_x
        self.env.sampler.states['vel-y'] = vel_y
        self.env.state = self.env.sampler.states
        state['vel-x'] = vel_x
        state['vel-y'] = vel_y
        return state, info
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
        
    
class PongPolicy(BaseAgent):

    def sample_action(self, state):
        ball_y = state['ball-y'][0]
        paddle_y = state['paddle-y']
        if ball_y < paddle_y + 0.05:    
            return {'move': -1 if np.random.rand() < 0.85 else 1}
        elif ball_y > paddle_y + 0.05:  
            return {'move': 1 if np.random.rand() < 0.85 else -1}
        else:                    
            return {'move': 0}


def vec_policy(states):
    ball_y = states['ball-y'][:, 0].detach().cpu().numpy()
    paddle_y = states['paddle-y'][:, 0].detach().cpu().numpy()
    actions = np.zeros((len(ball_y),), dtype=np.int32)
    actions[ball_y < paddle_y + 0.05] = np.where(
        np.random.rand((ball_y < paddle_y + 0.05).sum()) < 0.85, -1, 1)
    actions[ball_y > paddle_y + 0.05] = np.where(
        np.random.rand((ball_y > paddle_y + 0.05).sum()) < 0.85, 1, -1)
    return {'move': torch.from_numpy(actions)}    


def create_pong_data(episodes=500, max_steps=200, save_path='pong_data.pkl'):
    env = PongEnvWithRandomStarts()
    policy = PongPolicy()
    create_data(env, env_spec, policy, episodes, max_steps, save_path)


def plot_rollouts(model, batch_size=4):
    env = PongEnvWithRandomStarts()

    # rollout trajectories from the world model
    eval = WorldModelEvaluator(model)
    init_states = {k: torch.from_numpy(v).float().to('cuda')[None, None]
                   for k, v in env.reset()[0].items()}
    trajectories = eval.rollout(init_states, None, vec_policy, max_steps=200)
    trajectories = [{k: v[0].detach().cpu() for k, v in trajectories.items()}]
    
    # plot rollouts
    plot_data_trajectories('pong_data.pkl', batch_size, 'pong_data_rollouts.png')
    plot_trajectories(trajectories, 'pong_model_rollouts.png')

    # save rollout video
    def render_fn(state_dict):
        state = {'ball-x___b1': state_dict['ball-x'][0].item(),
                 'ball-y___b1': state_dict['ball-y'][0].item(), 
                 'paddle-y': state_dict['paddle-y'][0].item()}
        return env._visualizer.render(state)
    save_video(render_fn, trajectories, 'pong_model_rollout.gif')


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
                                     planner_type=planner_type, offline_data_dir=offline_data_dir,
                                     pretrained_wm_epoch=50, wm_batch_size=1024, wm_lr=0.001,
                                     seq_len=seq_len, device=device)
    offline_trainer.solve(plot_name='pong_random_shoot_trainer.gif', max_steps=200, episodes=1, save_frames=True)