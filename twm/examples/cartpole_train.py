import numpy as np
import torch
import pyRDDLGym
from pyRDDLGym.core.policy import BaseAgent

from twm.core.data import (
    create_data,
    get_dataloader,
    plot_data_trajectories,
    plot_trajectories,
    save_video,
)
from twm.core.model import WorldModel, WorldModelEvaluator
from twm.core.spec import EnvSpec, FluentSpec


state_spec = {
    'pos': FluentSpec(shape=(1,), prange='real'),
    'ang-pos': FluentSpec(shape=(1,), prange='real'),
    'vel': FluentSpec(shape=(1,), prange='real'),
    'ang-vel': FluentSpec(shape=(1,), prange='real'),
}
action_spec = {
    'force': FluentSpec(shape=(1,), prange='real', values=(-10.0, 10.0)),
}
env_spec = EnvSpec(state_spec=state_spec, action_spec=action_spec)

MODEL_PREFIX = "cartpole_world_model"
MAX_FORCE = 10.0


class CartPoleEnvWithRandomStarts:

    def __init__(self):
        self.env = pyRDDLGym.make("CartPole_Continuous_gym", "0", vectorized=True)
        self._visualizer = self.env._visualizer

    def reset(self):
        state, info = self.env.reset()
        randomized_state = {
            'pos': np.random.uniform(-0.15, 0.15, size=(1,)),
            'ang-pos': np.random.uniform(-0.12, 0.12, size=(1,)),
            'vel': np.random.uniform(-0.25, 0.25, size=(1,)),
            'ang-vel': np.random.uniform(-0.25, 0.25, size=(1,)),
        }
        for key, value in randomized_state.items():
            self.env.sampler.subs[key] = value
            self.env.sampler.states[key] = value
            state[key] = value
        self.env.state = self.env.sampler.states
        return state, info

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()


class CartPolePolicy(BaseAgent):

    @staticmethod
    def _compute_force(pos, ang_pos, vel, ang_vel):
        force = 9.0 * ang_pos + 4.5 * ang_vel + 0.8 * pos + 1.2 * vel
        return float(np.clip(force, -MAX_FORCE, MAX_FORCE))

    def sample_action(self, state):
        pos = float(state['pos'][0])
        ang_pos = float(state['ang-pos'][0])
        vel = float(state['vel'][0])
        ang_vel = float(state['ang-vel'][0])
        force = self._compute_force(pos, ang_pos, vel, ang_vel)
        force += float(np.random.normal(scale=0.75))
        force = float(np.clip(force, -MAX_FORCE, MAX_FORCE))
        return {'force': force}


def vec_policy(states):
    pos = states['pos'][:, 0].detach().cpu().numpy()
    ang_pos = states['ang-pos'][:, 0].detach().cpu().numpy()
    vel = states['vel'][:, 0].detach().cpu().numpy()
    ang_vel = states['ang-vel'][:, 0].detach().cpu().numpy()
    force = 9.0 * ang_pos + 4.5 * ang_vel + 0.8 * pos + 1.2 * vel
    force = np.clip(force, -MAX_FORCE, MAX_FORCE).astype(np.float32)
    return {'force': torch.from_numpy(force[:, None])}


def create_cartpole_data(episodes=500, max_steps=200, save_path="cartpole_data.pkl"):
    env = CartPoleEnvWithRandomStarts()
    policy = CartPolePolicy()
    create_data(env, env_spec, policy, episodes, max_steps, save_path)


def plot_rollouts(model, batch_size=4, max_steps=200):
    env = CartPoleEnvWithRandomStarts()

    evaluator = WorldModelEvaluator(model)
    init_states = {
        key: torch.from_numpy(value).float().to('cuda')[None, None]
        for key, value in env.reset()[0].items()
    }
    trajectories = evaluator.rollout(init_states, None, vec_policy, max_steps=max_steps)
    trajectories = [{key: value[0].detach().cpu() for key, value in trajectories.items()}]

    plot_data_trajectories(
        "cartpole_data.pkl",
        batch_size,
        'cartpole_data_rollouts.png',
    )
    plot_trajectories(trajectories, 'cartpole_model_rollouts.png')

    def render_fn(state_dict):
        state = {
            'pos': float(state_dict['pos'][0]),
            'ang-pos': float(state_dict['ang-pos'][0]),
            'vel': float(state_dict['vel'][0]),
            'ang-vel': float(state_dict['ang-vel'][0]),
        }
        return env._visualizer.render(state)

    save_video(render_fn, trajectories, 'cartpole_model_rollout.gif')


if __name__ == "__main__":
    create_cartpole_data()
    seq_len = 8
    fit = True

    if fit:
        train_loader, test_loader = get_dataloader(
            "cartpole_data.pkl", seq_len, batch_size=64, augment_starts=False
        )

        model = WorldModel(env_spec=env_spec, seq_len=seq_len).to('cuda')
        model.fit(
            train_loader,
            lr=0.001,
            epochs=50,
            test_data_loader=test_loader,
            model_name=f'{MODEL_PREFIX}_{seq_len}.pth',
        )

    else:
        model = WorldModel.load(f'{MODEL_PREFIX}_{seq_len}.pth').to('cuda')
        plot_rollouts(model)
