# pyRDDLGym-worldmodel

`pyRDDLGym-worldmodel` learns transformer-based dynamics models for
[pyRDDLGym](https://github.com/pyrddlgym-project/pyRDDLGym) environments and
uses them for planning. It supports vector-valued and
pixel observations, discrete and bounded continuous actions, and planning by
random shooting or differentiable optimisation.

## What this package provides
- A transformer `WorldModel` that predicts the next state from a history of
  states and actions.
- Data collection and sequence-dataset utilities for Gymnasium-compatible
  environments.
- `WorldModelEnv`, a Gymnasium environment whose transitions are supplied by a
  learned model.
- MPC planners:
  - `RandomShootingMPC` for finite discrete action spaces.
  - `ContinuousRandomShootingMPC` for bounded real-valued actions.
  - `PlanByBackpropMPC` for gradient-based planning, with SLP and DRP policy
    representations.
- Trainers for offline and online training of the world model and planner.
- Examples with Pong for discrete actions and CartPole for continuous actions.
## Setup
Run the following commands to install the package.

```bash
git clone https://github.com/linhlpv/pyRDDLGym-worldmodel.git
cd pyRDDLGym-worldmodel

conda create -n pyRDDLGym-worldmodel python=3.10
conda activate pyRDDLGym-worldmodel
python -m pip install --upgrade pip
python -m pip install -e .
```
## Quick start: Pong

From the repository root, train a world model on data collected from the
`Pong_arcade` pyRDDLGym environment:

```bash
python -m twm.examples.pong_train
```

The default script collects 500 episodes and trains for 600 epochs. Reduce the arguments
in `twm/examples/pong_train.py` when iterating locally. Collection writes
`twm/data/pong_data.pkl`; training saves
`twm/models/pong_world_model_8.pth`.

With that checkpoint in place, run differentiable MPC in the real Pong
environment:

```bash
python -m twm.examples.pong_solve
```

This writes a rollout GIF under `twm/plots/`. The example uses
`PlanByBackpropMPC` with an SLP policy by default. Uncomment the
random-shooting block in `pong_solve.py` to try `RandomShootingMPC` instead.

## Using the library with a new environment

Describe the observed state variables and actions with `FluentSpec`, combine
them into an `EnvSpec`, collect trajectories, and train a `WorldModel`.

```python
import torch

from twm.core.data import create_data, get_dataloader
from twm.core.model import WorldModel
from twm.core.spec import EnvSpec, FluentSpec

env_spec = EnvSpec(
    state_spec={
        "position": FluentSpec(shape=(2,), prange="real"),
    },
    action_spec={
        "move": FluentSpec(shape=(), prange="int", values=(-1, 1)),
    },
)

# `env` implements reset(), step(), and render(); `policy` implements
# sample_action(state). Both state and action dictionaries use the keys above.
create_data(env, env_spec, policy, episodes=100, max_steps=200,
            data_name="my_domain.pkl")

train_loader, test_loader = get_dataloader(
    "my_domain.pkl", seq_len=8, batch_size=64, augment_starts=False
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WorldModel(env_spec, seq_len=8).to(device)
model.fit(train_loader, epochs=50, test_data_loader=test_loader,
          model_name="my_domain_world_model.pth")
```

`prange` may be `real`, `int`, `bool`, or `pixel`. Discrete (`int`) fluents
need inclusive `(low, high)` `values`; bounded real actions should also define
`values` when using `ContinuousRandomShootingMPC`.

To plan with a fitted model, wrap it in `WorldModelEnv` using a tensor-valued
reward function and then construct a planner with the learned rollout
environment and the real evaluation environment:

```python
from twm.core.env import WorldModelEnv
from twm.planners.random_shooting import RandomShootingMPC

reward_fn = lambda state, action, next_state: -next_state["position"].square().sum(dim=-1)
rollout_env = WorldModelEnv(
    model, reward_fn, initial_state=initial_state, max_steps=200
)
planner = RandomShootingMPC(
    rollout_env, real_env, lookahead=40, num_parallel_evals=32
)
average_return = planner.run("my_domain.gif", episodes=1)
```

For continuous actions, substitute `ContinuousRandomShootingMPC`. For
differentiable planning, use `PlanByBackpropMPC`; its reward function must
return a `torch.Tensor` so gradients can flow through the imagined rollout.

## Code structure

```text
twm/
├── core/       # specs, data pipeline, model, and learned Gymnasium environment
├── planners/   # random-shooting and backpropagation-based MPC
├── trainer/    # offline and online training helpers
├── examples/   # Pong and CartPole experiments
├── data/       # generated trajectory datasets (created at runtime)
├── models/     # saved checkpoints (created at runtime)
└── plots/      # generated plots and GIFs (created at runtime)
```

## License

This project is released under the [MIT License](LICENSE).
