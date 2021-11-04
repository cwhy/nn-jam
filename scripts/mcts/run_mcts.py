import numpy as np
import torch
from mcts_python.config import n_mcts, n_eps, n_iters, exp_name, env_name, n_pools, train_from_last
from tqdm import tqdm
from mcts_python.env_utils import pit, RandomAgent
from mcts_python.mcts import Mcts
from mcts_python.memory_utils import NNMemoryAnyState
from mcts_python.games import envs, nets
from glob import glob

if n_pools > 0:
    import ray
    ray.init(num_cpus=n_pools)

env = envs[env_name]
neural_net_ = nets[env_name]
if train_from_last:
    saves = glob(f'saved_models/{exp_name}_*.pth')
    if len(saves) > 0:
        save_iters = [int(s.split('.')[0].split('_')[-1]) for s in saves]
        last_iter = max(save_iters)
        last_path = saves[save_iters.index(last_iter)]
        neural_net_.load_state_dict(torch.load(last_path))
        iters_ = 0
    else:
        print("Loading model failed: no saved model found, reinitilize neural network")
        iters_ = 0
else:
    iters_ = 0
CliAgent = env.cli_agent
print("Learning...")

for _ in tqdm(range(n_iters)):
    iters_ += 1
    memory_ = NNMemoryAnyState(neural_net_, env)
    mcts = Mcts(n_mcts, env, max_depth=100)

    if n_pools > 0:
        @ray.remote
        def do_episode_(i):
            return mcts.self_play(memory_, i)


        exps = ray.get([do_episode_.remote(i) for i in range(n_eps)])
    else:
        exps = [mcts.self_play(memory_, i) for i in range(n_eps)]
    exps_arrays = [np.concatenate([ex[i] for ex in exps], axis=0) for i in range(4)]
    neural_net_.train_(*exps_arrays)

net_save_path = f'saved_models/{exp_name}_{iters_}.pth'

print("Testing Against Random...")
memory_ = NNMemoryAnyState(neural_net_, env)
print(f"as {env.agent_symbols[0]}")
mcts_agent0 = Mcts(n_mcts, env).get_agent_decision_fn(memory_, 0)
for _ in range(5):
    pit(env, [mcts_agent0, RandomAgent(1, env).find_action])
    print('-----------------------------------------')

print(f"as {env.agent_symbols[1]}")
for _ in range(5):
    mcts_agent1 = Mcts(n_mcts, env).get_agent_decision_fn(memory_, 1)
    pit(env, [RandomAgent(0, env).find_action, mcts_agent1])
    print('-----------------------------------------')

torch.save(neural_net_.state_dict(), net_save_path)
while True:
    print("Testing Against Cli/Human...")
    print(f"as {env.agent_symbols[0]}")
    mcts_agent0 = Mcts(n_mcts, env).get_agent_decision_fn(memory_, 0)
    pit(env, [mcts_agent0, CliAgent(1)], render=True)
    mcts_agent1 = Mcts(n_mcts, env).get_agent_decision_fn(memory_, 1)
    print(f"as {env.agent_symbols[1]}")
    pit(env, [CliAgent(0), mcts_agent1], render=True)
