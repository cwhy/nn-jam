## Environment
env_name = 'WuZiQi'
player_symbols = ['*', 'o']
h = 12

## MCTS setting
n_iters = 0
n_eps = 32
n_mcts = 256
max_depth = 100

## Bandit setting
c_puct = 1.0

## Neural Network setting
max_batch_size = 1024
lr = 0.01

## Experiment setting
exp_name = f'{env_name}_{h}_n_mcts_{n_mcts}_net1'
train_from_last = True
device = 'cpu'
n_pools = 8
