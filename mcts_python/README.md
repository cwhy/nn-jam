## MCTS (actually PUCT) in Python

### Demo Games
* TicTacToe
* Reversi
* Wuziqi (Gomoku)
* Wuziqi-1swap ([五子棋一手交換規則](https://zh.wikipedia.org/wiki/%E4%BA%94%E5%AD%90%E6%A3%8B))
### Supports
* Custom environment with clear APIs
    - Examples are in `/games`
* Arbitrary number of agents with per-agent rewards
### Instructions
* run `run_mcts.py` to start
* look up `config.py` to change game/configurations
### Key parameters of MCTS(PUCT)
* `n_iters`: the larger the more clever neural network will be,
 will increase training time linearly.
* `n_eps`: the larger the more robust the training will be,
 will increase training time linearly
* `n_mcts`: the larger the larger the more brute-force search samples will be,
 will increase training time and testing time polynomially

### Possible Improvements
* add a Q head
* add \alpha for Dirichlet noise
* cyclic learning rate
* PPO for policy
* episodic memory for value
* population based training

### Interesting Literature:
* Loss related
	* [add a Q head](https://openreview.net/pdf?id=BJxvH1BtDS)
	* [use Q directly](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628)
	* [using value is better than both](https://pdfs.semanticscholar.org/b125/c8933d0264b9a103cb8fa80f226f8c9c3cdc.pdf)
* Sampling
	* [Position Averaging](https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a)
* Learning rate
	* [Cyclic Learning Rate](https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a)
* Meta-train/Hyperparams
	* [Population-based tuning](https://arxiv.org/pdf/2003.06212.pdf)
* Bandit
	* [Leena strategy](https://deepai.org/publication/accelerating-self-play-learning-in-go)
	* [Thompson Sampling](https://github.com/leela-zero/leela-zero/issues/860)

### Requirement
* Python 3.8 +
* Refer to `requirement.txt`
