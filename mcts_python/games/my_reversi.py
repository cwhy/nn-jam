from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, List

import numpy as np
from numpy.typing import NDArray

from mcts_python.protocols import State, Action, Env

from mcts_python.config import h, player_symbols, device
from mcts_python.games.gridboard_utils import rewards_all, rewards_winner_take_all, rewards_individual, \
    get_actions, move_along_in_dirs, GridBoard, BoardIndices, StateBoardTemplate
from mcts_python.games.web_agent import WebAgent
from mcts_python.torch_networks import BasicBoardNet

env_name = "Reversi"
board = GridBoard(h, h)
render_ = board.render_

# bad: -2, empty: -1, players: 0, 1, 2...
assert h % 2 == 0
init_grid = np.full((h, h), -1)
init_grid[h // 2, h // 2] = 0
init_grid[h // 2 - 1, h // 2] = 1
init_grid[h // 2, h // 2 - 1] = 1
init_grid[h // 2 - 1, h // 2 - 1] = 0

n_actions = h ** 2
n_players = len(player_symbols)

reversi_net = BasicBoardNet(device, env_name, h ** 2, n_players, h ** 2)


StateReversi = StateBoardTemplate

init_state: State = StateReversi.from_grid(init_grid, n_players)


def _hash(state: StateReversi, agent_id: int) -> Hashable:
    return hash((state.get_array.tobytes(), agent_id))


def update_array_(s_array: NDArray, action: Action, player: int):
    action_idx = (action // h, action % h)
    grid = s_array.reshape((h, h))
    to_eat = []
    for move_fn in move_along_in_dirs:
        pending = []
        pos = action_idx
        while True:
            pos = move_fn(*pos)
            if not board.check_bound(pos):
                break
            else:
                if grid[pos] == player:
                    to_eat += pending
                    break
                elif grid[pos] == -1:
                    break
                else:
                    pending.append(pos)
    s_array[[board.pos_to_arr_idx(i) for i in to_eat]] = player
    s_array[action] = player


def model(s: StateReversi, a: Action, player: int,
          render: bool = False) -> EnvOutput:
    new_array = s.get_array
    if new_array[a] != -1:
        reward_type = 'bad_position'
        rewards = rewards_individual(-10, player)
        done = True
    else:
        update_array_(new_array, a, player)
        if not any(new_array == -1):
            count = np.sum(new_array == player)
            if count > h ** 2 / 2:
                reward_type = 'win'
                rewards = rewards_winner_take_all(1, player)
            elif count > h ** 2 / 2:
                reward_type = 'lose'
                rewards = rewards_winner_take_all(1, (player + 1) % n_players)
            else:
                reward_type = 'draw'
                rewards = rewards_all(0)
            done = True
        else:
            reward_type = 'still_in_game'
            rewards = tuple(0 for _ in range(n_players))
            done = False
    next_player = (player + 1) % n_players
    message = f"{player_symbols[player]} {reward_type} " \
              f"with reward {rewards[player]}"
    s_new = StateReversi.from_array(new_array, (h, h), n_players)
    if render:
        render_(s_new)
    return EnvOutput(s_new, rewards, done, next_player, message)


reversi_env = Env(
    name=env_name,
    n_agents=n_players,
    n_actions=n_actions,
    init_state=lambda: (init_state, 0),
    model=model,
    state_utils=Env.StateUtils(
        hash=_hash,
        get_actions=get_actions,
        get_symmetries=lambda s, a: board.get_symmetries_4(s, a, wrapper=StateReversi),
        render_=render_
    ),
    agent_symbols=player_symbols,
    cli_agent=board.get_actor,
    web_agent=WebAgent(h, h).get_actor,
)
