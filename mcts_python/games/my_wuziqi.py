from typing import Hashable

import numpy as np
from numpy.typing import NDArray

from mcts_python.protocols import State, Action, Env, EnvOutput
from mcts_python.config import h, player_symbols, device
from mcts_python.games.gridboard_utils import rewards_all, rewards_winner_take_all, rewards_individual, \
    get_actions, move_along_in_dirs, GridBoard, StateBoardTemplate
from mcts_python.games.web_agent import WebAgent
from mcts_python.torch_networks import BasicBoardNet

env_name = "WuZiQi"
board = GridBoard(h, h)
render_ = board.render_

StateWZQ = StateBoardTemplate


def _hash(state: StateWZQ, agent_id: int) -> Hashable:
    return hash((state.get_array.tobytes(), agent_id))


# bad: -2, empty: -1, players: 0, 1, 2...
n_actions = h ** 2
n_players = len(player_symbols)
init_state: State = StateWZQ.from_array(np.full(h ** 2, -1), (h, h), n_players)

wzq_net = BasicBoardNet(device, env_name, h ** 2, n_players, h ** 2)


def check_win(s_array: NDArray, action: Action) -> bool:
    action_idx = (action // h, action % h)
    grid = s_array.reshape((h, h))
    for move_fn in move_along_in_dirs:
        count = 1
        pos = action_idx
        while True:
            pos = move_fn(*pos)
            if not board.check_bound(pos):
                break
            else:
                if grid[pos]:
                    count += 1
                else:
                    break
        if count >= 5:
            return True
    return False


def model(s: StateWZQ, a: Action, player: int,
          render: bool = False) -> EnvOutput:
    new_array = s.get_array
    if new_array[a] != -1:
        reward_type = 'bad_position'
        rewards = rewards_individual(-10, player)
        done = True
    else:
        new_array[a] = player
        p_pos = new_array == player
        if check_win(p_pos, a):
            reward_type = 'win'
            rewards = rewards_winner_take_all(1, player)
            done = True
        else:
            if not any(new_array == -1):
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
    s_new = StateWZQ.from_array(new_array, (h, h), n_players)
    if render:
        render_(s_new)
    return EnvOutput(s_new, rewards, done, next_player, message)


wuziqi_env = Env(
    name=env_name,
    n_agents=n_players,
    n_actions=n_actions,
    init_state=lambda: (init_state, 0),
    model=model,
    state_utils=Env.StateUtils(
        hash=_hash,
        get_actions=get_actions,
        get_symmetries=lambda s, a: board.get_symmetries_4(s, a,
                                                           wrapper=StateWZQ.get_wrapper((h, h), n_players)),
        render_=render_
    ),
    agent_symbols=player_symbols,
    cli_agent=board.get_actor,
    web_agent=WebAgent(h, h).get_actor
)
