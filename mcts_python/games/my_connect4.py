from dataclasses import dataclass, replace
from typing import Final, Tuple, List

import numpy as np
from numpy.typing import NDArray

from mcts_python.config import player_symbols, device
from mcts_python.games.gridboard_utils import rewards_all, rewards_winner_take_all, rewards_individual, \
    Empty, _hash, GridBoard, move_along_in_dir_pairs
from mcts_python.protocols import State, Action, Env, \
    EnvOutput, Actions
from mcts_python.torch_networks import BasicBoardNet

env_name = "Connect4"

h: Final[int] = 6
w: Final[int] = 7
board = GridBoard(h, w)
render_ = board.render_


@dataclass
class StateConn4:
    grid: np.array

    @property
    def get_array(self):
        return self.grid.ravel()


# bad: -2, empty: -1, players: 0, 1, 2...
init_state: State = StateConn4(np.full((h, w), -1))
n_actions = w
n_players = len(player_symbols)

conn4_net = BasicBoardNet(device, env_name, h * w, n_players, w)


def get_actions(state: StateConn4, agent_id: int) -> Actions:
    return np.where(state.grid[0, :] == Empty)[0]


def check_connected(grid: NDArray,
                    action_idx: Tuple[int, int],
                    threshold: int = 4) -> bool:
    for move_fns in move_along_in_dir_pairs:
        count_ = 1
        for move_fn in move_fns:
            pos = action_idx
            while True:
                pos = move_fn(*pos)
                if not board.check_bound(pos):
                    break
                else:
                    if grid[pos]:
                        count_ += 1
                    else:
                        break
        if count_ >= threshold:
            return True
    return False


def get_symmetries(state: StateConn4,
                   actions: Actions) -> Tuple[
                        List[StateConn4], List[Actions]]:
    # mirror
    boards = [replace(state, grid=np.fliplr(state.grid)), state]
    return boards, [actions, np.flip(actions)]


def model(s: StateConn4, a: Action, player: int,
          render: bool = False) -> EnvOutput:
    s_new = replace(s, grid=np.copy(s.grid))
    if s_new.grid[0, a] != Empty:
        reward_type = 'bad_position'
        rewards = rewards_individual(-10, player)
        done = True
    else:
        if (s_new.grid[:, a] == Empty).all():
            idx = (h - 1, a)
        else:
            idx = (np.argmax(s_new.grid[:, a] != Empty, axis=0) - 1, a)
        s_new.grid[idx] = player
        p_pos = s_new.grid == player
        if check_connected(p_pos, idx):
            reward_type = 'win'
            rewards = rewards_winner_take_all(1, player)
            done = True
        else:
            if not (s_new.grid == Empty).any():
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
    if render:
        render_(s_new)
    return EnvOutput(s_new, rewards, done, next_player, message)


class Conn4CliAgent:
    def __init__(self, _h: int, _w: int):
        self.h = _h
        self.w = _w

    def get_actor(self, ag_id: int):
        def _actor(s: StateConn4, render: bool = False) -> Action:
            print("Current Game State:")
            render_(s)
            print("Position numbers:")
            array = np.arange(0, self.w)
            actions = get_actions(s, ag_id)
            grid_w = 2 + round(np.log10(self.w))
            fmt = f"{{: ^{grid_w}}}"
            print(" ", end='')
            for i in range(w):
                if i in actions:
                    s = str(array[i])
                else:
                    s = '⮾'
                print("|" + fmt.format(s), end='')
            print("|")
            i = int(input("Enter your next move as position number:"))
            return i

        return _actor


conn4_env = Env(
    name=env_name,
    n_agents=n_players,
    n_actions=n_actions,
    init_state=lambda: (init_state, 0),
    model=model,
    state_utils=Env.StateUtils(
        hash=_hash,
        get_actions=get_actions,
        get_symmetries=get_symmetries,
        render_=render_
    ),
    agent_symbols=player_symbols,
    cli_agent=Conn4CliAgent(h, w).get_actor,
)
