from collections import Hashable
from dataclasses import replace, dataclass

import numpy as np
from numpy.typing import NDArray

from mcts_python.config import h, player_symbols, device
from mcts_python.games.gridboard_utils import rewards_all, rewards_winner_take_all, rewards_individual
from mcts_python.games.my_wuziqi import n_players, check_win, n_actions, render_, board
from mcts_python.protocols import Action, Env, EnvOutput, Actions
from mcts_python.torch_networks import BasicBoardNet


@dataclass
class StateWZQ1Swap:
    array: NDArray
    swapped: int

    @property
    def get_array(self) -> NDArray:
        return self.array


env_name = "WuZiQi_1Swap"
init_state = StateWZQ1Swap(np.full(h ** 2, -1), swapped=0)
wzq_net = BasicBoardNet(device, env_name, h ** 2, n_players, h ** 2)


def _hash(state: StateWZQ1Swap, agent_id: int) -> Hashable:
    return hash((state.array.tobytes(), agent_id, state.swapped))


def get_actions(state: StateWZQ1Swap, agent_id: int) -> Actions:
    if state.swapped == 0 and sum(state.array != -1) == 1:
        return np.array((0, 1))
    else:
        return np.where(state.array == -1)[0]


def model(s: StateWZQ1Swap, a: Action, player: int,
          render: bool = False) -> EnvOutput:
    s_new = replace(s, array=np.copy(s.array))
    message = ""
    if s_new.array[a] != -1:
        reward_type = 'bad_position'
        rewards = rewards_individual(-10, player)
        done = True
    elif s_new.swapped == 0 and sum(s_new.array != -1) == 1:
        assert a in (0, 1)
        if a == 0:
            message += f'{player_symbols[player]} choose not to swap\n'
            player = (player + 1) % n_players
        elif a == 1:
            message += f'{player_symbols[player]} choose to swap\n'
            next_player = (player + 1) % n_players
            np.place(s_new.array, s_new.array == next_player, player)
        reward_type = 'still_in_game'
        s_new.swapped = 1
        rewards = tuple(0 for _ in range(n_players))
        done = False
    else:
        s_new.array[a] = player
        p_pos = s_new.array == player
        if check_win(p_pos, a):
            reward_type = 'win'
            rewards = rewards_winner_take_all(1, player)
            done = True
        else:
            if not any(s_new.array == -1):
                reward_type = 'draw'
                rewards = rewards_all(0)
                done = True
            else:
                reward_type = 'still_in_game'
                rewards = tuple(0 for _ in range(n_players))
                done = False
    next_player = (player + 1) % n_players
    message += f"{player_symbols[player]} {reward_type} " \
               f"with reward {rewards[player]}"
    if render:
        render_(s_new)
    return EnvOutput(s_new, rewards, done, next_player, message)


class WZQ1SwapCliAgent:
    def __init__(self, _h: int, _w: int):
        self.h = _h
        self.w = _w

    def get_actor(self, ag_id: int):
        def _actor(s: StateWZQ1Swap, render: bool = False) -> Action:
            print("Current Game State:")
            render_(s)
            if s.swapped == 0 and sum(s.array != -1) == 1:
                print("Do you want to swap?")
                i = int(input("0 -> No, 1 -> Yes: "))
            else:
                print("Position numbers:")
                board.print_grid_(np.arange(0, self.w * self.h).reshape(self.h, self.w),
                                  to_str=lambda x: str(x).rjust(2))
                msg = f"Enter your next move as position number (as {player_symbols[ag_id]}):"
                i = int(input(msg))
            return i

        return _actor


wuziqi_env = Env(
    name=env_name,
    n_agents=n_players,
    n_actions=n_actions,
    init_state=lambda: (init_state, 0),
    model=model,
    state_utils=Env.StateUtils(
        hash=_hash,
        get_actions=get_actions,
        get_symmetries=lambda s, a: board.get_symmetries(s, a,
                                                         wrapper=lambda na: replace(s, array=na)),
        render_=render_
    ),
    agent_symbols=player_symbols,
    cli_agent=WZQ1SwapCliAgent(h, h).get_actor,
)
