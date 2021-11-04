from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

from mcts_python.config import player_symbols
from typing import List, Tuple, Protocol, Callable, Final, Hashable, TypeVar, Any, \
    Iterator, NamedTuple, Type

Empty: Final[int] = -1

T = TypeVar('T')


class ListLike(Protocol[T]):
    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator[T]:
        pass

    def __getitem__(self, item: T) -> T:
        pass


class BoardIndices(NamedTuple):
    player_id: int
    coord: Tuple[int, int]


class StateBoard(Protocol):
    @property
    def get_array(self) -> NDArray:
        raise NotImplementedError

    @property
    def get_indices(self) -> List[BoardIndices]:
        raise NotImplementedError


@dataclasses.dataclass
class StateBoardTemplate:
    _indices: List[BoardIndices]
    _shape: Tuple[int, int]

    @property
    def get_grid(self) -> NDArray:
        grid = np.full(self._shape, -1)
        for player, (x, y) in self._indices:
            grid[x, y] = player
        return grid

    @property
    def get_array(self) -> NDArray:
        return self.get_grid.ravel()

    @property
    def get_indices(self) -> List[BoardIndices]:
        return self._indices

    @classmethod
    def from_grid(cls, grid: NDArray, n_players: int) -> StateBoardTemplate:
        indices = [
            BoardIndices(player, tuple(coord_np.tolist()))
            for player in range(n_players)
            for coord_np in np.argwhere(grid == player)]
        return StateBoardTemplate(indices, grid.shape)

    @classmethod
    def from_array(cls,
                   array: NDArray,
                   shape: Tuple[int, int],
                   n_players: int) -> StateBoardTemplate:
        grid = array.reshape(shape)
        return StateBoardTemplate.from_grid(grid, n_players=n_players)

    @classmethod
    def get_wrapper(cls, shape: Tuple[int, int], n_players: int) -> Callable[[NDArray], StateBoardTemplate]:
        return lambda array: StateBoardTemplate.from_grid(array.reshape(shape), n_players=n_players)


def _hash(state: StateBoard, agent_id: int) -> Hashable:
    return hash((state.get_array.tobytes(), agent_id))


def get_actions(state: StateBoard, agent_id: int) -> Actions:
    return np.where(state.get_array == Empty)[0]


class GridBoard:
    def __init__(self, h: int, w: int):
        self.h = h
        self.w = w

    def print_grid_(self,
                    grid: NDArray,
                    to_str: Callable[[Any], str] = str):
        grid_w = len(to_str(Empty)) + 2
        bar = " " + "-" * (self.w * (1 + grid_w) + 1)
        print(bar)
        for i in range(self.h):
            print(" ", end='')
            for j in range(self.w):
                print("|" + to_str(grid[i, j]).center(grid_w), end='')
            print("|")
            print(bar)

    def render_(self, state: StateBoard) -> None:
        symbol_dict = {i: v for i, v in enumerate(player_symbols)}
        symbol_dict[Empty] = ' '
        self.print_grid_(np.array(state.get_array).reshape(self.h, self.w),
                         to_str=lambda i: symbol_dict[i])
        print()

    def get_actor(self, ag_id: int):
        def _actor(s: StateBoard, render: bool = False) -> Action:
            print("Current Game State:")
            self.render_(s)
            print("Position numbers:")
            self.print_grid_(np.arange(0, self.w * self.h).reshape(self.h, self.w),
                             to_str=lambda x: str(x).rjust(2))
            i = int(input("Enter your next move as position number:"))
            return i

        return _actor

    def get_symmetries_4(self,
                         state: StateBoard,
                         actions: Actions,
                         wrapper: Callable[[NDArray], StateBoard]) -> Tuple[
                                  List[StateBoard], List[Actions]]:
        # mirror, rotational
        h = self.h
        board = state.get_array.reshape(h, h)
        board_a = actions.reshape(h, h)
        boards = []
        boards_a = []
        for i in range(1, 5):
            for j in True, False:
                new_board = np.rot90(board, i)
                new_board_a = np.rot90(board_a, i)
                if j:
                    new_board = np.fliplr(new_board)
                    new_board_a = np.fliplr(new_board_a)
                new_board = new_board.flatten()
                boards.append(wrapper(new_board))
                boards_a.append(new_board_a.flatten())
        return boards, boards_a

    def pos_to_arr_idx(self, pos: Tuple[int, int]) -> int:
        i, j = pos
        return self.w * i + j

    def check_bound(self, pos: Tuple[int, int]) -> bool:
        i, j = pos
        return 0 <= i < self.h and 0 <= j < self.w


def rewards_winner_take_all(num: float, player: int):
    rewards = np.full(2, -num)
    rewards[player] += num * 2
    return rewards


def rewards_individual(num: float, player: int):
    rewards = np.full(2, 0)
    rewards[player] += num
    return rewards


def rewards_all(num: float):
    rewards = np.full(2, num)
    return rewards


move_along_in_dirs = [
    lambda ii, jj, i=i, j=j: (ii + i, jj + j)
    for i in range(-1, 2) for j in range(-1, 2) if not (i == 0 and j == 0)]

dir_pairs = set([frozenset({(i, j), (-i, -j)})
                 for i in range(-1, 2) for j in range(i, 2) if not (i == 0 and j == 0)])

move_along_in_dir_pairs = [
    (lambda ii, jj, i=i, j=j: (ii + i, jj + j),
     lambda ii, jj, k=k, kk=kk: (ii + k, jj + kk))
    for (i, j), (k, kk) in dir_pairs]
