from __future__ import annotations

import abc
from typing import NamedTuple, Protocol, Literal, Optional

from variable_protocols.base_variables import BaseVariable, Ord, Bounded


class NumStats(NamedTuple):
    mean: float
    std: float
    min: float
    max: float


class ColumnStats(NamedTuple):
    n: int
    n_unique: int
    n_null: int
    sorted_desc_unique_count: tuple[int, ...]
    num_stats: Optional[NumStats]  # if it is None, it is a string


StrategyType = Literal["anynet"]


class TabularColumnsConfig(Protocol):
    @property
    @abc.abstractmethod
    def strategy(self) -> StrategyType: ...

    def classify_column(self, stats: ColumnStats) -> BaseVariable: ...


class AnyNetStrategyConfig(NamedTuple):
    number_unique: int = 1024
    number_unique_ratio: float = 0.01
    str_unique_ratio: float = 0.8  # throw away if there are too many categories, can save by big top classes
    top5_unique_ratio: float = 0.2

    def classify_column(self, stats: ColumnStats) -> Optional[BaseVariable]:
        top5_count = sum(stats.sorted_desc_unique_count[:5])
        top5_count_ratio = top5_count / stats.n
        unique_ratio = stats.n_unique / stats.n
        null_ratio = stats.n_null / stats.n

        if stats.num_stats is None:
            if unique_ratio > self.str_unique_ratio:
                if top5_count_ratio > 1 - self.str_unique_ratio:
                    return Ord(n_category=stats.n_unique)
                else:
                    return None
            else:
                return Ord(n_category=stats.n_unique)
        else:
            if stats.n_unique <= self.number_unique and null_ratio <= self.number_unique_ratio:
                return Ord(n_category=stats.n_unique)
            else:
                # TODO logic to find distributions
                return Bounded(max=stats.num_stats.max, min=stats.num_stats.min)
