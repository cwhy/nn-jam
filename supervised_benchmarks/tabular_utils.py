from __future__ import annotations

import abc
from typing import NamedTuple, Protocol, Literal, Optional, Dict, List

import polars as pl  # type: ignore
from numpy.typing import NDArray

from variable_protocols.base_variables import BaseVariable, Bounded, Ordinal
from variable_protocols.labels import L, Labels
from variable_protocols.tensorhub import F, V, TensorHub


class NumStats(NamedTuple):
    mean: float
    std: float
    min: float
    max: float


class ColumnInfo(NamedTuple):
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

    def process_polars_column(self, col: pl.Series) -> Optional[BaseVariable]: ...


class AnyNetStrategyConfig(NamedTuple):
    number_unique: int = 1024
    number_unique_ratio: float = 0.01  # if there are less of that and less number_unique,
    # consider it as a categorical variable
    str_unique_ratio: float = 0.8  # if there are less of that, consider it as a categorical variable
    top5_unique_ratio: float = 0.2  # if str_unique_ratio check failed
    # and there are a lot of top 5 unique values, consider it as a categorical variable
    # else throw away the variable
    strategy: Literal["anynet"] = "anynet"

    def process_polars_column(self, col: pl.Series) -> Optional[BaseVariable]:
        if col.is_utf8():
            num_stats = None
        else:
            num_stats = NumStats(mean=col.mean(), std=col.std(), min=col.min(), max=col.max())

        stats = ColumnInfo(
            n=col.len(),
            n_unique=len(col.unique()),
            n_null=col.null_count(),
            sorted_desc_unique_count=col.unique_counts().sort(reverse=True).to_list(),
            num_stats=num_stats,
        )

        top5_count = sum(stats.sorted_desc_unique_count[:5])
        top5_count_ratio = top5_count / stats.n
        unique_ratio = stats.n_unique / stats.n
        null_ratio = stats.n_null / stats.n

        if stats.num_stats is None:
            if unique_ratio > self.str_unique_ratio:
                if top5_count_ratio > 1 - self.str_unique_ratio:
                    return Ordinal(n_category=stats.n_unique,
                                   labels=tuple(L(col.name, str(val)) for val in col.unique().to_list()))
                else:
                    return None
            else:
                return Ordinal(n_category=stats.n_unique,
                               labels=tuple(L(col.name, str(val)) for val in col.unique().to_list()))
        else:
            if stats.n_unique <= self.number_unique and null_ratio <= self.number_unique_ratio:
                return Ordinal(n_category=stats.n_unique,
                               labels=tuple(L(col.name, str(val)) for val in col.unique().to_list()))
            else:
                # TODO logic to find distributions
                return Bounded(max=stats.num_stats.max, min=stats.num_stats.min)


def parse_polars(column_config: TabularColumnsConfig, df: pl.DataFrame) -> TensorHub:
    cols = df.get_columns()
    variable_protocols = V.empty()
    for col in cols:
        base = column_config.process_polars_column(col)
        if base is not None:
            variable_protocols += F(base, col.name)
    return variable_protocols


# select and convert to numpy usable discrete
def polar_select_discrete(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
    return df.select(s.cast(pl.Utf8).cast(pl.Categorical).to_physical() for s in df.select(columns))


# Used for testing or directly used without Dataset
def anynet_load_polars(column_config: TabularColumnsConfig, df: pl.DataFrame) -> Dict[str, NDArray]:
    values = []
    symbols = []
    cols = df.get_columns()
    variable_protocols = V.empty()
    for col in cols:
        base = column_config.process_polars_column(col)
        if base is not None:
            variable_protocols += F(base, col.name)
            if isinstance(base, Ordinal):
                symbols.append(col.name)
            elif isinstance(base, Bounded):
                values.append(col.name)
            else:
                # Support for other types
                raise ValueError("Unknown variable type")
    value_df = df.select(values)
    symbol_df = polar_select_discrete(df, symbols)
    # pass label information
    return {"values": value_df.to_numpy(), "symbols": symbol_df.to_numpy()}


def anynet_get_discrete(df: pl.DataFrame, original_protocol: TensorHub, exclude: List[str]) -> NDArray:
    symbols = [col.name for col in df.get_columns()
               if isinstance(original_protocol[L(col.name)].base, Ordinal) and col.name not in exclude]
    if len(symbols) == 0:
        raise ValueError("No discrete variables found")
    return polar_select_discrete(df, symbols).to_numpy()


def anynet_get_continuous(df: pl.DataFrame, original_protocol: TensorHub, exclude: List[str]) -> NDArray:
    values = [col.name for col in df.get_columns()
              if isinstance(original_protocol[L(col.name)].base, Bounded) and col.name not in exclude]
    if len(values) == 0:
        raise ValueError("No continuous variables found")
    return df.select(values).to_numpy()
