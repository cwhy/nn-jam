# DataUnit is the only exposed array
# , which is just a mapping(dict) of Port:numpy.ndarray
# All calculations will be done in numpy

from __future__ import annotations

from abc import abstractmethod
from typing import Literal, Protocol, NamedTuple, Mapping, FrozenSet

from numpy.typing import NDArray

from supervised_benchmarks.ports import Port
from variable_protocols.variables import Variable

SupportedDatasetNames = Literal['MNIST', 'IRaven']
DataQuery = Mapping[Port, Variable]


FixedTrain: Literal['FixedTrain'] = 'FixedTrain'
FixedTest: Literal['FixedTest'] = 'FixedTest'
FixedValidation: Literal['FixedValidation'] = 'FixedValidation'
All: Literal['All'] = 'All'
FixedSubsetType = Literal['FixedTrain', 'FixedTest', 'FixedValidation', 'All']

DataUnit = Mapping[Port, NDArray]


class Subset(Protocol):
    @property
    @abstractmethod
    def tag(self) -> Literal[FixedSubsetType, 'Sampled']: ...

    @property
    @abstractmethod
    def len(self) -> int: ...


class FixedSubset(NamedTuple):
    tag: FixedSubsetType
    len: int

    @property
    def uid(self) -> FixedSubsetType:
        return self.tag


class SampledSubset(NamedTuple):
    tag: Literal['Sampled']
    intervals: frozenset[tuple[int, int]]  # in Python style(end-1)

    @property
    def len(self) -> int:
        return sum(map(lambda x: x[1] - x[0], self.intervals))


class DataSubset(NamedTuple):
    query: DataQuery
    subset: Subset
    content_map: DataUnit


class DataPool(Protocol):
    @property
    @abstractmethod
    def query(self) -> DataQuery: ...

    @property
    @abstractmethod
    def fixed_subsets(self) -> Mapping[FixedSubsetType, DataSubset]: ...

    def subset(self, subset: Subset) -> DataSubset: ...


class DataConfig(Protocol):
    @property
    @abstractmethod
    def query(self) -> DataQuery: ...

    @property
    @abstractmethod
    def type(self) -> Literal['DataConfig']: ...

    def get_data(self) -> DataPool: ...


class Dataset(Protocol):
    @property
    @abstractmethod
    def exports(self) -> FrozenSet[Port]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    def retrieve(self, query: DataQuery) -> DataPool: ...
