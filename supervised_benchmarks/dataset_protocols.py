from __future__ import annotations
from abc import abstractmethod
from typing import Literal, Protocol, NamedTuple, Mapping, FrozenSet, TypeVar

from supervised_benchmarks.ports import Port
from variable_protocols.variables import Variable

SupportedDatasetNames = Literal['MNIST', 'IRaven']
DataQuery = Mapping[Port, Variable]


class DataContentBound(Protocol):
    def __getitem__(self, index): ...

    def __len__(self): ...

    def __contains__(self, item): ...

    def __iter__(self): ...


DataContentCov = TypeVar('DataContentCov', bound=DataContentBound, covariant=True)
DataContentContra = TypeVar('DataContentContra', bound=DataContentBound, contravariant=True)
DataContent = TypeVar('DataContent', bound=DataContentBound)

FixedTrain = Literal['FixedTrain']
FixedTest = Literal['FixedTest']
FixedValidation = Literal['FixedValidation']
All = Literal['All']
FixedSubsetType = Literal[All, FixedTrain, FixedTest, FixedValidation]


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


class Data(Protocol[DataContentCov]):
    @property
    @abstractmethod
    def port(self) -> Port: ...

    @property
    @abstractmethod
    def protocol(self) -> Variable: ...

    @property
    @abstractmethod
    def subset(self) -> Subset: ...

    @property
    @abstractmethod
    def content(self) -> DataContentCov: ...


class DataPool(Protocol[DataContentCov]):
    @property
    @abstractmethod
    def port(self) -> Port: ...

    @property
    @abstractmethod
    def src_var(self) -> Variable: ...

    @property
    @abstractmethod
    def tgt_var(self) -> Variable: ...

    @property
    @abstractmethod
    def fixed_subsets(self) -> Mapping[FixedSubsetType, Data[DataContentCov]]: ...

    def subset(self, subset: Subset) -> Data[DataContentCov]: ...


DataPortMap = Mapping[Port, DataPool]


class DataConfig(Protocol):
    @property
    @abstractmethod
    def port_vars(self) -> DataQuery: ...

    @property
    @abstractmethod
    def type(self) -> Literal['DataConfig']: ...

    def get_data(self) -> DataPortMap: ...


class DataPair(Protocol):
    output: Data
    target: Data


class Dataset(Protocol):
    @property
    @abstractmethod
    def ports(self) -> FrozenSet[Port]: ...

    @property
    @abstractmethod
    def name(self) -> SupportedDatasetNames: ...

    def retrieve(self, query: DataQuery) -> Mapping[Port, DataPool]: ...
