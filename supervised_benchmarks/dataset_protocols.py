from abc import abstractmethod
from pathlib import Path

from variable_protocols.variables import Variable
from typing import Literal, Protocol, NamedTuple, Mapping, List, FrozenSet, TypeVar

Port = Literal['Input', 'Output']
Input: Literal['Input'] = 'Input'
Output: Literal['Output'] = 'Output'
SupportedDatasetNames = Literal['MNIST']
DataQuery = Mapping[Port, Variable]


class DataContentBound(Protocol):
    def __getitem__(self, index): ...

    def __len__(self): ...

    def __contains__(self, item): ...

    def __iter__(self): ...


DataContentCov = TypeVar('DataContentCov', bound=DataContentBound, covariant=True)
DataContentContra = TypeVar('DataContentContra', bound=DataContentBound, contravariant=True)
DataContent = TypeVar('DataContent', bound=DataContentBound)


class Subset(Protocol):
    @property
    @abstractmethod
    def tag(self) -> Literal['All', 'FixedTrain', 'FixedTest', 'RandomSample']: ...

    @property
    @abstractmethod
    def indices(self) -> List[int]: ...


class FixedSubset(NamedTuple):
    tag: Literal['FixedTrain', 'FixedTest', 'All']
    indices: List[int]

    @property
    def uid(self) -> Literal['All', 'FixedTrain', 'FixedTest']:
        return self.tag


class FixedSample(NamedTuple):
    indices: List[int]
    tag: Literal['RandomSample']


class DataConfig(Protocol):
    @property
    @abstractmethod
    def base_path(self) -> Path: ...

    @property
    @abstractmethod
    def type(self) -> Literal['DataConfig']: ...


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

    def subset(self, subset: Subset) -> Data[DataContentCov]: ...


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
