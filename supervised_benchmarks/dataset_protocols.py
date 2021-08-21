from abc import abstractmethod
from pathlib import Path

from variable_protocols.variables import Variable
from typing import Literal, Protocol, NamedTuple, Dict, List, Any, Set, FrozenSet

Port = Literal['Input', 'Output']
Input: Literal['Input'] = 'Input'
Output: Literal['Output'] = 'Output'
SupportedDatasetNames = Literal['MNIST']
DataQuery = Dict[Port, Variable]


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


class Data(Protocol):
    port: Port
    protocol: Variable
    subset: Subset
    content: Any


class DataPool(Protocol):
    port: Port
    src_var: Variable
    tgt_var: Variable

    def subset(self, subset: Subset) -> Data: ...


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

    def retrieve(self, query: DataQuery) -> Dict[Port, DataPool]: ...
