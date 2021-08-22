from abc import abstractmethod
from pathlib import Path

from variable_protocols.variables import Variable
from typing import Literal, Protocol, NamedTuple, Dict, List, Any, Set, FrozenSet, Iterator, TypeVar, Sequence, \
    runtime_checkable

Port = Literal['Input', 'Output']
Input: Literal['Input'] = 'Input'
Output: Literal['Output'] = 'Output'
SupportedDatasetNames = Literal['MNIST']
DataQuery = Dict[Port, Variable]
DataContent = TypeVar('DataContent', bound=Sequence)


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


class Data(Protocol[DataContent]):
    port: Port
    protocol: Variable
    subset: Subset
    content: DataContent


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


class Sampler(Protocol):
    @property
    @abstractmethod
    def tag(self) -> Literal['FixedEpochSampler', 'FullBatchSampler', 'MiniBatchSampler']: ...


@runtime_checkable
class MiniBatchSampler(Protocol[DataContent]):
    @property
    @abstractmethod
    def iter(self) -> Dict[Port, Iterator[DataContent]]: ...

    @property
    @abstractmethod
    def tag(self) -> Literal['FixedEpochSampler', 'MiniBatchSampler']: ...


@runtime_checkable
class FixedEpochSampler(Protocol[DataContent]):
    @property
    @abstractmethod
    def iter(self) -> Dict[Port, Iterator[DataContent]]: ...

    @property
    @abstractmethod
    def num_batches(self) -> int: ...

    @property
    @abstractmethod
    def tag(self) -> Literal['FixedEpochSampler']: ...


@runtime_checkable
class FullBatchSampler(Protocol[DataContent]):
    @property
    @abstractmethod
    def full_batch(self) -> Dict[Port, DataContent]: ...

    @property
    @abstractmethod
    def tag(self) -> Literal['FullBatchSampler']: ...
