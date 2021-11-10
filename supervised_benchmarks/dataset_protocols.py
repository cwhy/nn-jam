from abc import abstractmethod
from typing import Literal, Protocol, NamedTuple, Mapping, List, FrozenSet, TypeVar

from variable_protocols.variables import Variable

Port = Literal['Input', 'Output', 'Context', 'OutputOptions', 'AllVars']
Input: Literal['Input'] = 'Input'
Output: Literal['Output'] = 'Output'
AllVars: Literal['AllVars'] = 'AllVars'

Context: Literal['Context'] = 'Context'
OutputOptions: Literal['OutputOptions'] = 'OutputOptions'

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

FixedSubsetType = Literal['All', 'FixedTrain', 'FixedTest', 'FixedValidation']


class Subset(Protocol):
    @property
    @abstractmethod
    def tag(self) -> Literal[FixedSubsetType, 'RandomSample']: ...

    @property
    @abstractmethod
    def indices(self) -> List[int]: ...


class FixedSubset(NamedTuple):
    tag: FixedSubsetType
    indices: List[int]

    @property
    def uid(self) -> FixedSubsetType:
        return self.tag


class FixedSample(NamedTuple):
    indices: List[int]
    tag: Literal['RandomSample']


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
