from abc import abstractmethod
from typing import Literal, Protocol, NamedTuple, Dict

from variables import VariableGroup

Port = Literal['Input', 'Output']


class Subset(Protocol):
    @abstractmethod
    @property
    def tag(self) -> Literal['FixedTrain', 'FixedTest', 'RandomSample', 'RandomSampleSubset']: ...


class FixedTrain(NamedTuple):
    tag: Literal['FixedTrain']


class FixedTest(NamedTuple):
    tag: Literal['FixedTest']


class RandomSample(NamedTuple):
    uid: str
    tag: Literal['RandomSample']


class RandomSampleSubset(NamedTuple):
    uid: str
    subset_uid: str
    tag: Literal['RandomSampleSubset']


DataQuery = Dict[Port, VariableGroup]