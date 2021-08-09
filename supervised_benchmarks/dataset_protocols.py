from abc import abstractmethod
from typing import Literal, Protocol, NamedTuple, Dict

from variable_protocols.variables import Variable

Port = Literal['Input', 'Output']
Input: Literal['Input'] = 'Input'
Output: Literal['Output'] = 'Output'


class Subset(Protocol):
    @property
    @abstractmethod
    def tag(self) -> Literal['FixedTrain', 'FixedTest', 'RandomSample', 'RandomSampleSubset']: ...


class FixedSubset(NamedTuple):
    tag: Literal['FixedTrain', 'FixedTest']


FixedTrain = FixedSubset('FixedTrain')
FixedTest = FixedSubset('FixedTest')


class RandomSample(NamedTuple):
    uid: str
    tag: Literal['RandomSample']


class RandomSampleSubset(NamedTuple):
    uid: str
    subset_uid: str
    tag: Literal['RandomSampleSubset']


DataQuery = Dict[Port, Variable]
