from abc import abstractmethod
from typing import Literal, Protocol, NamedTuple, Dict, List

from variable_protocols.variables import Variable

Port = Literal['Input', 'Output']
Input: Literal['Input'] = 'Input'
Output: Literal['Output'] = 'Output'


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


DataQuery = Dict[Port, Variable]
