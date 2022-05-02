from __future__ import annotations

from abc import abstractmethod
from typing import Literal, Protocol, NamedTuple

PortType = Literal['Input', 'Output', 'Context', 'OutputOptions', 'AllVars']


class Port(Protocol):
    @property
    @abstractmethod
    def type(self) -> PortType: ...

    @property
    @abstractmethod
    def name(self) -> int: ...


class Input(NamedTuple):
    type: Literal['Input'] = 'Input'

    @property
    def name(self) -> Literal['Input']:
        return self.type


class Output(NamedTuple):
    type: Literal['Output'] = 'Output'

    @property
    def name(self) -> Literal['Output']:
        return self.type


class Context(NamedTuple):
    type: Literal['Context'] = 'Context'

    @property
    def name(self) -> Literal['Context']:
        return self.type


class OutputOptions(NamedTuple):
    type: Literal['OutputOptions'] = 'OutputOptions'

    @property
    def name(self) -> Literal['OutputOptions']:
        return self.type


class AllVars(NamedTuple):
    type: Literal['AllVars'] = 'AllVars'

    @property
    def name(self) -> Literal['AllVars']:
        return self.type
