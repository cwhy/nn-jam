from __future__ import annotations

from abc import abstractmethod
from typing import Literal, NamedTuple, Protocol

# If the port is both Input and Output, type it as Input
from variable_protocols.tensorhub import Tensor

PortType = Literal['Input', 'Output', 'Context']


class NewPort(NamedTuple):
    protocol: Tensor
    type: PortType
    name: str


class Port(Protocol):
    @property
    @abstractmethod
    def type(self) -> PortType: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def protocol(self) -> Tensor: ...


class InputPort(Protocol):
    @property
    @abstractmethod
    def type(self) -> Literal['Input']: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def protocol(self) -> Tensor: ...


class OutputPort(Protocol):
    @property
    @abstractmethod
    def type(self) -> Literal['Output']: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def protocol(self) -> Tensor: ...
