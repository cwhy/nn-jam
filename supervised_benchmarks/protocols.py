from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, Mapping, Literal, FrozenSet

from numpy.typing import NDArray

from supervised_benchmarks.dataset_protocols import DataUnit
from supervised_benchmarks.ports import Port
from variable_protocols.protocols import Variable


class ModelConfig(Protocol):
    """
    All benchmark related configs are here
    """

    @property
    @abstractmethod
    def repertoire(self) -> FrozenSet[Port]:
        """
        Output ports
        """
        ...

    @property
    @abstractmethod
    def ports(self) -> Mapping[Port, Variable]:
        """
        Variable Protocol of different ports
        """
        ...

    @property
    @abstractmethod
    def type(self) -> Literal['ModelConfig']: ...

    def prepare(self) -> Performer: ...


class Performer(Protocol):
    @property
    @abstractmethod
    def model(self) -> ModelConfig:
        """
        The model that the performer based on
        """
        ...

    def perform(self, data_src: DataUnit, tgt: Port) -> NDArray: ...

    def perform_batch(self,
                      data_src: DataUnit,
                      tgt: FrozenSet[Port]) -> DataUnit: ...
