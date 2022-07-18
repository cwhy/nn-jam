from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, Literal, FrozenSet

from numpy.typing import NDArray

from supervised_benchmarks.dataset_protocols import DataUnit, PortSpecs
from supervised_benchmarks.ports import Port


class ModelConfig(Protocol):
    """
    All benchmark related configs are here
    """

    @property
    @abstractmethod
    def input_ports(self) -> FrozenSet[Port]: ...

    @property
    @abstractmethod
    def output_ports(self) -> FrozenSet[Port]: ...

    @property
    @abstractmethod
    def type(self) -> Literal['ModelConfig']: ...

    def prepare(self) -> Performer: ...


class Performer(Protocol):
    @property
    @abstractmethod
    def repertoire(self) -> FrozenSet[Port]:
        """
        Output ports
        """
        ...

    def perform(self, data_src: DataUnit, tgt: Port) -> NDArray: ...

    def perform_batch(self,
                      data_src: DataUnit,
                      tgt: FrozenSet[Port]) -> DataUnit: ...
