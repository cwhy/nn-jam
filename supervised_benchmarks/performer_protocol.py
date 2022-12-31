from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, FrozenSet

from numpy.typing import NDArray

from supervised_benchmarks.dataset_protocols import DataUnit, PortSpecs
from supervised_benchmarks.ports import Port


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


class ModelConfig(Protocol):
    """
    All benchmark related configs are here
    """

    @staticmethod
    def get_ports() -> PortSpecs: ...

    def prepare(self, repertoire: FrozenSet[Port]) -> Performer: ...
