from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, Mapping, Literal, FrozenSet

from variable_protocols.protocols import Variable

from supervised_benchmarks.dataset_protocols import DataContent, DataPool, Dataset, DataConfig
from supervised_benchmarks.ports import Port


class ModelConfig(Protocol[DataContent]):
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

    def prepare(self) -> Performer[DataContent]: ...

# Mapping[Port, DataPool[DataContent]]


class Performer(Protocol[DataContent]):
    @property
    @abstractmethod
    def model(self) -> ModelConfig:
        """
        The model that the performer based on
        """
        ...

    def perform(self, data_src: Mapping[Port, DataContent], tgt: Port) -> DataContent: ...

    def perform_batch(self,
                      data_src: Mapping[Port, DataContent],
                      tgt: FrozenSet[Port]) -> Mapping[Port, DataContent]: ...


