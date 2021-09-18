from __future__ import annotations
from abc import abstractmethod
from typing import Protocol, Mapping, List, Literal, FrozenSet

from variable_protocols.protocols import Variable

from supervised_benchmarks.dataset_protocols import DataContent, Port, DataPool, DataContentContra, Data
from supervised_benchmarks.metric_protocols import PairMetric, MetricResult
from supervised_benchmarks.sampler import Sampler


class Benchmark(Protocol[DataContent]):
    @property
    @abstractmethod
    def sampler(self) -> Sampler[DataContent]: ...

    def measure(self, model: Model) -> List[MetricResult]: ...


class ModelConfig(Protocol):
    type: Literal['ModelConfig'] = 'ModelConfig'


class Model(Protocol[DataContent]):

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

    def perform(self, data_src: Mapping[Port, DataContent], tgt: Port) -> DataContent: ...

    def perform_batch(self,
                      data_src: Mapping[Port, DataContent],
                      tgt: FrozenSet[Port]) -> Mapping[Port, DataContent]: ...


class ModelUtils(Protocol[DataContent]):
    @staticmethod
    def prepare(config: ModelConfig,
                pool_dict: Mapping[Port, DataPool[DataContent]]) -> Model[DataContent]: ...
