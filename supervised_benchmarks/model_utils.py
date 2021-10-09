from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Mapping, Generic, Protocol, FrozenSet, Callable, Optional, Literal, Dict

from supervised_benchmarks.benchmark import Benchmark, BenchmarkConfig
from supervised_benchmarks.dataset_protocols import Subset, Port, DataPool, DataContent, Dataset, DataConfig, DataQuery
from supervised_benchmarks.dataset_utils import subset_all
from supervised_benchmarks.protocols import ModelConfig, Performer
from supervised_benchmarks.sampler import FixedEpochSamplerConfig, MiniBatchSampler

Probes = Literal['before_epoch_', 'after_epoch_']


@dataclass(frozen=True)
class Train(Generic[DataContent]):
    num_epochs: int
    batch_size: int
    bench_configs: List[BenchmarkConfig]
    model: TrainablePerformer[DataContent]
    data_subset: Subset
    data_config: DataConfig

    def run_(self):
        pool_dict: Mapping[Port, DataPool[DataContent]] = self.data_config.get_data()
        train_data = subset_all(pool_dict, self.data_subset)
        benchmarks: List[Benchmark] = [b.prepare(pool_dict) for b in self.bench_configs]
        train_sampler = FixedEpochSamplerConfig(self.batch_size).get_sampler(train_data)
        for epoch in range(self.num_epochs):
            if 'before_epoch_' in self.model.probe:
                self.model.probe['before_epoch_']()
            start_time = time.time()
            for _ in range(train_sampler.num_batches):
                self.model.update_(train_sampler)
            epoch_time = time.time() - start_time

            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            for b in benchmarks:
                b.log_measure_(self.model)
            if 'after_epoch_' in self.model.probe:
                self.model.probe['after_epoch_']()


class TrainablePerformer(Protocol[DataContent]):
    @property
    @abstractmethod
    def model(self) -> TrainableModelConfig:
        """
        The model that the performer based on
        """
        ...

    @property
    @abstractmethod
    def probe(self) -> Dict[Probes, Callable[[], None]]: ...

    def perform(self, data_src: Mapping[Port, DataContent], tgt: Port) -> DataContent: ...

    def perform_batch(self,
                      data_src: Mapping[Port, DataContent],
                      tgt: FrozenSet[Port]) -> Mapping[Port, DataContent]: ...

    def update_(self, sampler: MiniBatchSampler): ...


class TrainableModelConfig(ModelConfig, Protocol):
    @property
    @abstractmethod
    def train_data_config(self) -> DataConfig: ...
