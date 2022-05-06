from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Protocol, Callable, Literal, Dict

from supervised_benchmarks.benchmark import Benchmark, BenchmarkConfig
from supervised_benchmarks.dataset_protocols import Subset, DataPool, DataConfig, DataUnit
from supervised_benchmarks.protocols import ModelConfig, Performer
from supervised_benchmarks.sampler import FixedEpochSamplerConfig, MiniBatchSampler

Probes = Literal['before_epoch_', 'after_epoch_']


@dataclass(frozen=True)
class Train:
    num_epochs: int
    batch_size: int
    bench_configs: List[BenchmarkConfig]
    model: TrainablePerformer
    data_subset: Subset
    data_config: DataConfig

    def run_(self):
        pool_dict: DataPool = self.data_config.get_data()
        train_data = pool_dict.subset(self.data_subset)
        benchmarks: List[Benchmark] = [b.prepare(pool_dict) for b in self.bench_configs]
        train_sampler = FixedEpochSamplerConfig(self.batch_size).get_sampler(train_data)
        for epoch in range(self.num_epochs):
            if 'before_epoch_' in self.model.probe:
                self.model.probe['before_epoch_'](pool_dict)
            start_time = time.time()
            for _ in range(train_sampler.num_batches):
                self.model.update_(train_sampler)
            epoch_time = time.time() - start_time

            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            results = [b.log_measure_(self.model)
                       for b in benchmarks]
            if 'after_epoch_' in self.model.probe:
                self.model.probe['after_epoch_'](pool_dict)


class TrainablePerformer(Performer, Protocol):
    @property
    @abstractmethod
    def probe(self) -> Dict[Probes, Callable[[DataUnit], None]]: ...

    def update_(self, sampler: MiniBatchSampler): ...


class TrainableModelConfig(ModelConfig, Protocol):
    @property
    @abstractmethod
    def train_data_config(self) -> DataConfig: ...
