from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Mapping, Generic, List, Literal, Protocol
from supervised_benchmarks.dataset_protocols import DataContent, Port, DataPool, Data, Subset, DataConfig, \
    DataContentCov
from supervised_benchmarks.dataset_utils import subset_all
from supervised_benchmarks.metric_protocols import PairMetric, MetricResult
from supervised_benchmarks.mnist import FixedTest
from supervised_benchmarks.protocols import Performer, ModelConfig
from supervised_benchmarks.sampler import FullBatchSampler, FullBatchSamplerConfig, Sampler


# Using dataclass because NamedTuple does not support generics
@dataclass(frozen=True)
class BenchmarkConfig(Generic[DataContent]):
    metrics: Mapping[Port, PairMetric[DataContent]]
    on: Subset = FixedTest
    type: Literal['BenchmarkConfig'] = 'BenchmarkConfig'

    # noinspection PyTypeChecker
    # Because Pycharm sucks
    def prepare(self, data_pool: Mapping[Port, DataPool[DataContent]]) -> Benchmark[DataContent]:
        bench_data = subset_all(data_pool, self.on)
        return Benchmark(
            sampler=FullBatchSamplerConfig().get_sampler(bench_data),
            config=self
        )

    def bench(self, data_config: DataConfig, model_config: ModelConfig) -> List[MetricResult]:
        model = model_config.prepare()
        data_pool = data_config.get_data()
        benchmark = self.prepare(data_pool)
        return benchmark.measure(model)


# Using dataclass because NamedTuple does not support generics
@dataclass(frozen=True)
class Benchmark(Generic[DataContent]):
    sampler: Sampler[DataContent]
    config: BenchmarkConfig[DataContent]

    def measure(self, performer: Performer) -> List[MetricResult]:
        sampler: Sampler = self.sampler
        metrics = self.config.metrics
        assert all((k in performer.model.repertoire) for k in metrics)

        if sampler.tag == 'FullBatchSampler':
            assert isinstance(sampler, FullBatchSampler)
            return [
                metric.measure(
                    performer.perform(sampler.full_batch, tgt),
                    sampler.full_batch[tgt])
                for tgt, metric in metrics.items()]
        else:
            raise NotImplementedError

    def make_msg(self, results: List[MetricResult]) -> str:
        return f"{self.config.on.tag} result {results}"

    def log_measure_(self, performer: Performer) -> None:
        results = self.measure(performer)
        print(f"{self.config.on.tag} result {results}")
