from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Generic, List, Literal
from supervised_benchmarks.dataset_protocols import DataContent, Port, DataPool, Data
from supervised_benchmarks.metric_protocols import PairMetric, MetricResult
from supervised_benchmarks.protocols import Model, Benchmark
from supervised_benchmarks.sampler import Sampler, FullBatchSampler, FullBatchSamplerConfig


# Using dataclass because NamedTuple does not support generics
@dataclass(frozen=True)
class BenchmarkConfig(Generic[DataContent]):
    metrics: Mapping[Port, PairMetric[DataContent]]
    type: Literal['BenchmarkConfig'] = 'BenchmarkConfig'

    # noinspection PyTypeChecker
    # Because Pycharm sucks
    def prepare(self, data: Mapping[Port, Data[DataContent]]) -> Benchmark[DataContent]:
        return BenchmarkImp(
            sampler=FullBatchSamplerConfig().get_sampler(data),
            config=self
        )


# Using dataclass because NamedTuple does not support generics
@dataclass(frozen=True)
class BenchmarkImp(Generic[DataContent]):
    sampler: Sampler[DataContent]
    config: BenchmarkConfig[DataContent]

    def measure(self, model: Model) -> List[MetricResult]:
        sampler: Sampler = self.sampler
        metrics = self.config.metrics
        assert all((k in model.repertoire) for k in metrics)

        if sampler.tag == 'FullBatchSampler':
            assert isinstance(sampler, FullBatchSampler)
            return [
                metric.measure(
                    model.perform(sampler.full_batch, tgt),
                    sampler.full_batch[tgt])
                for tgt, metric in metrics.items()]
        else:
            raise NotImplementedError
