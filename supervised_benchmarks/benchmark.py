from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Generic, List, Literal
from supervised_benchmarks.dataset_protocols import DataContent, Port, DataPool, Subset, DataConfig
from supervised_benchmarks.dataset_utils import subset_all, merge_vec
from supervised_benchmarks.metric_protocols import PairMetric, MetricResult
from supervised_benchmarks.protocols import Performer
from supervised_benchmarks.sampler import FullBatchSampler, FullBatchSamplerConfig, Sampler, SamplerConfig, \
    FixedEpochSampler


# Using dataclass because NamedTuple does not support generics
@dataclass(frozen=True)
class BenchmarkConfig(Generic[DataContent]):
    metrics: Mapping[Port, PairMetric[DataContent]]
    on: Subset
    sampler_config: SamplerConfig = FullBatchSamplerConfig()
    type: Literal['BenchmarkConfig'] = 'BenchmarkConfig'

    # noinspection PyTypeChecker
    # Because Pycharm sucks
    def prepare(self, data_pool: Mapping[Port, DataPool[DataContent]]) -> Benchmark[DataContent]:
        bench_data = subset_all(data_pool, self.on)
        return Benchmark(
            sampler=self.sampler_config.get_sampler(bench_data),
            config=self
        )

    def bench(self, data_config: DataConfig, model: Performer) -> List[MetricResult]:
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
        elif sampler.tag == 'FixedEpochSampler':
            assert isinstance(sampler, FixedEpochSampler)
            results = defaultdict(list)
            targets = defaultdict(list)
            for _ in range(sampler.num_batches):
                data_map = {port: next(_iter) for port, _iter in sampler.iter.items()}
                for tgt in metrics:
                    result = performer.perform(data_map, tgt)
                    results[tgt].append(result)
                    targets[tgt].append(data_map[tgt])

            measures = []
            for tgt, metric in metrics.items():
                measure = metric.measure(merge_vec(results[tgt]), merge_vec(targets[tgt]))
                measures.append(measure)
            return measures

        else:
            raise NotImplementedError

    def make_msg(self, results: List[MetricResult]) -> str:
        return f"{self.config.on.tag} result {results}"

    def log_measure_(self, performer: Performer) -> None:
        results = self.measure(performer)
        print(f"{self.config.on.tag} result {results}")
