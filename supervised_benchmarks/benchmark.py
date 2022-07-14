from __future__ import annotations

from collections import defaultdict
from typing import Mapping, List, Literal, NamedTuple

from supervised_benchmarks.dataset_protocols import DataPool, DataConfig, FixedSubsetType
from supervised_benchmarks.metric_protocols import PairMetric, MetricResult
from supervised_benchmarks.numpy_utils import merge_vec
from supervised_benchmarks.ports import Port
from supervised_benchmarks.protocols import Performer
from supervised_benchmarks.sampler import FullBatchSampler, FullBatchSamplerConfig, Sampler, SamplerConfig, \
    FixedEpochSampler


class BenchmarkConfig(NamedTuple):
    metrics: Mapping[Port, PairMetric]
    on: FixedSubsetType
    sampler_config: SamplerConfig = FullBatchSamplerConfig()
    type: Literal['BenchmarkConfig'] = 'BenchmarkConfig'

    def prepare(self, data_pool: DataPool) -> Benchmark:
        bench_data = data_pool.fixed_subsets[self.on]
        return Benchmark(
            sampler=self.sampler_config.get_sampler(bench_data),
            config=self
        )

    def bench(self, data_config: DataConfig, model: Performer) -> List[MetricResult]:
        data_pool = data_config.get_data()
        benchmark = self.prepare(data_pool)
        return benchmark.measure(model)


class Benchmark(NamedTuple):
    sampler: Sampler
    config: BenchmarkConfig

    def measure(self, performer: Performer) -> List[MetricResult]:
        sampler: Sampler = self.sampler
        metrics = self.config.metrics
        assert all((k in performer.repertoire) for k in metrics)

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
                data_map = next(sampler.iter)
                for tgt in metrics:
                    result = performer.perform(data_map, tgt)
                    results[tgt].append(result)
                    targets[tgt].append(data_map[tgt])

            measures = []
            for tgt, metric in metrics.items():
                print("out_mean: ", merge_vec(results[tgt]).mean())
                measure = metric.measure(merge_vec(results[tgt]), merge_vec(targets[tgt]))
                measures.append(measure)
            return measures

        else:
            raise NotImplementedError

    def make_msg(self, results: List[MetricResult]) -> str:
        return f"{self.config.on.tag} result {results}"

    def log_measure_(self, performer: Performer) -> List[MetricResult]:
        results = self.measure(performer)
        print(f"{self.config.on.tag} result {results}")
        return results
