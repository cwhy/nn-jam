from typing import Literal, TypeVar, Protocol

from supervised_benchmarks.dataset_protocols import DataPair
from supervised_benchmarks.sampler import Sampler

Model = TypeVar('Model')


# Feature matching concept

class ModelConfig(Protocol):
    type: Literal['ModelConfig'] = 'ModelConfig'


class Metrics(Protocol):
    type: Literal['MetricConfig'] = 'MetricConfig'


class ModelUtils(Protocol[Model]):
    @staticmethod
    def brew(model_config: ModelConfig,
             train_sampler: Sampler,
             validate_sampler: Sampler) -> Model:
        pass

    @staticmethod
    def predict(model: Model, test_sampler: Sampler) -> DataPair:
        pass


# Output-metric matching concept

# class BenchUtils(Protocol):
#     @staticmethod
#     def init(model_config: ModelConfig) -> Model:
#         pass
#
#     @staticmethod
#     def bench(metric_queries: List[Metric],
#               data: Dataset,
#               model_utils: ModelUtils) -> List[MetricResult]: ...
#
#     @staticmethod
#     def test(metric_queries: List[Metric],
#              model: Model,
#              data: Dataset) -> List[MetricResult]: ...
#
#     @staticmethod
#     def measure(metric: Metric, data: DataPair) -> MetricResult: ...
#