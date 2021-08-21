from typing import Literal, TypeVar, Protocol, List
from supervised_benchmarks.dataset_protocols import Data, DataPair
from supervised_benchmarks.metric_protocols import Metric, MetricResult

FeatureTypeTag = TypeVar('FeatureTypeTag')
SubsetTypeTag = TypeVar('SubsetTypeTag')
Model = TypeVar('Model')


# Feature matching concept

class ModelConfig(Protocol):
    type: Literal['ModelConfig'] = 'ModelConfig'


class Metrics(Protocol):
    type: Literal['MetricConfig'] = 'MetricConfig'


class ModelUtils(Protocol[Model]):
    @staticmethod
    def init(model_config: ModelConfig) -> Model:
        pass

    @staticmethod
    def train(model: Model,
              metric_config: Metrics,
              data: Data) -> Model:
        pass

    @staticmethod
    def test(model: Model, data: Data) -> DataPair:
        pass


# Output-metric matching concept

class BenchUtils(Protocol):
    @staticmethod
    def init(model_config: ModelConfig) -> Model:
        pass

    @staticmethod
    def bench(metric_queries: List[Metric],
              data: Dataset,
              model_utils: ModelUtils) -> List[MetricResult]: ...

    @staticmethod
    def test(metric_queries: List[Metric],
             model: Model,
             data: Dataset) -> List[MetricResult]: ...

    @staticmethod
    def measure(metric: Metric, data: DataPair) -> MetricResult: ...
