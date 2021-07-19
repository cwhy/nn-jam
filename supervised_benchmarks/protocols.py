from pathlib import Path

from variables import VariableGroup, VariableTensor
from typing import Callable, NamedTuple, Literal, Generic, TypeVar, Protocol, Tuple, Generic, List, Final

from supervised_benchmarks.metric_protocols import Metric, MetricResult

FeatureTypeTag = TypeVar('FeatureTypeTag')
SubsetTypeTag = TypeVar('SubsetTypeTag')


# Feature matching concept

class Data(Protocol[FeatureTypeTag, SubsetTypeTag]):
    pass


class DataEnv(Protocol[FeatureTypeTag, SubsetTypeTag]):
    input_protocol: VariableGroup
    output_protocol: VariableGroup
    input: Data[FeatureTypeTag, SubsetTypeTag]
    output: Data[FeatureTypeTag, SubsetTypeTag]


class DataPair(Protocol[FeatureTypeTag, SubsetTypeTag]):
    output_protocol: VariableGroup
    output: Data[FeatureTypeTag, SubsetTypeTag]
    target: Data[FeatureTypeTag, SubsetTypeTag]


class ModelConfig(Protocol):
    type: Literal['ModelConfig'] = 'ModelConfig'


class DataConfig(Protocol):
    base_path: Path
    shuffle: bool
    type: Literal['DataConfig'] = 'DataConfig'


class Metrics(Protocol):
    type: Literal['MetricConfig'] = 'MetricConfig'


class Model(Protocol[ModelConfig]):
    pass


class DataSet(Protocol[DataConfig]):
    pass


class ModelUtils(Protocol[Model]):
    @staticmethod
    def init(model_config: ModelConfig) -> Model[ModelConfig]:
        pass

    @staticmethod
    def train(model: Model[ModelConfig],
              metric_config: Metrics,
              data: DataEnv) -> Model[ModelConfig]:
        pass

    @staticmethod
    def supervise(model: Model[ModelConfig], data: DataEnv) -> DataPair:
        pass

    @staticmethod
    def unsupervise(model: Model[ModelConfig], data: DataEnv) -> DataPair:
        pass


class DataSetUtils(Protocol[DataSet]):
    @staticmethod
    def init(data_config: DataConfig) -> DataSet[DataConfig]:
        pass

    @staticmethod
    def get_train(dataset: DataSet[DataConfig]) -> DataEnv:
        pass

    @staticmethod
    def get_test(dataset: DataSet[DataConfig]) -> DataEnv:
        pass


# Output-metric matching concept

class BenchUtils(Protocol):
    @staticmethod
    def bench(metric_queries: List[Metric],
              dataset_utils: DataSetUtils,
              model_utils: ModelUtils) -> List[MetricResult]:
        pass

    @staticmethod
    def test(metric_queries: List[Metric],
             model: Model[ModelConfig],
             data: DataSet[DataConfig]) -> List[MetricResult]:
        pass

    @staticmethod
    def measure(metric: Metric, data_pair: DataPair) -> MetricResult:
        pass
