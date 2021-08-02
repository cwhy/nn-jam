import abc
from pathlib import Path

from variable_protocols.variables import VariableGroup, OneHot, Bounded
from typing import Callable, NamedTuple, Literal, Generic, TypeVar, Protocol, Tuple, Generic, List, Final

from supervised_benchmarks.metric_protocols import Metric, MetricResult

FeatureTypeTag = TypeVar('FeatureTypeTag')
SubsetTypeTag = TypeVar('SubsetTypeTag')
Model = TypeVar('Model')


# Feature matching concept

class Data(Protocol[FeatureTypeTag, SubsetTypeTag]):
    pass


# noinspection PyPropertyDefinition
class DataEnv(Protocol[FeatureTypeTag, SubsetTypeTag]):
    @property
    def input_protocol(self) -> VariableGroup: ...

    @property
    def output_protocol(self) -> VariableGroup: ...

    @property
    def input(self) -> Data[FeatureTypeTag, SubsetTypeTag]: ...

    @property
    def output(self) -> Data[FeatureTypeTag, SubsetTypeTag]: ...


class DataPair(Protocol[FeatureTypeTag, SubsetTypeTag]):
    output_protocol: VariableGroup
    output: Data[FeatureTypeTag, SubsetTypeTag]
    target: Data[FeatureTypeTag, SubsetTypeTag]


class ModelConfig(Protocol):
    type: Literal['ModelConfig'] = 'ModelConfig'


# noinspection PyPropertyDefinition
class DataConfig(Protocol):
    @property
    @abc.abstractmethod
    def base_path(self) -> Path: ...

    @property
    @abc.abstractmethod
    def shuffle(self) -> bool: ...

    @property
    @abc.abstractmethod
    def type(self) -> Literal['DataConfig']: return 'DataConfig'


class Metrics(Protocol):
    type: Literal['MetricConfig'] = 'MetricConfig'


SupportedDatasetNames = Literal['MNIST']


class ModelUtils(Protocol[Model]):
    @staticmethod
    def init(model_config: ModelConfig) -> Model:
        pass

    @staticmethod
    def train(model: Model,
              metric_config: Metrics,
              data: DataEnv) -> Model:
        pass

    @staticmethod
    def supervise(model: Model, data: DataEnv) -> DataPair:
        pass

    @staticmethod
    def unsupervise(model: Model, data: DataEnv) -> DataPair:
        pass


class Dataset(Protocol):
    @abc.abstractmethod
    def __init__(self, data_config: DataConfig) -> None: ...

    @property
    @abc.abstractmethod
    def train(self) -> DataEnv: ...

    @property
    @abc.abstractmethod
    def test(self) -> DataEnv: ...

    @property
    @abc.abstractmethod
    def name(self) -> SupportedDatasetNames: ...


# Output-metric matching concept

class BenchUtils(Protocol):
    @staticmethod
    def bench(metric_queries: List[Metric],
              dataset: Dataset,
              model_utils: ModelUtils) -> List[MetricResult]:
        pass

    @staticmethod
    def test(metric_queries: List[Metric],
             model: Model,
             data: Dataset) -> List[MetricResult]:
        pass

    @staticmethod
    def measure(metric: Metric, data_pair: DataPair) -> MetricResult:
        pass
