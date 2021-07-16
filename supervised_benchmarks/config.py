from variables import VariableGroup, VariableTensor
from typing import Callable, NamedTuple, Literal, Generic, TypeVar, Protocol, Tuple, Generic, List, Final

FeatureLayouts = Literal["flatten"]
Input = TypeVar('Input')
Output = TypeVar('Output')
Results = TypeVar('Results')

mnist_in = VariableGroup(name="mnist_in",
                         variables={
                             VariableTensor(Bounded(max=1, min=0), (28, 28))
                         })
mnist_out = VariableGroup(name="mnist_out",
                          variables={
                              VariableTensor(OneHot(n_category=10), (1,))
                          })

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
    type: Literal['DataConfig'] = 'DataConfig'


class MetricConfig(Protocol):
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
              metric_config: MetricConfig,
              data: DataEnv) -> Tuple[Model[ModelConfig], Results[MetricConfig]]:
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


class BenchUtils(Protocol):
    @staticmethod
    def bench(metric_config: MetricConfig,
              dataset_utils: DataSetUtils,
              model_utils: ModelUtils) -> Metrics[MetricConfig]:
        pass

    @staticmethod
    def test(metric_config: MetricConfig,
             model: Model[ModelConfig],
             data: DataSet[DataConfig]) -> Metrics[MetricConfig]:
        pass

    @staticmethod
    def measure(data_pair: DataPair) -> Metrics[MetricConfig]:
        pass
