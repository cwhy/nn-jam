# Metric Graphs
from typing import Literal, Protocol, NamedTuple, List, TypeVar, FrozenSet

from variable_protocols.variables import Variable

from supervised_benchmarks.dataset_protocols import DataContentCov

MetricType = Literal['mean_acc', 'categorical_acc']

ResultContent = TypeVar('ResultContent', covariant=True)


class Metric(Protocol[DataContentCov, ResultContent]):
    protocols: FrozenSet[Variable]
    measure: Measure[DataContentCov, ResultContent]
    type: MetricType


class MetricResult(Protocol[ResultContent]):
    content: ResultContent
    result_type: MetricType


class MeanAcc(NamedTuple):
    result: float
    type: Literal['mean_acc'] = 'mean_acc'


class CategoricalAcc(NamedTuple):
    result: List[float]
    type: Literal['categorical_acc'] = 'categorical_acc'
