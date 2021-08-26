# Metric Graphs
from typing import Literal, Protocol, NamedTuple, Any, List

MetricType = Literal['mean_acc', 'categorical_acc']

DataContent = TypeVar('DataContent', bound=Sequence)
ResultContent = TypeVar('ResultContent')

class Metric(Protocol[DataContent, ResultContent]):
    protocol: Variable
    measure: Measure[DataContent, ResultContent]
    type: MetricType

class MetricResult(Protocol[ResultContent]):
    content: ResultContent
    type: MetricType


class MeanAcc(NamedTuple):
    result: float
    type: Literal['mean_acc'] = 'mean_acc'


class CategoricalAcc(NamedTuple):
    result: List[float]
    type: Literal['categorical_acc'] = 'categorical_acc'
