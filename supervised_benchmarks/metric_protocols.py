# Metric Graphs
from typing import Literal, Protocol, NamedTuple, Any, List

Metric = Literal['mean_acc', 'categorical_acc']


class ResultType(Protocol):
    result: Any
    type: Metric


class MetricResult(Protocol):
    result: ResultType
    result_type: Metric


class MeanAcc(NamedTuple):
    result: float
    type: Literal['mean_acc'] = 'mean_acc'


class CategoricalAcc(NamedTuple):
    result: List[float]
    type: Literal['categorical_acc'] = 'categorical_acc'
