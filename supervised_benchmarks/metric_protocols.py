from __future__ import annotations

from abc import abstractmethod
from typing import Literal, Protocol, NamedTuple, List, TypeVar, Tuple, Any, Callable

from numpy.typing import NDArray
from variable_protocols.variables import Variable

PairMetricType = Literal['mean_acc', 'categorical_acc']

VarParam = TypeVar('VarParam', bound=Tuple)


class PairMetric(Protocol):
    @property
    @abstractmethod
    def protocol(self) -> Variable: ...

    @property
    @abstractmethod
    def type(self) -> PairMetricType: ...

    @property
    @abstractmethod
    def measure(self) -> Measure: ...


class MeanAccResult(NamedTuple):
    content: float
    result_type: Literal['mean_acc'] = 'mean_acc'


class CategoricalAccResult(NamedTuple):
    content: List[float]
    result_type: Literal['categorical_acc'] = 'categorical_acc'


class MetricResult(NamedTuple):
    content: Any
    result_type: PairMetricType


Measure = Callable[[NDArray, NDArray], MetricResult]


class PairMetricImp(NamedTuple):
    protocol: Variable
    type: PairMetricType
    measure: Measure
