from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol, NamedTuple, List, TypeVar, Tuple, Generic, Any, Callable

from variable_protocols.variables import Variable, var_scalar, ordinal

from supervised_benchmarks.dataset_protocols import DataContentContra, DataContent

PairMetricType = Literal['mean_acc', 'categorical_acc']

ResultContentCov = TypeVar('ResultContentCov', covariant=True)
ResultContent = TypeVar('ResultContent')
VarParam = TypeVar('VarParam', bound=Tuple)


class Measure(Protocol[DataContentContra]):
    def __call__(self, output: DataContentContra, target: DataContentContra) -> MetricResult: ...


class PairMetric(Protocol[DataContentContra]):
    @property
    @abstractmethod
    def protocol(self) -> Variable: ...

    @property
    @abstractmethod
    def type(self) -> PairMetricType: ...

    @property
    @abstractmethod
    def measure(self) -> Measure[DataContentContra]: ...


class MeanAccResult(NamedTuple):
    content: float
    result_type: Literal['mean_acc'] = 'mean_acc'


class CategoricalAccResult(NamedTuple):
    content: List[float]
    result_type: Literal['categorical_acc'] = 'categorical_acc'


@dataclass(frozen=True)
class PairMetricImp(Generic[DataContentContra]):
    protocol: Variable
    type: PairMetricType
    measure: Measure[DataContentContra]


class MetricResult(NamedTuple):
    content: Any
    result_type: PairMetricType

