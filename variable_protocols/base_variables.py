from __future__ import annotations
import abc
from typing import NamedTuple, Literal, FrozenSet, Protocol, Optional

from variable_protocols.labels import Labels

BaseVariableType = Literal['BaseVariable']


class BaseVariable(Protocol):
    @property
    @abc.abstractmethod
    def type_name(self) -> str: ...

    @property
    @abc.abstractmethod
    def type(self) -> BaseVariableType: ...

    @abc.abstractmethod
    def check(self) -> None: ...

    @abc.abstractmethod
    def fmt(self) -> str: ...

    @abc.abstractmethod
    def _asdict(self) -> dict: ...


class OneSideSupported(NamedTuple):
    bound: float
    min_or_max: Literal["min", "max"]
    type_name: str = 'one_side_supported'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        if self.min_or_max == 'min':
            return f"OneSideSupported(>={self.bound})"
        else:
            assert self.min_or_max == 'max'
            return f"OneSideSupported(<={self.bound})"

    def check(self) -> None:
        if self.min_or_max not in ['min', 'max']:
            raise ValueError(f"min_or_max must be 'min' or 'max', not {self.min_or_max}")


class Gamma(NamedTuple):
    alpha: float
    beta: float
    type_name: str = 'gamma'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        return f"Gamma({self.alpha}, {self.beta})"

    def is_valid(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(f"alpha and beta must be positive, not {self.alpha} and {self.beta}")


class Bounded(NamedTuple):
    max: float
    min: float
    type_name: str = 'bounded'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        return f"Bounded({self.min}, {self.max})"

    def check(self) -> None:
        if self.min >= self.max:
            raise ValueError(f"min must be less than max, but min: {self.min} >= max: {self.max}")


class OneHot(NamedTuple):
    n_category: int
    type_name: str = '1hot'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        return f"OneHot({self.n_category})"

    def check(self) -> None:
        if self.n_category <= 0:
            raise ValueError(f"n_category must be positive, not {self.n_category}")


class IDs(NamedTuple):
    id_type: Literal['int', 'str']
    type_name: str = 'ids'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def check(self) -> None:
        if self.id_type not in ['int', 'str']:
            raise ValueError(f"id_type must be 'int' or 'str', not {self.id_type}")

    def fmt(self) -> str:
        return f"IDs({self.id_type})"


class Real(NamedTuple):
    type_name: str = 'real'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def check(self) -> None:
        pass

    def fmt(self) -> str:
        return "Real()"


class Ordinal(NamedTuple):
    n_category: int
    labels: tuple[Labels, ...] = ()
    type_name: str = 'ordinal'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        if len(self.labels) == 0:
            return f"Ordinal({self.n_category})"
        else:
            return f"Ordinal({[labels.fmt() for labels in self.labels]})"

    def check(self) -> None:
        if self.n_category <= 0:
            raise ValueError(f"n_category must be positive, not {self.n_category}")
        if len(self.labels) != 0:
            for label in self.labels:
                label.check()
            if len(self.labels) != self.n_category:
                raise ValueError(
                    f"n_category({self.n_category}) must be equal to the number of labels({len(self.labels)})")

    def clear_labels(self) -> Ordinal:
        return Ordinal(self.n_category, Labels.empty())


class CategoricalVector(NamedTuple):
    n_category: int
    n_embedding: int
    type_name: str = '2vec'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        return f"CategoricalVector({self.n_category}, n_embedding={self.n_embedding})"

    def check(self) -> None:
        if self.n_category <= 0:
            raise ValueError(f"n_category must be positive, not {self.n_category}")
        if self.n_embedding <= 0:
            raise ValueError(f"n_embedding must be positive, not {self.n_embedding}")


class Gaussian(NamedTuple):
    mean: float = 0
    var: float = 1
    type_name: str = 'gaussian'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        return f"Gaussian({self.mean}, {self.var})"

    def check(self) -> None:
        if self.var <= 0:
            raise ValueError(f"var must be positive, not {self.var}")


Γ = Gamma
N = Gaussian
Ord = Ordinal
