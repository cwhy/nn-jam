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


class CategoryIds(NamedTuple):
    max_id_len: int
    labels: Labels = Labels.empty()
    type_name: str = 'category_ids'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        if self.labels is None:
            return f"CategoryIds(max_id_len={self.max_id_len})"
        else:
            return f"CategoryIds({self.labels.fmt()})"

    def check(self) -> None:
        if self.max_id_len <= 0:
            raise ValueError(f"max_id_len must be positive, not {self.max_id_len}")
        if self.labels is not None:
            self.labels.check()
            if len(self.labels) != self.max_id_len:
                raise ValueError(f"max_id_len({self.max_id_len}) must be equal to the number of labels({len(self.labels)})")

    def clear_labels(self) -> CategoryIds:
        return CategoryIds(self.max_id_len, Labels.empty())


class Ordinal(NamedTuple):
    n_category: int
    type_name: str = 'ordinal'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        return f"Ordinal({self.n_category})"

    def check(self) -> None:
        if self.n_category <= 0:
            raise ValueError(f"n_category must be positive, not {self.n_category}")


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
