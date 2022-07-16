import abc
from typing import Literal, Protocol, NamedTuple, FrozenSet, runtime_checkable

BaseVariableType = Literal['BaseVariable']


@runtime_checkable
class BaseVariable(Protocol):
    @property
    @abc.abstractmethod
    def type_name(self) -> str: ...

    @property
    @abc.abstractmethod
    def type(self) -> BaseVariableType: ...

    @abc.abstractmethod
    def fmt(self) -> str: ...

    @abc.abstractmethod
    def _asdict(self) -> dict: ...


@runtime_checkable
class CustomHashBaseVariable(Protocol):
    @property
    @abc.abstractmethod
    def type_name(self) -> str: ...

    @property
    @abc.abstractmethod
    def type(self) -> BaseVariableType: ...

    @abc.abstractmethod
    def fmt(self) -> str: ...

    @abc.abstractmethod
    def struct_hash(self, ignore_names: bool) -> str: ...


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


class Gamma(NamedTuple):
    alpha: float
    beta: float
    type_name: str = 'gamma'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        return f"Gamma({self.alpha}, {self.beta})"


class Bounded(NamedTuple):
    max: float
    min: float
    type_name: str = 'bounded'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        return f"Bounded({self.min}, {self.max})"


class OneHot(NamedTuple):
    n_category: int
    type_name: str = '1hot'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        return f"OneHot({self.n_category})"


class NamedCategorical(NamedTuple):
    names: FrozenSet[str]
    type_name: str = 'named_categorical'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        return f"Categorical({', '.join(self.names)})"

    def struct_hash(self, ignore_names: bool) -> str:
        content = (
            str(len(self.names)) if ignore_names
            else f"[{','.join(self.names)}]")
        return f"B[{self.type_name}|{content}]"


class CategoryIds(NamedTuple):
    max_id_len: int
    type_name: str = 'category_ids'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        return f"CategoryIds(max_id_len={self.max_id_len})"


class Ordinal(NamedTuple):
    n_category: int
    type_name: str = 'ordinal'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        return f"Ordinal({self.n_category})"


class CategoricalVector(NamedTuple):
    n_category: int
    n_embedding: int
    type_name: str = '2vec'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        return f"CategoricalVector({self.n_category}, n_embedding={self.n_embedding})"


class Gaussian(NamedTuple):
    mean: float = 0
    var: float = 1
    type_name: str = 'gaussian'
    type: Literal['BaseVariable'] = 'BaseVariable'

    def fmt(self) -> str:
        return f"Gaussian({self.mean}, {self.var})"


def struct_hash_base_variable(var: BaseVariable, ignore_names: bool) -> str:
    if isinstance(var, CustomHashBaseVariable):
        return var.struct_hash(ignore_names)
    else:
        # noinspection PyProtectedMember
        # Because Pycharm sucks
        var_dict = var._asdict()
        var_dict.pop("type")
        var_type = var_dict.pop("type_name")
        content = "|".join(str(int(v)) if isinstance(v, bool) else str(v)
                           for _, v in var_dict.items())
        return f"B[{var_type}|{content}]"
