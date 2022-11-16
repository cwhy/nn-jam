from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Container, Protocol, Literal, Iterable, Union

from variable_protocols.base_variables import BaseVariable
from variable_protocols.labels import Labels, L


class DimensionFamily(NamedTuple):
    len: Optional[int] = None
    labels: Labels = Labels.empty()
    positioned: bool = True
    n_members: int = 1

    def fmt(self) -> str:
        family = f"*{self.n_members}" if self.n_members > 1 else ""
        positioned = "|shuffle-able" if not self.positioned else ""
        return f"{self.labels.fmt()}[{self.len}{positioned}]{family}"


# For syntactic sugar
# returns a checked tensor
@dataclass(frozen=True)
class Dimensions:
    dims: frozenset[DimensionFamily] = frozenset()

    @classmethod
    def from_dict(cls,
                  dims: dict[str, Optional[int]],
                  positioned_dims: Optional[Container[str]] = None,
                  n_members: Optional[dict[str, int]] = None) -> Dimensions:
        dimensions = set()
        for label, length in dims.items():
            positioned = positioned_dims is not None and label in positioned_dims
            n_member = n_members.get(label, 1) if n_members is not None else 1
            dimensions.add(DimensionFamily(length, L(label), positioned, n_member))
        return Dimensions(frozenset(dimensions))

    @classmethod
    def empty(cls) -> Dimensions:
        return cls(frozenset())

    def __radd__(self, other: Dimensions) -> Dimensions:
        return Dimensions(self.dims | other.dims)

    def __add__(self, other: Dimensions) -> Dimensions:
        return self.__radd__(other)

    def add1(self, dim: DimensionFamily) -> Dimensions:
        return Dimensions(self.dims | {dim})

    def __rmul__(self, b: BaseVariable) -> Tensor:
        b.check()
        return Tensor(b, self.dims)

    def __mul__(self, b: BaseVariable) -> Tensor:
        return self.__rmul__(b)


class Tensor(NamedTuple):
    base: BaseVariable
    dims: frozenset[DimensionFamily] = frozenset()
    labels: Labels = Labels.empty()

    def add_labels(self, labels: Union[Labels, str]) -> Tensor:
        return Tensor(self.base, self.dims, self.labels + labels)

    def fmt(self, indent: int = 2, curr_indent: int = 0) -> str:
        var_type = self.base.fmt()
        if len(self.dims) == 0:
            return f"{curr_indent * ' '}{self.labels.fmt()}#{var_type}"
        else:
            header = f"Tensor{self.labels.fmt()}#"
            dims = ", ".join(d.fmt() for d in self.dims)
            if len(header) + len(var_type) + len(": ") + len(dims) > 70:
                indent_spaces = (curr_indent + len(header) + indent) * ' '
                return f"{header}{var_type}:\n{indent_spaces}{dims}"
            else:
                return f"{header}{var_type}: {dims}"


@dataclass(frozen=True)
class TensorHub:
    tensors: frozenset[Tensor]

    def __post_init__(self) -> None:
        try:
            for t in self.tensors:
                t.base.check()
        except ValueError as e:
            raise ValueError(f"Failed to validate TensorHub: {e}")

    def fmt(self, indent: int = 2, curr_indent: int = 0) -> str:
        tensors = "\n".join(t.fmt(indent, curr_indent + indent) for t in self.tensors)
        return f"TensorHub:\n{curr_indent * ' '}{tensors}"

    def __add__(self, other: TensorHub) -> TensorHub:
        return TensorHub(self.tensors | other.tensors)

    def __radd__(self, other: TensorHub) -> TensorHub:
        return self.__add__(other)

    @classmethod
    def empty(cls) -> TensorHub:
        return cls(frozenset())


def F(base: BaseVariable, *tags: str) -> TensorHub:
    return TensorHub(frozenset({Tensor(base, Dimensions.empty().dims, L(*tags))}))


DimFam = DimensionFamily
Dim = Dimensions.from_dict
FeatureDim = Dim({"Feature": None})
V = TensorHub
