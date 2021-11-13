from __future__ import annotations
from abc import abstractmethod
from typing import Protocol, List, Optional, NamedTuple

from variable_protocols.protocols import Variable, struct_check, fmt


# TODO: Hole oriented design
# Compare Variables and Variables with holes
class Transformation(Protocol):
    @property
    @abstractmethod
    def source(self) -> Variable: ...

    @property
    @abstractmethod
    def target(self) -> Variable: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


Tr = Transformation


class NewTransformation(NamedTuple):
    source: Variable
    target: Variable
    name: str


def new_transformation(source: Variable, target: Variable, name: str) -> Tr:
    # noinspection PyTypeChecker
    # cus Pycharm sucks
    return NewTransformation(source, target, name)


def check(transformation: Transformation,
          source: Variable,
          target: Variable,
          ignore_names: bool = False) -> bool:
    return struct_check(transformation.source, source, ignore_names) and \
           struct_check(transformation.target, target, ignore_names)


def transform(transformation: Transformation,
              source: Variable) -> Variable:
    if not struct_check(transformation.source, source):
        raise ValueError(f"Expected source of \"{fmt(transformation.source)}\","
                         f" got \"{fmt(source)}\"")
    return transformation.target


def pipe(trs: List[Transformation], name: Optional[str] = None) -> Tr:
    wrong_list = []
    for back, front in zip(trs[:-1], trs[:-1]):
        if not back.target == front.source:
            wrong_list.append((back.target, front.source))

    if len(wrong_list) > 0:
        mismatch_msgs = "".join(f"\n  {v[0]} != {v[1]}" for v in wrong_list)
        raise ValueError(f"Unable to make pipe. "
                         f"Found {len(wrong_list)} mismatching Transformations: {mismatch_msgs}")
    else:
        name = "".join(t.name for t in trs) if not name else name
        return new_transformation(
            source=trs[0].source,
            target=trs[-1].target,
            name=name
        )
