from __future__ import annotations

from typing import List, Dict, TypeVar
from typing import NamedTuple, Callable

from numpy import typing as npt

from tests.jax_random_utils import ArrayParamTree, ArrayTree


# Wait for mypy to support recursive types


# TODO Generics?
class Component(NamedTuple):
    params: ArrayParamTree
    process: Callable[[ArrayTree, npt.NDArray], npt.NDArray]


CompVar = TypeVar("CompVar", bound=str)


def sequential(components: Dict[CompVar, Component],
               weights: ArrayTree,
               sequence: List[CompVar],
               flow_: npt.NDArray) -> npt.NDArray:
    for comp in sequence:
        flow_ = components[comp].process(weights[comp], flow_)
    return flow_
