from __future__ import annotations

from typing import Any, Mapping, List, Dict, TypeVar
from typing import NamedTuple, Tuple, Union, Literal, Callable

from numpy import typing as npt


class WeightParams(NamedTuple):
    # from in to out
    shape: Tuple[int, ...]
    init: Union[Literal['kaiming'], int, float] = "kaiming"
    scale: float = 1


# Wait for mypy to support recursive types
WeightsParams = Union[Mapping[str, Any], WeightParams]
Weights = Union[Mapping[str, Any], npt.NDArray]


class Inputs(NamedTuple):
    noise: Union[Mapping[str, Any], npt.NDArray]
    x: npt.NDArray


# TODO Generics?
class Component(NamedTuple):
    weight_params: WeightsParams
    process: Callable[[Weights, npt.NDArray], npt.NDArray]


CompVar = TypeVar("CompVar", bound=str)


def sequential(components: Dict[CompVar, Component],
               weights: Weights,
               sequence: List[CompVar],
               flow_: npt.NDArray) -> npt.NDArray:
    for comp in sequence:
        flow_ = components[comp].process(weights[comp], flow_)
    return flow_
