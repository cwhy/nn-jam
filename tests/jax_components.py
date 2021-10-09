from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, TypeVar, Literal, Optional, Generic, Callable, Any, Mapping

from jax import random
from numpy import typing as npt

from tests.jax_random_utils import ArrayTree, RNGKey

WeightVar = TypeVar("WeightVar")
CompVar = TypeVar("CompVar", bound=str)


# TODO Generics?
@dataclass()
class Component(Generic[CompVar, WeightVar]):
    params: Dict[WeightVar, Any]
    process: Callable[[Mapping[CompVar, Any], npt.NDArray, Optional[RNGKey]], npt.NDArray]
    type: Literal["Random", "Fixed"] = "Random"

    @property
    def fixed_process(self):
        if self.type == 'Fixed':
            process = self.process

            def _fn(weights: WeightVar, x: npt.NDArray) -> npt.NDArray:
                return process(weights, x, None)

            return _fn
        else:
            raise ValueError("fixed_process is not available to Random components")

    @classmethod
    def from_fixed_process(cls,
                           params: Dict[WeightVar, Any],
                           param_f: Callable[[Mapping[CompVar, Any], npt.NDArray],
                                             npt.NDArray]) -> Component[CompVar, WeightVar]:
        def _fn(weights: Mapping[CompVar, Any], x: npt.NDArray, rng: RNGKey) -> npt.NDArray:
            return param_f(weights, x)

        return cls(params, _fn, type="Fixed")


def sequential(components: Dict[CompVar, Component[CompVar, WeightVar]],
               weights: Mapping[CompVar, Any],
               sequence: List[CompVar],
               flow_: npt.NDArray,
               rng: RNGKey) -> npt.NDArray:
    for comp_name in sequence:
        key, rng = random.split(rng)
        process = components[comp_name].process
        flow_ = process(weights[comp_name], flow_, key)
    return flow_
