from __future__ import annotations

from dataclasses import dataclass
from typing import List, TypeVar, Literal, Optional, Generic, Callable, Mapping, FrozenSet, Set

from jax import random
from numpy.typing import NDArray

from jax_make.jax_random_utils import ArrayTree, RNGKey, ArrayParamTree, ArrayTreeMapping

WeightVar = TypeVar("WeightVar")
CompVar = TypeVar("CompVar", bound=str)
X: Literal['X'] = 'X'


@dataclass
class Component(Generic[CompVar, WeightVar]):
    ports_in: FrozenSet[str]
    ports_out: FrozenSet[str]
    params: Mapping[CompVar, ArrayParamTree]
    process: Callable[[Mapping[CompVar, ArrayTree],
                       ArrayTreeMapping,
                       RNGKey],
                      ArrayTreeMapping]
    type: Literal["Random", "Fixed"] = "Random"

    @property
    def is_pipeline(self) -> bool:
        return X in (self.ports_in & self.ports_out)

    @property
    def pipeline(self) -> Callable[[Mapping[CompVar, ArrayTree], NDArray, RNGKey],
                                   NDArray]:
        if self.is_pipeline:
            def _fn(weights: Mapping[CompVar, ArrayTree],
                    x: NDArray, key: RNGKey) -> NDArray:
                return self.process(weights, {X: x}, key)[X]

            return _fn
        else:
            raise ValueError(f"pipeline is not available when \"{X}\" is not in both ports_in and ports_out")

    @property
    def fixed_pipeline(self) -> Callable[[Mapping[CompVar, ArrayTree], NDArray],
                                         NDArray]:
        if self.is_pipeline:
            def _fn(weights: Mapping[CompVar, ArrayTree],
                    x: NDArray) -> NDArray:
                return self.process(weights, {X: x}, None)[X]

            return _fn
        else:
            raise ValueError(f"pipeline is not available when \"{X}\" is not in both ports_in and ports_out")

    @property
    def fixed_process(self) -> Callable[[Mapping[CompVar, ArrayTree], ArrayTreeMapping],
                                        ArrayTreeMapping]:
        if self.type == 'Fixed':
            process = self.process

            def _fn(weights: Mapping[CompVar, ArrayTree],
                    x: ArrayTreeMapping) -> ArrayTreeMapping:
                return process(weights, x, None)

            return _fn
        else:
            raise ValueError("fixed_process is not available to Random components")

    @classmethod
    def from_pipeline(cls,
                      params: Mapping[CompVar, ArrayParamTree],
                      param_f: Callable[[Mapping[CompVar, ArrayTree],
                                         NDArray, RNGKey],
                                        NDArray]) -> Component[CompVar, WeightVar]:
        def _fn(weights: Mapping[CompVar, ArrayTree],
                x: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping:
            return {X: param_f(weights, x[X], rng)}

        return cls(frozenset([X]), frozenset([X]), params, _fn, type="Random")

    @classmethod
    def from_fixed_process(cls,
                           ports_in: Set[str],
                           ports_out: Set[str],
                           params: Mapping[CompVar, ArrayParamTree],
                           param_f: Callable[[Mapping[CompVar, ArrayTree],
                                              ArrayTreeMapping],
                                             ArrayTreeMapping]) -> Component[CompVar, WeightVar]:
        def _fn(weights: Mapping[CompVar, ArrayTree],
                x: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping:
            return param_f(weights, x)

        return cls(frozenset(ports_in), frozenset(ports_out), params, _fn, type="Fixed")

    @classmethod
    def from_fixed_pipeline(cls,
                            params: Mapping[CompVar, ArrayParamTree],
                            param_f: Callable[[Mapping[CompVar, ArrayTree],
                                               NDArray],
                                              NDArray]) -> Component[CompVar, WeightVar]:
        def _fn(weights: Mapping[CompVar, ArrayTree],
                x: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping:
            return {X: param_f(weights, x[X])}

        return cls(frozenset([X]), frozenset([X]), params, _fn, type="Fixed")


def merge_params(
        components: Mapping[CompVar, Component[CompVar, WeightVar]]
) -> Mapping[CompVar, ArrayParamTree]:
    return {k: v.params for k, v in components.items()}


def connect_all(components: Mapping[CompVar, Component[CompVar, WeightVar]],
                sequence: List[CompVar]) -> Callable[[Mapping[CompVar, ArrayTree], ArrayTreeMapping, RNGKey],
                                                     ArrayTreeMapping]:
    def _fn(weights: Mapping[CompVar, ArrayTree],
            flow_: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping:
        pervious_comp = None
        for comp_name in sequence:
            key, rng = random.split(rng)
            if pervious_comp is not None:
                if components[pervious_comp].ports_out != components[comp_name].ports_in:
                    raise Exception((f"Trying to sequence components with port mismatch:"
                                     f" from {pervious_comp}:{components[pervious_comp].ports_out}"
                                     f" to {comp_name}:{components[comp_name].ports_in}"
                                     ))
            # noinspection PyTypeChecker
            # Because pycharm sucks
            flow_ = components[comp_name].process(weights[comp_name], flow_, key)
            pervious_comp = comp_name
        return flow_

    return _fn


def sequential(components: Mapping[CompVar, Component[CompVar, WeightVar]],
               sequence: List[CompVar]) -> Callable[[Mapping[CompVar, ArrayTree], NDArray, RNGKey],
                                                    NDArray]:

    if all(components[comp_name].is_pipeline for comp_name in sequence):
        def _fn(weights: Mapping[CompVar, ArrayTree],
                flow_: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping:
            for comp_name in sequence:
                key, rng = random.split(rng)
                # noinspection PyTypeChecker
                # Because pycharm sucks
                flow_ = components[comp_name].pipeline(weights[comp_name], flow_, key)
            return flow_
        return _fn
    else:
        not_list = [comp_name
                    for comp_name in sequence if not components[comp_name].is_pipeline]
        raise Exception(("The list of component is not sequence-able because component "
                         f"{not_list} are not pipelines (contain \"{X}\" in ports_in and ports_out)"))
