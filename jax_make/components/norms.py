from abc import abstractmethod
from typing import NamedTuple, Protocol, Tuple, Union

import jax.numpy as xp
import numpy.typing as npt

from jax_make.component_protocol import Component
from jax_make.params import WeightParams, ArrayTreeMapping
import jax_make.params as p


class LayerNormConfigs(Protocol):
    @property
    @abstractmethod
    def norm_axis(self) -> Union[Tuple[int, ...], int]:
        ...

    @property
    @abstractmethod
    def eps(self) -> float:
        ...


class LayerNorm(NamedTuple):
    norm_axis: Union[Tuple[int, ...], int]
    eps: float

    @staticmethod
    def make(config: LayerNormConfigs) -> Component:
        components = {
            'a': WeightParams(shape=(1,)),
            'b': WeightParams(shape=(1,))
        }

        def _fn(weights: ArrayTreeMapping, x: npt.NDArray) -> npt.NDArray:
            mean = xp.mean(x, axis=config.norm_axis, keepdims=True)
            centered = x - mean
            variance = xp.mean(centered ** 2, axis=config.norm_axis, keepdims=True)
            a, b = p.get_arr(weights, 'a'), p.get_arr(weights, 'b')
            return a * centered / xp.sqrt(variance + config.eps) + b

        return Component.from_fixed_pipeline(components, _fn)
