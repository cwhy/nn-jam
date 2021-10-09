from typing import NamedTuple, TypedDict, Protocol, Tuple, Union

import jax.numpy as xp
import numpy.typing as npt

from tests.jax_components import Component
from tests.jax_random_utils import WeightParams, ArrayTree


class LayerNormWeights(TypedDict):
    a: npt.NDArray
    b: npt.NDArray


class LayerNormConfigs(Protocol):
    norm_axis: Union[Tuple[int, ...], int]
    eps: float


class LayerNorm(NamedTuple):
    norm_axis: Union[Tuple[int, ...], int]
    eps: float

    @staticmethod
    def make(config: LayerNormConfigs) -> Component:
        components = {
            'a': WeightParams(shape=(1,)),
            'b': WeightParams(shape=(1,))
        }

        def _fn(params: ArrayTree, x: npt.NDArray, _) -> npt.NDArray:
            mean = xp.mean(x, axis=config.norm_axis, keepdims=True)
            centered = x - mean
            variance = xp.mean(centered ** 2, axis=config.norm_axis, keepdims=True)
            return params['a'] * centered / xp.sqrt(variance + config.eps) + params['b']

        return Component(components, _fn)
