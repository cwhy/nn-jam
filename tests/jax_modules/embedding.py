from typing import NamedTuple, TypedDict, Protocol

import numpy.typing as npt
from numpy.typing import NDArray

from tests.jax_components import Component, X
from tests.jax_random_utils import WeightParams, ArrayTree


class EmbeddingsWeights(TypedDict):
    dict: npt.NDArray


class EmbeddingsConfigs(Protocol):
    dict_size: int
    dim_model: int


class Embeddings(NamedTuple):
    dict_size: int
    dim_model: int

    @staticmethod
    def make(config: EmbeddingsConfigs) -> Component:
        components = {
            'dict': WeightParams(shape=(config.dict_size, config.dim_model), init="embedding"),
        }

        # int -> float K
        def _fn(params: ArrayTree, x: NDArray) -> NDArray:
            return params['dict'][x, :]

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return Component.from_fixed_pipeline(components, _fn)
