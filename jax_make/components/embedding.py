from abc import abstractmethod
from typing import NamedTuple, Protocol

from jax import Array as NDArray

import jax_make.params as p
from jax_make.component_protocol import Component
from jax_make.params import WeightParams, ArrayTreeMapping


class EmbeddingsConfigs(Protocol):
    @property
    @abstractmethod
    def dict_size(self) -> int: ...

    @property
    @abstractmethod
    def dim_model(self) -> int: ...

    @property
    @abstractmethod
    def dict_init_scale(self) -> float: ...


class Embeddings(NamedTuple):
    dict_size: int
    dim_model: int
    dict_init_scale: float

    @staticmethod
    def make(config: EmbeddingsConfigs) -> Component:
        components = {
            'dict': WeightParams(shape=(config.dict_size, config.dim_model),
                                 init="embedding",
                                 scale=config.dict_init_scale),
        }

        # int -> float K
        def _fn(weights: ArrayTreeMapping, x: NDArray) -> NDArray:
            return p.get_arr(weights, 'dict')[x, :]

        return Component.from_fixed_pipeline(components, _fn)
