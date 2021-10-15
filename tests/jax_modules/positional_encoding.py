from functools import reduce
from typing import NamedTuple, TypedDict, Protocol, Tuple, Literal
import jax.numpy as xp

import numpy.typing as npt
from jax import vmap, lax

from tests.jax_components import Component
from tests.jax_random_utils import WeightParams, ArrayTree


class PositionalEncodingConfigs(Protocol):
    input_shape: Tuple[int, ...]
    input_channels: int
    output_channels: int
    dim_encoding: int
    positional_encode_strategy: Literal['dot']


class PositionalEncoding(NamedTuple):
    input_shape: Tuple[int, ...]
    input_channels: int
    output_channels: int
    dim_encoding: int
    positional_encode_strategy: Literal['dot']

    @staticmethod
    def make(config: PositionalEncodingConfigs) -> Component:
        components = {
            f'encoding_dim_{i}': WeightParams(shape=(config.dim_encoding, dim), init="embedding")
            for i, dim in enumerate(config.input_shape)
        }
        assert config.positional_encode_strategy == 'dot'
        assert config.output_channels == config.dim_encoding == config.input_channels

        # [input_channels, *input_shape] -> [output_channels, prod(input_shape)]
        def _fn(params: ArrayTree, x: npt.NDArray) -> npt.NDArray:
            def _t_outer(a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
                return a[..., None] @ b[None, :]

            pos_encode = reduce(vmap(_t_outer, (0, 0), 0),
                                [params[f'encoding_dim_{i}'] for i in range(len(config.input_shape))])
            return x * pos_encode

        return Component.from_fixed_process(components, _fn)
