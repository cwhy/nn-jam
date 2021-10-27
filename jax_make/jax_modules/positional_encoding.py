from functools import reduce
from math import prod
from typing import NamedTuple, Protocol, Tuple, Literal

import numpy.typing as npt
from jax import vmap, jit

from jax_make.components import Component, X, FixedProcess
from jax_make.params import WeightParams, ArrayTree


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
        def _fn(params: ArrayTree, inputs: ArrayTree) -> ArrayTree:
            x = inputs[X]

            x *= dot_product_encode(params, len(config.input_shape))
            # [output_channels, *input_shape]

            return {X: x.reshape(config.output_channels, prod(config.input_shape))}

        # noinspection PyTypeChecker
        # Because Pycharm sucks
        return Component.from_fixed_process({X}, {X}, components, _fn)


# {} -> [output_channels, *input_shape]
def dot_product_encode(params: ArrayTree, input_n_dims: int) -> npt.NDArray:
    def _t_outer(a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
        return a[..., None] @ b[None, :]

    pos_encode = reduce(vmap(_t_outer, (0, 0), 0),
                        [params[f'encoding_dim_{i}'] for i in range(input_n_dims)])
    return pos_encode
