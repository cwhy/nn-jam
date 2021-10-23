from functools import reduce
from math import prod
from typing import NamedTuple, Protocol, Tuple, Literal

import numpy.typing as npt
from jax import vmap

from tests.jax_components import Component, X
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
        def _fn(params: ArrayTree, inputs: ArrayTree) -> ArrayTree:
            x = inputs[X]

            def _t_outer(a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
                return a[..., None] @ b[None, :]

            pos_encode = reduce(vmap(_t_outer, (0, 0), 0),
                                [params[f'encoding_dim_{i}'] for i in range(len(config.input_shape))])
            x *= pos_encode
            # [output_channels, *input_shape]

            return {X: x.reshape(config.output_channels, prod(config.input_shape)),
                    'pos_enc': pos_encode.reshape(config.output_channels, prod(config.input_shape))}

        # noinspection PyTypeChecker
        # Because Pycharm sucks
        return Component.from_fixed_process({X}, {X, 'pos_enc'}, components, _fn)
