from typing import NamedTuple, TypedDict, Protocol, Tuple, Literal

import numpy.typing as npt
from jax import vmap

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
            f'encoding_dim_{i}': WeightParams(shape=(dim, config.dim_encoding), init="embedding")
            for i, dim in enumerate(config.input_shape)
        }
        assert config.positional_encode_strategy == 'dot'
        assert config.output_channels == config.dim_encoding == config.input_channels

        # [input_channels, *input_shape] -> [output_channels, prod(input_shape)]
        def _fn(params: ArrayTree, x: npt.NDArray) -> npt.NDArray:
            for i in range(len(config.input_shape)):
                def __fn(_x: npt.NDArray) -> npt.NDArray:
                    return _x * params[f'encoding_dim_{i}']
                for _ in reversed(range(i)):
                    __fn = vmap(__fn, 0, -1)
                for _ in range(i, len(config.input_shape)):
                    __fn = vmap(__fn, -1, -1)
                x = __fn(x)
            return x.reshape((config.output_channels, -1))

        return Component.from_fixed_process(components, _fn)
