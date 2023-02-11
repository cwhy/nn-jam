from abc import abstractmethod
from typing import NamedTuple, Protocol, Literal

import jax.numpy as xp
from jax import Array

import jax_make.params as p
from jax_make.component_protocol import Component, FixedPipeline, make_ports, Output, fixed_pipeline2process
from jax_make.params import WeightParams, ArrayTreeMapping

PositionalEncodeStrategies = Literal['dot', 'sum']


class PositionalEncodingConfigs(Protocol):
    @property
    @abstractmethod
    def max_input_length(self) -> int: ...

    @property
    @abstractmethod
    def dim_encoding(self) -> int: ...

    @property
    @abstractmethod
    def positional_encode_strategy(self) -> PositionalEncodeStrategies: ...

    @property
    @abstractmethod
    def init_scale(self) -> float: ...

    @property
    @abstractmethod
    def dynamic_length(self) -> bool: ...


class PositionalEncoding(NamedTuple):
    max_input_length: int
    dim_encoding: int
    positional_encode_strategy: PositionalEncodeStrategies
    init_scale: float
    dynamic_length: bool

    @staticmethod
    def make(config: PositionalEncodingConfigs) -> Component:
        components = {
            f'embedding': WeightParams(shape=(config.dim_encoding, config.max_input_length),
                                       init="embedding",
                                       scale=config.init_scale)
        }

        def make_fn(*, dynamic: bool) -> FixedPipeline:
            # [dim_encoding, <max_encoding_length] -> [dim_encoding, <max_encoding_length]
            def _fn(weights: ArrayTreeMapping, x: Array) -> Array:
                pos_embed = p.get_arr(weights, 'embedding')
                if dynamic:
                    pos_embed = pos_embed[:, :x.shape[-1]]
                if config.positional_encode_strategy == 'dot':
                    result = pos_embed * x
                elif config.positional_encode_strategy == 'sum':
                    result = pos_embed + x
                else:
                    raise ValueError(f"Positional encoding type {config.positional_encode_strategy} is not supported")
                return result
            return _fn

        if config.dynamic_length:
            return Component.from_fixed_pipeline(components, make_fn(dynamic=True))
        else:
            return Component.from_fixed_pipeline(components, make_fn(dynamic=False))
