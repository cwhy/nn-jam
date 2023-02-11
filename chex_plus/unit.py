from __future__ import annotations

from typing import Dict, NamedTuple, Sequence, TypedDict

import jax
from jax import Array

from variable_protocols.tensorhub import Dimensions

ArrDict = Dict[str, Array]


class Unit:
    def __init__(self,
                 weight_configs: dict[str, Dimensions],
                 input_configs: dict[str, Dimensions],
                 output_configs: dict[str, Dimensions],
                 pipeline_fn):
        self.weight_configs = weight_configs
        self.input_configs = input_configs
        self.output_configs = output_configs

    def compile(self, weights, inputs):
        return self.pipeline_fn(weights, inputs)


class Linear:
    array_shapes = {
        'w': "1",
        'b': "1",
        'x': "1"
    }

    class Weights(TypedDict):
        w: Array
        b: Array

    @staticmethod
    def _f(x: Array, w: Array, b: Array) -> Array:
        return w * x + b

    @staticmethod
    def f(x: Array, w: Array, b: Array) -> Array:


class Mlp:
    class Weights(TypedDict):
        layers: list[Linear.Weights]

    @staticmethod
    def f(x: Array, layers: list[Linear.Weights]) -> Array:
        x.shape
        for l_w in layers:
            x = Linear.f(x, **l_w)
        return x


class AutoEncoder(NamedTuple):
    @staticmethod
    def fn(x: Array, encoder_weight: Mlp.Weights, decoder_weight: Mlp.Weights) -> Array:
        encoded = Mlp.f(x, **encoder_weight)
        return Mlp.f(encoded, **decoder_weight)


class Attention(NamedTuple):
    @staticmethod
    def f(q: Array, k: Array, v: Array) -> Array:
        return jax.nn.softmax(q.T @ k) * v


class SelfAttentionWeights(NamedTuple):
    q: Array
    k: Array
    v: Array


class SelfAttention(NamedTuple):
    @staticmethod
    def self_attention(x: Array, to_q: Array, to_k: Array, to_v: Array) -> Array:
        return Attention.f(to_q @ x, to_k @ x, to_v @ x)


class LinearComponent(NamedTuple):
    w: LinearWeights
    f: LinearProtocol


class MlpComponents(NamedTuple):
    layers: list[LinearComponent]


def mlp(w: MlpWeights, x: LinearInputs, c: MlpComponents) -> Array:
    x_ = x.x
    for l in c.layers:
        x_ = l.f(x_)
    return x_
