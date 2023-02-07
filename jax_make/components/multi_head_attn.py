from abc import abstractmethod
from math import sqrt
from typing import NamedTuple, Protocol

import jax.numpy as xp
from jax import Array
from jax import vmap

import jax_make.params as p
from jax_make.component_protocol import Component, merge_params, make_ports, Input, Output, \
    fixed_pipeline2processes
from jax_make.params import WeightParams, ArrayTreeMapping
from jax_make.utils.functions import softmax
from jax_make.utils.elementary_components import linear

masked_mha_port = make_ports((Input, 'mask'), (Output, 'attn'))


class SelfMultiHeadAttnConfigs(Protocol):
    @property
    @abstractmethod
    def n_heads(self) -> int: ...  # H

    @property
    @abstractmethod
    def dim_model(self) -> int: ...  # k

    @property
    @abstractmethod
    def dim_input(self) -> int: ...  # x


class SelfMultiHeadAttn(NamedTuple):
    n_heads: int  # H
    dim_model: int  # k
    dim_input: int  # x

    @staticmethod
    def make(config: SelfMultiHeadAttnConfigs) -> Component:
        # TODO check if need bias
        components = {
            "kqv": Component.from_fixed_pipeline(
                {"w": WeightParams(shape=(config.dim_input, config.dim_model * 3)),
                 "b": WeightParams(shape=(config.dim_model * 3,), init=0)},
                linear),
            "out": Component.from_fixed_pipeline(
                {"w": WeightParams(shape=(config.dim_model, config.dim_model)),
                 "b": WeightParams(shape=(config.dim_model,), init=0)},
                linear),

        }

        def _dot_attention(q: Array, k: Array, v: Array) -> Array:
            x = softmax(xp.einsum('mt,ms->ts', q, k) / sqrt(config.dim_model))
            return xp.einsum('ts,mt->ms', x, v)

        def _dot_attention_mask(q: Array, k: Array, mask: Array) -> Array:
            scores = xp.einsum('mt,ms,t->ts', q, k, mask)
            return softmax(scores / sqrt(config.dim_model))

        # [C] -> [3*K] -> [3,H,W]
        def _separate(weights: ArrayTreeMapping, x: Array) -> Array:
            W = config.dim_model // config.n_heads
            return components['kqv'].fixed_pipeline(weights, x).reshape((3, config.n_heads, W))

        # [HW] -> [K] -> [K]
        def _combine(weights: ArrayTreeMapping, x: Array) -> Array:
            return components["out"].fixed_pipeline(weights, x.ravel())

        # [CT] -> [KT]
        def _fn(weights: ArrayTreeMapping, x: Array) -> Array:
            # Split into heads ([C] -> [3*K] -> [3,H,W]) * T
            q, k, v = vmap(_separate, (None, 1), -1)(p.get_mapping(weights, "kqv"), x)

            # attention H * (3*[W,T] -> [W,T])
            values = vmap(_dot_attention, (0, 0, 0))(q, k, v)

            # Merge back the heads and output ([HW] -> [K] -> [K])*T
            x = vmap(_combine, (None, -1), -1)(p.get_mapping(weights, "out"), values)
            return x

        # [CT] -> [KT]
        def _fn_mask(weights: ArrayTreeMapping, inputs: ArrayTreeMapping, rng=None) -> ArrayTreeMapping:
            x, mask = p.get_arr(weights, Input), p.get_arr(weights, 'mask')
            # Split into heads ([C] -> [3*K] -> [3,H,W]) * T
            # W: latent dims
            q, k, v = vmap(_separate, (None, 1), -1)(p.get_mapping(weights, "kqv"), x)

            # attention H * (3*[W,T] -> [T,T]) first T is masked
            attn = vmap(_dot_attention_mask, (0, 0, None))(q, k, mask)

            # [H, T, T] [H, W, T] -> [H, W, T]
            values = xp.einsum('hts, hws -> hws', attn, v)

            # Merge back the heads and output ([HW] -> [K] -> [K])*T
            x = vmap(_combine, (None, -1), -1)(p.get_mapping(weights, "out"), values)
            return {'attn': attn, Output: x}

        processes = fixed_pipeline2processes(_fn)
        processes[masked_mha_port] = _fn_mask
        return Component(merge_params(components), processes)
