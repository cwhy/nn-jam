from math import sqrt
from typing import NamedTuple, List, Dict, TypedDict, Literal, Protocol

import jax.numpy as xp
import numpy.typing as npt
from einops import rearrange
from jax import vmap

from tests.einops_utils import MixWeights, mix
from tests.jax_components import Component, merge_params
from tests.jax_paramed_functions import linear
from tests.jax_random_utils import ArrayTree, RNGKey, WeightParams
from tests.jax_utils import softmax


class SelfMultiHeadAttnConfigs(Protocol):
    n_heads: int  # H
    dim_model: int  # k
    dim_input: int  # x


class SelfMultiHeadAttn(NamedTuple):
    n_heads: int  # H
    dim_model: int  # k
    dim_input: int  # x

    @staticmethod
    def make(config: SelfMultiHeadAttnConfigs) -> Component:
        # TODO check if need bias
        # noinspection PyTypeChecker
        # Because pycharm sucks
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

        def _dot_attention(q: npt.NDArray, k: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
            x = softmax(xp.einsum('mt,ms->ts', q, k)/sqrt(config.dim_model))
            return xp.einsum('ts,mt->ms', x, v)

        # [C] -> [3*K] -> [3,H,W]
        def _separate(weights: ArrayTree, x: npt.NDArray) -> npt.NDArray:
            W = config.dim_model // config.n_heads
            return components['kqv'].fixed_process(weights, x).reshape((3, config.n_heads, W))

        # [HW] -> [K] -> [K]
        def _combine(weights: ArrayTree, x: npt.NDArray) -> npt.NDArray:
            return components["out"].fixed_process(weights, x.ravel())

        # [CT] -> [KT]
        def _fn(weights: ArrayTree, x: npt.NDArray) -> npt.NDArray:
            # Split into heads ([C] -> [3*K] -> [3,H,W]) * T
            q, k, v = vmap(_separate, (None, 1), -1)(weights['kqv'], x)

            # attention H * (3*[W,T] -> [W,T])
            values = vmap(_dot_attention, (0, 0, 0))(q, k, v)

            # Merge back the heads and output ([HW] -> [K] -> [K])*T
            x = vmap(_combine, (None, -1), -1)(weights['out'], values)
            return x

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return Component.from_fixed_pipeline(merge_params(components), _fn)
