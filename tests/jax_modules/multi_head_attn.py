from typing import NamedTuple, List, Dict, TypedDict, Literal, Protocol

import jax.numpy as xp
import numpy.typing as npt
from einops import rearrange
from jax import vmap

from tests.einops_utils import MixWeights, mix
from tests.jax_components import Component
from tests.jax_paramed_functions import linear
from tests.jax_random_utils import ArrayTree, RNGKey, WeightParams
from tests.jax_utils import softmax


class MultiHeadAttnWeights(TypedDict):
    k: MixWeights  # K, C
    q: MixWeights  # K, C
    v: MixWeights  # K, C
    out: MixWeights  # K, K


class MultiHeadAttnConfigs(Protocol):
    n_seq: int  # T
    n_data: int  # N
    n_heads: int  # H
    dim_model: int  # k
    dim_input: int  # x


class MultiHeadAttn(NamedTuple):
    n_seq: int  # T
    n_data: int  # N
    n_heads: int  # H
    dim_model: int  # k
    dim_input: int  # x

    # TODO Implement non-self version, move to jax style
    @staticmethod
    def make(config: MultiHeadAttnConfigs) -> Component:
        attn_in: List[Literal["k", "q", "v"]] = ["k", "q", "v"]
        components: Dict[Literal["k", "q", "v", "out"], Component] = {
            name: mix("x N T -> k N T", weight_shape='x k', bias_shape='k',
                      x=config.dim_input, k=config.dim_model).make()
            for name in attn_in}
        # add affine output layer on top [kk][kNT] -> [kNT]
        components["out"] = mix("k N T -> k2 N T", weight_shape="k k2", bias_shape="k2",
                                k=config.dim_model, k2=config.dim_model).make()

        def _fn(weights: ArrayTree, x: npt.NDArray, rng: RNGKey) -> npt.NDArray:
            # [CNT] -> [KNT]
            act = {
                k: rearrange(components[k].process(weights[k], x, rng),
                             "(h m) N T -> h m N T", h=config.n_heads, m=config.dim_model // config.n_heads)
                for i, k in enumerate(attn_in)
            }

            # reduce K, outer product T, for each N, H. Result is [HTTN]
            attention = softmax(xp.einsum('hmnt,hmns->htsn', act['q'], act['k']))
            # then compute the weighted values [HTTN][HMNT]=[HMNT]
            values = xp.einsum('htsn,hmnt->hmns', attention, act['v'])
            # [HMNT] -> [KNT]
            values = rearrange(values, "h m N T -> (h m) N T")
            return components["out"].process(weights['out'], values, rng)

        return Component({k: v.params for k, v in components.items()}, _fn)


def dot_attention(q: npt.NDArray, k: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
    x = softmax(xp.einsum('mt,ms->ts', q, k))
    return xp.einsum('ts,mt->ms', x, v)


class SelfMultiHeadAttnConfigs(Protocol):
    n_seq: int  # T
    n_heads: int  # H
    dim_model: int  # k
    dim_input: int  # x


class SelfMultiHeadAttn(NamedTuple):
    n_seq: int  # T
    n_heads: int  # H
    dim_model: int  # k
    dim_input: int  # x

    @staticmethod
    def make(config: SelfMultiHeadAttnConfigs) -> Component:
        # TODO check if need bias
        components = {
            "kqv": Component.from_fixed_process(
                {"w": WeightParams(shape=(config.dim_input, config.dim_model * 3)),
                 "b": WeightParams(shape=(config.dim_model * 3,), init=0)},
                linear),
            "out": Component.from_fixed_process(
                {"w": WeightParams(shape=(config.dim_model, config.dim_model)),
                 "b": WeightParams(shape=(config.dim_model,), init=0)},
                linear),

        }

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
            values = vmap(dot_attention, (0, 0, 0))(q, k, v)

            # Merge back the heads and output ([HW] -> [K] -> [K])*T
            return vmap(_combine, (None, -1), -1)(weights['out'], values)

        return Component.from_fixed_process({k: v.params for k, v in components.items()}, _fn)
