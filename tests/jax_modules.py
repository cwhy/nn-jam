from functools import cached_property, lru_cache
from typing import NamedTuple, Callable

import jax.numpy as xp
import numpy.typing as npt
from einops import reduce, rearrange
from jax import jit

from tests.einops_utils import MixWeights, mix
from tests.jax_utils import softmax


# TODO einops
class MultiHeadAttnBK(NamedTuple):
    n_seq: int  # T
    n_data: int  # N
    n_heads: int  # H
    dim_model: int  # K
    dim_input: int  # C
    w_k: npt.NDArray  # K, C
    w_q: npt.NDArray  # K, C
    w_v: npt.NDArray  # K, C
    out_linear: npt.NDArray  # K, K

    def forward_self(self, x: npt.NDArray) -> npt.NDArray:
        # [CNT] -> [KNT]
        dim_act = (self.n_heads, self.dim_model // self.n_heads, self.n_data, self.n_seq)  # hidden state per head.
        # first multiply the query against the keys, [KNT] = [KC] [CNT], [KNT] -> reshape [HMNT], M = K//H
        act_K = xp.einsum('kc, cnt -> knt', self.w_k, x).reshape(dim_act)
        act_Q = xp.einsum('kc, cnt -> knt', self.w_q, x).reshape(dim_act)
        act_V = xp.einsum('kc, cnt -> knt', self.w_v, x).reshape(dim_act)
        # reduce K, outer product T, for each N, H. Result is [HTTN]
        attention = xp.einsum('hmnt,hmns->htsn', act_Q, act_K)
        # softmax over sequence (T) dimension
        attention = softmax(attention)
        # then compute the weighted values [HTTN][HMNT]=[HMNT]
        attention = xp.einsum('htsn,hmnt->hmns', attention, act_V)
        # [HMNT] -> [KNT]
        attention = attention.reshape((self.dim_model, self.n_data, self.n_seq))
        # add affine output layer on top [KK][KNT] -> [KNT]
        attention = xp.einsum('kx, xnt -> knt', self.out_linear, attention)
        return attention


class MultiHeadAttnWeights(NamedTuple):
    k: MixWeights  # K, C
    q: MixWeights  # K, C
    v: MixWeights  # K, C
    out_linear: MixWeights  # K, K


class MultiHeadAttn(NamedTuple):
    n_seq: int  # T
    n_data: int  # N
    n_heads: int  # H
    dim_model: int  # k
    dim_input: int  # x

    @property
    @lru_cache()
    def process_self(self) -> Callable[[MultiHeadAttnWeights, npt.NDArray], npt.NDArray]:
        def _get_attn_head(_weights: MixWeights, _x: npt.NDArray) -> npt.NDArray:
            # TODO check if need bias
            return rearrange(
                mix("x N T -> k N T",
                    weight_shape='x k', bias_shape='k', x=self.dim_input, k=self.dim_model).process(_weights, _x),
                "k N T -> h m N T", h=self.n_heads, m=self.dim_model // self.n_heads)

        def _fn(weights: MultiHeadAttnWeights, x: npt.NDArray) -> npt.NDArray:
            # [CNT] -> [KNT]
            act_Q = _get_attn_head(weights.q, x)
            act_K = _get_attn_head(weights.k, x)
            act_V = _get_attn_head(weights.v, x)

            # reduce K, outer product T, for each N, H. Result is [HTTN]
            attention = softmax(xp.einsum('hmnt,hmns->htsn', act_Q, act_K))
            # then compute the weighted values [HTTN][HMNT]=[HMNT]
            values = xp.einsum('htsn,hmnt->hmns', attention, act_V)
            # [HMNT] -> [KNT]
            values = rearrange(values, "h m N T -> (h m) N T")
            # add affine output layer on top [kk][kNT] -> [kNT]
            # TODO check if need bias
            return mix("k N T -> k2 N T",
                       weight_shape="k k2",
                       bias_shape="k2",
                       x=self.dim_model,
                       x2=self.dim_model).process(weights.out_linear, values)
        return jit(_fn)
