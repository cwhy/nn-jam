from typing import NamedTuple, List, Dict, TypedDict, Literal, Protocol

import jax.numpy as xp
import numpy.typing as npt
from einops import rearrange

from tests.einops_utils import MixWeights, mix
from tests.jax_protocols import Component, Weights
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

    @staticmethod
    def make(config: MultiHeadAttnConfigs) -> Component:
        # TODO check if need bias
        attn_in: List[Literal["k", "q", "v"]] = ["k", "q", "v"]
        components: Dict[Literal["k", "q", "v", "out"], Component] = {
            name: mix("x N T -> k N T", weight_shape='x k', bias_shape='k',
                      x=config.dim_input, k=config.dim_model).make()
            for name in attn_in}
        # add affine output layer on top [kk][kNT] -> [kNT]
        components["out"] = mix("k N T -> k2 N T", weight_shape="k k2", bias_shape="k2",
                                k=config.dim_model, k2=config.dim_model).make()

        def _fn(weights: Weights, x: npt.NDArray) -> npt.NDArray:
            # [CNT] -> [KNT]

            act = {
                k: rearrange(components[k].process(weights[k], x),
                             "(h m) N T -> h m N T", h=config.n_heads, m=config.dim_model // config.n_heads)
                for k in attn_in
            }

            # reduce K, outer product T, for each N, H. Result is [HTTN]
            attention = softmax(xp.einsum('hmnt,hmns->htsn', act['q'], act['k']))
            # then compute the weighted values [HTTN][HMNT]=[HMNT]
            values = xp.einsum('htsn,hmnt->hmns', attention, act['v'])
            # [HMNT] -> [KNT]
            values = rearrange(values, "h m N T -> (h m) N T")
            return components["out"].process(weights['out'], values)

        return Component({k: v.weight_params for k, v in components.items()}, _fn)
