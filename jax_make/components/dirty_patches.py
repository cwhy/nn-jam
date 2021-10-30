from typing import NamedTuple, Protocol, List

import jax
import jax.numpy as xp
from einops import rearrange
from jax import vmap
from numpy.typing import NDArray

from jax_make.utils.activations import Activation
from jax_make.component_protocol import Component, merge_params
from jax_make.components.mlp import Mlp
from jax_make.params import ArrayTree, RNGKey


class DirtyPatchesConfigs(Protocol):
    dim_out: int
    n_sections_w: int
    n_sections_h: int
    w: int
    h: int
    ch: int
    mlp_n_hidden: List[int]
    mlp_activation: Activation
    dropout_keep_rate: float


class DirtyPatches(NamedTuple):
    dim_out: int
    n_sections_w: int
    n_sections_h: int
    w: int
    h: int
    ch: int
    mlp_n_hidden: List[int]
    mlp_activation: Activation
    dropout_keep_rate: float

    @staticmethod
    def make(config: DirtyPatchesConfigs) -> Component:
        assert config.w % config.n_sections_w == 0
        assert config.h % config.n_sections_h == 0
        dim_w = config.w // config.n_sections_w
        dim_h = config.h // config.n_sections_h
        components = {
            'mlp': Mlp.make(Mlp(
                n_hidden=config.mlp_n_hidden,
                activation=config.mlp_activation,
                dropout_keep_rate=config.dropout_keep_rate,
                n_in=dim_w * dim_h,
                n_out=config.dim_out
            ))
        }

        # [h, w, ch] -> [dim_out, n_sections_h, n_sections_w, ch]
        def _fn(params: ArrayTree, x: NDArray, rng: RNGKey) -> NDArray:
            x = xp.expand_dims(x, axis=0)
            # n h w c
            patches = jax.lax.conv_general_dilated_patches(
                lhs=x,
                filter_shape=(dim_h, dim_w),
                window_strides=(dim_h, dim_w),
                padding='VALID',
                rhs_dilation=(1, 1),
                dimension_numbers=("NHWC", "OIHW", "NHWC")
            ).squeeze(0)
            # n_patches_h, n_patches_w, ch*dim_h*dim_w
            patches = rearrange(patches, 'h w (c ph pw) -> (h w c) (ph pw)',
                                c=config.ch, ph=dim_h, pw=dim_w)

            features = vmap(components['mlp'].pipeline, (None, 0, None), 0)(params['mlp'], patches, rng)
            return rearrange(features, '(h w c) out -> out h w c',
                             out=config.dim_out, h=config.n_sections_h, w=config.n_sections_w, c=config.ch)

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return Component.from_pipeline(merge_params(components), _fn)
