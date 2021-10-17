from functools import reduce
from typing import NamedTuple, Protocol, Tuple, Literal, List

import jax
import numpy.typing as npt
from jax import vmap

from tests.jax_activations import Activation
from tests.jax_components import Component
from tests.jax_modules.mlp import MlpConfigs, Mlp
from tests.jax_random_utils import WeightParams, ArrayTree, RNGKey


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
                n_in=dim_w * dim_h * config.ch,
                n_out=config.dim_out
            ))
        }

        # [w, h, ch] -> [dim_out, n_sections_w*n_sections_h]
        def _fn(params: ArrayTree, x: npt.NDArray, rng: RNGKey) -> npt.NDArray:

            # n c h w
            channels = images.shape[1]
            patches = jax.lax.conv_general_dilated_patches(
                lhs=images,
                filter_shape=sizes[1:-1],
                window_strides=strides[1:-1],
                padding=padding,
                rhs_dilation=rates[1:-1],
            )
            patches = jnp.transpose(patches, [0, 2, 3, 1])

            # `conv_general_dilated_patches returns patches` is channel-major order,
            # rearrange to match interface of `tf.image.extract_patches`.
            patches = jnp.reshape(patches,
                                  patches.shape[:3] + (channels, sizes[1], sizes[2]))
            patches = jnp.transpose(patches, [0, 1, 2, 4, 5, 3])
            patches = jnp.reshape(patches, patches.shape[:3] + (-1,))
            
            components['mlp'].process(, rng)
            return x

        return Component(components, _fn)
