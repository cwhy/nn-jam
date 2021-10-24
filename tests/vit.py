from typing import Tuple, List, NamedTuple

from jax import numpy as xp, random, vmap
from numpy import typing as npt

from tests.jax_activations import Activation
from tests.jax_components import Component, merge_params
from tests.jax_modules.dirty_patches import DirtyPatches
from tests.jax_modules.embedding import Embeddings
from tests.jax_modules.norms import LayerNorm
from tests.jax_modules.positional_encoding import PositionalEncoding
from tests.jax_random_utils import ArrayTree
from tests.transformer import TransformerEncoderConfigs, TransformerEncoder


class VitConfigs(TransformerEncoderConfigs):
    hwc: Tuple[int, int, int]  # x
    n_patches_side: int
    mlp_n_hidden_patches: List[int]

    dim_output: int
    dict_size_output: int

    input_keep_rate: float


class Vit(NamedTuple):
    n_tfe_layers: int
    n_heads: int  # H
    dim_model: int  # k
    pos_t: int
    dropout_keep_rate: float
    eps: float
    mlp_n_hidden: List[int]
    mlp_activation: Activation

    hwc: Tuple[int, int, int]  # x
    n_patches_side: int
    mlp_n_hidden_patches: List[int]

    dim_output: int
    dict_size_output: int

    input_keep_rate: float

    @staticmethod
    def make(configs: VitConfigs) -> Component:
        components = {
            'mask_embedding': Embeddings.make(Embeddings(dict_size=1,
                                                         dim_model=configs.dim_model)),
            'out_embedding': Embeddings.make(Embeddings(dict_size=configs.dict_size_output,
                                                        dim_model=configs.dim_model)),
            'patching': DirtyPatches.make(DirtyPatches(
                dim_out=configs.dim_model,
                n_sections_w=configs.n_patches_side,
                n_sections_h=configs.n_patches_side,
                h=configs.hwc[0],
                w=configs.hwc[1],
                ch=configs.hwc[2],
                mlp_n_hidden=configs.mlp_n_hidden_patches,
                mlp_activation=configs.mlp_activation,
                dropout_keep_rate=configs.dropout_keep_rate
            )),
            'positional_encoding': PositionalEncoding.make(PositionalEncoding(
                input_shape=(configs.n_patches_side,
                             configs.n_patches_side,
                             configs.hwc[-1]),
                input_channels=configs.dim_model,
                output_channels=configs.dim_model,
                dim_encoding=configs.dim_model,
                positional_encode_strategy='dot'
            )),
            'positional_encoding_y': PositionalEncoding.make(PositionalEncoding(
                input_shape=(1,),
                input_channels=configs.dim_model,
                output_channels=configs.dim_model,
                dim_encoding=configs.dim_model,
                positional_encode_strategy='dot'
            )),
            'encoder': TransformerEncoder.make(configs),
            'norm': LayerNorm.make(LayerNorm(
                eps=configs.eps,
                norm_axis=0)),
        }
        T = configs.n_patches_side ** 2 * configs.hwc[-1]

        # [D, D, 1] -> [D]
        def mask_with_default(x: npt.NDArray, m: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
            ds = xp.repeat(d, T)
            ds *= 1 - m
            x *= m
            return ds + x

        # [H, W, C] -> [dim_model, T]
        def _fn(weights: ArrayTree, x: npt.NDArray, rng) -> npt.NDArray:
            rng, key = random.split(rng)
            x = components['patching'].pipeline(weights['patching'], x, key)
            rng, key = random.split(rng)
            # [dim_model, h, w, C]

            x_flattened = x.reshape((-1, T))
            missing_embed = weights['mask_embedding']['dict'].T
            if 0 < configs.input_keep_rate < 1:
                keep_idx = random.bernoulli(key, configs.input_keep_rate, (T,))
                masked_x = vmap(mask_with_default, (0, None, 0), 0)(x_flattened, keep_idx, missing_embed)
            elif configs.input_keep_rate == 1:
                keep_idx = xp.ones((T,))
                masked_x = x
            else:
                raise Exception("~~@@!!....")

            masked_x = components['positional_encoding'].fixed_pipeline(
                weights['positional_encoding'],
                masked_x.reshape((-1,
                                  configs.n_patches_side,
                                  configs.n_patches_side,
                                  configs.hwc[-1])))
            # [dim_model, (h, w, C)]
            y = components['positional_encoding_y'].fixed_pipeline(weights['positional_encoding_y'], missing_embed)
            yx = xp.c_[y, masked_x]
            yx = components['norm'].pipeline(weights['norm'], yx, key)
            # [dim_model, dim_out + (h, w, C)]

            rng, key = random.split(rng)
            yx = components['encoder'].pipeline(weights['encoder'], yx, key)
            # [dim_model, dim_out + (h, w, C)]
            return yx

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return Component.from_pipeline(merge_params(components), _fn)