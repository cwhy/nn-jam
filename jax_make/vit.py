from typing import Tuple, List, NamedTuple, Callable

from jax import numpy as xp, random, vmap
from numpy import typing as npt

from jax_make.utils.activations import Activation
from jax_make.component_protocol import Component, merge_params
from jax_make.components.dirty_patches import DirtyPatches
from jax_make.components.embedding import Embeddings
from jax_make.components.mlp import Mlp
from jax_make.components.norms import LayerNorm
from jax_make.components.positional_encoding import PositionalEncoding
from jax_make.utils.functions import get_cosine_similarity
from jax_make.params import ArrayTree
from jax_make.transformer import TransformerEncoderConfigs, TransformerEncoder


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
                raise Exception("Invalide input keep rate")

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


class VitReconstruct(NamedTuple):
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
        if configs.hwc[-1] == 0:
            ch = 1
        else:
            ch = configs.hwc[-1]

        h_patch = configs.hwc[0] // configs.n_patches_side
        w_patch = configs.hwc[1] // configs.n_patches_side
        patch_size = h_patch * w_patch * configs.hwc[2]
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
                ch=ch,
                mlp_n_hidden=configs.mlp_n_hidden_patches,
                mlp_activation=configs.mlp_activation,
                dropout_keep_rate=configs.dropout_keep_rate
            )),
            'x_reconstruct': Mlp.make(
                Mlp(n_in=configs.dim_model,
                    n_hidden=configs.mlp_n_hidden_patches,
                    n_out=patch_size,
                    activation=configs.mlp_activation,
                    dropout_keep_rate=configs.dropout_keep_rate
                    )
            ),
            'y_reconstruct': Mlp.make(
                Mlp(n_in=configs.dim_model,
                    n_hidden=configs.mlp_n_hidden,
                    n_out=configs.dim_model,
                    activation=configs.mlp_activation,
                    dropout_keep_rate=configs.dropout_keep_rate
                    )
            ),
            'positional_encoding': PositionalEncoding.make(PositionalEncoding(
                input_shape=(configs.n_patches_side,
                             configs.n_patches_side,
                             ch),
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
        T = configs.n_patches_side ** 2 * ch

        # [D, D, 1] -> [D]
        def mask_with_default(x: npt.NDArray, m: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
            ds = xp.repeat(d, T)
            ds *= 1 - m
            x *= m
            return ds + x

        # x: [H, W, C] -> x_rec: [dim_model, T], y_rec: dim_model, x_mask: T
        def _fn(weights: ArrayTree, inputs: ArrayTree, rng) -> ArrayTree:
            x, y = inputs['x'], inputs['y']
            if configs.hwc[-1] == 0:
                x = xp.expand_dims(x, -1)
            rng, key = random.split(rng)
            x = components['patching'].pipeline(weights['patching'], x, key)
            rng, key = random.split(rng)
            # [dim_model, h, w, C]

            x_flattened = x.reshape((-1, T))
            missing_embed = weights['mask_embedding']['dict'].T
            if 0 < configs.input_keep_rate < 1:
                x_mask = random.bernoulli(key, configs.input_keep_rate, (T,))
                masked_x = vmap(mask_with_default, (0, None, 0), 0)(x_flattened, x_mask, missing_embed)
            elif configs.input_keep_rate == 1:
                x_mask = xp.ones((T,))
                masked_x = x
            else:
                raise Exception("Invalid input keep rate")

            masked_x = components['positional_encoding'].fixed_pipeline(
                weights['positional_encoding'],
                masked_x.reshape((-1,
                                  configs.n_patches_side,
                                  configs.n_patches_side,
                                  ch)))
            # [dim_model, (h, w, C)]
            y_masked_emb = components['positional_encoding_y'].fixed_pipeline(weights['positional_encoding_y'], missing_embed)
            yx = xp.c_[y_masked_emb, masked_x]
            yx = components['norm'].pipeline(weights['norm'], yx, key)
            # [dim_model, dim_out + (h, w, C)]

            rng, key = random.split(rng)
            yx = components['encoder'].pipeline(weights['encoder'], yx, key)
            # [dim_model, dim_out + (h, w, C)]

            y_rec, x = yx[:, 0], yx[:, 1:]
            rng, key1, key2 = random.split(rng, 2)
            x_rec = vmap(components['x_reconstruct'].pipeline, (None, 1, None), 1)(weights['x_reconstruct'], x, key1)
            y_rec = vmap(components['y_reconstruct'].pipeline, (None, 1, None), 1)(weights['x_reconstruct'], y_rec, key2)

            cos = get_cosine_similarity(configs.eps)
            y_emb = components['out_embedding'].fixed_pipeline(weights['out_embedding'], y)
            y_emb = components['positional_encoding_y'].fixed_pipeline(weights['positional_encoding_y'], y_emb)
            loss_x = xp.sum(vmap(cos, (-1, -1), -1)(x_flattened, x_rec) * (1 - x_mask)) / xp.sum((1 - x_mask))
            loss_y = cos(y_emb, y_rec)
            rec_loss = loss_x + loss_y
            return {'y': y, 'rec_loss': rec_loss}

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return Component({'x', 'y'}, {'x_rec', 'y_rec', 'x_mask'}, merge_params(components), _fn)


def get_reconstruct_loss(eps: float) -> Callable[[npt.NDArray, npt.NDArray,
                                                  npt.NDArray, npt.NDArray, npt.NDArray], npt.NDArray]:
    # x, x_rec: [dim_model, T]| y, y_rec: dim_model| x_mask: T
    def _fn(x: npt.NDArray,
            x_rec: npt.NDArray,
            y: npt.NDArray,
            y_rec: npt.NDArray,
            x_mask: npt.NDArray) -> npt.NDArray:
        return

    return _fn
