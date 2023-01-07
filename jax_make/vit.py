from abc import abstractmethod
from typing import Tuple, List, NamedTuple, Protocol

import einops
import numpy as np
from jax import numpy as xp, random, vmap, nn
from jax.nn import logsumexp
from numpy import typing as npt

import jax_make.params as p
from jax_make.component_protocol import Component, merge_params, pipeline2processes, make_ports, Input, \
    Output
from jax_make.components.dirty_patches import DirtyPatches
from jax_make.components.embedding import Embeddings
from jax_make.components.mlp import Mlp
from jax_make.components.norms import LayerNorm
from jax_make.components.positional_encoding import PositionalEncoding
from jax_make.params import ArrayTreeMapping
from jax_make.transformer import TransformerEncoderConfigs, TransformerEncoder
from jax_make.utils.activations import Activation
from jax_make.utils.functions import sigmoid_cross_entropy_loss


class VitConfigs(TransformerEncoderConfigs, Protocol):
    @property
    @abstractmethod
    def hwc(self) -> Tuple[int, int, int]: ...

    @property
    @abstractmethod
    def n_patches_side(self) -> int: ...

    @property
    @abstractmethod
    def mlp_n_hidden_patches(self) -> List[int]: ...

    @property
    @abstractmethod
    def dim_output(self) -> int: ...

    @property
    @abstractmethod
    def dict_size_output(self) -> int: ...

    @property
    @abstractmethod
    def input_keep_rate(self) -> float: ...

    @property
    @abstractmethod
    def pos_init_scale(self) -> float: ...


class VitComponentWeights(NamedTuple):
    patching: ArrayTreeMapping
    mask_embedding: ArrayTreeMapping
    positional_encoding: ArrayTreeMapping
    positional_encoding_y: ArrayTreeMapping
    out_embedding: ArrayTreeMapping
    encoder: ArrayTreeMapping
    norm: ArrayTreeMapping

    @classmethod
    def from_mapping(cls, mapping: ArrayTreeMapping):
        return cls(*(p.get_mapping(mapping, s) for s in cls._fields))


class Vit(NamedTuple):
    # For encoder
    universal: bool
    n_tfe_layers: int
    n_heads: int  # H
    dim_model: int  # k
    pos_t: int
    dropout_keep_rate: float
    eps: float
    mlp_n_hidden: List[int]
    mlp_activation: Activation
    dict_init_scale: float

    # Vit specific
    hwc: Tuple[int, int, int]  # x
    n_patches_side: int
    mlp_n_hidden_patches: List[int]

    dim_output: int
    dict_size_output: int

    input_keep_rate: float
    pos_init_scale: float

    @staticmethod
    def make(configs: VitConfigs) -> Component:
        components = {
            'mask_embedding': Embeddings.make(Embeddings(dict_size=1,
                                                         dim_model=configs.dim_model,
                                                         dict_init_scale=configs.dict_init_scale)),
            'out_embedding': Embeddings.make(Embeddings(dict_size=configs.dict_size_output,
                                                        dim_model=configs.dim_model,
                                                        dict_init_scale=configs.dict_init_scale)),
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
                positional_encode_strategy='dot',
                init_scale=0.001,
            )),
            'positional_encoding_y': PositionalEncoding.make(PositionalEncoding(
                input_shape=(1,),
                input_channels=configs.dim_model,
                output_channels=configs.dim_model,
                dim_encoding=configs.dim_model,
                positional_encode_strategy='dot',
                init_scale=0.001,
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
        def _fn(weights: ArrayTreeMapping, x: npt.NDArray, rng) -> npt.NDArray:
            w = VitComponentWeights.from_mapping(weights)
            rng, key = random.split(rng)
            x = components['patching'].pipeline(w.patching, x, key)
            rng, key = random.split(rng)
            # [dim_model, h, w, C]

            x_flattened = x.reshape((-1, T))
            missing_embed = w.mask_embedding['dict'].T
            if 0 < configs.input_keep_rate < 1:
                keep_idx = random.bernoulli(key, configs.input_keep_rate, (T,))
                assert isinstance(keep_idx, np.ndarray)
                masked_x = vmap(mask_with_default, (0, None, 0), 0)(x_flattened, keep_idx, missing_embed)
            elif configs.input_keep_rate == 1:
                keep_idx = xp.ones((T,))
                masked_x = x
            else:
                raise Exception("Invalid input keep rate")

            masked_x = components['positional_encoding'].fixed_pipeline(
                w.positional_encoding,
                masked_x.reshape((-1,
                                  configs.n_patches_side,
                                  configs.n_patches_side,
                                  configs.hwc[-1])))
            # [dim_model, (h, w, C)]
            y = components['positional_encoding_y'].fixed_pipeline(w.positional_encoding_y, missing_embed)
            yx = xp.c_[y, masked_x]
            yx = components['norm'].pipeline(w.norm, yx, key)
            # [dim_model, dim_out + (h, w, C)]

            rng, key = random.split(rng)
            yx = components['encoder'].pipeline(w.encoder, yx, key)
            # [dim_model, dim_out + (h, w, C)]
            return yx

        return Component.from_pipeline(merge_params(components), _fn)


class VitReconstructComponentWeights(NamedTuple):
    patching: ArrayTreeMapping
    mask_embedding: ArrayTreeMapping
    positional_encoding: ArrayTreeMapping
    positional_encoding_y: ArrayTreeMapping
    out_embedding: ArrayTreeMapping
    encoder: ArrayTreeMapping
    norm: ArrayTreeMapping
    x_reconstruct: ArrayTreeMapping
    y_reconstruct: ArrayTreeMapping

    @classmethod
    def from_mapping(cls, mapping: ArrayTreeMapping):
        return cls(*(p.get_mapping(mapping, s) for s in cls._fields))


class VitReconstruct(NamedTuple):
    universal: bool
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
    dict_init_scale: float

    @staticmethod
    def make(configs: VitConfigs) -> Component:
        if configs.hwc[-1] == 0:
            ch = 1
        else:
            ch = configs.hwc[-1]

        h_patch = configs.hwc[0] // configs.n_patches_side
        w_patch = configs.hwc[1] // configs.n_patches_side
        patch_size = h_patch * w_patch * ch
        components = {
            'mask_embedding': Embeddings.make(Embeddings(dict_size=1,
                                                         dim_model=configs.dim_model,
                                                         dict_init_scale=configs.dict_init_scale)),
            'out_embedding': Embeddings.make(Embeddings(dict_size=configs.dict_size_output,
                                                        dim_model=configs.dim_model,
                                                        dict_init_scale=configs.dict_init_scale)),
            'patching': DirtyPatches.make(DirtyPatches(
                dim_out=configs.dim_model,
                n_sections_w=configs.n_patches_side,
                n_sections_h=configs.n_patches_side,
                h=configs.hwc[0],
                w=configs.hwc[1],
                ch=ch,
                mlp_n_hidden=[],
                mlp_activation=configs.mlp_activation,
                dropout_keep_rate=configs.dropout_keep_rate
            )),
            'x_reconstruct': Mlp.make(
                Mlp(
                    n_in=configs.dim_model,
                    n_hidden=[],
                    n_out=patch_size,
                    # activation='tanh',
                    activation=configs.mlp_activation,
                    dropout_keep_rate=configs.dropout_keep_rate,
                )
            ),
            'y_reconstruct': Mlp.make(
                Mlp(n_in=configs.dim_model,
                    n_hidden=[],
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
                positional_encode_strategy='naive_sum',
                init_scale=configs.pos_init_scale,
            )),
            'positional_encoding_y': PositionalEncoding.make(PositionalEncoding(
                input_shape=(1,),
                input_channels=configs.dim_model,
                output_channels=configs.dim_model,
                dim_encoding=configs.dim_model,
                positional_encode_strategy='naive_sum',
                init_scale=configs.pos_init_scale
            )),
            'encoder': TransformerEncoder.make(configs),
            'norm': LayerNorm.make(LayerNorm(
                eps=configs.eps,
                norm_axis=0)),
            'x_rec_norm': LayerNorm.make(LayerNorm(
                eps=configs.eps,
                norm_axis=0)),
        }
        T = configs.n_patches_side ** 2 * ch

        # [D, D, 1] -> [D]
        def mask_x_with_default(x: npt.NDArray, m: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
            ds = xp.repeat(d, T)
            ds *= 1 - m
            x *= m
            return ds + x

        # [D, D, 1] -> [D]
        def mask_y_with_default(x: npt.NDArray, m: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
            d *= 1 - m
            x *= m
            return d + x

        def _pre_process_x(w_patching: ArrayTreeMapping, x: npt.NDArray, key) -> npt.NDArray:
            if configs.hwc[-1] == 0:
                x = xp.expand_dims(x, -1)
            x = components['patching'].pipeline(w_patching, x, key)
            # [dim_model, h, w, C]

            return x

        # x: [H, W, C] -> x_rec: [dim_model, T], y_rec: dim_model, x_mask: T
        def _fn(weights: ArrayTreeMapping, inputs: ArrayTreeMapping, rng) -> ArrayTreeMapping:
            w = VitReconstructComponentWeights.from_mapping(weights)
            x, y = p.get_arr(inputs, Input), p.get_arr(inputs, Output)
            rng, key = random.split(rng)
            if configs.hwc[-1] == 0:
                x = xp.expand_dims(x, -1)
            patches_out: ArrayTreeMapping = components['patching'].processes[
                make_ports(Input, ("patches", Output))
            ](w.patching, {Input: x}, key)
            # [dim_model, h, w, C]
            x_flattened = p.get_arr(patches_out, Output).reshape((configs.dim_model, T))
            y_emb = components['out_embedding'].fixed_pipeline(w.out_embedding, y)

            missing_embed = w.mask_embedding['dict'].T
            if 0 < configs.input_keep_rate < 1:
                rng, key_x, key_y = random.split(rng, 3)
                x_mask = random.bernoulli(key_x, configs.input_keep_rate, (T,))
                assert isinstance(x_mask, np.ndarray)
                masked_x = vmap(mask_x_with_default, (0, None, 0), 0)(x_flattened, x_mask, missing_embed)
                y_mask = random.bernoulli(key_y, configs.input_keep_rate, (1,))
                assert isinstance(y_mask, np.ndarray)
                masked_y = vmap(mask_y_with_default, (0, None, 0), 0)(y_emb, y_mask, missing_embed)
            elif configs.input_keep_rate == 1:
                x_mask = xp.ones((T,))
                y_mask = xp.zeros(1)
                masked_x = x_flattened
                masked_y = y_emb
            else:
                raise Exception("Invalid input keep rate")
            # sd around 0.3
            masked_x = components['positional_encoding'].fixed_pipeline(
                w.positional_encoding,
                masked_x.reshape((configs.dim_model,
                                  configs.n_patches_side,
                                  configs.n_patches_side,
                                  ch)))
            # [dim_model, (h, w, C)]

            masked_y = components['positional_encoding_y'].fixed_pipeline(
                w.positional_encoding_y, masked_y)
            yx = xp.c_[masked_y, masked_x]
            yx = components['norm'].pipeline(w.norm, yx, key)
            # [dim_model, dim_out + (h, w, C)]

            rng, key = random.split(rng)
            yx = components['encoder'].pipeline(w.encoder, yx, key)
            # [dim_model, dim_out + (h, w, C)]

            y_rec, x = yx[:, 0], yx[:, 1:]
            rng, key1, key2 = random.split(rng, 3)
            x_rec = vmap(components['x_reconstruct'].pipeline, (None, 1, None), 1)(w.x_reconstruct, x, key1)
            # x_rec = components['x_rec_norm'].fixed_pipeline(weights['x_rec_norm'], x_rec)
            # [patch_size, T]

            y_rec = components['y_reconstruct'].pipeline(w.y_reconstruct, y_rec, key2)
            logits = weights['out_embedding']['dict'] @ y_rec
            loss_y = -xp.mean((logits - logsumexp(logits, keepdims=True)) * nn.one_hot(y, configs.dict_size_output))
            loss_y *= 1 - y_mask  # Use this for precise y converge

            # loss_y = loss_fn(y_emb, y_rec)

            x_patches = p.get_arr(patches_out, 'patches').T
            # loss_fn = get_cosine_similarity_loss(configs.eps)
            loss_fn = vmap(sigmoid_cross_entropy_loss, (0, 0), 0)
            loss_x_all = vmap(loss_fn, (-1, -1), -1)(x_rec, x_patches)
            print(loss_x_all.shape, "\n*************************")
            # [patch_size, T]
            # loss_x = xp.mean(loss_x_all)
            loss_x_m = xp.sum(xp.mean(loss_x_all, axis=0) * (1 - x_mask)) / xp.maximum((1 - x_mask).sum(), 1)
            loss_x_nm = xp.sum(xp.mean(loss_x_all, axis=0) * x_mask) / xp.maximum(x_mask.sum(), 1)
            loss_x = loss_x_m + loss_x_nm
            # loss_x = loss_x_m
            rec_loss = loss_x + loss_y
            x_rec_img = einops.rearrange(x_rec, '(dh dw) (h w) -> (h dh) (w dw)', dh=h_patch, h=configs.n_patches_side)
            return {'rec_loss': rec_loss, 'x_rec_img': x_rec_img}

        # [H, W, C] -> [dim_model]
        def _x2y(weights: ArrayTreeMapping, x: npt.NDArray, rng) -> npt.NDArray:
            w = VitReconstructComponentWeights.from_mapping(weights)
            rng, key = random.split(rng)
            x = _pre_process_x(w.patching, x, key)
            x = components['positional_encoding'].fixed_pipeline(
                w.positional_encoding, x)

            missing_embed = w.mask_embedding['dict'].T
            # [dim_model, (h, w, C)]
            y_masked_emb = components['positional_encoding_y'].fixed_pipeline(w.positional_encoding_y,
                                                                              missing_embed)
            yx = xp.c_[y_masked_emb, x]
            yx = components['norm'].fixed_pipeline(w.norm, yx)
            # [dim_model, dim_out + (h, w, C)]

            yx = components['encoder'].pipeline(w.encoder, yx, rng)
            # [dim_model, dim_out + (h, w, C)]

            y_rec = yx[:, 0]
            y_rec = components['y_reconstruct'].pipeline(w.y_reconstruct, y_rec, rng)
            return y_rec

        processes = pipeline2processes(_x2y)
        processes[make_ports((Input, Output), ('rec_loss', 'x_rec_img'))] = _fn
        return Component(merge_params(components), processes)
