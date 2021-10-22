from typing import NamedTuple, List, Tuple, Protocol

import numpy.typing as npt
from jax import random, vmap
from jax import numpy as xp

from tests.jax_activations import Activation
from tests.jax_components import Component, sequential
from tests.jax_modules.dirty_patches import DirtyPatches
from tests.jax_modules.dropout import Dropout
from tests.jax_modules.embedding import Embeddings
from tests.jax_modules.mlp import Mlp
from tests.jax_modules.multi_head_attn import SelfMultiHeadAttn
from tests.jax_modules.norms import LayerNorm
from tests.jax_modules.positional_encoding import PositionalEncoding
from tests.jax_random_utils import ArrayTree, RNGKey


class TransformerEncoderConfigs(Protocol):
    n_tfe_layers: int
    n_heads: int  # H
    dim_model: int  # k
    pos_t: int
    dropout_keep_rate: float
    eps: float
    mlp_n_hidden: List[int]
    mlp_activation: Activation


class TransformerLayer(NamedTuple):
    @staticmethod
    def make(configs: TransformerEncoderConfigs) -> Component:
        components = {
            'mha': SelfMultiHeadAttn.make(SelfMultiHeadAttn(
                n_heads=configs.n_heads,
                dim_model=configs.dim_model,
                dim_input=configs.dim_model
            )),
            'norm': LayerNorm.make(LayerNorm(
                eps=configs.eps,
                norm_axis=0)),
            'dropout': Dropout.make(configs),
            'mlp': Mlp.make(Mlp(n_in=configs.dim_model,
                                n_hidden=configs.mlp_n_hidden,
                                n_out=configs.dim_model,
                                activation=configs.mlp_activation,
                                dropout_keep_rate=configs.dropout_keep_rate
                                ))
        }

        def _fn(weights: ArrayTree, x: npt.NDArray, rng: RNGKey) -> npt.NDArray:
            rng, key1, key2 = random.split(rng, 3)
            x = sequential(components, ['norm', 'mha', 'dropout'])(weights, x, key1) + x
            x = vmap(sequential(components, ['norm', 'mlp', 'dropout']),
                     (None, configs.pos_t, None), configs.pos_t)(weights, x, key2) + x
            return x

        return Component({k: v.params for k, v in components.items()}, _fn)


class TransformerEncoder(NamedTuple):

    @staticmethod
    def make(configs: TransformerEncoderConfigs) -> Component:
        components = {
            f"tfe_layer_{i}": TransformerLayer.make(configs)
            for i in range(configs.n_tfe_layers)
        }
        components['norm'] = LayerNorm.make(LayerNorm(
            eps=configs.eps,
            norm_axis=0))

        def _fn(weights: ArrayTree, x: npt.NDArray, rng) -> npt.NDArray:
            rng, key = random.split(rng)
            x = sequential(components, [f"tfe_layer_{i}" for i in range(configs.n_tfe_layers)])(weights, x, key)
            x = vmap(components['norm'].process, (None, configs.pos_t, None), configs.pos_t)(weights['norm'], x, key)
            return x

        return Component({k: v.params for k, v in components.items()}, _fn)


class TransformerConfigs(TransformerEncoderConfigs):
    n_seq: int  # T
    dim_input: int  # x
    dict_size: int


class Transformer(NamedTuple):
    n_tfe_layers: int
    n_seq: int  # T
    n_heads: int  # H
    dim_model: int  # k
    pos_t: int
    dropout_keep_rate: float
    eps: float
    mlp_n_hidden: List[int]
    mlp_activation: Activation
    dim_input: int  # x
    dict_size: int

    @staticmethod
    def make(configs: TransformerConfigs) -> Component:
        components = {
            'embedding': Embeddings.make(configs),
            'positional_encoding': PositionalEncoding.make(PositionalEncoding(
                input_shape=(configs.n_seq,),
                input_channels=configs.dim_input,
                output_channels=configs.dim_model,
                dim_encoding=configs.dim_model,
                positional_encode_strategy='dot'
            )),
            'encoder': TransformerEncoder.make(configs)
        }

        # (int)[T] -> [dim_model, T]
        def _fn(weights: ArrayTree, x: npt.NDArray, rng) -> npt.NDArray:
            x = vmap(components['embedding'].fixed_process, (None, configs.pos_t), configs.pos_t)(weights['embedding'],
                                                                                                  x)
            print(x.shape)
            x = components['positional_encoding'].fixed_process(weights['positional_encoding'], x)
            rng, key = random.split(rng)
            x = components['encoder'].process(weights['encoder'], x, key)
            return x

        return Component({k: v.params for k, v in components.items()}, _fn)


class VitConfigs(TransformerEncoderConfigs):
    hwc: Tuple[int, int, int]  # x
    n_patches_side: int
    mlp_n_hidden_patches: List[int]

    dim_output: int
    dict_size_output: int


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

    @staticmethod
    def make(configs: VitConfigs) -> Component:
        components = {
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
            'encoder': TransformerEncoder.make(configs),
        }

        # [H, W, C] -> [dim_model, T]
        def _fn(weights: ArrayTree, x: npt.NDArray, rng) -> npt.NDArray:
            rng, key = random.split(rng)
            x = components['patching'].process(weights['patching'], x, key)
            # [dim_model, h, w, C]
            x = components['positional_encoding'].fixed_process(weights['positional_encoding'], x)
            # [dim_model, (h, w, C)]

            out_embed = weights['out_embedding']['dict']
            # [dim_model, dim_out + (h, w, C)]
            x = xp.c_[xp.zeros((configs.dim_model, configs.dim_output)), x]
            rng, key = random.split(rng)
            x = components['encoder'].process(weights['encoder'], x, key)
            # [dim_model, dim_out + (h, w, C)]
            return x

        return Component({k: v.params for k, v in components.items()}, _fn)
