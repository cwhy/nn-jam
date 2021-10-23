from typing import NamedTuple, List, Protocol

import numpy.typing as npt
from jax import random, vmap

from tests.jax_activations import Activation
from tests.jax_components import Component, sequential, merge_params
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
            'norm1': LayerNorm.make(LayerNorm(
                eps=configs.eps,
                norm_axis=0)),
            'norm2': LayerNorm.make(LayerNorm(
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
            # noinspection PyTypeChecker
            # Because pycharm sucks
            x = sequential(components, ['norm1', 'mha', 'dropout'])(weights, x, key1) + x
            # noinspection PyTypeChecker
            # Because pycharm sucks
            x = vmap(sequential(components, ['norm2', 'mlp', 'dropout']),
                     (None, configs.pos_t, None), configs.pos_t)(weights, x, key2) + x
            return x

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return Component.from_pipeline(merge_params(components), _fn)


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
            # noinspection PyTypeChecker
            # Because pycharm sucks
            x = sequential(components, [f"tfe_layer_{i}" for i in range(configs.n_tfe_layers)])(weights, x, key)
            x = vmap(components['norm'].process, (None, configs.pos_t, None), configs.pos_t)(weights['norm'], x, key)
            return x

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return Component.from_pipeline(merge_params(components), _fn)


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
            x = vmap(components['embedding'].fixed_pipeline,
                     (None, configs.pos_t),
                     configs.pos_t)(weights['embedding'], x)
            print(x.shape)
            x = components['positional_encoding'].fixed_pipeline(weights['positional_encoding'], x)
            rng, key = random.split(rng)
            x = components['encoder'].pipeline(weights['encoder'], x, key)
            return x

        # noinspection PyTypeChecker
        # Because Pycharm sucks
        return Component.from_pipeline(merge_params(components.items), _fn)


