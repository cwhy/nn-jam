from abc import abstractmethod
from typing import NamedTuple, List, Protocol

import jax.numpy as xp
from jax import random, vmap, Array

import jax_make.params as p
from jax_make.component_protocol import Component, sequential, merge_component_params, pipeline2processes, Input, \
    Output
from jax_make.components.dropout import Dropout
from jax_make.components.embedding import Embeddings
from jax_make.components.mlp import Mlp
from jax_make.components.multi_head_attn import SelfMultiHeadAttn, masked_mha_port
from jax_make.components.norms import LayerNorm
from jax_make.components.positional_encoding import PositionalEncoding
from jax_make.components.tensor_positional_encoding import TensorPositionalEncoding
from jax_make.params import RNGKey, ArrayTreeMapping
from jax_make.utils.activations import Activation


class RwkvEncoderConfigs(Protocol):
    @property
    @abstractmethod
    def universal(self) -> bool: ...

    @property
    @abstractmethod
    def n_encoder_layers(self) -> int: ...

    @property
    @abstractmethod
    def n_heads(self) -> int: ...

    @property
    @abstractmethod
    def dim_model(self) -> int: ...

    @property
    @abstractmethod
    def pos_t(self) -> int: ...

    @property
    @abstractmethod
    def dropout_keep_rate(self) -> float: ...

    @property
    @abstractmethod
    def eps(self) -> float: ...

    @property
    @abstractmethod
    def mlp_n_hidden(self) -> List[int]: ...

    @property
    @abstractmethod
    def mlp_activation(self) -> Activation: ...

    @property
    @abstractmethod
    def dict_init_scale(self) -> float: ...


class RwkvLayer(NamedTuple):
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

        def _fn(weights: ArrayTreeMapping, x: Array, rng: RNGKey) -> Array:
            rng, key1, key2 = random.split(rng, 3)
            x = sequential(components, ['norm1', 'mha', 'dropout'])(weights, x, key1) + x
            x = vmap(sequential(components, ['norm2', 'mlp', 'dropout']),
                     (None, configs.pos_t, None), configs.pos_t)(weights, x, key2) + x
            return x

        def _fn_mask(weights: ArrayTreeMapping, inputs: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping:
            rng, key1, key2, key3 = random.split(rng, 4)
            x = p.get_arr(inputs, Input)
            x = sequential(components, ['norm1'])(weights, x, key1)

            w_mha, w_dropout = p.get_mapping(weights, 'mha'), p.get_mapping(weights, 'dropout')

            attn_out = components['mha'].processes[masked_mha_port](w_mha, inputs, None)
            attned_x = p.get_arr(attn_out, Output)
            x += sequential(components, ['dropout'])(w_dropout, attned_x, key2)

            x = vmap(sequential(components, ['norm2', 'mlp', 'dropout']),
                     (None, configs.pos_t, None), configs.pos_t)(weights, x, key3) + x
            return {Output: x, 'attn': attn_out['attn']}

        processes = pipeline2processes(_fn)
        processes[masked_mha_port] = _fn_mask
        return Component(merge_component_params(components), processes)
