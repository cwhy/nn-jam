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


class TransformerEncoderConfigs(Protocol):
    @property
    @abstractmethod
    def universal(self) -> bool: ...

    @property
    @abstractmethod
    def n_tfe_layers(self) -> int: ...

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


class TransformerEncoder(NamedTuple):

    @staticmethod
    def make(configs: TransformerEncoderConfigs) -> Component:
        if configs.universal:
            components = {
                "tfe_layer": TransformerLayer.make(configs),
                'norm': LayerNorm.make(LayerNorm(
                    eps=configs.eps,
                    norm_axis=0))
            }
            get_layer_name = lambda _: "tfe_layer"
        else:
            components = {
                f"tfe_layer_{i}": TransformerLayer.make(configs)
                for i in range(configs.n_tfe_layers)
            }
            components['norm'] = LayerNorm.make(LayerNorm(
                eps=configs.eps,
                norm_axis=0))
            get_layer_name = lambda i: f"tfe_layer_{i}"

        def _fn(weights: ArrayTreeMapping, x: Array, rng) -> Array:
            rng, key = random.split(rng)
            w_norm = p.get_mapping(weights, 'norm')
            x = sequential(components, [get_layer_name(i) for i in range(configs.n_tfe_layers)])(weights, x, key)
            x = vmap(components['norm'].pipeline, (None, configs.pos_t, None), configs.pos_t)(w_norm, x, key)
            return x

        def _fn_mask(weights: ArrayTreeMapping, inputs: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping:
            x, mask = p.get_arr(inputs, Input), p.get_arr(inputs, 'mask')
            rng, key = random.split(rng)
            all_attns = []
            for i in range(configs.n_tfe_layers):
                _l = get_layer_name(i)
                layer_weights = p.get_mapping(weights, _l)
                outputs = components[_l].processes[masked_mha_port](
                    layer_weights, {Input: x, 'mask': mask}, key)
                x = p.get_arr(outputs, Output)
                all_attns.append(p.get_arr(outputs, 'attn'))
            w_norm = p.get_mapping(weights, 'norm')
            x = vmap(components['norm'].pipeline, (None, configs.pos_t, None), configs.pos_t)(w_norm, x, key)
            return {Output: x, 'attn': xp.stack(all_attns)}

        processes = pipeline2processes(_fn)
        processes[masked_mha_port] = _fn_mask
        return Component(merge_component_params(components), processes)


class TransformerConfigs(TransformerEncoderConfigs, Protocol):
    @property
    @abstractmethod
    def n_seq(self) -> int: ...  # T

    @property
    @abstractmethod
    def dim_input(self) -> int: ...  # x

    @property
    @abstractmethod
    def dict_size(self) -> int: ...

    @property
    @abstractmethod
    def pos_init_scale(self) -> float: ...


class TensorTransformer(NamedTuple):
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

    n_seq: int  # T
    dim_input: int  # x
    dict_size: int
    pos_init_scale: float = 0.001

    @staticmethod
    def make(configs: TransformerConfigs) -> Component:
        components = {
            'embedding': Embeddings.make(configs),
            'positional_encoding': TensorPositionalEncoding.make(TensorPositionalEncoding(
                input_shape=(configs.n_seq,),
                input_channels=configs.dim_input,
                output_channels=configs.dim_model,
                dim_encoding=configs.dim_model,
                positional_encode_strategy='dot',
                init_scale=configs.pos_init_scale,
            )),
            'encoder': TransformerEncoder.make(configs)
        }

        # (int)[T] -> [dim_model, T]
        def _fn(weights: ArrayTreeMapping, x: Array, rng: RNGKey) -> Array:
            w_embedding, w_pos_encoding, w_encoder = p.get_mapping(weights, 'embedding'), \
                p.get_mapping(weights, 'positional_encoding'), \
                p.get_mapping(weights, 'encoder')
            x = vmap(components['embedding'].fixed_pipeline,
                     (None, configs.pos_t),
                     configs.pos_t)(w_embedding, x)
            x = components['positional_encoding'].fixed_pipeline(w_pos_encoding, x)
            rng, key = random.split(rng)
            x = components['encoder'].pipeline(w_encoder, x, key)
            return x

        def _fn_mask(weights: ArrayTreeMapping, inputs: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping:
            x, mask = p.get_arr(inputs, Input), p.get_arr(inputs, 'mask')
            w_embedding, w_pos_encoding, w_encoder = p.get_mapping(weights, 'embedding'), \
                p.get_mapping(weights, 'positional_encoding'), \
                p.get_mapping(weights, 'encoder')
            x = vmap(components['embedding'].fixed_pipeline,
                     (None, configs.pos_t),
                     configs.pos_t)(w_embedding, x)
            x = components['positional_encoding'].fixed_pipeline(w_pos_encoding, x)
            rng, key = random.split(rng)
            outputs = components['encoder'].processes[masked_mha_port](
                w_encoder, {Input: x, 'mask': mask}, key)
            return outputs

        processes = pipeline2processes(_fn)
        processes[masked_mha_port] = _fn_mask
        return Component(merge_component_params(components), processes)


class DynamicTransformer(NamedTuple):
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

    n_seq: int  # T
    dim_input: int  # x
    dict_size: int
    pos_init_scale: float = 0.001

    @staticmethod
    def make(configs: TransformerConfigs) -> Component:
        components = {
            'embedding': Embeddings.make(configs),
            'positional_encoding': PositionalEncoding.make(PositionalEncoding(
                max_input_length=configs.n_seq,
                dynamic_length=True,
                dim_encoding=configs.dim_model,
                positional_encode_strategy='sum',
                init_scale=configs.pos_init_scale,
            )),
            'encoder': TransformerEncoder.make(configs)
        }

        # (int)[T] -> [dim_model, T]
        def _fn(weights: ArrayTreeMapping, x: Array, rng: RNGKey) -> Array:
            w_embedding, w_pos_encoding, w_encoder = p.get_mapping(weights, 'embedding'), \
                p.get_mapping(weights, 'positional_encoding'), \
                p.get_mapping(weights, 'encoder')
            x = vmap(components['embedding'].fixed_pipeline,
                     (None, configs.pos_t),
                     configs.pos_t)(w_embedding, x)
            x = components['positional_encoding'].fixed_pipeline(w_pos_encoding, x)
            rng, key = random.split(rng)
            x = components['encoder'].pipeline(w_encoder, x, key)
            return x

        def _fn_mask(weights: ArrayTreeMapping, inputs: ArrayTreeMapping, rng: RNGKey) -> ArrayTreeMapping:
            x, mask = p.get_arr(inputs, Input), p.get_arr(inputs, 'mask')
            w_embedding, w_pos_encoding, w_encoder = p.get_mapping(weights, 'embedding'), \
                p.get_mapping(weights, 'positional_encoding'), \
                p.get_mapping(weights, 'encoder')
            x = vmap(components['embedding'].fixed_pipeline,
                     (None, configs.pos_t),
                     configs.pos_t)(w_embedding, x)
            x = components['positional_encoding'].fixed_pipeline(w_pos_encoding, x)
            rng, key = random.split(rng)
            outputs = components['encoder'].processes[masked_mha_port](
                w_encoder, {Input: x, 'mask': mask}, key)
            return outputs

        processes = pipeline2processes(_fn)
        processes[masked_mha_port] = _fn_mask
        return Component(merge_component_params(components), processes)
