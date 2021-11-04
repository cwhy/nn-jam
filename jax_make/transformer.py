from typing import NamedTuple, List, Protocol

import numpy.typing as npt
from jax import random, vmap
import jax.numpy as xp

from jax_make.utils.activations import Activation
from jax_make.component_protocol import Component, sequential, merge_params, pipeline2processes, make_ports, Input, \
    Output
from jax_make.components.dropout import Dropout
from jax_make.components.embedding import Embeddings
from jax_make.components.mlp import Mlp
from jax_make.components.multi_head_attn import SelfMultiHeadAttn, masked_mha_port
from jax_make.components.norms import LayerNorm
from jax_make.components.positional_encoding import PositionalEncoding
from jax_make.params import ArrayTree, RNGKey


class TransformerEncoderConfigs(Protocol):
    universal: bool
    n_tfe_layers: int
    n_heads: int  # H
    dim_model: int  # k
    pos_t: int
    dropout_keep_rate: float
    eps: float
    mlp_n_hidden: List[int]
    mlp_activation: Activation
    init_scale: float


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

        def _fn_mask(weights: ArrayTree, inputs: ArrayTree, rng: RNGKey) -> ArrayTree:
            rng, key1, key2, key3 = random.split(rng, 4)
            x = inputs[Input]
            # noinspection PyTypeChecker
            # Because pycharm sucks
            inputs[Input] = sequential(components, ['norm1'])(weights, inputs[Input], key1)
            attn_out = components['mha'].processes[masked_mha_port](weights['mha'], inputs, key2)
            attned_x = attn_out[Output]
            x += sequential(components, ['dropout'])(weights['dropout'], attned_x, key1)

            # noinspection PyTypeChecker
            # Because pycharm sucks
            x = vmap(sequential(components, ['norm2', 'mlp', 'dropout']),
                     (None, configs.pos_t, None), configs.pos_t)(weights, x, key2) + x
            return {Output: x, 'attn': attn_out['attn']}

        # noinspection PyTypeChecker
        # Because pycharm sucks
        processes = pipeline2processes(_fn)
        processes[masked_mha_port] = _fn_mask
        return Component(merge_params(components), processes)


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

        def _fn(weights: ArrayTree, x: npt.NDArray, rng) -> npt.NDArray:
            rng, key = random.split(rng)
            # noinspection PyTypeChecker
            # Because pycharm sucks
            x = sequential(components, [get_layer_name(i) for i in range(configs.n_tfe_layers)])(weights, x, key)
            x = vmap(components['norm'].pipeline, (None, configs.pos_t, None), configs.pos_t)(weights['norm'], x, key)
            return x

        def _fn_mask(weights: ArrayTree, inputs: ArrayTree, rng: RNGKey) -> ArrayTree:
            x = inputs[Input]
            mask = inputs['mask']
            rng, key = random.split(rng)
            all_attns = []
            for i in range(configs.n_tfe_layers):
                _l = get_layer_name(i)
                outputs = components[_l].processes[masked_mha_port](
                    weights[_l], {Input: x, 'mask': mask}, key)
                x = outputs[Output]
                all_attns.append(outputs['attn'])
            # noinspection PyTypeChecker
            # Because pycharm sucks
            x = vmap(components['norm'].pipeline, (None, configs.pos_t, None), configs.pos_t)(weights['norm'], x, key)
            return {Output: x, 'attn': xp.stack(all_attns)}

        # noinspection PyTypeChecker
        # Because pycharm sucks
        processes = pipeline2processes(_fn)
        processes[masked_mha_port] = _fn_mask
        return Component(merge_params(components), processes)


class TransformerConfigs(TransformerEncoderConfigs):
    n_seq: int  # T
    dim_input: int  # x
    dict_size: int
    init_scale: float


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
    init_scale: float

    @staticmethod
    def make(configs: TransformerConfigs) -> Component:
        components = {
            'embedding': Embeddings.make(configs),
            'positional_encoding': PositionalEncoding.make(PositionalEncoding(
                input_shape=(configs.n_seq,),
                input_channels=configs.dim_input,
                output_channels=configs.dim_model,
                dim_encoding=configs.dim_model,
                positional_encode_strategy='dot',
                init_scale=0.001,
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
        return Component.from_pipeline(merge_params(components), _fn)
