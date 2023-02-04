from typing import List, NamedTuple, Tuple, Protocol

from jax import Array as NDArray
from jax import numpy as xp, random, vmap, nn
from jax.nn import logsumexp

import jax_make.params as p
from jax_make.component_protocol import Component, merge_params, random_process2process
from jax_make.component_protocol import make_ports, Input, Output
from jax_make.components.embedding import Embeddings
from jax_make.components.multi_head_attn import masked_mha_port
from jax_make.params import ArrayTree, WeightParams
from jax_make.transformer import TransformerEncoderConfigs, TransformerEncoder
from jax_make.utils.activations import Activation

QUERY_SYMBOL_MASK = 0
PAD_MASK = -1
FLOAT_OFFSET = -2
SYMBOLS = slice(1, -2)

QUERY_VALUE = 0
VALUE_SYMBOL = 0

FLOAT_IN = 0
FLOAT_OUT = 1


class AnyNetConfigs(TransformerEncoderConfigs, Protocol):
    init_embed_scale: int
    n_symbols: int
    n_positions: int
    input_keep_rate: float
    max_inputs: int


class AnyNet(NamedTuple):
    universal: bool
    n_tfe_layers: int
    n_heads: int  # H
    dim_model: int  # k
    pos_t: int
    dropout_keep_rate: float
    eps: float
    mlp_n_hidden: List[int]
    mlp_activation: Activation

    init_embed_scale: int
    n_symbols: int
    n_positions: int
    input_keep_rate: float
    max_inputs: int

    @staticmethod
    def make(configs: AnyNetConfigs) -> Component:
        components = {
            # MASK: 0 for masked features to reconstruct
            # PAD: -1 for padded empty features for ragged batches
            # FLOAT_OFFSET: -2 for float offset embedding
            'symbol_embedding': Embeddings.make(Embeddings(
                dict_size=configs.n_symbols + 3,
                dim_model=configs.dim_model,
                dict_init_scale=configs.init_embed_scale
            )),
            'positional_embedding': Embeddings.make(Embeddings(
                dict_size=configs.n_positions,
                dim_model=configs.dim_model,
                dict_init_scale=configs.init_embed_scale
            )),
            'encoder': TransformerEncoder.make(configs)
        }

        def _parse(weights: p.ArrayTreeMapping,
                   pos: NDArray,
                   index: NDArray,
                   value: NDArray) -> NDArray:
            positional_embedding = p.get_mapping(weights, 'positional_embedding')
            symbol_embedding = p.get_mapping(weights, 'symbol_embedding')
            float_embedding = p.get_arr(weights, 'float_embedding')
            pos_embed = components['positional_embedding'].fixed_pipeline(
                positional_embedding, pos)
            index_embed = components['symbol_embedding'].fixed_pipeline(
                symbol_embedding, index)
            val_embed = float_embedding[FLOAT_IN, :] * value
            return pos_embed + index_embed + val_embed

        # out_embed[dim_model], int[], float[] -> float[]
        def _calc_loss(weights: p.ArrayTreeMapping, out_embed: NDArray, index: NDArray, val: NDArray,
                       mask: NDArray) -> NDArray:

            symbol_embedding = p.get_arr(weights, 'symbol_embedding')
            float_embedding = p.get_arr(weights, 'float_embedding')

            symbol_dict = symbol_embedding['dict'][SYMBOLS, :]
            logits = symbol_dict @ out_embed
            out_val = (symbol_dict[FLOAT_OFFSET, :]
                       + float_embedding[FLOAT_OUT, :]) @ out_embed
            int_loss = -xp.mean(
                (logits - logsumexp(logits, keepdims=True)) * nn.one_hot(index, configs.n_symbols)) * mask
            float_loss = mask * (out_val - val) ** 2
            return int_loss + float_loss

        def _calc_output(weights: p.ArrayTreeMapping, out_embed: NDArray) -> Tuple[NDArray, NDArray]:
            symbol_embedding = p.get_arr(weights, 'symbol_embedding')
            float_embedding = p.get_arr(weights, 'float_embedding')

            logits = symbol_embedding['dict'] @ out_embed
            out_val = (symbol_embedding['dict'][FLOAT_OFFSET, :]
                       + float_embedding[FLOAT_OUT, :]) @ out_embed
            return xp.argmax(logits), out_val

        # input_pos: (int)[T], input: (int)[T, value: (float)[T]] -> float
        def _loss(weights: p.ArrayTreeMapping, inputs: p.ArrayTreeMapping, rng: p.RNGKey) -> p.ArrayTreeMapping:
            input_pos = p.get_arr(inputs, 'input_pos')
            x = p.get_arr(inputs, 'Input')
            input_value = p.get_arr(inputs, 'value')

            if 0 < configs.input_keep_rate < 1:
                rng, key = random.split(rng)
                pred_mask = random.bernoulli(key, configs.input_keep_rate, (configs.max_inputs,))
            else:
                pred_mask = xp.ones((configs.max_inputs,))
            indices = pred_mask * inputs[Input]
            values = pred_mask * inputs['value']
            embeds = vmap(_parse, (None, 0, 0, 0), configs.pos_t)(weights, input_pos, indices, values)

            rng, key = random.split(rng)
            encoder_inputs = {Input: embeds, 'mask': inputs['mask']}
            encoder_weights = p.get_mapping(weights, 'encoder')
            out_embeds = p.get_arr(components['encoder'].processes[masked_mha_port](
                encoder_weights, encoder_inputs, key), Output)

            all_losses = vmap(_calc_loss, (None, configs.pos_t, 0, 0, 0), configs.pos_t)(
                weights, out_embeds, x, input_value, pred_mask)
            return {'loss': all_losses.mean()}

        # input_pos: (int)[T], input: (int)[T, value: (float)[T]] -> [int, T], [float, T]
        def _query(weights: p.ArrayTreeMapping, inputs: p.ArrayTreeMapping, rng: p.RNGKey) -> p.ArrayTreeMapping:
            input_pos = p.get_arr(inputs, 'input_pos')
            intput_ = p.get_arr(inputs, 'Input')
            input_value = p.get_arr(inputs, 'value')
            embeds = vmap(_parse, (None, 0, 0, 0), configs.pos_t)(weights, input_pos, intput_, input_value)
            rng, key = random.split(rng)
            encoder_weights = p.get_mapping(weights, 'encoder')
            out_embed = components['encoder'].pipeline(encoder_weights, embeds, key)
            class_i, value = vmap(_calc_output, (None, configs.pos_t), configs.pos_t)(weights, out_embed)
            return {"symbol": class_i, 'value': value}

        # FLOAT_IN: 0 for in , FLOAT_OUT: 1 for out
        params: p.ArrayParamMapping = {
            'float_embedding': WeightParams(
                shape=(2, configs.dim_model),
                init="embedding",
                scale=configs.init_embed_scale),
            **merge_params(components),
        }
        processes = {
            loss_ports: random_process2process(_loss),
            inference_ports: random_process2process(_query)
        }
        return Component(params, processes)


loss_ports = make_ports((Input, 'mask', 'input_pos', 'value'), 'loss')
inference_ports = make_ports((Input, 'input_pos', 'value'), ('symbol', 'value'))
