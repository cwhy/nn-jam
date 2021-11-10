from typing import List, NamedTuple, Tuple

import numpy.typing as npt
from jax import numpy as xp, random, vmap, nn
from jax._src.nn.functions import logsumexp
from numpy.typing import NDArray

from jax_make.component_protocol import Component, merge_params
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


class AnyNetConfigs(TransformerEncoderConfigs):
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
                init_scale=configs.init_embed_scale
            )),
            'positional_embedding': Embeddings.make(Embeddings(
                dict_size=configs.n_positions,
                dim_model=configs.dim_model,
                init_scale=configs.init_embed_scale
            )),
            'encoder': TransformerEncoder.make(configs)
        }
        # FLOAT_IN: 0 for in , FLOAT_OUT: 1 for out
        params = {
            'float_embedding': WeightParams(
                shape=(2, configs.dim_model),
                init="embedding",
                scale=configs.init_embed_scale),
        }

        def _parse(weights: ArrayTree,
                   pos: NDArray,
                   index: NDArray,
                   value: NDArray) -> NDArray:
            pos_embed = components['positional_embedding'].fixed_pipeline(
                weights['positional_embedding'], pos)
            index_embed = components['symbol_embedding'].fixed_pipeline(
                weights['symbol_embedding'], index)
            val_embed = weights['float_embedding'][FLOAT_IN, :] * value
            return pos_embed + index_embed + val_embed

        # out_embed[dim_model], int[], float[] -> float[]
        def _calc_loss(weights: ArrayTree, out_embed: NDArray, index: NDArray, val: NDArray,
                       mask: NDArray) -> NDArray:
            symbol_dict = weights['symbol_embedding']['dict'][SYMBOLS, :]
            logits = symbol_dict @ out_embed
            out_val = (symbol_dict[FLOAT_OFFSET, :]
                       + weights['float_embedding'][FLOAT_OUT, :]) @ out_embed
            int_loss = -xp.mean(
                (logits - logsumexp(logits, keepdims=True)) * nn.one_hot(index, configs.n_symbols)) * mask
            float_loss = mask * (out_val - val) ** 2
            return int_loss + float_loss

        def _calc_output(weights: ArrayTree, out_embed: NDArray) -> Tuple[NDArray, NDArray]:
            logits = weights['symbol_embedding']['dict'] @ out_embed
            out_val = (weights['symbol_embedding']['dict'][FLOAT_OFFSET, :]
                       + weights['float_embedding'][FLOAT_OUT, :]) @ out_embed
            return xp.argmax(logits), out_val

        # input_pos: (int)[T], input: (int)[T, value: (float)[T]] -> float
        def _loss(weights: ArrayTree, inputs: ArrayTree, rng) -> ArrayTree:
            if 0 < configs.input_keep_rate < 1:
                rng, key = random.split(rng)
                pred_mask = random.bernoulli(key, configs.input_keep_rate, (configs.max_inputs,))
            else:
                pred_mask = xp.ones((configs.max_inputs,))
            indices = pred_mask * inputs[Input]
            values = pred_mask * inputs['value']
            embeds = vmap(_parse, (None, 0, 0, 0), configs.pos_t)(weights, inputs['input_pos'], indices, values)

            rng, key = random.split(rng)
            encoder_inputs = {Input: embeds, 'mask': inputs['mask']}
            out_embeds = components['encoder'].processes[masked_mha_port](
                weights['encoder'], encoder_inputs, key)[Output]

            all_losses = vmap(_calc_loss, (None, configs.pos_t, 0, 0, 0), configs.pos_t)(
                weights, out_embeds, inputs[Input], inputs['value'], pred_mask)
            return {'loss': all_losses.mean()}

        # input_pos: (int)[T], input: (int)[T, value: (float)[T]] -> [int, T], [float, T]
        def _query(weights: ArrayTree, inputs: ArrayTree, rng) -> ArrayTree:
            embeds = vmap(_parse, (None, 0, 0, 0), configs.pos_t)(weights, inputs['input_pos'], inputs[Input], inputs['value'])
            rng, key = random.split(rng)
            out_embed = components['encoder'].pipeline(weights['encoder'], embeds, key)
            class_i, value = vmap(_calc_output, (None, configs.pos_t), configs.pos_t)(weights, out_embed)
            return {"symbol": class_i, 'value': value}

        # noinspection PyTypeChecker
        # Because Pycharm sucks
        params.update(merge_params(components))
        processes = {
            loss_ports: _loss,
            inference_ports: _query
        }
        # noinspection PyTypeChecker
        # Because Pycharm sucks
        return Component(params, processes)


loss_ports = make_ports((Input, 'mask', 'input_pos', 'value'), 'loss')
inference_ports = make_ports((Input, 'input_pos', 'value'), ('symbol', 'value'))
