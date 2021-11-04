from typing import List, NamedTuple, Tuple

import numpy.typing as npt
from jax import numpy as xp, random, vmap

from jax_make.component_protocol import Component, merge_params
from jax_make.component_protocol import make_ports, Input, Output
from jax_make.components.embedding import Embeddings
from jax_make.params import ArrayTree, WeightParams
from jax_make.transformer import TransformerEncoderConfigs, TransformerEncoder
from jax_make.utils.activations import Activation


class AnyNetConfigs(TransformerEncoderConfigs):
    init_embed_scale: int
    n_classes: int
    n_positions: int
    input_keep_rate: float
    max_n_ints: int
    max_n_floats: int


class AnyNet(NamedTuple):
    n_tfe_layers: int
    n_seq: int  # T
    n_heads: int  # H
    dim_model: int  # k
    pos_t: int
    dropout_keep_rate: float
    eps: float
    mlp_n_hidden: List[int]
    mlp_activation: Activation

    init_embed_scale: int
    n_classes: int
    n_positions: int
    input_keep_rate: float
    max_n_ints: int
    max_n_floats: int

    @staticmethod
    def make(configs: AnyNetConfigs) -> Component:
        components = {
            # 0 for padded empty features for ragged batches
            'embedding': Embeddings.make(Embeddings(
                dict_size=configs.n_classes + 1,
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
        # one for in , one for out
        params = {
            'float_embedding': WeightParams(
                shape=(configs.dim_model, 2),
                init="embedding",
                scale=configs.init_embed_scale),
            'missing_embedding': WeightParams(
                shape=(configs.dim_model, 1),
                init="embedding",
                scale=configs.init_embed_scale),
        }

        def _parse_int(weights: ArrayTree, pos: npt.NDArray, value: npt.NDArray) -> npt.NDArray:
            pos_embed = components['positional_embedding'].fixed_pipeline(
                weights['positional_embedding'], pos)
            val_embed = components['embedding'].fixed_pipeline(
                weights['embedding'], value)
            return pos_embed + val_embed

        def _parse_float(weights: ArrayTree, pos: npt.NDArray, value: npt.NDArray) -> npt.NDArray:
            pos_embed = components['positional_embedding'].fixed_pipeline(
                weights['positional_embedding'], pos)
            val_embed = weights['float_embedding'][:, 0] * value
            return pos_embed + val_embed

        # ints: (int, int)[A], floats: (int, float)[B] -> [dim_model, A+B]
        def _fn(weights: ArrayTree, inputs: ArrayTree, rng) -> npt.NDArray:
            int_embeds = vmap(_parse_int, (None, 0, 0), 0)(weights, inputs['ints_pos'], inputs['ints'])
            float_embeds = vmap(_parse_float, (None, 0, 0), 0)(weights, inputs['floats_pos'], inputs['floats'])
            embeds = xp.c_[int_embeds, float_embeds]
            rng, key = random.split(rng)
            return components['encoder'].pipeline(weights['encoder'], embeds, key)


        # ints: (int, int)[A], floats: (int, float)[B] -> float
        def _loss(weights: ArrayTree, inputs: ArrayTree, rng) -> ArrayTree:
            int_embeds = vmap(_parse_int, (None, 0, 0), 0)(weights, inputs['ints_pos'], inputs['ints'])
            float_embeds = vmap(_parse_float, (None, 0, 0), 0)(weights, inputs['floats_pos'], inputs['floats'])
            embeds = xp.c_[int_embeds, float_embeds]
            len_inputs = configs.max_n_ints + configs.max_n_floats
            if 0 < configs.input_keep_rate < 1:
                rng, key_i, key_f = random.split(rng, 3)
                mask = random.bernoulli(key_i, configs.input_keep_rate, (len_inputs,))

                # [D, D, 1] -> [D]
                def mask_with_default(x: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
                    ds = xp.repeat(d, len_inputs)
                    return ds * (1 - mask) + x * mask
                masked_embeds = vmap(mask_with_default, (0, 0), 0)(embeds, weights['missing_embedding'])
            else:
                mask = xp.ones((len_inputs,))
                masked_embeds = embeds

            rng, key = random.split(rng)
            out_embeds = components['encoder'].processes[](weights['encoder'], masked_embeds, key)

            logits = weights['embedding']['dict'] @ out_embeds
            loss_y = -xp.mean((logits - logsumexp(logits, keepdims=True)) * nn.one_hot(y, configs.dict_size_output))

            return

        # [H, W, C] -> [dim_model]
        def _x2y(weights: ArrayTree, inputs: ArrayTree, rng) -> npt.NDArray:
            ints, floats = inputs['ints'], inputs['floats']
            rng, key = random.split(rng)

            yx = xp.c_[y_masked_emb, x]
            yx = components['norm'].fixed_pipeline(weights['norm'], yx)
            # [dim_model, dim_out + (h, w, C)]

            yx = components['encoder'].pipeline(weights['encoder'], yx, rng)
            # [dim_model, dim_out + (h, w, C)]

            y_rec = yx[:, 0]
            y_rec = components['y_reconstruct'].pipeline(weights['y_reconstruct'], y_rec, rng)
            return y_rec

        # noinspection PyTypeChecker
        # Because Pycharm sucks
        params.update(merge_params(components))
        processes = {
            make_ports(Input, 'loss'): _fn,
            make_ports((Input, "query"), Output): _x2y
        }
        # noinspection PyTypeChecker
        # Because Pycharm sucks
        return Component(params, processes)
