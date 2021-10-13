from typing import NamedTuple, List

import numpy.typing as npt
from jax import random, vmap

from tests.jax_activations import Activation
from tests.jax_components import Component, sequential
from tests.jax_modules.dropout import Dropout
from tests.jax_modules.embedding import Embeddings
from tests.jax_modules.mlp import Mlp
from tests.jax_modules.multi_head_attn import SelfMultiHeadAttn
from tests.jax_modules.norms import LayerNorm
from tests.jax_modules.positional_encoding import PositionalEncoding
from tests.jax_random_utils import ArrayTree, RNGKey


class TransformerConfigs(NamedTuple):
    n_tfe_layers: int
    n_seq: int  # T
    n_heads: int  # H
    dim_model: int  # k
    dim_input: int  # x
    dict_size: int
    pos_t: int
    dropout_keep_rate: float
    eps: float
    mlp_n_hidden: List[int]
    mlp_activation: Activation


class TransformerLayer(NamedTuple):
    configs: TransformerConfigs

    @staticmethod
    def make(config: TransformerConfigs) -> Component:
        components = {
            'mha': SelfMultiHeadAttn.make(SelfMultiHeadAttn(
                n_seq=config.n_seq,
                n_heads=config.n_heads,
                dim_model=config.dim_model,
                dim_input=config.dim_model
            )),
            'norm': LayerNorm.make(LayerNorm(
                eps=config.eps,
                norm_axis=0)),
            'dropout': Dropout.make(config),
            'mlp': Mlp.make(Mlp(n_in=config.dim_model,
                                n_hidden=config.mlp_n_hidden,
                                n_out=config.dim_model,
                                activation=config.mlp_activation,
                                dropout_keep_rate=config.dropout_keep_rate
                                ))
        }

        def _fn(weights: ArrayTree, x: npt.NDArray, rng: RNGKey) -> npt.NDArray:
            rng, key1, key2 = random.split(rng, 3)
            x = sequential(components, ['norm', 'mha', 'dropout'])(weights, x, key1) + x
            x = vmap(sequential(components, ['norm', 'mlp', 'dropout']),
                     (None, config.pos_t, None), config.pos_t)(weights, x, key2) + x
            return x

        return Component({k: v.params for k, v in components.items()}, _fn)


class TransformerEncoder(NamedTuple):
    configs: TransformerConfigs

    @staticmethod
    def make(configs: TransformerConfigs) -> Component:
        components = {
            f"tfe_layer_{i}": TransformerLayer.make(configs)
            for i in range(configs.n_tfe_layers)
        }

        def _fn(weights: ArrayTree, x: npt.NDArray, rng) -> npt.NDArray:
            rng, key = random.split(rng)
            x = sequential(components, [f"tfe_layer_{i}" for i in range(configs.n_tfe_layers)])(weights, x, key)
            return x

        return Component({k: v.params for k, v in components.items()}, _fn)


class Transformer(NamedTuple):
    configs: TransformerConfigs

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
            x = vmap(components['embedding'].fixed_process, (None, configs.pos_t), configs.pos_t)(weights['embedding'], x)
            x = components['positional_encoding'].fixed_process(weights['positional_encoding'], x)
            rng, key = random.split(rng)
            x = components['encoder'].process(weights['encoder'], x, key)
            return x

        return Component({k: v.params for k, v in components.items()}, _fn)
