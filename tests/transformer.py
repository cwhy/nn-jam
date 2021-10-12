from typing import NamedTuple, List

import numpy.typing as npt
from jax import random, vmap

from tests.jax_activations import Activation
from tests.jax_components import Component, sequential
from tests.jax_modules.dropout import Dropout
from tests.jax_modules.mlp import Mlp
from tests.jax_modules.multi_head_attn import SelfMultiHeadAttn
from tests.jax_modules.norms import LayerNorm
from tests.jax_random_utils import ArrayTree, RNGKey


class TransformerConfigs(NamedTuple):
    n_tfe_layers: int
    n_seq: int  # T
    n_heads: int  # H
    dim_model: int  # k
    dim_input: int  # x
    dropout_keep_rate: float
    eps: float
    mlp_n_hidden: List[int]
    mlp_activation: Activation


class TransformerLayer(NamedTuple):
    configs: TransformerConfigs

    @staticmethod
    def make(config: TransformerConfigs) -> Component:
        components = {
            'mha': SelfMultiHeadAttn.make(config),
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
            x = vmap(sequential(components, ['norm', 'mlp', 'dropout']), (None, -1, None), -1)(weights, x, key2) + x
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
