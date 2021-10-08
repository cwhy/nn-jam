from typing import NamedTuple, TypedDict, List
import numpy.typing as npt

from tests.jax_activations import Activation
from tests.jax_modules.dropout import Dropout, DropoutWeights
from tests.jax_modules.mlp import Mlp
from tests.jax_components import Component, sequential
from tests.jax_random_utils import ArrayTree
from tests.jax_modules.multi_head_attn import MultiHeadAttnWeights, MultiHeadAttn
from tests.jax_modules.norms import LayerNorm, LayerNormWeights


class TransformerConfigs(NamedTuple):
    n_seq: int  # T
    n_data: int  # N
    n_heads: int  # H
    dim_model: int  # k
    dim_input: int  # x
    dropout_keep_rate: float
    eps: float
    mlp_n_hidden: List[int]
    mlp_activation: Activation


class TransformerLayerWeights(TypedDict):
    mha: MultiHeadAttnWeights
    dropout: DropoutWeights
    norm: LayerNormWeights


class TransformerLayer(NamedTuple):
    configs: TransformerConfigs

    def make(self) -> Component:
        config = self.configs
        norm_input_shape = (config.dim_model, config.n_data, config.n_seq)
        components = {
            'mha': MultiHeadAttn.make(config),
            'norm': LayerNorm.make(LayerNorm(
                eps=config.eps,
                norm_axis=1)),
            'dropout': Dropout.make(
                Dropout(dropout_keep_rate=config.dropout_keep_rate,
                        input_shape=norm_input_shape)
            ),
            'mlp': Mlp(
                input_shape=norm_input_shape,
                on_axis=0,
                n_hidden=config.mlp_n_hidden,
                n_out=config.dim_model,
                activation=config.mlp_activation,
                dropout_keep_rate=config.dropout_keep_rate
            ).make()
        }

        def _fn(weights: ArrayTree, x: npt.NDArray) -> npt.NDArray:
            x = sequential(components, weights,
                           ['norm', 'mha', 'dropout'], x) + x
            x = sequential(components, weights,
                           ['norm', 'mlp', 'dropout'], x) + x
            return x

        return Component(components, _fn)
