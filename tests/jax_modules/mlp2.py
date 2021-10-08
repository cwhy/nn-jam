from typing import NamedTuple, List, Protocol

import numpy.typing as npt

from tests.jax_activations import Activation, get_activation
from tests.jax_modules.dropout import Dropout
from tests.jax_components import Component
from tests.jax_random_utils import WeightParams, ArrayTree


class MlpConfigs(Protocol):
    n_in: int
    n_hidden: List[int]
    n_out: int
    activation: Activation
    dropout_keep_rate: float


class Mlp(NamedTuple):
    n_in: int
    n_hidden: List[int]
    n_out: int
    activation: Activation
    dropout_keep_rate: float

    @staticmethod
    def make(config: MlpConfigs) -> Component:
        u_ins = [config.n_in] + config.n_hidden
        u_outs = config.n_hidden + [config.n_out]
        components = {
            f"layer_{i}": Component({"w": WeightParams(shape=(_in, _out)),
                                     "b": WeightParams(shape=(_out,), init=0)},
                                    lambda params, x: x @ params['w'] + params['b']
                                    )
            for i, (_in, _out) in enumerate(zip(u_ins, u_outs))
        }
        if config.dropout_keep_rate != 1:
            for i, _out in enumerate(u_outs):
                components[f'dropout_{i}'] = Dropout.make(Dropout(
                    dropout_keep_rate=config.dropout_keep_rate,
                    single_input_shape=(_out,)
                ))

        def _fn(weights: ArrayTree, flow_: npt.NDArray) -> npt.NDArray:
            activation = get_activation(config.activation)
            for layer_i in range(len(config.n_hidden)):
                layer_name = f"layer_{layer_i}"
                flow_ = components[layer_name].process(weights[layer_name], flow_)
                flow_ = activation(flow_)
                if config.dropout_keep_rate != 1:
                    flow_ = components[f"dropout_{layer_i}"].process(weights[layer_name], flow_)
            output_layer = f"layer_{len(config.n_hidden)}"
            flow_ = components[output_layer].process(weights[output_layer], flow_)
            return flow_

        return Component({k: v.params for k, v in components.items()}, _fn)
