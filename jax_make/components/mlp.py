from typing import NamedTuple, List, Protocol

from jax import random, vmap
from numpy.typing import NDArray

from jax_make.components.norms import LayerNorm
from jax_make.utils.activations import Activation, get_activation
from jax_make.components.dropout import Dropout
from jax_make.component_protocol import Component, merge_params
from jax_make.utils.pipelines import linear
from jax_make.params import WeightParams, ArrayTree, RNGKey


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
        # noinspection PyTypeChecker
        # Because pycharm sucks
        components = {
            f"layer_{i}": Component.from_fixed_pipeline(
                {"w": WeightParams(shape=(_in, _out)),
                 "b": WeightParams(shape=(_out,), init=0)},
                linear
            )
            for i, (_in, _out) in enumerate(zip(u_ins, u_outs))
        }
        if config.dropout_keep_rate != 1:
            for i, _out in enumerate(u_outs):
                components[f'dropout_{i}'] = Dropout.make(config)

        # n_in -> n_out
        def _fn(weights: ArrayTree, flow_: NDArray, rng: RNGKey) -> NDArray:
            activation = get_activation(config.activation)
            n_layers = len(config.n_hidden)
            keys = random.split(rng, n_layers)
            for layer_i in range(n_layers):
                layer_name = f"layer_{layer_i}"
                flow_ = components[layer_name].fixed_pipeline(weights[layer_name], flow_)
                flow_ = activation(flow_)
                if config.dropout_keep_rate != 1:
                    flow_ = components[f"dropout_{layer_i}"].pipeline(weights[layer_name], flow_, keys[layer_i])
            output_layer = f"layer_{len(config.n_hidden)}"
            flow_ = components[output_layer].fixed_pipeline(weights[output_layer], flow_)
            return flow_

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return Component.from_pipeline(merge_params(components), _fn)


class MlpLayerNormConfigs(Protocol):
    norm_dim_size: int
    norm_axis: int
    n_in: int
    n_hidden: List[int]
    n_out: int
    activation: Activation
    dropout_keep_rate: float
    eps: float


# Only support 2D now
class MlpLayerNorm(NamedTuple):
    norm_dim_size: int
    norm_axis: int
    n_in: int
    n_hidden: List[int]
    n_out: int
    activation: Activation
    dropout_keep_rate: float
    eps: float

    @staticmethod
    def make(config: MlpLayerNormConfigs) -> Component:
        u_ins = [config.n_in] + config.n_hidden
        u_outs = config.n_hidden + [config.n_out]
        # noinspection PyTypeChecker
        # Because pycharm sucks
        components_layers = {
            f"layer_{i}_linear": Component.from_fixed_pipeline(
                {"w": WeightParams(shape=(_in, _out)),
                 "b": WeightParams(shape=(_out,), init=0)},
                linear
            )
            for i, (_in, _out) in enumerate(zip(u_ins, u_outs))
        }

        components_norms = {
            f'layer_{i}_norm': LayerNorm.make(LayerNorm(
                eps=config.eps,
                norm_axis=config.norm_axis))
            for i in range(len(config.n_hidden))
        }
        components = z = {**components_norms, **components_layers}

        if config.dropout_keep_rate != 1:
            for i, _out in enumerate(u_outs):
                components[f'dropout_{i}'] = Dropout.make(config)

        # dim_norm, n_in -> dim_norm, n_out
        # or n_in, dim_norm -> n_out, dim_norm
        def _fn(weights: ArrayTree, flow_: NDArray, rng: RNGKey) -> NDArray:
            dn = config.norm_axis
            activation = get_activation(config.activation)
            n_layers = len(config.n_hidden)
            keys = random.split(rng, n_layers)
            for layer_i in range(n_layers):
                layer_name = f"layer_{layer_i}_linear"
                layer_norm_name = f"layer_{layer_i}_norm"
                flow_ = vmap(components[layer_name].fixed_pipeline, (None, dn), dn)(weights[layer_name], flow_)
                flow_ = components[layer_norm_name].fixed_pipeline(weights[layer_norm_name], flow_)
                flow_ = activation(flow_)
                if config.dropout_keep_rate != 1:
                    do_keys = random.split(keys[layer_i], config.norm_dim_size)
                    flow_ = vmap(
                        components[f"dropout_{layer_i}"].pipeline, (None, dn, 0), dn)(
                        weights[layer_name], flow_, do_keys)
            output_layer = f"layer_{len(config.n_hidden)}_linear"
            flow_ = vmap(components[output_layer].fixed_pipeline, (None, dn), dn)(weights[output_layer], flow_)
            return flow_

        # noinspection PyTypeChecker
        # Because pycharm sucks
        return Component.from_pipeline(merge_params(components), _fn)
