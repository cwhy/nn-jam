from abc import abstractmethod
from typing import NamedTuple, List, Protocol

from jax import random, vmap
from jax import Array as NDArray

from jax_make.components.norms import LayerNorm
from jax_make.utils.activations import Activation, get_activation
from jax_make.components.dropout import Dropout
from jax_make.component_protocol import Component, merge_component_params
from jax_make.utils.elementary_components import linear, linear_component
from jax_make.params import WeightParams, ArrayTree, RNGKey
import jax_make.params as p


class MlpConfigs(Protocol):
    @property
    @abstractmethod
    def n_in(self) -> int:
        ...

    @property
    @abstractmethod
    def n_hidden(self) -> List[int]:
        ...

    @property
    @abstractmethod
    def n_out(self) -> int:
        ...

    @property
    @abstractmethod
    def activation(self) -> Activation:
        ...

    @property
    @abstractmethod
    def dropout_keep_rate(self) -> float:
        ...


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
            f"layer_{i}": linear_component(_in, _out)
            for i, (_in, _out) in enumerate(zip(u_ins, u_outs))
        }
        if config.dropout_keep_rate != 1:
            for i, _out in enumerate(u_outs):
                components[f'dropout_{i}'] = Dropout.make(config)

        # n_in -> n_out
        def _fn(weights: p.ArrayTreeMapping, x: NDArray, rng: RNGKey) -> NDArray:
            activation = get_activation(config.activation)
            n_layers = len(config.n_hidden)
            keys = random.split(rng, n_layers)
            for layer_i in range(n_layers):
                layer_name = f"layer_{layer_i}"
                layer_weights = p.get_mapping(weights, layer_name)
                x = components[layer_name].fixed_pipeline(layer_weights, x)
                x = activation(x)
                if config.dropout_keep_rate != 1:
                    x = components[f"dropout_{layer_i}"].pipeline(layer_weights, x, keys[layer_i])
            output_layer = f"layer_{len(config.n_hidden)}"
            output_weights = p.get_mapping(weights, output_layer)
            x = components[output_layer].fixed_pipeline(output_weights, x)
            return x

        return Component.from_pipeline(merge_component_params(components), _fn)


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
        components_layers = {
            f"layer_{i}_linear": linear_component(_in, _out)
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
        def _fn(weights: p.ArrayTreeMapping, x: NDArray, rng: RNGKey) -> NDArray:
            dn = config.norm_axis
            activation = get_activation(config.activation)
            n_layers = len(config.n_hidden)
            keys = random.split(rng, n_layers)
            for layer_i in range(n_layers):
                layer_name = f"layer_{layer_i}_linear"
                layer_norm_name = f"layer_{layer_i}_norm"
                layer_weights = p.get_mapping(weights, layer_name)
                layer_norm_weights = p.get_mapping(weights, layer_norm_name)
                x = vmap(components[layer_name].fixed_pipeline, (None, dn), dn)(layer_weights, x)
                x = components[layer_norm_name].fixed_pipeline(layer_norm_weights, x)
                x = activation(x)
                if config.dropout_keep_rate != 1:
                    do_keys = random.split(keys[layer_i], config.norm_dim_size)
                    x = vmap(
                        components[f"dropout_{layer_i}"].pipeline, (None, dn, 0), dn)(
                        layer_weights, x, do_keys)
            output_layer = f"layer_{len(config.n_hidden)}_linear"
            output_weights = p.get_mapping(weights, output_layer)
            x = vmap(components[output_layer].fixed_pipeline, (None, dn), dn)(output_weights, x)
            return x

        return Component.from_pipeline(merge_component_params(components), _fn)


