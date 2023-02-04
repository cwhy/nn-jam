from typing import Literal, Callable
import jax.nn
from jax import Array

Activation = Literal['relu', 'tanh', 'gelu']
ActivationFn = Callable[[Array], Array]


def get_activation(a: Activation) -> ActivationFn:
    if a == 'relu':
        return jax.nn.relu
    elif a == 'tanh':
        return jax.nn.tanh
    elif a == 'gelu':
        return jax.nn.gelu
    else:
        raise NotImplementedError(f"Activation function {a} is not supported")
