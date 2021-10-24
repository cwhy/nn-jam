from typing import Literal, Callable
import numpy.typing as npt
import jax.nn

Activation = Literal['relu']
ActivationFn = Callable[[npt.NDArray], npt.NDArray]


def get_activation(a: Activation) -> ActivationFn:
    if a == 'relu':
        return jax.nn.relu
    elif a == 'tanh':
        return jax.nn.tanh
    else:
        raise NotImplementedError(f"Activation function {a} is not supported")
