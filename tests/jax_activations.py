from typing import Literal, Callable
import numpy.typing as npt
import jax.nn

Activation = Literal['relu']
ActivationFn = Callable[[npt.NDArray], npt.NDArray]


def get_activation(a: Activation) -> ActivationFn:
    if a == 'relu':
        return jax.nn.relu
    else:
        raise NotImplementedError(f"Activation function {a} is not supported")
