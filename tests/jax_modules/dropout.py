from typing import NamedTuple, TypedDict, Protocol, Tuple

import jax.numpy as xp
import numpy as np
import numpy.typing as npt

from tests.jax_protocols import WeightParams, Component, Weights


class DropoutNoise(TypedDict):
    dropout_keep: npt.NDArray


class DropoutConfigs(Protocol):
    dropout_keep_rate: float
    input_shape: Tuple[int, ...]


class Dropout(NamedTuple):
    dropout_keep_rate: float
    input_shape: Tuple[int, ...]

    @staticmethod
    def make(config: DropoutConfigs) -> Component:
        components = {
            'dropout_keep': WeightParams(shape=config.input_shape),
        }

        def _fn(weights: Weights, inputs: Inputs) -> npt.NDArray:
            return xp.where(weights['dropout_keep'], x / config.dropout_keep_rate, 0)

        return Component(components, _fn)


def dropout_gen(keep_rate: float, shape: Tuple[int, ...]):
    return np.random.binomial(1, keep_rate, shape)

