from typing import Tuple

import numpy.typing as npt
import jax.numpy as xp
from jax.scipy.special import logsumexp
import numpy as np


def relu(x: npt.NDArray) -> npt.NDArray:
    """ stax relu"""
    return xp.maximum(x, 0.)


def softmax(inputs: npt.NDArray) -> npt.NDArray:
    # wait for good jax typing
    # noinspection PyArgumentList
    un_normalized = xp.exp(inputs - inputs.max(axis=1, keepdims=True))
    return un_normalized / xp.sum(un_normalized, axis=1, keepdims=True)


def softmax_cross_entropy(logits: npt.NDArray, target: npt.NDArray) -> npt.NDArray:
    """
    softmax_cross_entropy.

    Arguments:
        logits: network prediction. Dimensions are [C, NT]
        target: one-hot targets across 81 characters

    Returns:
        cross_entropy: Loss vector over NT

    TODO: Register custom gradient to avoid numerical instability
    """
    log_softmax = logits - logsumexp(logits, axis=0, keepdims=True)
    cross_entropy = - xp.sum(target * log_softmax, axis=0)
    return cross_entropy


def kaiming_init(sd: float, shape: Tuple[int, ...]) -> npt.NDArray:
    """
    Generate randomly initialized weight matrix with Kaiming initalization:
    Normally distributed scaled by sqrt(2/fan_in)

    Arguments:
        :param sd:  standard deviation for initialization
        :param shape:  = (n_in, ..., n_out)
            where
            n_in is number of inputs to the layer
            n_out is number of outputs from the layer

    Returns:
        weight matrix of shape [n_in, n_out]
    """
    n_in = shape[0]
    return xp.array(np.sqrt(2 / n_in) * np.random.normal(0, sd, shape))
