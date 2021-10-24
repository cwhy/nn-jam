import numpy.typing as npt
import jax.numpy as xp
from jax.scipy.special import logsumexp


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


