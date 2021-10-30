from typing import Callable

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


def get_cosine_similarity(eps: float) -> Callable[[npt.NDArray, npt.NDArray], npt.NDArray]:
    def _fn(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
        y /= xp.maximum(xp.linalg.norm(y, axis=1, keepdims=True), xp.sqrt(eps))
        x /= xp.maximum(xp.linalg.norm(x, axis=1, keepdims=True), xp.sqrt(eps))
        return -xp.sum(x * y, axis=1)
    return _fn
