from typing import Callable

import numpy.typing as npt
import jax.numpy as xp
from jax import nn
from jax.scipy.special import logsumexp


def softmax(inputs: npt.NDArray) -> npt.NDArray:
    # wait for good jax typing
    # noinspection PyArgumentList
    un_normalized = xp.exp(inputs - inputs.max(axis=1, keepdims=True))
    return un_normalized / xp.sum(un_normalized, axis=1, keepdims=True)


def get_cosine_similarity_loss(eps: float) -> Callable[[npt.NDArray, npt.NDArray], npt.NDArray]:
    # x: [D], y: [D] -> []
    def _fn(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
        y /= xp.maximum(xp.linalg.norm(y), xp.sqrt(eps))
        x /= xp.maximum(xp.linalg.norm(x), xp.sqrt(eps))
        return -xp.mean(x * y)

    return _fn


def l2loss(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    return xp.mean(xp.square(x - y))


def l1loss(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    return xp.mean(xp.abs(x - y))


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


# [], [] -> []
def sigmoid_cross_entropy_loss(logits: npt.NDArray, targets: npt.NDArray) -> npt.NDArray:
    # Optax implementation:
    # log_p = nn.log_sigmoid(logits)
    # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
    # log_not_p = nn.log_sigmoid(-logits)
    # return -targets * log_p - (1. - targets) * log_not_p
    return xp.mean(xp.maximum(logits, 0) - logits * targets + nn.softplus(-xp.abs(logits)))
