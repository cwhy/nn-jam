from typing import Callable

import jax.numpy as xp
from jax import nn, Array, lax
from jax.scipy.special import logsumexp


def softmax(inputs: Array) -> Array:
    # wait for good jax typing
    # noinspection PyArgumentList
    un_normalized = xp.exp(inputs - inputs.max(axis=1, keepdims=True))
    return un_normalized / xp.sum(un_normalized, axis=1, keepdims=True)


def get_cosine_similarity_loss(eps: float) -> Callable[[Array, Array], Array]:
    # x: [D], y: [D] -> []
    # smaller is better
    def _fn(x: Array, y: Array) -> Array:
        y /= xp.maximum(xp.linalg.norm(y), xp.sqrt(eps))
        x /= xp.maximum(xp.linalg.norm(x), xp.sqrt(eps))
        return -xp.mean(x * y)

    return _fn


def l2loss(x: Array, y: Array) -> Array:
    return xp.mean(xp.square(x - y))


def l1loss(x: Array, y: Array) -> Array:
    return xp.mean(xp.abs(x - y))


def softmax_cross_entropy(logits: Array, target: Array) -> Array:
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


def softmax_cross_entropy_with_integer_labels(
        logits: Array,
        labels: Array,
) -> Array:
    """From Optax:
  Computes softmax cross entropy between sets of logits and integer labels.
  Measures the probability error in discrete classification tasks in which
  the classes are mutually exclusive (each entry is in exactly one class).
  For example, each CIFAR-10 image is labeled with one and only one label:
  an image can be a dog or a truck, but not both.
  References:
    [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)
  Args:
    logits: Unnormalized log probabilities, with shape `[..., num_classes]`.
    labels: Integers specifying the correct class for each input, with shape
      `[...]`.
  Returns:
    Cross entropy between each prediction and the corresponding target
    distributions, with shape `[...]`.
  """
    # This is like jnp.take_along_axis(jax.nn.log_softmax(...), ...) except that
    # we avoid subtracting the normalizer from all values, just from the values
    # for the correct labels.
    logits_max = xp.max(logits, axis=-1, keepdims=True)
    logits -= lax.stop_gradient(logits_max)
    label_logits = xp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    log_normalizers = xp.log(xp.sum(xp.exp(logits), axis=-1))
    return log_normalizers - label_logits


# [], [] -> []
def sigmoid_cross_entropy_loss(logits: Array, targets: Array) -> Array:
    # Optax implementation:
    # log_p = nn.log_sigmoid(logits)
    # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
    # log_not_p = nn.log_sigmoid(-logits)
    # return -targets * log_p - (1. - targets) * log_not_p
    return xp.mean(xp.maximum(logits, 0) - logits * targets + nn.softplus(-xp.abs(logits)))


def new_Gelu(x: Array) -> Array:
    return 0.5 * x * (1 + xp.tanh(xp.sqrt(2 / xp.pi) * (x + 0.044715 * xp.power(x, 3))))
