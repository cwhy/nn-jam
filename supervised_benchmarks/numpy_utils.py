import numpy as np
from numpy import ndarray


# x can be any dimension, returns x with extra axis of 1hot
def ordinal_to_1hot(x: ndarray) -> ndarray:
    n_category = x.max(initial=0) + 1
    out = np.zeros((x.size, n_category), dtype=np.uint8)
    out[np.arange(x.size), x.ravel()] = 1
    out.shape = x.shape + (n_category,)
    return out


def ordinal_from_1hot(x: ndarray) -> ndarray:
    return np.argmax(x, axis=-1)
