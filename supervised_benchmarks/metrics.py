import numpy as np
from numpy.typing import NDArray


def calc_acc_numpy(output: NDArray, target: NDArray):
    return np.sum(np.square(output - target))
