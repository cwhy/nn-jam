import numpy as np


def calc_acc_numpy(output: np.ndarray, target: np.ndarray):
    return np.sum(np.square(output - target))
