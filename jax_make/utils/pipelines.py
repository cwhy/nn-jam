from typing import TypedDict

import numpy.typing as npt

Arr = npt.NDArray


class LinearWeights(TypedDict):
    w: Arr  # ab
    b: Arr  # b


# w:ab, b:b, a -> b
def linear(params: LinearWeights, x: Arr) -> Arr:
    return x @ params['w'] + params['b']

