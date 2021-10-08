from abc import abstractmethod
from typing import Tuple, NamedTuple, Union, Literal, Mapping, Any, Protocol, Optional

import numpy as np
from jax import numpy as xp
from numpy import typing as npt

ArrayGen = Literal['kaiming', 'dropout']


class ArrayParams(Protocol):
    # from in to out
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]: ...

    @property
    @abstractmethod
    def init(self) -> Union[ArrayGen, int, float]: ...

    @property
    @abstractmethod
    def scale(self) -> float: ...


class WeightParams(NamedTuple):
    # from in to out
    shape: Tuple[int, ...]
    init: Union[ArrayGen, int, float] = "kaiming"
    scale: float = 1


class RandomParams(NamedTuple):
    # from in to out
    shape: Tuple[int, ...]
    init: Union[ArrayGen, int, float]
    scale: float


ArrayParamTree = Union[Mapping[str, Any], ArrayParams]

ArrayTree = Union[Mapping[str, Any], npt.NDArray]


def dropout_gen(keep_rate: float, shape: Tuple[int, ...]):
    return np.random.binomial(1, keep_rate, shape)


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


def array_gen(params: ArrayParams) -> npt.NDArray:
    if isinstance(params.init, int) or isinstance(params.init, float):
        return xp.full(params.shape, float(params.init))
    elif params.init == 'kaiming':
        return kaiming_init(params.scale, params.shape)
    elif params.init == 'dropout':
        return dropout_gen(params.scale, params.shape)
    else:
        raise NotImplementedError("unsupported init type")


# TODO proper jax way to do this
def init_random(params: ArrayParamTree) -> Optional[ArrayTree]:
    if isinstance(params, RandomParams):
        # noinspection PyTypeChecker
        # Because pycharm sucks
        return array_gen(params)
    elif isinstance(params, WeightParams):
        return None
    else:
        assert isinstance(params, dict)
        return {
            k: init_random(v)
            for k, v in params.items() if not isinstance(v, WeightParams)
        }


def init_weights(params: ArrayParamTree) -> Optional[ArrayTree]:
    if isinstance(params, WeightParams):
        # noinspection PyTypeChecker
        # Because pycharm sucks
        return array_gen(params)
    elif isinstance(params, RandomParams):
        return None
    else:
        assert isinstance(params, dict)
        return {
            k: init_weights(v)
            for k, v in params.items() if not isinstance(v, RandomParams)
        }


def init_array(params: ArrayParamTree) -> ArrayTree:
    if isinstance(params, WeightParams) or isinstance(params, RandomParams):
        # noinspection PyTypeChecker
        # Because pycharm sucks
        return array_gen(params)
    else:
        assert isinstance(params, dict)
        return {
            k: init_array(v)
            for k, v in params.items()
        }
