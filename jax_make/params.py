from __future__ import annotations
from abc import abstractmethod
from typing import Tuple, NamedTuple, Union, Literal, Mapping, Any, Protocol, Optional, TypeVar, Iterator, Collection, \
    Dict, ItemsView, ValuesView, KeysView

import numpy as np
from jax import numpy as xp
from jax.interpreters.xla import DeviceArray
from numpy import typing as npt

ArrayGen = Literal['kaiming', 'dropout', 'embedding', 'normal']
RNGKey = DeviceArray

KT = TypeVar('KT')
VT_co = TypeVar('VT_co', covariant=True)


class RecursiveMapping(Protocol[KT, VT_co]):
    def __getitem__(self, key: KT) -> Union[VT_co, RecursiveMapping[KT, VT_co]]: ...

    def items(self) -> ItemsView[KT, Union[VT_co, RecursiveMapping[KT, VT_co]]]: ...

    def __len__(self) -> int: ...


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


ArrayParamMapping = RecursiveMapping[str, ArrayParams]
ArrayParamTree = Union[RecursiveMapping[str, ArrayParams], ArrayParams]
ArrayTreeMapping = RecursiveMapping[str, npt.NDArray]
ArrayTree = Union[RecursiveMapping[str, npt.NDArray], npt.NDArray]


class WeightParams(NamedTuple):
    # from in to out
    shape: Tuple[int, ...]
    init: Union[ArrayGen, int, float] = "kaiming"
    scale: float = 1


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


def embedding_init(scale: float, shape: Tuple[int, ...]) -> npt.NDArray:
    """
    Arguments:
    :param scale:  standard deviation for initialization
        :param shape:  = (dict_size, ..., dim_model)
    where

    Returns:
    weight matrix of shape (dict_size, ..., dim_model)
    """
    dim_model = shape[-1]
    return xp.array(np.random.normal(0, scale * np.sqrt(dim_model), shape))


def normal_init(sd: float, shape: Tuple[int, ...]) -> npt.NDArray:
    return xp.array(np.random.normal(0, sd, shape))


def array_gen(params: ArrayParams) -> npt.NDArray:
    if isinstance(params.init, int) or isinstance(params.init, float):
        return xp.full(params.shape, float(params.init))
    elif params.init == 'kaiming':
        return kaiming_init(params.scale, params.shape)
    elif params.init == 'embedding':
        return embedding_init(params.scale, params.shape)
    elif params.init == 'normal':
        return normal_init(params.scale, params.shape)
    elif params.init == 'dropout':
        return dropout_gen(params.scale, params.shape)
    else:
        raise NotImplementedError("unsupported init type")


# noinspection PyTypeChecker
# Because pycharm sucks
def init_weights_helper(params: ArrayParamTree) -> ArrayTree:
    if isinstance(params, WeightParams):
        return array_gen(params)
    else:
        assert isinstance(params, dict)
        return {
            k: init_weights_helper(v)
            for k, v in params.items() if v is not None
        }


def make_weights(params: ArrayParamMapping) -> ArrayTreeMapping:
    return {
        k: init_weights_helper(v)
        for k, v in params.items() if v is not None
    }
