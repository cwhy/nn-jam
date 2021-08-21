from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import Dict, Tuple, Callable, NamedTuple, Literal, Protocol, Set

import numpy as np
from einops import rearrange
from variable_protocols.base_variables import BaseVariable
from variable_protocols.protocols import fmt
from variable_protocols.variables import Variable, ordinal, bounded_float, var_scalar, one_hot
from variable_protocols.variables import dim, var_tensor

from supervised_benchmarks.numpy_utils import ordinal_from_1hot, ordinal_to_1hot


class MnistConfig(Protocol):
    @property
    @abstractmethod
    def type(self) -> Literal['MnistConfig']: ...


class MnistConfigIn(NamedTuple):
    is_float: bool
    is_flat: bool
    type: Literal['MnistConfig'] = 'MnistConfig'

    # noinspection PyTypeChecker
    # because pyCharm sucks
    def get_var(self) -> Variable:
        if self.is_float:
            base: BaseVariable = bounded_float(0, 1)
        else:
            base = ordinal(256)
        dims = {dim("h", 28), dim("w", 28)} if not self.is_flat else {dim("hw", 28 * 28)}
        return var_tensor(base, dims)


class MnistConfigOut(NamedTuple):
    is_1hot: bool

    type: Literal['MnistConfig'] = 'MnistConfig'

    # noinspection PyTypeChecker
    # because pyCharm sucks
    def get_var(self) -> Variable:
        if self.is_1hot:
            return var_scalar(one_hot(10))
        else:
            return var_scalar(ordinal(10))


# Can be removed after hole based variable config
transformations: Dict[Tuple[Variable, Variable],
                      Callable[[np.ndarray], np.ndarray]] = dict()

look_up_forward_: Dict[Variable, Set[Variable]] = defaultdict(set)


def register_(src: Variable, tgt: Variable, fn: Callable[[np.ndarray], np.ndarray]):
    assert src != tgt
    transformations[(src, tgt)] = fn
    look_up_forward_[src].add(tgt)
    if tgt in look_up_forward_:
        for new_tgt in look_up_forward_[tgt]:
            assert new_tgt != tgt
            if new_tgt != src:
                if new_tgt not in look_up_forward_[src]:
                    look_up_forward_[src].add(new_tgt)
                    new_fn = transformations[(tgt, new_tgt)]
                    transformations[(src, new_tgt)] = lambda x: new_fn(fn(x))


len_table_ = len(transformations)
old_len_table_ = None
while len_table_ != old_len_table_:
    print(len_table_, old_len_table_)
    for is_float in True, False:
        register_(MnistConfigIn(is_float, False).get_var(),
                  MnistConfigIn(is_float, True).get_var(),
                  lambda x: rearrange(x, 'b h w -> b (h w)'))
        register_(MnistConfigIn(is_float, True).get_var(),
                  MnistConfigIn(is_float, False).get_var(),
                  lambda x: rearrange(x, 'b (h w) -> b h w', h=28, w=28))

    for is_flat in True, False:
        register_(MnistConfigIn(True, is_flat).get_var(),
                  MnistConfigIn(False, is_flat).get_var(),
                  lambda x: np.round(x * 255).astype(np.uint8))
        register_(MnistConfigIn(False, is_flat).get_var(),
                  MnistConfigIn(True, is_flat).get_var(),
                  lambda x: x / 255)
    register_(MnistConfigOut(True).get_var(),
              MnistConfigOut(False).get_var(),
              ordinal_from_1hot)
    register_(MnistConfigOut(False).get_var(),
              MnistConfigOut(True).get_var(),
              ordinal_to_1hot)
    old_len_table_ = len_table_
    len_table_ = len(transformations)


def get_transformations(protocols: Tuple[Variable, Variable]) -> Callable[[np.ndarray], np.ndarray]:
    s, t = protocols
    # TODO after support struct-check
    if s == t:
        return lambda x: x
    else:
        return transformations[(s, t)]
