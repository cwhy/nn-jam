from __future__ import annotations

from typing import List, Dict, Set, Literal, Iterable, Optional

from variable_protocols.base_variables import BaseVariable, \
    OneSideSupported, Gamma, Bounded, OneHot, NamedCategorical, \
    CategoryIds, Ordinal, CategoricalVector, Gaussian
from variable_protocols.protocols import VariableTensor, Variable, \
    VariableGroup, DimensionFamily, TensorBase


def bounded_float(min_val: float, max_val: float) -> Bounded:
    if min_val > max_val:
        raise ValueError(f"min_val > max_val: {min_val} > {max_val}")
    return Bounded(max_val, min_val)


def one_side_supported(bound: int, max_or_min: Literal["max", "min"]) -> OneSideSupported:
    return OneSideSupported(bound, max_or_min)


def positive_float() -> OneSideSupported:
    return OneSideSupported(0, "min")


def negative_float() -> OneSideSupported:
    return OneSideSupported(0, "max")


def gamma(alpha: float, beta: float) -> Gamma:
    if alpha > 0 and beta > 0:
        return Gamma(alpha, beta)
    else:
        raise ValueError(f"Invalid alpha or beta value {alpha}, {beta}")


def erlang(k: int, beta: float) -> Gamma:
    if k > 0:
        return gamma(k, beta)
    else:
        raise ValueError(f"Invalid k:{k}")


def exponential(lambda_: float) -> Gamma:
    if lambda_ > 0:
        return erlang(1, lambda_)
    else:
        raise ValueError(f"Invalid lambda(l) value {lambda_}")


def ordinal(n_category: int) -> Ordinal:
    if n_category > 0:
        return Ordinal(n_category)
    else:
        raise ValueError(f"Invalid n_category value: {n_category}")


def one_hot(n_category: int) -> OneHot:
    if n_category > 0:
        return OneHot(n_category)
    else:
        raise ValueError(f"Invalid n_category value: {n_category}")


def cat_vec(n_category: int, n_embedding: int) -> CategoricalVector:
    if n_category > 0:
        if n_embedding > 0:
            return CategoricalVector(n_category, n_embedding)
        else:
            raise ValueError(f"Invalid n_embedding value: {n_embedding}")
    else:
        raise ValueError(f"Invalid n_category value: {n_category}")


def cat_from_names(names: Iterable[str]) -> NamedCategorical:
    return NamedCategorical(frozenset(names))


def cat_ids(max_id_len: int) -> CategoryIds:
    return CategoryIds(max_id_len)


def gaussian(mean: float, var: float) -> Gaussian:
    if var > 0:
        return Gaussian(mean, var)
    else:
        raise ValueError(f"Invalid variance(var) value: {var}")


def dim(label: str,
        length: Optional[int],
        positioned: bool = True,
        n_members: int = 1) -> DimensionFamily:
    return DimensionFamily(label=label, len=length,
                           n_members=n_members, positioned=positioned)


def var_tensor(var: TensorBase, dims: Set[DimensionFamily]) -> VariableTensor:
    return VariableTensor(var, frozenset(dims))


def var_array(var: TensorBase,
              length: int,
              dim_label: str) -> VariableTensor:
    return VariableTensor(var, frozenset({dim(length=length, label=dim_label)}))


def var_scalar(var: BaseVariable) -> VariableTensor:
    return VariableTensor(var, frozenset())


def var_group(vars_set: Set[Variable]) -> VariableGroup:
    assert isinstance(vars_set, set)
    if len(vars_set) <= 1:
        raise ValueError("A variable set/group must contain more than one variable")
    return VariableGroup(frozenset(vars_set))


def var_unique(var: TensorBase, name: str) -> VariableTensor:
    return VariableTensor(var, frozenset(), label=name)


def var_dict(vars_dict: Dict[Variable, str]) -> VariableGroup:
    return var_group({
        var_unique(var, name)
        for var, name in vars_dict.items()
    })


def var_ordered(vars_list: List[Variable]) -> VariableGroup:
    if not len(vars_list) == len(set(vars_list)):
        raise ValueError("There are duplicate variables in input,"
                         "use var_tensor to combine them first,"
                         "or var_unique to make them unique")
    return var_group({
        var_unique(var, str(i))
        for i, var in enumerate(vars_list)
    })
