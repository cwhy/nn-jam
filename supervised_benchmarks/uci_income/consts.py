from pathlib import Path
from typing import NamedTuple, Dict, List

from supervised_benchmarks.ports import Port
from variable_protocols.variables import Variable, ordinal, dim, var_tensor, gaussian

FLOAT_OFFSET = -2
VALUE_SYMBOL = 0
variable_names = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]
row_width = len(variable_names)
n_tokens = 108


class TabularDataInfo(NamedTuple):
    n_rows_tr: int
    n_rows_tst: int
    tr_path: Path
    tst_path: Path
    symbol_id_table: Dict[str, int]
    is_digits: List[bool]
    # Values that are common
    common_values: Dict[str, float]


AnyNetContinuous = Port(type='Input', name='AnyNetContinuous')
AnyNetDiscrete = Port(type='Input', name='AnyNetDiscrete')
AnyNetDiscreteOut = Port(type='Output', name='AnyNetDiscreteOut')


def get_anynet_feature(_dict_size: int, _n_features: int, continuous: bool) -> Variable:
    if continuous:
        return var_tensor(gaussian(0, 1), {dim("Feature", _n_features)})
    else:
        return var_tensor(ordinal(_dict_size), {dim("Feature", _n_features)})
