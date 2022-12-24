from pathlib import Path
from typing import NamedTuple, Dict, List

from supervised_benchmarks.ports import NewPort
from variable_protocols.base_variables import N, IDs
from variable_protocols.tensorhub import GenericFeatureDim

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


AnyNetContinuous = NewPort(N(0, 1) * GenericFeatureDim,
                           type='Input',
                           name='AnyNetContinuous')
AnyNetDiscrete = NewPort(IDs("int") * GenericFeatureDim,
                         type='Input',
                         name='AnyNetDiscrete')
AnyNetDiscreteOut = NewPort(IDs("int") * GenericFeatureDim, type='Output', name='AnyNetDiscreteOut')
