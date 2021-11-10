from typing import NamedTuple, Dict, List
from pathlib import Path

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


class TabularDataInfo(NamedTuple):
    n_rows_tr: int
    n_rows_tst: int
    tr_path: Path
    tst_path: Path
    symbol_id_table: Dict[str, int]
    is_digits: List[bool]
    special_values: Dict[str, float]