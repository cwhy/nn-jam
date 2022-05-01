from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Literal, Mapping, FrozenSet, Dict, TypedDict

import numpy.typing as npt

from supervised_benchmarks.dataset_protocols import Port, Subset, DataQuery, Input, Output, \
    Data, DataPortMap, OutputOptions, Context, FixedSubset, AllVars
from supervised_benchmarks.uci_income.consts import row_width
from supervised_benchmarks.uci_income.utils import analyze_data, load_data
from variable_protocols.variables import Variable, ordinal, dim, var_tensor, var_group, gaussian

name: Literal["UciIncome"] = "UciIncome"


class NpDataContent(TypedDict):
    symbols: npt.ArrayLike
    values: npt.ArrayLike


class UciIncomeData(NamedTuple):
    port: Port
    protocol: Variable
    subset: Subset
    content: NpDataContent


class UciIncomeDataPool(NamedTuple):
    array_dict: Mapping[str, npt.ArrayLike]
    port: Port
    src_var: Variable
    tgt_var: Variable

    def subset(self, subset: Subset) -> Data[npt.ArrayLike]:
        assert self.src_var == self.tgt_var
        if subset == FixedTrain:
            prefix = "tr"
        elif subset == FixedTest:
            prefix = "tst"
        else:
            raise ValueError(f"Unsupported subset: {subset}")
        if self.port == AllVars:
            data_array = NpDataContent(
                symbols=self.array_dict[f'{prefix}_symbol'],
                values=self.array_dict[f'{prefix}_value']
            )
        else:
            raise ValueError(f"Unsupported port: {self.port}")

        # noinspection PyTypeChecker
        # because pyCharm sucks
        return UciIncomeData(self.port, self.tgt_var, subset, data_array)


data_info = analyze_data()
n_samples_tr = data_info.n_rows_tr
n_samples_tst = data_info.n_rows_tst
n_samples = n_samples_tr + n_samples_tst

FixedTrain = FixedSubset('FixedTrain', list(range(n_samples_tr)))
FixedTest = FixedSubset('FixedTest', list(range(n_samples_tst)))
FixedAll = FixedSubset('All', list(range(n_samples)))

dict_size = len(data_info.symbol_id_table) + 3


# noinspection PyTypeChecker
# because pyCharm sucks
def get_anynet_feature(_dict_size: int, _n_features: int) -> Variable:
    return var_group(
        {var_tensor(gaussian(0, 1), {dim("Feature", _n_features)}),
         var_tensor(ordinal(_dict_size), {dim("Feature", _n_features)})})


uci_income_all_anynet = get_anynet_feature(dict_size, row_width)
uci_income_in_anynet = get_anynet_feature(dict_size, row_width - 1)
uci_income_out_anynet = get_anynet_feature(dict_size, 1)


class UciIncome:
    @property
    def ports(self) -> FrozenSet[Port]:
        return frozenset({Input, Output, AllVars})

    def __init__(self, base_path: Path) -> None:
        # TODO implement download logic
        symbol_table_tr, value_table_tr = load_data(data_info, is_train=True)
        symbol_table_tst, value_table_tst = load_data(data_info, is_train=False)

        self.array_dict: Dict[str, npt.NDArray] = {
            'tr_symbol': symbol_table_tr,
            'tr_value': value_table_tr,
            'tst_symbol': symbol_table_tst,
            'tst_value': value_table_tst
        }

        # noinspection PyTypeChecker
        # because pyCharm sucks
        self.protocols: Mapping[str, Variable] = {
            Input: uci_income_in_anynet,
            Output: uci_income_out_anynet,
            AllVars: uci_income_all_anynet
        }

    @property
    def name(self) -> Literal['UciIncome']:
        return name

    def retrieve(self, query: DataQuery) -> Mapping[Port, UciIncomeDataPool]:
        assert all(port in self.ports for port in query)
        return {
            port: UciIncomeDataPool(
                self.array_dict,
                port,
                src_var=self.protocols[port],
                tgt_var=variable_protocol)
            for port, variable_protocol in query.items()
        }


# noinspection PyTypeChecker
# Because pycharm sucks
class UciIncomeDataConfig(NamedTuple):
    base_path: Path
    port_vars: DataQuery
    type: Literal['DataConfig'] = 'DataConfig'

    def get_data(self) -> DataPortMap:
        return UciIncome(self.base_path).retrieve(self.port_vars)
