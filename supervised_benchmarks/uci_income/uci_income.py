from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import NamedTuple, Literal, Mapping, FrozenSet, Dict, TypedDict

import numpy.typing as npt

from supervised_benchmarks.dataset_protocols import Subset, DataQuery, Data, DataPortMap, FixedSubset, FixedSubsetType, FixedTrain, FixedTest
from supervised_benchmarks.ports import Port, Input, Output, Context, OutputOptions, AllVars
from supervised_benchmarks.uci_income.consts import row_width, TabularDataInfo
from supervised_benchmarks.uci_income.utils import analyze_data, load_data
from variable_protocols.variables import Variable, ordinal, dim, var_tensor, var_group, gaussian

name: Literal["UciIncome"] = "UciIncome"
# support column names


class TabularDataContentNp(TypedDict):
    symbol_names: tuple[str]
    value_names: tuple[str]
    symbols: npt.ArrayLike
    values: npt.ArrayLike


class UciIncomeData(NamedTuple):
    port: Port
    protocol: Variable
    subset: Subset
    content: TabularDataContentNp


class UciIncomeDataPool(NamedTuple):
    data_info: TabularDataInfo
    array_dict: Mapping[str, npt.ArrayLike]
    fixed_datasets: dict[FixedSubsetType, Data[TabularDataContentNp]]
    port: Port
    src_var: Variable
    tgt_var: Variable

    def subset(self, subset: Subset) -> Data[TabularDataContentNp]:
        assert self.src_var == self.tgt_var
        raise NotImplementedError

        # noinspection PyTypeChecker
        # because pyCharm sucks
        # return UciIncomeData(self.port, self.tgt_var, subset, data_array)


# noinspection PyTypeChecker
# because pyCharm sucks
def get_anynet_feature(_dict_size: int, _n_features: int) -> Variable:
    return var_group(
        {var_tensor(gaussian(0, 1), {dim("Feature", _n_features)}),
         var_tensor(ordinal(_dict_size), {dim("Feature", _n_features)})})


dict_size = 10

uci_income_all_anynet = get_anynet_feature(dict_size, row_width)
uci_income_in_anynet = get_anynet_feature(dict_size, row_width - 1)
uci_income_out_anynet = get_anynet_feature(dict_size, 1)


class UciIncome:
    @property
    def ports(self) -> FrozenSet[Port]:
        return frozenset({Input, Output, AllVars})

    def __init__(self, base_path: Path) -> None:
        # TODO implement download logic
        self.data_info = analyze_data(base_path)
        pprint(self.data_info.common_values)
        pprint(self.data_info.symbol_id_table)


        # dict_size = len(data_info.symbol_id_table) + 3

        symbol_table_tr, value_table_tr = load_data(self.data_info, is_train=True)
        symbol_table_tst, value_table_tst = load_data(self.data_info, is_train=False)

        self.array_dict: Dict[str, npt.ArrayLike] = {
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

    def get_fixed_datasets(self, port: Port, variable: Variable) -> Dict[FixedSubsetType, Data[TabularDataContentNp]]:
        assert port in self.ports
        n_samples_tr = self.data_info.n_rows_tr
        n_samples_tst = self.data_info.n_rows_tst
        n_samples = n_samples_tr + n_samples_tst

        # noinspection PyTypeChecker
        # because pyCharm sucks https://youtrack.jetbrains.com/issue/PY-49439
        fixed_datasets: Dict[FixedSubsetType, UciIncomeData] = {
            FixedTrain: UciIncomeData(port,
                                      variable,
                                      FixedSubset(FixedTrain, n_samples_tr),
                                      TabularDataContentNp(
                                          symbols=self.array_dict[f'tr_symbol'],
                                          values=self.array_dict[f'tr_value']
                                      ))
        }
        # noinspection PyTypeChecker
        # because pyCharm sucks https://youtrack.jetbrains.com/issue/PY-49439
        return fixed_datasets

    def retrieve(self, query: DataQuery) -> Mapping[Port, UciIncomeDataPool]:
        assert all(port in self.ports for port in query)

        return {
            port: UciIncomeDataPool(
                array_dict=self.array_dict,
                data_info=self.data_info,
                fixed_datasets=self.get_fixed_datasets(port, variable_protocol),
                port=port,
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
