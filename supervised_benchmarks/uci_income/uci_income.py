from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import NamedTuple, Literal, Mapping, FrozenSet, Dict

import numpy.typing as npt

from supervised_benchmarks.dataset_protocols import Subset, DataQuery, DataSubset, FixedSubset, FixedSubsetType, \
    FixedTrain, FixedTest
from supervised_benchmarks.ports import Port, Input, Output
from supervised_benchmarks.uci_income.consts import row_width, TabularDataInfo, get_anynet_feature, AnyNetDiscrete, \
    AnyNetContinuous, AnyNetDiscreteOut
from supervised_benchmarks.uci_income.utils import analyze_data, load_data
from variable_protocols.variables import Variable

name: Literal["UciIncome"] = "UciIncome"


# support column names


class UciIncomeDataPool(NamedTuple):
    data_info: TabularDataInfo
    array_dict: Mapping[str, npt.NDArray]
    fixed_datasets: Mapping[FixedSubsetType, DataSubset]
    port: Port
    src_var: Variable
    tgt_var: Variable

    def subset(self, subset: Subset) -> DataSubset:
        assert self.src_var == self.tgt_var
        raise NotImplementedError

        # return UciIncomeData(self.port, self.tgt_var, subset, data_array)


dict_size = 10

uci_income_in_anynet_discrete = get_anynet_feature(dict_size, row_width - 1, continuous=False)
uci_income_in_anynet_continuous = get_anynet_feature(dict_size, row_width - 1, continuous=True)
uci_income_out_anynet_discrete = get_anynet_feature(dict_size, 1, continuous=False)


class UciIncome:
    @property
    def ports(self) -> FrozenSet[Port]:
        return frozenset({AnyNetDiscrete, AnyNetContinuous, AnyNetDiscreteOut})

    def __init__(self, base_path: Path) -> None:
        # TODO implement download logic
        self.data_info = analyze_data(base_path)
        pprint(self.data_info.common_values)
        pprint(self.data_info.symbol_id_table)

        # dict_size = len(data_info.symbol_id_table) + 3

        symbol_table_tr, value_table_tr = load_data(self.data_info, is_train=True)
        symbol_table_tst, value_table_tst = load_data(self.data_info, is_train=False)

        self.array_dict: Dict[str, npt.NDArray] = {
            'tr_symbol': symbol_table_tr,
            'tr_value': value_table_tr,
            'tst_symbol': symbol_table_tst,
            'tst_value': value_table_tst
        }

        # noinspection PyTypeChecker
        # because pyCharm sucks
        self.protocols: Mapping[Port, Variable] = {
            AnyNetDiscrete: uci_income_in_anynet_discrete,
            AnyNetContinuous: uci_income_in_anynet_continuous,
            AnyNetDiscreteOut: uci_income_out_anynet_discrete
        }

    @property
    def name(self) -> Literal['UciIncome']:
        return name

    def get_fixed_datasets(self, variable: Variable) -> Mapping[FixedSubsetType, DataSubset]:
        n_samples_tr = self.data_info.n_rows_tr
        n_samples_tst = self.data_info.n_rows_tst

        fixed_datasets: Dict[FixedSubsetType, DataSubset] = {
            FixedTrain: DataSubset(variable,
                                   FixedSubset(FixedTrain, n_samples_tr),
                                   {
                                       AnyNetDiscrete: self.array_dict['tr_symbol'][:-1],
                                       AnyNetContinuous: self.array_dict['tr_value'],
                                       AnyNetDiscreteOut: self.array_dict['tr_value'][-1, :]
                                   }),
            FixedTest: DataSubset(variable,
                                  FixedSubset(FixedTest, n_samples_tst),
                                  {
                                      AnyNetDiscrete: self.array_dict['tr_symbol'][:-1],
                                      AnyNetContinuous: self.array_dict['tr_value'],
                                      AnyNetDiscreteOut: self.array_dict['tr_value'][-1, :]
                                  })
        }
        return fixed_datasets

    def retrieve(self, query: DataQuery) -> UciIncomeDataPool:
        assert all(port in self.ports for port in query)

        return UciIncomeDataPool(
                array_dict=self.array_dict,
                data_info=self.data_info,
                fixed_datasets=self.get_fixed_datasets(variable_protocol),
                src_var=self.protocols[port],
                tgt_var=variable_protocol)


class UciIncomeDataConfig(NamedTuple):
    base_path: Path
    port_vars: DataQuery
    type: Literal['DataConfig'] = 'DataConfig'

    def get_data(self) -> UciIncomeDataPool:
        return UciIncome(self.base_path).retrieve(self.port_vars)
