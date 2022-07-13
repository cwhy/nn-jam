from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Literal, Mapping, FrozenSet, Dict

import numpy.typing as npt
from numpy.typing import NDArray

from supervised_benchmarks.dataset_protocols import Subset, DataQuery, DataSubset, FixedSubset, FixedSubsetType, \
    FixedTrain, FixedTest
from supervised_benchmarks.ports import Port, Input, Output
from supervised_benchmarks.uci_income.consts import row_width, n_tokens, TabularDataInfo, get_anynet_feature, \
    AnyNetDiscrete, \
    AnyNetContinuous, AnyNetDiscreteOut
from supervised_benchmarks.uci_income.utils import analyze_data, load_data
from variable_protocols.variables import Variable

name: Literal["UciIncome"] = "UciIncome"


# support column names


class UciIncomeDataPool(NamedTuple):
    data_info: TabularDataInfo
    array_dict: Mapping[str, npt.NDArray]
    fixed_subsets: Mapping[FixedSubsetType, DataSubset]
    query: DataQuery

    def subset(self, subset: Subset) -> DataSubset:
        raise NotImplementedError

        # return UciIncomeData(self.port, self.tgt_var, subset, data_array)


uci_income_in_anynet_discrete = get_anynet_feature(n_tokens, row_width - 1, continuous=False)
uci_income_in_anynet_continuous = get_anynet_feature(n_tokens, row_width - 1, continuous=True)
uci_income_out_anynet_discrete = get_anynet_feature(n_tokens, 1, continuous=False)


class UciIncome:
    @property
    def exports(self) -> FrozenSet[Port]:
        return frozenset({AnyNetDiscrete, AnyNetContinuous, AnyNetDiscreteOut})

    def __init__(self, base_path: Path) -> None:
        # TODO implement download logic
        # TODO implement checkvar logic after VarProtocols are revamped
        self.data_info = analyze_data(base_path)

        dict_size = len(self.data_info.symbol_id_table) + 3

        symbol_table_tr, value_table_tr = load_data(self.data_info, is_train=True)
        symbol_table_tst, value_table_tst = load_data(self.data_info, is_train=False)

        self.array_dict: Dict[str, npt.NDArray] = {
            'tr_symbol': symbol_table_tr,
            'tr_value': value_table_tr,
            'tst_symbol': symbol_table_tst,
            'tst_value': value_table_tst
        }

        self.protocols: Mapping[Port, Variable] = {
            AnyNetDiscrete: uci_income_in_anynet_discrete,
            AnyNetContinuous: uci_income_in_anynet_continuous,
            AnyNetDiscreteOut: uci_income_out_anynet_discrete
        }

    @property
    def name(self) -> Literal['UciIncome']:
        return name

    def get_fixed_datasets(self, query: DataQuery) -> Mapping[FixedSubsetType, DataSubset]:
        assert set(query.keys()).issubset(self.exports)
        n_samples_tr = self.data_info.n_rows_tr
        n_samples_tst = self.data_info.n_rows_tst

        def get_data(port: Port, is_train: bool) -> NDArray:
            # TODO test, may not right
            prefix = 'tr' if is_train else 'tst'
            if port is AnyNetDiscrete:
                return self.array_dict[f'{prefix}_symbol'][:, :-1]
            elif port is AnyNetContinuous:
                return self.array_dict[f'{prefix}_value']
            elif port is AnyNetDiscreteOut:
                return self.array_dict[f'{prefix}_symbol'][:, -1]
            else:
                raise ValueError(f'Unknown port {port}')

        fixed_datasets: Dict[FixedSubsetType, DataSubset] = {
            FixedTrain: DataSubset(query,
                                   FixedSubset(FixedTrain, n_samples_tr),
                                   {port: get_data(port, is_train=True) for port in query}),
            FixedTest: DataSubset(query,
                                  FixedSubset(FixedTest, n_samples_tst),
                                  {port: get_data(port, is_train=False) for port in query})
        }
        return fixed_datasets

    def retrieve(self, query: DataQuery) -> UciIncomeDataPool:
        assert all(port in self.exports for port in query)

        return UciIncomeDataPool(
            array_dict=self.array_dict,
            data_info=self.data_info,
            fixed_subsets=self.get_fixed_datasets(query),
            query=query)


class UciIncomeDataConfig(NamedTuple):
    base_path: Path
    query: DataQuery
    type: Literal['DataConfig'] = 'DataConfig'

    def get_data(self) -> UciIncomeDataPool:
        return UciIncome(self.base_path).retrieve(self.query)
