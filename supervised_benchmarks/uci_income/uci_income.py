from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Literal, Mapping, FrozenSet, Dict

from numpy.typing import NDArray

from supervised_benchmarks.dataset_protocols import Subset, PortSpecs, DataSubset, FixedSubset, FixedSubsetType, \
    FixedTrain, FixedTest
from supervised_benchmarks.ports import Port
from supervised_benchmarks.uci_income.consts import TabularDataInfo, AnyNetDiscrete, \
    AnyNetContinuous, AnyNetDiscreteOut, variable_names
from supervised_benchmarks.uci_income.utils import analyze_data, load_data
from variable_protocols.labels import Labels
from variable_protocols.tensorhub import TensorHub

name: Literal["UciIncome"] = "UciIncome"


# support column names


class UciIncomeDataPool(NamedTuple):
    data_info: TabularDataInfo
    array_dict: Mapping[str, NDArray]
    fixed_subsets: Mapping[FixedSubsetType, DataSubset]
    query: PortSpecs

    def subset(self, subset: Subset) -> DataSubset:
        raise NotImplementedError

        # return UciIncomeData(self.port, self.tgt_var, subset, data_array)


class UciIncome:
    def __init__(self, base_path: Path) -> None:
        # TODO implement download logic
        self.data_info = analyze_data(base_path)
        print(self.data_info)

        #      self._format =

        symbol_table_tr, value_table_tr = load_data(self.data_info, is_train=True)
        symbol_table_tst, value_table_tst = load_data(self.data_info, is_train=False)

        self.array_dict: Dict[str, NDArray] = {
            'tr_symbol': symbol_table_tr,
            'tr_value': value_table_tr,
            'tst_symbol': symbol_table_tst,
            'tst_value': value_table_tst
        }

    @property
    def data_format(self) -> Literal['UciIncome']:
        return self._format

    @property
    def name(self) -> Literal['UciIncome']:
        return name

    def get_fixed_datasets(self, query: PortSpecs) -> Mapping[FixedSubsetType, DataSubset]:
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
            FixedTrain: DataSubset(FixedSubset(FixedTrain, n_samples_tr),
                                   {port: get_data(port, is_train=True) for port in query}),
            FixedTest: DataSubset(FixedSubset(FixedTest, n_samples_tst),
                                  {port: get_data(port, is_train=False) for port in query})
        }
        return fixed_datasets

    def retrieve(self, query: PortSpecs) -> UciIncomeDataPool:
        assert all(port in self.exports for port in query)

        return UciIncomeDataPool(
            array_dict=self.array_dict,
            data_info=self.data_info,
            fixed_subsets=self.get_fixed_datasets(query),
            query=query)


class UciIncomeDataConfig(NamedTuple):
    base_path: Path
    port_allocation: Mapping[Labels, Port]
    type: Literal['DataConfig'] = 'DataConfig'

    def get_data(self) -> UciIncomeDataPool:
        data_class = UciIncome(self.base_path)
        query: dict[Port, TensorHub] = {}
        data_class.data_info
        for feature, port in self.port_allocation.items():
            query[port] = query.get(port, TensorHub.empty()) + feature
        return data_class.retrieve(query)
