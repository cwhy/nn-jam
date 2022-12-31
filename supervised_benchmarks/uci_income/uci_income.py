from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Literal, Mapping, FrozenSet, Dict

import polars as pl
from numpy.typing import NDArray

from supervised_benchmarks.dataset_protocols import Subset, PortSpecs, DataSubset, FixedSubset, FixedSubsetType, \
    FixedTrain, FixedTest
from supervised_benchmarks.ports import Port
from supervised_benchmarks.tabular_utils import anynet_load_polars, AnyNetStrategyConfig, TabularColumnsConfig, \
    parse_polars, anynet_get_discrete, polar_select_discrete, anynet_get_continuous
from supervised_benchmarks.uci_income.consts import TabularDataInfo, AnyNetDiscrete, \
    AnyNetContinuous, AnyNetDiscreteOut, variable_names
from supervised_benchmarks.uci_income.utils import analyze_data, load_data
from variable_protocols.labels import Labels
from variable_protocols.tensorhub import TensorHub

name: Literal["UciIncome"] = "UciIncome"


# support column names


class UciIncomeDataPool(NamedTuple):
    data_info: TabularDataInfo
    polars_dict: Mapping[str, pl.DataFrame]
    fixed_subsets: Mapping[FixedSubsetType, DataSubset]
    query: PortSpecs

    def subset(self, subset: Subset) -> DataSubset:
        raise NotImplementedError

        # return UciIncomeData(self.port, self.tgt_var, subset, data_array)



class UciIncome:
    def __init__(self, base_path: Path, column_config: TabularColumnsConfig) -> None:
        # TODO implement download logic
        self.data_info = analyze_data(base_path)
        print(self.data_info)

        tr_data_polars = pl.read_csv(self.data_info.tr_path, delimiter=',', has_header=False,
                                     new_columns=variable_names)
        tst_data_polars = pl.read_csv(self.data_info.tst_path, delimiter=',', has_header=False,
                                      new_columns=variable_names, skip_rows=1)

        self.anynet_config = AnyNetStrategyConfig()
        self.column_config = column_config

        self.testing_columns = variable_names[-1:]

        # TODO: deal with training-testing inconsistency
        self._format = parse_polars(column_config, tr_data_polars)
        self.polars_dict: Dict[str, pl.DataFrame] = {
            'tr': tr_data_polars,
            'tst': tst_data_polars
        }

    @property
    def data_format(self) -> TensorHub:
        return self._format

    @property
    def name(self) -> Literal['UciIncome']:
        return name

    def get_fixed_datasets(self, query: PortSpecs) -> Mapping[FixedSubsetType, DataSubset]:
        n_samples_tr = self.data_info.n_rows_tr
        n_samples_tst = self.data_info.n_rows_tst

        def get_data(port: Port, is_train: bool) -> NDArray:
            data_polars = self.polars_dict['tr' if is_train else 'tst']

            if port is AnyNetDiscrete:
                return anynet_get_discrete(data_polars, self.data_format, self.testing_columns)
            elif port is AnyNetContinuous:
                return anynet_get_continuous(data_polars, self.data_format, self.testing_columns)
            elif port is AnyNetDiscreteOut:
                out = polar_select_discrete(data_polars, self.testing_columns).to_numpy()
                return out
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

        return UciIncomeDataPool(
            polars_dict=self.polars_dict,
            data_info=self.data_info,
            fixed_subsets=self.get_fixed_datasets(query),
            query=query)


class UciIncomeDataConfig(NamedTuple):
    base_path: Path
    column_config: TabularColumnsConfig
    type: Literal['DataConfig'] = 'DataConfig'

    def get_data(self, query: PortSpecs) -> UciIncomeDataPool:
        data_class = UciIncome(self.base_path, column_config=self.column_config)
        return data_class.retrieve(query)
