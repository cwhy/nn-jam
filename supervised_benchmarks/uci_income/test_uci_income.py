from pathlib import Path

from supervised_benchmarks.dataset_protocols import AllVars
from supervised_benchmarks.uci_income.uci_income import UciIncomeDataConfig, uci_income_all_anynet, FixedTrain, \
    FixedTest


def test_get_data():
    data_config = UciIncomeDataConfig(base_path=Path(""), port_vars={AllVars: uci_income_all_anynet})
    data = data_config.get_data()
    assert AllVars in data
    data_pool = data[AllVars]
    assert data_pool.port == AllVars
    assert data_pool.src_var == data_pool.tgt_var == uci_income_all_anynet
    print(data_pool.subset(FixedTrain))
    print(data_pool.subset(FixedTest))

    assert True
