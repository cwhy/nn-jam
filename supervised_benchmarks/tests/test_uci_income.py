from pathlib import Path

from supervised_benchmarks.dataset_protocols import FixedTrain, FixedTest
from supervised_benchmarks.uci_income.consts import AnyNetDiscrete
from supervised_benchmarks.uci_income.uci_income import UciIncomeDataConfig, uci_income_in_anynet_discrete


def test_get_data():
    data_config = UciIncomeDataConfig(base_path=Path('/Data/uci'),
                                      query={AnyNetDiscrete: uci_income_in_anynet_discrete})
    data_pool = data_config.get_data()
    assert data_pool.src_var == data_pool.tgt_var == uci_income_in_anynet_discrete
    tr = data_pool.fixed_datasets[FixedTrain]
    tst = data_pool.fixed_datasets[FixedTest]
    assert AnyNetDiscrete in tr
    assert AnyNetDiscrete in tst
    assert tr.protocol == uci_income_in_anynet_discrete

    assert True
