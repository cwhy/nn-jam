from collections import defaultdict
from pathlib import Path

import numpy as np
import csv

from supervised_benchmarks.uci_income.consts import FLOAT_OFFSET, VALUE_SYMBOL, variable_names, row_width, \
    TabularDataInfo


def analyze_data():
    is_digits = [True] * row_width
    number_counts = [defaultdict(int) for _ in range(row_width)]
    string_baskets = [set() for _ in range(row_width)]
    tr_path = Path('/Data/uci/adult.data')
    tst_path = Path('/Data/uci/adult.test')

    with tr_path.open('r') as f:
        uci_reader = csv.reader(f, delimiter=',')
        n_rows_tr = 0
        for row in uci_reader:
            if len(row) > 0:
                n_rows_tr += 1
                assert len(row) == row_width
                for i, entry in enumerate(row):
                    entry = entry.strip()
                    if entry.isnumeric() and is_digits[i]:
                        number_counts[i][float(entry)] += 1
                    else:
                        is_digits[i] = False
                        string_baskets[i].add(entry)

    len_string_baskets = [len(b) for b in string_baskets]

    is_cat = [b_l != 0 and b_l < n_rows_tr / 10 for b_l in len_string_baskets]

    all_symbol = set()
    for i, basket in enumerate(string_baskets):
        if is_cat[i]:
            all_symbol |= basket
    all_symbols_list = list(all_symbol)
    symbol_id_table = {v: i for i, v in enumerate(all_symbols_list)}

    special_values = {}
    for i, s in enumerate(number_counts):
        if is_digits[i]:
            special = max(s.values())
            if special > sum(sorted(s.values())[-5:-1]):
                special_values[f"{variable_names[i]}_offset_{int(special)}"] = special
                symbol_id_table[f"{variable_names[i]}_offset_{int(special)}"] = len(symbol_id_table)

    with tst_path.open('r') as f:
        uci_reader = csv.reader(f, delimiter=',')
        n_rows_tst = 0
        # skip header
        next(uci_reader)
        for row in uci_reader:
            if len(row) > 0:
                assert len(row) == row_width
                n_rows_tst += 1

    return TabularDataInfo(n_rows_tr, n_rows_tst, tr_path, tst_path, symbol_id_table, is_digits, special_values)


def load_data(info: TabularDataInfo, is_train: bool):
    if is_train:
        n_rows = info.n_rows_tr
        path = info.tr_path
    else:
        n_rows = info.n_rows_tst
        path = info.tst_path
    symbol_table = FLOAT_OFFSET * np.ones((n_rows, row_width), dtype=int)
    value_table = VALUE_SYMBOL * np.ones((n_rows, row_width), dtype=float)

    def load_symbol_(_entry: str, row_id: int, row_pos: int):
        symbol_table[row_id, row_pos] = info.symbol_id_table[_entry]

    def load_number_(_entry: float, row_id: int, row_pos: int):
        offset_name = f"{variable_names[row_pos]}_offset_{int(_entry)}"
        if info.special_values.get(offset_name, None) == _entry:
            symbol_table[row_id, row_pos] = info.symbol_id_table[offset_name]
        else:
            value_table[row_id, row_pos] = _entry

    with path.open('r') as f:
        uci_reader = csv.reader(f, delimiter=',')
        if not is_train:
            next(uci_reader)
        n_rows = 0
        for row in uci_reader:
            if len(row) > 0:
                assert len(row) == row_width
                for i, entry in enumerate(row):
                    entry = entry.strip().strip('.')
                    if info.is_digits[i]:
                        load_number_(float(entry), n_rows, i)
                    else:
                        load_symbol_(entry, n_rows, i)
                n_rows += 1
    return symbol_table, value_table
