from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Literal, Mapping, FrozenSet, Dict

import numpy as np
import numpy.typing as npt
from tqdm import trange
from variable_protocols.variables import Variable, ordinal, dim, var_scalar, var_tensor, var_group

from supervised_benchmarks.dataset_protocols import Port, Subset, DataQuery, Input, Output, \
    Data, DataPortMap, OutputOptions, Context, FixedSubset
from supervised_benchmarks.dataset_utils import download_resources, get_data_dir
from supervised_benchmarks.download_utils import check_integrity

name: Literal["IRaven"] = "IRaven"

tasks = {'center_single',
         'distribute_four',
         'distribute_nine',
         'in_center_single_out_center_single',
         'in_distribute_four_out_center_single',
         'left_center_single_right_center_single',
         'up_center_single_down_center_single'}

n_samples_tr = 6000
n_samples_val = 2000
n_samples_tst = 2000
n_samples = n_samples_tr + n_samples_val + n_samples_tst


class IravenData(NamedTuple):
    port: Port
    protocol: Variable
    subset: Subset
    content: npt.NDArray


class IravenDataPool(NamedTuple):
    array_dict: Mapping[str, npt.NDArray]
    port: Port
    src_var: Variable
    tgt_var: Variable

    def subset(self, subset: Subset) -> Data[npt.NDArray]:
        assert self.src_var == self.tgt_var
        if self.port is OutputOptions:
            data_array = self.array_dict['images'][subset.indices, 8:, :, :]
        elif self.port is Context:
            data_array = self.array_dict['images'][subset.indices, :8, :, :]
        else:
            if self.port is Input:
                port_tag = 'images'
                data_array = self.array_dict[port_tag][subset.indices, :, :, :]
            else:
                assert self.port is Output
                port_tag = 'targets'
                data_array = self.array_dict[port_tag][subset.indices]
        # noinspection PyTypeChecker
        # because pyCharm sucks
        return IravenData(self.port, self.tgt_var, subset, data_array)


n_rows = 3
n_cols = 3
n_options = 8
n_pics = n_rows * n_cols - 1 + n_options  # 16
H = 160
W = 160
# noinspection PyTypeChecker
# because pyCharm sucks
pic = var_tensor(ordinal(256), {dim("h", H), dim("w", W)})
# noinspection PyTypeChecker
# because pyCharm sucks
iraven_in_raw = var_group({
    var_tensor(pic, {dim("row", n_rows), dim("col", n_cols)}),
    var_tensor(pic, {dim("options", n_options)})})
# noinspection PyTypeChecker
# because pyCharm sucks
iraven_out_raw = var_scalar(ordinal(n_options))

FixedTrain = FixedSubset('FixedTrain', list(filter(lambda x: x % 10 <= 5, range(n_samples))))
FixedValidation = FixedSubset('FixedValidation', list(filter(lambda x: x % 10 in (8, 9), range(n_samples))))
FixedTest = FixedSubset('FixedTest', list(filter(lambda x: x % 10 in (6, 7), range(n_samples))))
FixedAll = FixedSubset('All', list(range(n_samples)))


def get_iraven_(base_path: Path, version: str, size: int, task: str) -> Dict[str, npt.NDArray]:
    assert size % 5 == 0
    resources = [(f"iraven{size}v{version}.zip", None)]
    version_tag = f"{size}v{version}"
    processed_cache = get_data_dir(base_path, name, 'processed').joinpath(version_tag, 'array_dict.npz')
    if not check_integrity(processed_cache):
        mirrors = [f"https://github.com/cwhy/i-raven/releases/download/{version}"]
        download_resources(base_path, name, resources, mirrors, f"{size}v{version}")
        data_path = get_data_dir(base_path, name, 'raw').joinpath(version_tag, "dataset", task)
        data_dict = {
            "images": np.empty((size, n_pics, H, W), dtype=np.uint8),
            "targets": np.empty(size, dtype=np.uint8)
        }
        for index in trange(size):
            if index % 10 <= 5:
                subset = 'train'
            elif index % 10 in (6, 7):
                subset = 'val'
            else:
                assert index % 10 in (8, 9)
                subset = 'test'

            data_file_path = data_path.joinpath(f"RAVEN_{index}_{subset}.npz")
            with np.load(str(data_file_path)) as data:
                data_dict["images"][index, :, :, :] = data['image']
                data_dict["targets"][index] = data['target']
        processed_path = get_data_dir(base_path, name, 'processed').joinpath(version_tag)
        processed_path.mkdir(exist_ok=True)
        np.savez_compressed(processed_cache, **data_dict)
        return data_dict
    else:
        print("loading dataset... please wait...")
        return dict(np.load(str(processed_cache)))


class Iraven:
    @property
    def ports(self) -> FrozenSet[Port]:
        return frozenset({Input, Output})

    def __init__(self, base_path: Path, version: str, size: int, task: str) -> None:
        self.array_dict: Dict[str, npt.NDArray] = get_iraven_(base_path, version, size, task)
        # noinspection PyTypeChecker
        # because pyCharm sucks
        self.protocols: Mapping[str, Variable] = {
            Input: iraven_in_raw,
            Output: iraven_out_raw
        }

    @property
    def name(self) -> Literal['IRaven']:
        return name

    def retrieve(self, query: DataQuery) -> Mapping[Port, IravenDataPool]:
        assert all(port in self.ports for port in query)
        return {
            port: IravenDataPool(
                self.array_dict,
                port,
                src_var=self.protocols[port],
                tgt_var=variable_protocol)
            for port, variable_protocol in query.items()
        }


# noinspection PyTypeChecker
# Because pycharm sucks
class IravenDataConfig(NamedTuple):
    task: str
    base_path: Path
    version: str
    size: int
    port_vars: DataQuery
    type: Literal['DataConfig'] = 'DataConfig'

    def get_data(self) -> DataPortMap:
        return Iraven(self.base_path, self.version, self.size, self.task).retrieve(self.port_vars)
