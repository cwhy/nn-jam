"""

Partial of the code are from github.com/pytorch/vision
  with the following license:

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
import codecs
from pathlib import Path

import numpy as np


SN3_PASCALVINCENT_TYPEMAP = {
    8: (np.uint8, np.uint8),
    9: (np.int8, np.int8),
    11: (np.dtype('>i2'), 'i2'),
    12: (np.dtype('>i4'), 'i4'),
    13: (np.dtype('>f4'), 'f4'),
    14: (np.dtype('>f8'), 'f8')
}


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)


def read_sn3_pascalvincent_ndarray(path: Path, strict: bool = True) -> npt.NDArray:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    print(nd, ty, path)
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[0], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.astype(m[1], copy=False).reshape(*s)


def read_label_file(path: Path) -> npt.NDArray:
    x = read_sn3_pascalvincent_ndarray(path, strict=False)
    assert (x.dtype == np.uint8)
    assert (len(x.shape) == 1)
    return x.astype(dtype=np.int64)


def read_image_file(path: Path) -> npt.NDArray:
    x = read_sn3_pascalvincent_ndarray(path, strict=False)
    assert (x.dtype == np.uint8)
    assert (len(x.shape) == 3)
    return x
