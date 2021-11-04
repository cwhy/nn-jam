import numpy as np

from supervised_benchmarks.numpy_utils import ordinal_from_1hot, ordinal_to_1hot

a = np.array([1, 2, 3, 4, 5, 2, 1])
b = np.array([[1, 2, 3], [5, 2, 1]])

ap = ordinal_to_1hot(a)
app = ordinal_from_1hot(ap)

bp = ordinal_to_1hot(b)
bpp = ordinal_from_1hot(bp)

assert np.all(app == a)
assert np.all(bpp == b)
print(app, bpp)


