# %%
import jax
import numpy as np

from scripts.nanogpt.utils import infinite_jax_keys
import jax.numpy as xp
from jax import random, vmap, Array

key_gen = infinite_jax_keys(0)
B, T, C = 4, 8, 2
x = random.normal(next(key_gen), (B, T, C))
print(x.shape)

# %%
wei = xp.tril(xp.ones((T, T)))
wei1 = wei / xp.sum(wei, axis=1, keepdims=True)
xbow = wei1 @ x

# %%
tril = xp.tril(xp.ones((T, T)))
wei = xp.zeros((T, T))
wei = xp.where(tril == 0, -xp.inf, wei)
wei2 = jax.nn.softmax(wei, axis=-1)

print(xp.allclose(wei1, wei2))

# %%
head_size = 16
query = random.normal(next(key_gen), (C, head_size))
key = random.normal(next(key_gen), (C, head_size))


def attn(q: Array, k: Array, _x: Array) -> Array:
    return x @ q @ (_x @ k).T


wei = vmap(attn, (None, None, 0))(query, key, x)
tril = xp.tril(xp.ones((T, T)))
wei = xp.where(tril == 0, -xp.inf, wei)
wei2 = jax.nn.softmax(wei, axis=-1)
wei2np = np.array(wei2)