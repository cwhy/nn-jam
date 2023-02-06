# %% imports
import optax
import os
from collections import Counter
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKeyArray

from jax_make.component_protocol import FixedPipeline
from jax_make.components.embedding import Embeddings
from jax_make.params import make_weights, ArrayTreeMapping, get_arr
from jax_make.utils.functions import softmax_cross_entropy_with_integer_labels

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"

# %% load text
dataset = "english"  # "play"
path = Path("/Data/nlp/") / dataset
books = [f for f in os.listdir(path) if f.endswith('.txt')]


def read_book_file(filename: str):
    print(f"reading {filename}...")
    try:
        with open(path / filename, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(path / filename, 'r', encoding='gb2312') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(path / filename, 'r', encoding='gbk') as f:
                    return f.read()
            except UnicodeDecodeError:
                try:
                    with open(path / filename, 'r', encoding='big5') as f:
                        return f.read()
                except UnicodeDecodeError:
                    try:
                        with open(path / filename, 'r', encoding='utf-16') as f:
                            return f.read()
                    except UnicodeDecodeError:
                        try:
                            with open(path / filename, 'r', encoding='gb18030') as f:
                                return f.read()
                        except UnicodeDecodeError:
                            raise Exception(f"Failed to read {filename} with many encodings")


text = "\n\n".join(f"{book_name}\n\n {read_book_file(book_name)}" for book_name in books)
chars = [ch for ch, c in Counter(text).most_common()]
print(chars[:100])
vocab_size = len(chars)

# %% tokenize
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}


def encode(_text: str):
    return [stoi[c] for c in _text]


def decode(_encoded: list):
    return "".join(itos[i] for i in _encoded)


print(encode("hii there"))
print(decode(encode("hii there")))

# %% get into jax and save encoded_jax

cache_path = path / 'encoded_jax.npy'
try:
    with open(cache_path, 'rb') as f:
        encoded_jax = jnp.load(f)
except FileNotFoundError:
    encoded = encode(text)
    encoded_jax = jnp.array(encoded, dtype=jnp.int16)
    print(encoded_jax.shape, encoded_jax.dtype)
    with open(cache_path, 'wb') as fw:
        jnp.save(fw, encoded_jax)

# %% train validate split
print(encoded_jax[:1000])
n = int(len(encoded_jax) * 0.9)
train_data = encoded_jax[:n]
valid_data = encoded_jax[n:]

# %% data loader
block_size = 8
batch_size = 32


def infinite_jax_keys(seed: int):
    init_key = jax.random.PRNGKey(seed)
    while True:
        init_key, key = jax.random.split(init_key)
        yield key


key_gen = infinite_jax_keys(0)


def get_batch(data: Array, _key: PRNGKeyArray) -> Dict[str, Array]:
    """optimized version of
        def get_batch(data: Array, _key: PRNGKeyArray) -> Dict[str, Array]:
            ix = jax.random.randint(key=_key, minval=0, maxval=len(data) - block_size, shape=(batch_size,))
            return {
                'inputs': jnp.stack([data[i: i + block_size] for i in ix]),
                'targets': jnp.stack([data[i + 1: i + block_size + 1] for i in ix])
            }
    """
    ix = jax.random.randint(key=_key, minval=0, maxval=len(data) - block_size, shape=(batch_size,))
    inputs = data[(ix[:, jnp.newaxis] + jnp.arange(block_size)[jnp.newaxis, :])]
    targets_ = data[(ix[:, jnp.newaxis] + jnp.arange(1, block_size + 1)[jnp.newaxis, :])]
    return {'inputs': inputs, 'targets': targets_}


key = next(key_gen)
results = get_batch(train_data, key)
xb, yb = results['inputs'], results['targets']

# %% bigram model

embeddings = Embeddings.make(Embeddings(dict_size=vocab_size, dim_model=vocab_size, dict_init_scale=0.01))

# def model(x: Array):
init_weights = make_weights(embeddings.weight_params)
logits = embeddings.fixed_pipeline(init_weights, xb[0])
print(logits.shape)


# %%
def batch_fy(fixed_pipeline: FixedPipeline) -> FixedPipeline:
    return jax.vmap(fixed_pipeline, in_axes=(None, 0), out_axes=0)


# combine first two dimensions of a numpy array
def flatten_token(array: Array) -> Array:
    return jnp.concatenate(array)


logits = flatten_token(batch_fy(embeddings.fixed_pipeline)(init_weights, xb))
targets = flatten_token(yb)
loss = softmax_cross_entropy_with_integer_labels(logits, targets).mean()
guess_loss = -jnp.log(1 / (vocab_size + 1))

print(loss, guess_loss)


def loss_fn(params: ArrayTreeMapping, batch_: Dict[str, Array]) -> Array:
    logits_ = flatten_token(batch_fy(embeddings.fixed_pipeline)(params, batch_['inputs']))
    targets_ = flatten_token(batch_['targets'])
    return softmax_cross_entropy_with_integer_labels(logits_, targets_).mean()


print(loss_fn(init_weights, results))


# %% generation
def generate(weights_: ArrayTreeMapping, idx: int, max_new_tokens: int) -> Array:
    new_tokens = [idx]
    idx_next = idx
    for _ in range(max_new_tokens):
        logits_ = embeddings.fixed_pipeline(weights_, jnp.array([idx_next]))
        probs = jax.nn.softmax(logits_).flatten()
        idx_next = jax.random.choice(key=next(key_gen), a=jnp.arange(0, vocab_size), p=probs, shape=()).item()
        new_tokens.append(idx_next)
    return jnp.array(new_tokens, dtype=jnp.int16)


generated = decode(generate(init_weights, 0, 100).tolist())
print(generated)

# %% train 1 step reset
weights = init_weights

# %% train 1 step
key = next(key_gen)
batch = get_batch(train_data, key)
print(loss_fn(weights, batch))

grads = jax.grad(loss_fn)(weights, batch)
optimiser = optax.adamw(1e-3)
opt_state = optimiser.init(weights)
updates, opt_state = optimiser.update(grads, opt_state, weights)
weights = optax.apply_updates(weights, updates)

print(loss_fn(weights, batch))

# %% train many step

weights = init_weights
optimiser = optax.adamw(1e-4)
opt_state = optimiser.init(weights)
# %% train many step continued
grad_static = jax.jit(jax.grad(loss_fn))
n_steps = 20000
keys = jax.random.split(next(key_gen), n_steps)

for step in range(n_steps):
    batch = get_batch(train_data, keys[step])
    if step % 50 == 0:
        print(loss_fn(weights, batch))
    grads = grad_static(weights, batch)
    updates, opt_state = optimiser.update(grads, opt_state, weights)
    weights = optax.apply_updates(weights, updates)

    if step % 50 == 0:
        print(loss_fn(weights, batch))
    if step % 50 == 0:
        generated = decode(generate(weights, 0, 100).tolist())
        print(generated)

print(weights)

# %% debug weights
