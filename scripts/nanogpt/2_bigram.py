# %% imports
from __future__ import annotations
import os
from typing import Dict, Callable

import jax
import jax.numpy as xp
import optax
from jax import Array

from jax_make.components.embedding import Embeddings
from jax_make.params import make_weights, ArrayTreeMapping, RNGKey
from jax_make.utils.functions import softmax_cross_entropy_with_integer_labels
from scripts.nanogpt.my_nlp_dataset import load_jax_cached
from scripts.nanogpt.utils import infinite_jax_keys, flatten_token, batch_fy, jax_calc_updates

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"

block_size = 8
batch_size = 32
max_iters = 20000
eval_interval = 300
learning_rate_ = 1e-3
eval_iters = 200
seed = 0
key_gen = infinite_jax_keys(seed)
init_scale_ = 0.01


# dataset = "english"
dataset = "play"
encoded_jax, encode, decode, vocab_size_ = load_jax_cached(dataset=dataset)

n = int(len(encoded_jax) * 0.9)
train_data = encoded_jax[:n]
valid_data = encoded_jax[n:]


def get_batch(data: Array, rng_key: RNGKey) -> Dict[str, Array]:
    """optimized version of
        def get_batch(data: Array, _key: PRNGKeyArray) -> Dict[str, Array]:
            ix = jax.random.randint(key=_key, minval=0, maxval=len(data) - block_size, shape=(batch_size,))
            return {
                'inputs': jnp.stack([data[i: i + block_size] for i in ix]),
                'targets': jnp.stack([data[i + 1: i + block_size + 1] for i in ix])
            }
    """
    ix = jax.random.randint(key=rng_key, minval=0, maxval=len(data) - block_size, shape=(batch_size,))
    inputs = data[(ix[:, xp.newaxis] + xp.arange(block_size)[xp.newaxis, :])]
    targets_ = data[(ix[:, xp.newaxis] + xp.arange(1, block_size + 1)[xp.newaxis, :])]
    return {'inputs': inputs, 'targets': targets_}


class BigramLanguageModel:
    def __init__(self, rng_key: RNGKey, learning_rate: float, vocab_size: int, init_scale: float):
        self.vocab_size = vocab_size

        self.embeddings = Embeddings.make(
            Embeddings(dict_size=vocab_size, dim_model=vocab_size, dict_init_scale=init_scale))
        self.init_weights = make_weights(rng_key, self.embeddings.weight_params)
        self.weights_: ArrayTreeMapping = self.init_weights

        self.optimiser = optax.adamw(learning_rate)
        self.opt_state_ = self.optimiser.init(self.init_weights)

        self.guess_loss = -xp.log(1 / (vocab_size + 1))
        self.forward1 = self.embeddings.fixed_pipeline
        self.forward = batch_fy(self.forward1)
        self.loss_fn = self.make_loss_fn()

    def loss(self, batch: Dict[str, Array]) -> Array:
        return jax.jit(self.loss_fn)(self.weights_, batch)

    def make_loss_fn(self) -> Callable[[ArrayTreeMapping, Dict[str, Array]], Array]:
        def _loss_fn(weight: ArrayTreeMapping, batch: Dict[str, Array]) -> Array:
            logits = flatten_token(self.forward(weight, batch['inputs']))
            targets = flatten_token(batch['targets'])
            return softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        return _loss_fn

    def generate(self, idx: int, max_new_tokens: int) -> list[int]:
        new_tokens = [idx]
        idx_next = idx
        for _ in range(max_new_tokens):
            logits = self.forward1(self.weights_, xp.array([idx_next]))
            probs = jax.nn.softmax(logits).flatten()
            idx_next = jax.random.choice(key=next(key_gen), a=xp.arange(0, self.vocab_size), p=probs, shape=()).item()
            new_tokens.append(idx_next)
        return new_tokens

    def train1_(self, batch: Dict[str, Array]):
        self.weights_, self.opt_state_ = jax_calc_updates(self.optimiser, self.loss_fn,
                                                          self.weights_,
                                                          batch,
                                                          self.opt_state_)


def estimate_loss(model: BigramLanguageModel) -> None:
    for split in 'train', 'val':
        total_eval_loss = 0
        for key in jax.random.split(next(key_gen), eval_iters):
            eval_batch = get_batch(train_data if split == 'train' else valid_data, key)
            total_eval_loss += model.loss(eval_batch).item()
        print(f"Estimated {split} loss: {total_eval_loss / eval_iters}")


# tests to replicate 1_notebook.py
# batch_ = get_batch(train_data, next(key_gen))
# xb, yb = batch_['inputs'], batch_['targets']
# model_ = BigramLanguageModel(next(key_gen), learning_rate_, vocab_size_, init_scale_)
# loss = model_.loss(batch_)
# print(loss, model_.guess_loss)
# generated = decode(model_.generate(0, 100))
# print(generated)
# batch_ = get_batch(train_data, next(key_gen))
# loss = model_.loss(batch_)
# print(f"before step, batch loss {loss}")
# model_.train1_(batch_)
# loss = model_.loss(batch_)
# print(f"after step, batch loss {loss}")

# %%
model_ = BigramLanguageModel(next(key_gen), learning_rate_, vocab_size_, init_scale_)
keys = jax.random.split(next(key_gen), max_iters)
for step in range(max_iters):
    batch_ = get_batch(train_data, keys[step])
    if step % eval_interval == 0:
        loss = model_.loss(batch_)
        print(f"===step {step} is an eval step===")
        print(f"before step {step}, batch loss {loss}")
    model_.train1_(batch_)
    if step % eval_interval == 0:
        loss = model_.loss(batch_)
        print(f"after step {step}, batch loss {loss}")
        estimate_loss(model_)
        generated = model_.generate(encode(",")[0], max_new_tokens=100)
        print(decode(generated))

# TODO fix the bug of getting nan
