# %% imports
from __future__ import annotations

from typing import Dict, Callable, Optional

import jax
import jax.numpy as xp
import optax
from jax import Array

from jax_make.component_protocol import merge_component_params, Input, Output, FixedPipeline
from jax_make.components.multi_head_attn import masked_mha_port
from jax_make.params import make_weights, ArrayTreeMapping, RNGKey, get_mapping, get_arr
from jax_make.transformer import TensorTransformer, DynamicTransformer
from jax_make.utils.elementary_components import linear_component
from jax_make.utils.functions import softmax_cross_entropy_with_integer_labels
from scripts.nanogpt.my_nlp_dataset import load_jax_cached
from scripts.nanogpt.utils import infinite_jax_keys, flatten_token, batch_fy, jax_calc_updates

block_size = 16
batch_size = 16
max_iters = 20000
eval_interval = 100
learning_rate_ = 1e-4
eval_iters = 200
seed = 0
key_gen = infinite_jax_keys(seed)
init_scale_ = 0.01
n_embd_ = 64

dataset = "english"
# dataset = "play"
encoded_jax, encode, decode, vocab_size_ = load_jax_cached(dataset=dataset)

print(encode("hii there"))
print(decode(encode("hii there")))

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


# %%


class TFLanguageModel:
    def __init__(self, rng_key: RNGKey,
                 learning_rate: float,
                 vocab_size: int,
                 init_scale: float,
                 n_embd: int):
        self.vocab_size = vocab_size

        transformer_config = TensorTransformer(
            n_heads=4,
            dim_model=n_embd_,
            dropout_keep_rate=1.,
            eps=0.00001,
            mlp_n_hidden=[24],
            mlp_activation="gelu",
            pos_t=-1,
            n_tfe_layers=2,
            universal=True,
            n_seq=block_size,
            dim_input=n_embd_,
            dict_size=vocab_size_,

            dict_init_scale=init_scale,
            pos_init_scale=init_scale
        )
        self.transformer = DynamicTransformer.make(configs=transformer_config)
        self.lm_head = linear_component(n_embd, vocab_size)
        self.mask = xp.tril(xp.ones((block_size, block_size)))

        self.weight_params = self.transformer.weight_params
        self.weight_params = merge_component_params({'transformer': self.transformer,
                                                     'lm_head': self.lm_head})

        self.init_weights = make_weights(rng_key, self.weight_params)
        self.weights_: ArrayTreeMapping = self.init_weights
        self.optimiser = optax.adamw(learning_rate)
        self.opt_state_ = self.optimiser.init(self.init_weights)

        self.guess_loss = -xp.log(1 / (vocab_size + 1))
        self.loss_fn = self.make_loss_fn()
        self.fixed_key = next(key_gen)  # for simplicity

    def get_forward1(self, mask: Optional[Array]) -> FixedPipeline:
        @jax.jit
        def forward1(weights: ArrayTreeMapping, x: Array) -> Array:
            key = self.fixed_key
            tsfm_weights = get_mapping(weights, 'transformer')
            if mask is not None:
                process_out = self.transformer.processes[masked_mha_port](tsfm_weights,
                                                                          {Input: x, 'mask': mask},
                                                                          key)
                tsfm_out = get_arr(process_out, Output).T
            else:
                tsfm_out = self.transformer.pipeline(tsfm_weights, x, key).T
            return self.lm_head.fixed_pipeline(get_mapping(weights, 'lm_head'), tsfm_out)

        return forward1

    def loss(self, batch: Dict[str, Array]) -> Array:
        return self.loss_fn(self.weights_, batch)

    def make_loss_fn(self) -> Callable[[ArrayTreeMapping, Dict[str, Array]], Array]:
        @jax.jit
        def _loss_fn(weight: ArrayTreeMapping, batch: Dict[str, Array]) -> Array:
            logits = flatten_token(batch_fy(self.get_forward1(self.mask))(weight, batch['inputs']))
            targets = flatten_token(batch['targets'])
            return softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        return _loss_fn

    def probs1(self, x: Array, mask: Optional[Array]) -> Array:
        return self.get_forward1(mask)(self.weights_, x)[-1, :].squeeze()

    def probs1_debug(self, x: Array, mask: Optional[Array]) -> Array:
        print("input:  ", decode(x.flatten().tolist()))
        k = self.get_forward1(mask)(self.weights_, x)
        print("output: ", decode(k.argmax(axis=-1).squeeze().tolist()))
        logits = k[-1, :].squeeze()
        return logits

    def generate(self, idx: int, max_new_tokens: int, debug: bool = False) -> list[int]:
        new_tokens = [idx]
        keys = jax.random.split(next(key_gen), max_new_tokens)
        for rng_key in keys:
            x = xp.array(new_tokens[-block_size:])
            mask = xp.tril(xp.ones((x.size, x.size)))
            if debug:
                logits = self.probs1_debug(x, mask)
            else:
                logits = jax.jit(self.probs1)(x, mask)
            idx_next = jax.random.categorical(key=rng_key, logits=logits).item()
            new_tokens.append(idx_next)
        return new_tokens

    def debug(self, batch: Dict[str, Array]):
        logits = self.get_forward1(self.mask)(self.weights_, batch['inputs'][0, :])
        print(logits.shape)
        return logits.argmax(axis=-1).flatten().tolist(), batch['targets'][0, :].flatten().tolist()

    def train1_(self, batch: Dict[str, Array]):
        self.weights_, self.opt_state_ = jax_calc_updates(self.optimiser, self.loss_fn,
                                                          self.weights_,
                                                          batch,
                                                          self.opt_state_)


def estimate_loss(model: TFLanguageModel) -> None:
    for split in 'train', 'val':
        total_eval_loss = 0
        for key in jax.random.split(next(key_gen), eval_iters):
            eval_batch = get_batch(train_data if split == 'train' else valid_data, key)
            total_eval_loss += model.loss(eval_batch).item()
        print(f"Estimated {split} loss: {total_eval_loss / eval_iters}")


# %%
model_ = TFLanguageModel(next(key_gen), learning_rate_, vocab_size_, init_scale_, n_embd_)
keys_ = jax.random.split(next(key_gen), max_iters)
for step in range(max_iters):
    batch_ = get_batch(train_data, keys_[step])
    if step % eval_interval == 0:
        loss = model_.loss(batch_)
        print(f"===step {step} is an eval step===")
        print(f"before step {step}, batch loss {loss}")
    model_.train1_(batch_)
    if step % eval_interval == 0:
        loss = model_.loss(batch_)
        print(f"after step {step}, batch loss {loss}")
        estimate_loss(model_)
        generated = model_.generate(0, max_new_tokens=100, debug=False)
        print(decode(generated), flush=True)
        a, b = model_.debug(batch_)
        print(decode(a), flush=True)
        print(decode(b), flush=True)


#     if step % (eval_interval * 5) == 0:
#         generated = model_.generate(0, max_new_tokens=100, debug=True)
#         print(decode(generated), flush=True)

# debug generation

