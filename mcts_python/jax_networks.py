from __future__ import annotations

from typing import Tuple

import optax as optax
from jax import grad, random
from optax._src.alias import adamw
from optax._src.base import OptState

from jax_make.params import RNGKey
from jax_make.transformer import Transformer
from mcts_python.config import lr, max_batch_size


class JaxMcts:
    def __init__(self, dict_size: int):
        eps = 0.00001
        weight_decay = 0.0001
        dim_model = 64
        optimiser = adamw(lr, 0.9, 0.98, eps, weight_decay)
        AnyNet(
            n_heads=8,
            dim_model=dim_model,
            dropout_keep_rate=1,
            eps=eps,
            mlp_n_hidden=[100],
            mlp_activation='gelu',
            pos_t=-1,
            n_tfe_layers=8,
            dict_size=dict_size,
            dim_input=
        )
        net_train = Transformer.make()
        net_test = Transformer.make()

        def update_adam(params, batch, _state: Tuple[RNGKey, OptState]):
            rng, _opt_state = _state
            key, rng = random.split(rng)
            grads = grad(loss)(params, batch, key)
            updates, _opt_state = optimiser.update(grads, _opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, (rng, _opt_state)
