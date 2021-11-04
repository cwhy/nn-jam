import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from typing import Dict, TypeVar, Hashable, Callable

from mcts_python.protocols import State, Env
from mcts_python.jax_networks import JaxMcts

T = TypeVar('T')


class FlatMemory:
    def __init__(self, env: Env):
        self.env = env
        self.n_actions = env.n_actions
        pass

    def add_(self, _, __, ___):
        pass

    def add_with_symmetry_(self, ag_id, s, policy, symmetry):
        pass

    def assign_values_(self, _):
        pass

    def clear_(self):
        pass

    def get_p(self, _, __):
        return np.ones(self.n_actions) / self.n_actions

    @staticmethod
    def get_v(_, __):
        return 0


class NNMemoryAnyState(FlatMemory):
    def __init__(self, model: nn.Module, env: Env):
        super().__init__(env)
        self.model = model.to(torch.float)
        self.ps_: Dict[Hashable, NDArray] = {}
        self.vs_: Dict[Hashable, NDArray] = {}
        self.hash = env.state_utils.hash

    def get_val(self, fn: Callable[[State, torch.Tensor], torch.Tensor],
                cache: Dict[Hashable, T], state: State, ag_id: int) -> T:
        state_hash = self.hash(state, ag_id)
        if state_hash in cache:
            return cache[state_hash]
        else:
            torch_agid = torch.tensor(ag_id).unsqueeze(0).to(self.model.device).long()
            val = fn(state, torch_agid).flatten().cpu().numpy()
            cache[state_hash] = val
            return val

    def get_p(self, state: State, ag_id: int) -> NDArray:
        return self.get_val(self.model.forward_p, self.ps_, state, ag_id)

    def get_v(self, state: State, ag_id: int) -> NDArray:
        return self.get_val(self.model.forward_v, self.vs_, state, ag_id)
