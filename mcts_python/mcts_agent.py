import numpy as np
from typing import Dict
from collections import defaultdict

from numpy.typing import NDArray

from mcts_python.protocols import StateID, Actions
from mcts_python.math_calc import ucb_all
from mcts_python.config import c_puct


class MctsAgent:
    def __init__(self, agent_id: int, n_actions: int):
        self.ag_id = agent_id
        self.visits: Dict[StateID, NDArray] \
            = defaultdict(lambda: np.zeros(n_actions))
        self.qs: Dict[StateID, NDArray] \
            = defaultdict(lambda: np.zeros(n_actions))

    def selection(self, s_id: int, avail_a: Actions, ps: NDArray) -> int:
        if len(avail_a) == 1:
            return avail_a[0]
        else:
            ucbs = ucb_all(qs=self.qs[s_id],
                           c_puct_normed_by_sum=c_puct * np.sqrt(
                               self.visits[s_id][avail_a].sum()),
                           ps=ps / ps[avail_a].sum(),
                           n_as=self.visits[s_id])
            # print(self.visits[s_id])
            # print(self.qs[s_id])
            # print(ucbs)
            # print("--")
            return avail_a[np.argmax(ucbs[avail_a])]

    def update_qn_(self, s_id, action, v):
        n_sa = self.visits[s_id][action]
        self.qs[s_id][action] = (n_sa * self.qs[s_id][action] + v) / (n_sa + 1)
        self.visits[s_id][action] += 1
        return v

    def find_policy(self, s_id: int, avail_a: Actions, render=False):
        policy_count = self.visits[s_id]
        if render:
            print("Q: ", self.qs[s_id])
            print("N: ", self.visits[s_id])

        if not policy_count.any():
            policy_count[avail_a] += 1 / len(avail_a)
            return policy_count
        else:
            return policy_count / policy_count.sum()
