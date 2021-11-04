from typing import List
import numpy as np
from mcts_python.protocols import Env, State


def pit(env: Env, actors: List[Env.Actor], render: bool = False):
    s, ag_id = env.init_state()
    current_actor_ = actors[ag_id]
    done = False
    env_output = action = None
    while not done:
        if render:
            print(f'agent {env.agent_symbols[ag_id]} turn')
        # render_(s)
        action = current_actor_(s, render)
        env_output = env.model(s, action, ag_id, render=False)
        current_actor_ = actors[env_output.next_agent_id]
        ag_id = env_output.next_agent_id
        done = env_output.done
        if not done:
            s = env_output.next_state

    if render:
        print("done")
        env.state_utils.render_(s)
        print(action)
    env.state_utils.render_(env_output.next_state)
    print(env_output.rewards)
    print(env_output.message)
    return env_output.rewards


class RandomAgent:
    def __init__(self, agent_id: int, env: Env):
        self.ag_id = agent_id
        self.get_actions = env.state_utils.get_actions

    def find_action(self, s: State, render=False):
        avail_a = self.get_actions(s, self.ag_id)
        return np.random.choice(avail_a)
