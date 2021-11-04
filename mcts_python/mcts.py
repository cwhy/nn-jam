import numpy as np
from typing import List, Tuple, Dict, Optional, Hashable

from numpy.typing import NDArray

from mcts_python.protocols import Action, StateID, Env, State
from mcts_python.mcts_agent import MctsAgent
from mcts_python.memory_utils import FlatMemory


class Mcts:
    def __init__(self,
                 n_mcts: int,
                 env: Env,
                 max_depth: Optional[int] = None):
        self.n_mcts = n_mcts
        self.states_: List[State] = []
        self.state_ids_: Dict[Hashable, StateID] = {}
        self.env = env
        self.max_depth = max_depth
        self.agents: List[MctsAgent] = [
            MctsAgent(i, env.n_actions)
            for i in range(env.n_agents)
        ]
        self.debug = False

    def search_(self, memory: FlatMemory,
                start_state: State,
                ag_id: int):
        env_model = self.env.model
        curr_agent_ = self.agents[ag_id]
        n_agents = len(self.agents)
        history_: List[Tuple[int, StateID, Action, float]] = []

        s_ = start_state
        done_ = False
        depth_ = 0
        s_vs = np.zeros(n_agents)
        current_states_ = set()
        while not (done_ or (self.max_depth is not None and depth_ > self.max_depth)):
            sb = self.env.state_utils.hash(s_, curr_agent_.ag_id)
            if sb not in current_states_:
                current_states_.add(sb)
            else:
                # prevent cycle
                if self.env.cycle_reward is not None:
                    s_vs += self.env.cycle_reward
                break
            if sb not in self.state_ids_:
                self.states_.append(s_)
                self.state_ids_[sb] = len(self.states_) - 1
                s_vs = memory.get_v(s_, curr_agent_.ag_id)
                break
            else:
                s_id = self.state_ids_[sb]
                avail_a = self.env.state_utils.get_actions(s_, curr_agent_.ag_id)
                action = curr_agent_.selection(s_id, avail_a,
                                               memory.get_p(s_, curr_agent_.ag_id))
                env_output = env_model(s_, action, curr_agent_.ag_id, render=False)
                history_.append((curr_agent_.ag_id, s_id, action, env_output.rewards))
                curr_agent_ = self.agents[env_output.next_agent_id]
                s_ = env_output.next_state
                depth_ += 1
                done_ = env_output.done

        if depth_ > 5 and self.debug:
            print(done_)
            self.env.state_utils.render_(s_)
        for ag_id, s_id, a, v in reversed(history_):
            s_vs += v
            if depth_ > 5 and self.debug:
                print(ag_id, s_id, s_vs)
                self.env.state_utils.render_(self.states_[s_id])
            self.agents[ag_id].update_qn_(s_id, a, s_vs[ag_id])
        if depth_ > 5 and self.debug:
            self.debug = False
            #raise Exception("sdfa")

    def get_agent_decision_fn(self, memory_: FlatMemory, ag_id: int):
        def decision(s: State, render=False):
            for _ in range(self.n_mcts):
                self.search_(memory_, s, ag_id)
            if render:
                np.set_printoptions(precision=3, suppress=True)
                print("P: ", memory_.get_p(s, ag_id))
                print("V: ", memory_.get_v(s, ag_id))

            avail_a = self.env.state_utils.get_actions(s, ag_id)
            if len(avail_a) == 1:
                return avail_a[0]
            else:
                sb = self.env.state_utils.hash(s, ag_id)
                assert sb in self.state_ids_
                s_id = self.state_ids_[sb]
                policy = self.agents[ag_id].find_policy(s_id, avail_a, render=render)
                action = np.argmax(policy)
                return action

        return decision

    def self_play(self, memory_: FlatMemory, seed: int = 0) -> Tuple[NDArray,
                                                                     NDArray,
                                                                     NDArray,
                                                                     NDArray]:
        self.debug = False
        random_gen = np.random.Generator(np.random.PCG64(seed))
        ag_ids = []
        states = []
        policies = []
        s_, ag_id_init = self.env.init_state()
        curr_agent_ = self.agents[ag_id_init]
        total_rewards = np.zeros(len(self.agents))
        done_ = False
        depth_ = 0
        while not (done_ or (self.max_depth is not None and depth_ > self.max_depth)):
            for _ in range(self.n_mcts):
                self.search_(memory_, s_, curr_agent_.ag_id)

            avail_a = self.env.state_utils.get_actions(s_, curr_agent_.ag_id)
            sb = self.env.state_utils.hash(s_, curr_agent_.ag_id)
            assert sb in self.state_ids_
            s_id = self.state_ids_[sb]

            policy = curr_agent_.find_policy(s_id, avail_a)
            action = random_gen.choice(len(policy), p=policy)
            env_output = self.env.model(s_, action, curr_agent_.ag_id, render=False)
            total_rewards += env_output.rewards

            if self.env.state_utils.get_symmetries is not None:
                _states, _policies = self.env.state_utils.get_symmetries(s_, policy)
                states += _states
                policies += _policies
                ag_ids += len(_states) * [curr_agent_.ag_id]
            else:
                states.append(s_)
                policies.append(policy)
                ag_ids.append(curr_agent_.ag_id)

            curr_agent_ = self.agents[env_output.next_agent_id]
            s_ = env_output.next_state
            depth_ += 1
            done_ = env_output.done
        # print(total_rewards)
        return (np.stack(ag_ids, axis=0)[:, np.newaxis],
                np.stack(states, axis=0),
                np.stack(policies, axis=0),
                np.repeat(total_rewards[np.newaxis, :], len(states), axis=0))

# agent_ids: NDArray  int, (length,)
# states: NDArray  int/float, (length, dim_states)
# policies: NDArray  float, (length, n_possible_policies)
# total_rewards: NDArray  float, (length, n_agents)
