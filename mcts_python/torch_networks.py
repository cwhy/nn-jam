import numpy as np
from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# TODO action query
# TODO batch normalization
from mcts_python.config import lr, max_batch_size


def state_process(device: str, env_name: str):
    from_np = lambda x: torch.tensor(x).to(device)
    if env_name == 'TicTacToe':
        return lambda s: torch.cat(tensors=(from_np(s.get_array),
                                            from_np((s.turn,))), dim=-1)
    else:
        return lambda s: from_np(s.get_array)


class BasicBoardNet(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self,
                 device: str,
                 env_name: str,
                 len_states: int,
                 n_agents: int,
                 max_actions: int,
                 hidden_dim: int = 16,
                 agent_embed_dim: int = 4):
        super().__init__()
        self.device = device
        self.state_t = nn.Linear(len_states, hidden_dim)
        self.agent_embed = nn.Embedding(n_agents + 1, agent_embed_dim)
        self.get_v = nn.Linear(hidden_dim * agent_embed_dim, n_agents)
        self.get_p_logits = nn.Linear(hidden_dim * agent_embed_dim, max_actions)
        self.state_process = state_process(device, env_name)
        self.optimizer = torch.optim.Adam(lr=lr, params=self.parameters())
        self.to(device)

    def get_embed(self, s, ag_id) -> Tensor:
        if isinstance(s, list):
            ss = [self.state_process(_s) for _s in s]
            s = torch.stack(ss)
        else:
            s = self.state_process(s)
        a_embd = self.agent_embed(ag_id + 1)
        s_embd = self.agent_embed(s + 1).transpose(-1, -2)
        s_embd = self.state_t(s_embd).transpose(-1, -2)
        final_embs = (s_embd * a_embd.unsqueeze(-2)).flatten(-2, -1)
        final_embs_relu = F.relu(final_embs)
        return final_embs_relu

    def forward(self, s, ag_id):
        embd = self.get_embed(s, ag_id)
        return self.get_p_logits(embd), self.get_v(embd)

    def forward_p(self, s, ag_id) -> Tensor:
        self.eval()
        with torch.no_grad():
            embd = self.get_embed(s, ag_id)
            return F.softmax(self.get_p_logits(embd), dim=-1)

    def forward_v(self, s, ag_id) -> Tensor:
        self.eval()
        with torch.no_grad():
            embd = self.get_embed(s, ag_id)
            return self.get_v(embd)

    def train_batch_(self, states: list, ag_ids, policies, values):
        self.optimizer.zero_grad()
        p_logits, v = self.forward(states, ag_ids)
        loss_p = F.kl_div(F.log_softmax(p_logits, dim=-1),
                          policies, reduction='batchmean')
        loss_v = F.mse_loss(v, values)
        print("P_loss: ", loss_p.cpu().item(), " V_loss", loss_v.cpu().item())
        (loss_p + loss_v).backward()
        self.optimizer.step()

    def train_(self, _ag_ids, _states, _policies, _values):
        self.train()
        mem_size = len(_values)
        ag_ids = torch.tensor(_ag_ids).flatten()
        policies = torch.tensor(_policies).float()
        values = torch.tensor(_values).float()
        # value_currs = _values[torch.arange(mem_size), ag_ids]
        # values = torch.tensor(value_currs).float()
        if 0 < mem_size < max_batch_size:
            shuffle = torch.randperm(values.shape[0])
            self.train_batch_(
                states=np.array(_states)[shuffle].tolist(),
                ag_ids=ag_ids[shuffle].to(self.device).long(),
                policies=policies[shuffle, :].to(self.device),
                values=values[shuffle].to(self.device))
        else:
            n_rounds = int(np.ceil(mem_size / max_batch_size)) + 2
            for _ in range(n_rounds):
                sample = torch.randint(mem_size, (max_batch_size,))
                self.train_batch_(
                    states=np.array(_states)[sample].tolist(),
                    ag_ids=ag_ids[sample].to(self.device).long(),
                    policies=policies[sample, :].to(self.device),
                    values=values[sample].to(self.device))
