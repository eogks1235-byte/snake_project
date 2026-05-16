"""PyTorch 버전 brain — MLP / GRU 옵션, 정책-가치(actor-critic) 헤드.

numpy Brain은 그대로 두고, PPO 학습 시 이걸 사용. 두 구조:

  arch='mlp'  : 기존 2층 MLP과 호환 (시뮬에 in-lifetime 학습으로 끼워넣기 가능)
  arch='gru'  : 시간적 의존성 학습 (1-layer GRU + linear heads)

actor: 가우시안 정책 — 평균 + 학습 가능 log_std
critic: 스칼라 value V(s)
"""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchBrain(nn.Module):
    """공유 정책. 모든 개체가 한 인스턴스를 같이 씀 (multi-agent shared policy)."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 arch: str = 'mlp'):
        super().__init__()
        self.arch = arch
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out

        if arch == 'mlp':
            self.encoder = nn.Sequential(
                nn.Linear(n_in, n_hidden),
                nn.Tanh(),
            )
            enc_out = n_hidden
        elif arch == 'gru':
            self.rnn = nn.GRU(n_in, n_hidden, batch_first=True)
            enc_out = n_hidden
        else:
            raise ValueError(f'unknown arch {arch}')

        # actor head: action mean
        self.actor_mean = nn.Linear(enc_out, n_out)
        # learnable log_std (state-independent) — PPO 표준 트릭
        self.actor_log_std = nn.Parameter(torch.zeros(n_out) - 0.5)
        # critic head
        self.critic = nn.Linear(enc_out, 1)

        # 가중치 초기화 — orthogonal (안정성)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def encode(self, obs: torch.Tensor,
               hidden: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.arch == 'mlp':
            return self.encoder(obs), None
        # GRU: obs shape (B, T, n_in)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)   # (B, 1, n_in)
        out, h = self.rnn(obs, hidden)
        return out.squeeze(1), h     # (B, hidden), h

    def forward(self, obs: torch.Tensor,
                hidden: Optional[torch.Tensor] = None):
        """반환: action_mean, action_std, value, new_hidden."""
        z, h = self.encode(obs, hidden)
        mean = torch.tanh(self.actor_mean(z))           # -1~1
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(z).squeeze(-1)
        return mean, std, value, h

    def act(self, obs: torch.Tensor,
            hidden: Optional[torch.Tensor] = None,
            deterministic: bool = False):
        """샘플 + log_prob 반환 — 행동 + 학습 데이터 수집용."""
        mean, std, value, h = self.forward(obs, hidden)
        if deterministic:
            action = mean
            log_prob = None
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        # 행동 클램프 (-1, 1)
        action = torch.clamp(action, -1.0, 1.0)
        return action, log_prob, value, h

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor,
                 hidden: Optional[torch.Tensor] = None):
        """PPO 업데이트용 — 주어진 (obs, action)에 대해 log_prob, entropy, value."""
        mean, std, value, _ = self.forward(obs, hidden)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value
