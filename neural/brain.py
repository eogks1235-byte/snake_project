"""작은 MLP — 시야 4채널 + 메모리 셀 + 종 거리 + 직렬화.

입력 (총 40):
  0..7   : 음식 8섹터 closeness (0~1)
  8..15  : 포식자 8섹터 closeness
  16..23 : 동족 8섹터 closeness
  24..31 : 페로몬 8섹터 closeness (life 비례)
  32     : 에너지 비율 (0~1)
  33     : 현재 vx 정규화 (-1~1)
  34     : 현재 vy 정규화 (-1~1)
  35..38 : 메모리 셀 (이전 틱 출력의 피드백, -1~1)
  39     : bias 1.0
출력 (총 6):
  0      : 다음 vx (tanh)
  1      : 다음 vy (tanh)
  2..5   : 다음 메모리 셀 (tanh)
"""
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import random


N_SECTORS = 8
N_CHANNELS = 4         # food / predator / kin / pheromone (prey 기준)
N_MEMORY = 4

# Prey brain
N_IN = N_SECTORS * N_CHANNELS + 3 + N_MEMORY + 1  # 32 + 3 + 4 + 1 = 40
N_HIDDEN = 24
N_OUT = 2 + N_MEMORY                              # 6

# Predator brain — 입력: prey 8섹터 + 동료 predator 8섹터 + 에너지/vx/vy + 메모리 + bias
PRED_N_IN = N_SECTORS * 2 + 3 + N_MEMORY + 1      # 16 + 3 + 4 + 1 = 24
PRED_N_HIDDEN = 16
PRED_N_OUT = 2 + N_MEMORY                         # 6

OUT_VX = 0
OUT_VY = 1
OUT_MEM_START = 2

INIT_SCALE = 0.7
MUTATION_STD = 0.15
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.5


class Brain:
    __slots__ = ('W1', 'b1', 'W2', 'b2')

    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    @staticmethod
    def random(rng: random.Random,
               n_in: int = N_IN, n_hidden: int = N_HIDDEN,
               n_out: int = N_OUT) -> 'Brain':
        np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
        return Brain(
            W1=np_rng.randn(n_in, n_hidden).astype(np.float32) * INIT_SCALE,
            b1=np_rng.randn(n_hidden).astype(np.float32) * 0.1,
            W2=np_rng.randn(n_hidden, n_out).astype(np.float32) * INIT_SCALE,
            b2=np_rng.randn(n_out).astype(np.float32) * 0.1,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.W1 + self.b1)
        return np.tanh(h @ self.W2 + self.b2)

    def forward_with_hidden(self, x: np.ndarray):
        """Hebbian 학습용 — hidden 활성도까지 반환."""
        h = np.tanh(x @ self.W1 + self.b1)
        out = np.tanh(h @ self.W2 + self.b2)
        return out, h

    def hebbian_update(self, x: np.ndarray, h: np.ndarray, out: np.ndarray,
                       reward: float, lr: float = 0.0015,
                       clip: float = 3.0) -> None:
        """보상-조절 Hebbian: Δw = lr · r · pre · post. 보상 클 때만 갱신."""
        self.W1 += (lr * reward * np.outer(x, h)).astype(np.float32)
        self.W2 += (lr * reward * np.outer(h, out)).astype(np.float32)
        np.clip(self.W1, -clip, clip, out=self.W1)
        np.clip(self.W2, -clip, clip, out=self.W2)

    def mutate(self, rng: random.Random) -> 'Brain':
        np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))

        def jitter(arr):
            mask = np_rng.rand(*arr.shape) < MUTATION_RATE
            noise = np_rng.randn(*arr.shape).astype(np.float32) * MUTATION_STD
            return arr + noise * mask

        return Brain(jitter(self.W1), jitter(self.b1),
                     jitter(self.W2), jitter(self.b2))

    @staticmethod
    def crossover(parent_a: 'Brain', parent_b: 'Brain',
                  rng: random.Random) -> 'Brain':
        np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))

        def mix(a, b):
            mask = np_rng.rand(*a.shape) < CROSSOVER_RATE
            return np.where(mask, a, b).astype(np.float32)

        def jitter(arr):
            mask = np_rng.rand(*arr.shape) < MUTATION_RATE
            noise = np_rng.randn(*arr.shape).astype(np.float32) * MUTATION_STD
            return arr + noise * mask

        return Brain(
            jitter(mix(parent_a.W1, parent_b.W1)),
            jitter(mix(parent_a.b1, parent_b.b1)),
            jitter(mix(parent_a.W2, parent_b.W2)),
            jitter(mix(parent_a.b2, parent_b.b2)),
        )

    @staticmethod
    def distance(a: 'Brain', b: 'Brain') -> float:
        """가중치 공간에서의 L2 거리. 종 분화 기준 + 다양성 지표."""
        return float(np.linalg.norm(a.W1 - b.W1)
                     + np.linalg.norm(a.W2 - b.W2))

    def signature(self) -> Tuple[int, int, int]:
        """가중치 → RGB. 같은 계통은 비슷한 색."""
        w1_sum = float(self.W1.sum())
        w2_sum = float(self.W2.sum())
        w_var = float(self.W1.var() + self.W2.var())

        def ch(v):
            return int(80 + (np.tanh(v / 8.0) * 0.5 + 0.5) * 175)

        return (ch(w1_sum), ch(w2_sum), ch(w_var * 2 - 1))

    def save(self, path: Path, meta: dict) -> None:
        arrays = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        for k, v in meta.items():
            arrays[f'meta_{k}'] = np.array(v)
        np.savez(str(path), **arrays)

    @staticmethod
    def load(path: Path,
             expected_shape: Optional[Tuple[int, int]] = (N_IN, N_HIDDEN)) -> Tuple['Brain', dict]:
        with np.load(str(path)) as data:
            W1 = data['W1'].astype(np.float32)
            if expected_shape is not None and W1.shape != expected_shape:
                raise ValueError(
                    f'incompatible brain dims: file has W1{W1.shape}, '
                    f'expected {expected_shape}')
            brain = Brain(
                W1=W1,
                b1=data['b1'].astype(np.float32),
                W2=data['W2'].astype(np.float32),
                b2=data['b2'].astype(np.float32),
            )
            meta = {}
            for k in data.files:
                if k.startswith('meta_'):
                    arr = data[k]
                    meta[k[5:]] = arr.item() if arr.shape == () else arr.tolist()
        return brain, meta
