"""
snake_env.py
────────────
SnakeGame을 gymnasium.Env 형식으로 래핑한 강화학습 환경.

관찰 공간 (AI가 보는 정보): 20차원 float32 벡터
  ── 즉각 위험 감지 (1칸 앞) ──────────────────────
  [0]  직진 방향 바로 앞 위험 (0=안전, 1=위험)
  [1]  우회전 방향 바로 앞 위험
  [2]  좌회전 방향 바로 앞 위험

  ── 확장 위험 감지 (2칸 앞) ── [난이도 하] ────────
  [3]  직진 방향 2칸 앞 위험
  [4]  우회전 방향 2칸 앞 위험
  [5]  좌회전 방향 2칸 앞 위험

  ── 현재 이동 방향 one-hot ───────────────────────
  [6]  위쪽으로 이동 중
  [7]  아래쪽으로 이동 중
  [8]  왼쪽으로 이동 중
  [9]  오른쪽으로 이동 중

  ── 먹이 상대 방향 ───────────────────────────────
  [10] 먹이가 위에 있음
  [11] 먹이가 아래에 있음
  [12] 먹이가 왼쪽에 있음
  [13] 먹이가 오른쪽에 있음

  ── A* 길찾기 결과 ── [난이도 상] ─────────────────
  [14] A*가 추천하는 방향: 위 (몸통을 피한 최단경로 첫 스텝)
  [15] A*가 추천하는 방향: 아래
  [16] A*가 추천하는 방향: 왼쪽
  [17] A*가 추천하는 방향: 오른쪽
  [18] 경로 존재 여부 (1=경로있음, 0=막힘)
  [19] 경로 길이 정규화 (0~1, 짧을수록 1에 가까움)

행동 공간: Discrete(3) → 0=직진, 1=우회전, 2=좌회전

보상 구조:
  +10  : 먹이를 먹었을 때
  -50  : 죽었을 때
  +0.1 : 먹이에 가까워졌을 때
  -0.1 : 먹이에서 멀어졌을 때
  -0.01: 매 스텝 패널티 (원 돌기 방지)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from heapq import heappush, heappop   # A* 우선순위 큐에 사용
from game.snake_game import SnakeGame
from game.constants import GRID_W, GRID_H, DIRECTIONS


class SnakeEnv(gym.Env):

    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, render_mode: str = "none", fps: int = None):
        super().__init__()
        self.render_mode = render_mode
        self.game = SnakeGame(render=(render_mode == "human"), fps=fps)

        self.action_space = spaces.Discrete(3)

        # 20차원 벡터: 즉각위험(3) + 확장위험(3) + 방향(4) + 먹이방향(4) + A*(6)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(20,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_state(), {}

    def step(self, action: int):
        hx, hy = self.game.snake[0]
        fx, fy = self.game.food
        prev_dist = abs(fx - hx) + abs(fy - hy)

        reward, done, score = self.game.step(int(action))

        if done:
            reward = -50
        else:
            if reward == 0:
                hx2, hy2 = self.game.snake[0]
                fx2, fy2 = self.game.food
                curr_dist = abs(fx2 - hx2) + abs(fy2 - hy2)
                reward += 0.1 if curr_dist < prev_dist else -0.1
                reward -= 0.01

        return self._get_state(), reward, done, False, {"score": score}

    def render(self):
        pass

    # ── A* 길찾기 알고리즘 ────────────────────────
    def _astar(self):
        """
        A*(에이스타) 알고리즘: 머리에서 먹이까지 최단 경로 탐색.

        GPS 내비게이션과 같은 원리:
          - 현재 위치(머리)에서 목적지(먹이)까지
          - 장애물(몸통)을 피해서
          - 가장 짧은 경로를 계산

        반환값:
          first_dir  : 최단 경로의 첫 번째 이동 방향 (dx, dy) 또는 None
          path_length: 경로의 총 길이 (스텝 수)

        A* 동작 원리:
          f(n) = g(n) + h(n)
          g(n) = 시작점에서 현재 노드까지 실제 비용 (이동 스텝 수)
          h(n) = 현재 노드에서 목표까지 예상 비용 (맨해튼 거리)
          → f가 작은 노드부터 탐색 (우선순위 큐)
        """
        start = self.game.snake[0]   # 탐색 시작: 머리
        goal  = self.game.food        # 목적지: 먹이

        # 장애물: 몸통 전체 (꼬리 제외 - 다음 스텝에 꼬리는 사라지므로)
        # 꼬리를 포함하면 꼬리 바로 앞으로 가는 경로를 잘못 막을 수 있음
        obstacles = set(list(self.game.snake)[:-1])

        # 우선순위 큐: (f점수, g점수, 현재위치, 첫번째이동방향)
        # f점수가 같으면 g점수 기준 정렬 (tie-breaking)
        heap = [(abs(start[0]-goal[0]) + abs(start[1]-goal[1]), 0, start, None)]
        visited = set()

        while heap:
            f, g, pos, first_dir = heappop(heap)

            if pos in visited:
                continue
            visited.add(pos)

            # 목표 도달: 첫 번째 이동 방향과 경로 길이 반환
            if pos == goal:
                return first_dir, g

            # 상하좌우 4방향 탐색
            for d in [(0, -1), (1, 0), (0, 1), (-1, 0)]:   # UP, RIGHT, DOWN, LEFT
                nx, ny = pos[0] + d[0], pos[1] + d[1]
                npos = (nx, ny)

                if npos in visited:
                    continue
                # 격자 범위 벗어나면 건너뜀
                if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H:
                    continue
                # 몸통과 겹치면 건너뜀
                if npos in obstacles:
                    continue

                ng = g + 1   # 실제 이동 비용 +1
                nf = ng + abs(nx - goal[0]) + abs(ny - goal[1])   # f = g + h
                # 첫 번째 이동 방향: 시작점에서 처음 이동한 방향을 끝까지 유지
                nd = first_dir if first_dir is not None else d
                heappush(heap, (nf, ng, npos, nd))

        return None, 0   # 경로 없음 (몸통에 완전히 막힌 경우)

    # ── 상태: 20차원 벡터 ────────────────────────
    def _get_state(self) -> np.ndarray:
        hx, hy = self.game.snake[0]
        dir_   = self.game.direction
        idx    = DIRECTIONS.index(dir_)

        dir_straight = DIRECTIONS[idx]
        dir_right    = DIRECTIONS[(idx + 1) % 4]
        dir_left     = DIRECTIONS[(idx - 1) % 4]

        body_set = set(self.game.snake)   # 충돌 검사용 집합 (매번 생성 최소화)

        def danger(d, steps=1):
            """
            해당 방향으로 steps칸 이동했을 때 위험 여부 반환.
            steps=1: 바로 앞 (즉각 위험)
            steps=2: 2칸 앞 (예측 위험) → 코너에서 막히는 상황 미리 감지
            """
            nx, ny = hx + d[0] * steps, hy + d[1] * steps
            if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H:
                return 1.0
            if (nx, ny) in body_set:
                return 1.0
            return 0.0

        fx, fy = self.game.food

        # ── A* 실행 ──────────────────────────────
        # 매 스텝마다 A*로 최단 경로 계산
        # first_dir: 다음 스텝에 갈 방향 (dx, dy) 또는 None (경로 없음)
        # path_len : 경로 총 길이
        first_dir, path_len = self._astar()

        # A* 결과를 관찰 벡터용으로 변환
        path_exists = 1.0 if first_dir is not None else 0.0

        # 경로 길이 정규화: 최대 가능 길이(GRID_W+GRID_H)로 나눔
        # 짧은 경로일수록 1에 가깝게 → AI가 경로 길이를 0~1 범위로 인식
        max_path = GRID_W + GRID_H
        path_len_norm = 1.0 - min(path_len / max_path, 1.0) if path_exists else 0.0

        return np.array([
            # ── 즉각 위험 (1칸 앞, 3개) ──────────
            danger(dir_straight, 1),   # [0] 직진하면 바로 죽는가?
            danger(dir_right,    1),   # [1] 우회전하면 바로 죽는가?
            danger(dir_left,     1),   # [2] 좌회전하면 바로 죽는가?

            # ── 예측 위험 (2칸 앞, 3개) ──────────
            # 1칸 앞은 안전해도 2칸 앞이 막혀있으면 코너에 갇힐 수 있음
            danger(dir_straight, 2),   # [3] 직진 방향 2칸 앞 위험?
            danger(dir_right,    2),   # [4] 우회전 방향 2칸 앞 위험?
            danger(dir_left,     2),   # [5] 좌회전 방향 2칸 앞 위험?

            # ── 현재 이동 방향 one-hot (4개) ─────
            1.0 if dir_ == (0, -1) else 0.0,   # [6]  위
            1.0 if dir_ == (0,  1) else 0.0,   # [7]  아래
            1.0 if dir_ == (-1, 0) else 0.0,   # [8]  왼쪽
            1.0 if dir_ == (1,  0) else 0.0,   # [9]  오른쪽

            # ── 먹이 상대 방향 (4개) ─────────────
            1.0 if fy < hy else 0.0,   # [10] 먹이가 위
            1.0 if fy > hy else 0.0,   # [11] 먹이가 아래
            1.0 if fx < hx else 0.0,   # [12] 먹이가 왼쪽
            1.0 if fx > hx else 0.0,   # [13] 먹이가 오른쪽

            # ── A* 길찾기 결과 (6개) ─────────────
            # AI가 "GPS가 이쪽으로 가라고 한다"는 정보를 직접 받음
            # 몸통에 막힌 경우 A*가 우회 경로를 계산해서 알려줌
            1.0 if first_dir == (0, -1) else 0.0,   # [14] A* 추천: 위
            1.0 if first_dir == (0,  1) else 0.0,   # [15] A* 추천: 아래
            1.0 if first_dir == (-1, 0) else 0.0,   # [16] A* 추천: 왼쪽
            1.0 if first_dir == (1,  0) else 0.0,   # [17] A* 추천: 오른쪽
            path_exists,                             # [18] 경로 존재 (1=있음, 0=막힘)
            path_len_norm,                           # [19] 경로 길이 (짧을수록 높음)
        ], dtype=np.float32)
