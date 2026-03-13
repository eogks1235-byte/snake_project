"""
snake_game.py
─────────────
Snake 게임의 순수한 로직 + pygame 렌더링을 담당하는 클래스.

역할:
  - 뱀의 이동, 먹이 섭취, 충돌 판정 등 게임 규칙을 구현
  - pygame을 이용해 화면에 격자, 뱀, 먹이, 점수를 그림
  - 사람이 직접 키보드로 플레이하거나,
    RL 환경(snake_env.py)이 step()으로 호출해서 학습에 사용

구조:
  SnakeEnv (snake_env.py)
    └─ SnakeGame (snake_game.py)  ← 이 파일
         └─ pygame, constants.py

직접 실행 시 (사람 플레이):
    python game/snake_game.py
"""

import sys
import os
# 상위 디렉토리를 경로에 추가 → constants.py를 'game.constants'로 import 가능
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import random
from collections import deque   # deque: 앞뒤 삽입/삭제가 O(1)인 자료구조 → 뱀 몸통 관리에 최적
from game.constants import *


# ══════════════════════════════════════════════
#  SnakeGame 클래스
# ══════════════════════════════════════════════
class SnakeGame:
    """
    Snake 게임의 모든 상태와 로직을 담당하는 클래스.

    Attributes
    ----------
    snake     : deque  – 뱀 몸통 좌표 리스트. [0]이 머리, [-1]이 꼬리
    direction : tuple  – 현재 이동 방향 (dx, dy)
    food      : tuple  – 먹이 좌표 (x, y)
    score     : int    – 현재 점수 (먹은 먹이 수)
    steps     : int    – 마지막으로 먹이를 먹은 이후 이동 횟수 (루프 방지용)
    """

    def __init__(self, render: bool = False, fps: int = None):
        """
        Parameters
        ----------
        render : True  → pygame 창을 띄워 시각화 (사람 플레이 / AI 관람용)
                 False → 창 없이 빠르게 실행 (학습 중엔 이 모드 사용)
        fps    : 렌더링 속도. None이면 FPS_HUMAN 사용
        """
        self.render_mode = render
        self.fps    = fps if fps is not None else FPS_HUMAN
        self.screen = None
        self.clock  = None

        if self.render_mode:
            # pygame 초기화 및 창 생성 (render=True일 때만)
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()
            self.font  = pygame.font.SysFont("consolas", 18)   # 점수 표시용 폰트

        self.reset()   # 게임 초기 상태 설정

    # ── 초기화 ──────────────────────────────────
    def reset(self):
        """
        게임을 초기 상태로 리셋한다.
        새로운 에피소드 시작마다 snake_env.py에서 호출.
        """
        cx, cy = GRID_W // 2, GRID_H // 2   # 격자 중앙 좌표

        # 뱀: 중앙에서 오른쪽을 향해 3칸짜리로 시작
        # deque: [(머리), (몸1), (몸2)] 순서
        self.snake     = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.direction = RIGHT   # 초기 이동 방향: 오른쪽
        self.score     = 0       # 먹이를 먹을 때마다 +1
        self.steps     = 0       # 루프 방지 카운터: 먹이를 먹으면 0으로 초기화
        self.game_over = False

        self.food = self._place_food()   # 초기 먹이 위치 설정

    # ── 먹이 배치 ────────────────────────────────
    def _place_food(self):
        """
        뱀 몸통이 없는 빈 칸에 랜덤으로 먹이를 배치한다.
        set()으로 변환해 O(1) 검색 → 뱀이 긴 상황에서도 빠름.
        """
        snake_set = set(self.snake)   # 리스트를 집합으로 변환 (검색 O(n) → O(1))
        while True:
            pos = (random.randint(0, GRID_W - 1),
                   random.randint(0, GRID_H - 1))
            if pos not in snake_set:   # 뱀 위치와 겹치지 않는 칸이면 배치
                return pos

    # ── 한 스텝 진행 ─────────────────────────────
    def step(self, action: int):
        """
        게임을 한 프레임 진행한다.
        snake_env.py의 step()에서 호출됨.

        Parameters
        ----------
        action : 0 = 직진, 1 = 우회전, 2 = 좌회전

        Returns
        -------
        reward    : float – 기본 보상 (-10=죽음, 0=이동, +10=먹이)
                    snake_env.py에서 추가 보정됨
        game_over : bool  – True면 에피소드 종료
        score     : int   – 현재 점수 (먹은 먹이 수)
        """
        # (1) 렌더링 중이면 창 닫기 이벤트 처리
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        # (2) 행동에 따라 이동 방향 업데이트
        self.direction = self._turn(action)

        # (3) 현재 머리 위치에서 새 머리 위치 계산
        hx, hy   = self.snake[0]
        dx, dy   = self.direction
        new_head = (hx + dx, hy + dy)

        # (4) 충돌 검사: 벽 또는 자기 몸과 충돌하면 게임 오버
        reward = 0
        if self._is_collision(new_head):
            self.game_over = True
            reward = -10   # 기본 충돌 패널티 (snake_env.py에서 -50으로 강화됨)
            return reward, True, self.score

        # (5) 머리를 deque 맨 앞에 추가 (이동)
        self.snake.appendleft(new_head)

        # (6) 먹이 먹었는지 확인
        if new_head == self.food:
            # 먹이 먹음: 꼬리를 제거하지 않아 뱀 길이 +1
            self.score += 1
            self.steps  = 0                    # 루프 방지 카운터 초기화
            reward      = 10                   # 먹이 보상
            self.food   = self._place_food()   # 새 먹이 배치
        else:
            # 먹이 못 먹음: 꼬리 제거 → 실제 이동 (길이 유지)
            self.snake.pop()
            self.steps += 1
            reward      = 0

        # (7) 루프 방지: 너무 오래 먹이를 못 먹으면 강제 종료
        # 뱀 길이 × 80 또는 최소 400스텝 초과 시 타임아웃
        # (원 돌기 같은 비생산적 행동 방지)
        if self.steps > max(400, 80 * len(self.snake)):
            self.game_over = True
            reward = -10
            return reward, True, self.score

        # (8) 렌더링 (render=True일 때만)
        if self.render_mode:
            self._draw()

        return reward, False, self.score

    # ── 방향 전환 계산 ───────────────────────────
    def _turn(self, action: int):
        """
        DIRECTIONS = [UP, RIGHT, DOWN, LEFT] (시계 방향 정렬)
        현재 방향의 인덱스를 기준으로 +1/-1 해서 회전 방향 결정.

        예) 현재 RIGHT(인덱스 1), action=1(우회전) → 인덱스 2 = DOWN
        """
        idx = DIRECTIONS.index(self.direction)

        if action == 1:      # 오른쪽 회전: 시계 방향 +1
            idx = (idx + 1) % 4
        elif action == 2:    # 왼쪽 회전: 반시계 방향 -1
            idx = (idx - 1) % 4
        # action == 0 이면 idx 그대로 (직진)

        return DIRECTIONS[idx]

    # ── 충돌 검사 ────────────────────────────────
    def _is_collision(self, pos=None):
        """
        pos가 벽 또는 뱀 몸통과 충돌하는지 확인.
        pos 생략 시 현재 머리 위치로 검사.

        주의: appendleft 이전에 호출하므로 현재 snake 전체(머리 포함)와 비교.
        """
        if pos is None:
            pos = self.snake[0]

        x, y = pos

        # 벽 충돌: 격자 범위를 벗어나면 충돌
        if x < 0 or x >= GRID_W or y < 0 or y >= GRID_H:
            return True

        # 몸통 충돌: 새 머리가 현재 몸통 위치와 겹치면 충돌
        if pos in set(self.snake):
            return True

        return False

    # ── 키보드 입력 → action 변환 ─────────────────
    def get_human_action(self):
        """
        사람이 플레이할 때 키보드 입력을 action(0/1/2)으로 변환.
        방향키(↑↓←→) 또는 WASD 모두 지원.
        180° 반전(예: 오른쪽 가다가 왼쪽 누르기)은 무시하고 직진.
        """
        keys = pygame.key.get_pressed()
        dx, dy = self.direction

        # 원하는 방향 파악 (현재 방향의 반대 방향 입력은 무시)
        if   (keys[pygame.K_UP]    or keys[pygame.K_w]) and dy != 1:  desired = UP
        elif (keys[pygame.K_DOWN]  or keys[pygame.K_s]) and dy != -1: desired = DOWN
        elif (keys[pygame.K_LEFT]  or keys[pygame.K_a]) and dx != 1:  desired = LEFT
        elif (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and dx != -1: desired = RIGHT
        else: return 0   # 입력 없거나 반전 입력 → 직진

        # 원하는 방향과 현재 방향의 인덱스 차이로 action 결정
        cur_idx  = DIRECTIONS.index(self.direction)
        want_idx = DIRECTIONS.index(desired)
        diff = (want_idx - cur_idx) % 4

        if   diff == 0: return 0   # 직진 (같은 방향)
        elif diff == 1: return 1   # 우회전
        elif diff == 3: return 2   # 좌회전
        else:           return 0   # 180° 반전 → 직진으로 처리

    # ── 화면 그리기 ──────────────────────────────
    def _draw(self):
        """
        pygame 화면에 현재 게임 상태를 그린다.
        순서: 배경 → 격자선 → 먹이 → 뱀 몸통 → 점수
        """
        self.screen.fill(BLACK)   # 배경을 검정으로 초기화

        # 격자선 그리기 (20px 간격)
        for x in range(0, WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (WINDOW_WIDTH, y))

        # 먹이 그리기 (빨간 사각형, 2px 안쪽 여백)
        fx, fy = self.food
        pygame.draw.rect(self.screen, RED,
                         (fx * CELL_SIZE + 2, fy * CELL_SIZE + 2,
                          CELL_SIZE - 4, CELL_SIZE - 4))

        # 뱀 몸통 그리기 (머리=밝은 초록, 몸통=어두운 초록, 둥근 모서리)
        for i, (sx, sy) in enumerate(self.snake):
            color = GREEN if i == 0 else DGREEN   # i==0이 머리
            pygame.draw.rect(self.screen, color,
                             (sx * CELL_SIZE + 1, sy * CELL_SIZE + 1,
                              CELL_SIZE - 2, CELL_SIZE - 2),
                             border_radius=4)

        # 점수 텍스트 좌상단에 표시
        txt = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(txt, (8, 8))

        pygame.display.flip()        # 그린 내용을 실제 화면에 반영
        self.clock.tick(self.fps)    # FPS 속도 제한

    # ── 사람 플레이 메인 루프 ─────────────────────
    def play_human(self):
        """
        키보드 입력을 받아 사람이 직접 Snake를 플레이한다.
        게임 오버 시 안내 텍스트를 표시하고 키 입력을 기다린다.
        """
        self.reset()
        while not self.game_over:
            action = self.get_human_action()
            _, done, score = self.step(action)
            if done:
                break

        # 게임 오버 화면 표시
        if self.render_mode:
            over_txt = self.font.render(
                f"Game Over!  Score: {self.score}  (아무 키나 누르면 재시작)",
                True, WHITE)
            self.screen.blit(over_txt, (10, WINDOW_HEIGHT // 2 - 10))
            pygame.display.flip()

            # 키 입력 대기 (아무 키나 누르면 재시작)
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        waiting = False
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return


# ══════════════════════════════════════════════
#  직접 실행: 사람이 플레이 모드
#  python game/snake_game.py 로 실행하면 키보드로 직접 게임 가능
# ══════════════════════════════════════════════
if __name__ == "__main__":
    game = SnakeGame(render=True)
    while True:
        game.play_human()
        game.reset()
