"""
play.py
───────
════════════════════════════════════════════════
  이 프로젝트의 의미
════════════════════════════════════════════════
  Snake 게임을 통해 강화학습(Reinforcement Learning)을 직접 구현한 프로젝트.

  강화학습이란?
    - 에이전트(뱀)가 환경(격자판)과 상호작용하면서
      보상(먹이 +10, 죽음 -50)을 최대화하도록 스스로 학습하는 AI 기법.
    - 사람이 규칙을 가르치는 게 아니라, 수십만 번의 시행착오를 통해
      AI가 스스로 "어떻게 움직이면 오래 살고 먹이를 많이 먹는지" 터득한다.

  왜 Snake인가?
    - 규칙이 단순해서 환경 구현이 쉬움
    - 처음엔 쉽고 (먹이 찾기), 갈수록 어려워짐 (몸이 길어질수록 피하기 어려움)
    - 단기 목표(먹이 먹기)와 장기 목표(죽지 않기)가 충돌 → 흥미로운 학습 문제
    - 강화학습 교과서에 자주 등장하는 대표적인 예제

  사용 기술:
    - PPO (Proximal Policy Optimization): 안정적인 강화학습 알고리즘
    - MLP (다층 퍼셉트론): 20차원 벡터를 입력받아 행동을 결정하는 신경망 (256→128→행동)
    - gymnasium: RL 환경 표준 인터페이스
    - stable-baselines3: PPO 구현체
    - pygame: 게임 렌더링
    - A* (에이스타): 몸통을 피해 먹이까지 최단 경로를 계산하는 길찾기 알고리즘
                     매 스텝마다 실행 → AI에게 "GPS 경로" 정보를 관찰 벡터로 제공

  AI가 보는 정보 (20차원 벡터):
    [0~2]  즉각 위험: 직진/우/좌 바로 앞 1칸이 벽이나 몸통인가?
    [3~5]  예측 위험: 직진/우/좌 2칸 앞이 위험한가? (코너 함정 예방)
    [6~9]  현재 이동 방향 (위/아래/왼/오른 중 하나만 1.0)
    [10~13] 먹이가 어느 방향에 있는가? (위/아래/왼/오른)
    [14~17] A*가 추천하는 다음 이동 방향 (몸통 우회 고려)
    [18]   A* 경로 존재 여부 (먹이까지 갈 수 있는가?)
    [19]   A* 경로 길이 (짧을수록 1에 가까움)
════════════════════════════════════════════════

학습된 모델을 불러와서 AI가 Snake를 플레이하는 것을 시각화한다.

실행:
    python play.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from env.snake_env import SnakeEnv
from game.constants import FPS_WATCH
import time

# EvalCallback이 학습 중 가장 높은 점수를 기록한 모델을 자동 저장하는 경로
MODEL_PATH = "models/best/best_model"

# ── 환경 및 모델 로드 ─────────────────────────
# render_mode="human" → pygame 창을 띄워 시각화
env   = SnakeEnv(render_mode="human", fps=FPS_WATCH)
model = PPO.load(MODEL_PATH, env=env)   # 저장된 모델 가중치 로드

print(f"모델 로드 완료: {MODEL_PATH}")
print("AI 플레이 시작! (창 닫으면 종료)\n")

# ── AI 플레이 루프 ───────────────────────────
# 게임이 끝나면 자동으로 다음 에피소드를 시작하며 무한 반복
episode = 0
while True:
    obs, _ = env.reset()    # 게임 초기화, 첫 번째 관찰값 받기
    done   = False
    score  = 0
    episode += 1

    while not done:
        # deterministic=True: 가장 확률이 높은 행동만 선택 (탐험 없이 최선의 플레이)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done  = terminated or truncated   # 죽거나 타임아웃이면 에피소드 종료
        score = info["score"]             # 실제 먹은 먹이 수

    print(f"에피소드 {episode:3d} | 점수: {score}")
    time.sleep(0.5)   # 에피소드 사이 잠깐 멈춤 (다음 판 시작 전 여유)
