"""
train.py
────────
PPO + MlpPolicy + 병렬 환경(8개)으로 Snake 에이전트를 무한 학습.
Ctrl+C로 중단하면 자동 저장.

AI가 보는 관찰 벡터 (20차원):
  [0~2]  즉각 위험 (1칸 앞: 직진/우/좌)
  [3~5]  예측 위험 (2칸 앞: 직진/우/좌) ← 코너 함정 예방
  [6~9]  현재 이동 방향 one-hot (위/아래/왼/오른)
  [10~13] 먹이 상대 방향 (위/아래/왼/오른)
  [14~17] A* 추천 방향 (몸통 우회 최단경로 첫 스텝)
  [18]   A* 경로 존재 여부
  [19]   A* 경로 길이 정규화

신경망 구조: 입력(20) → 256 → 128 → 출력(3가지 행동)

학습 흐름:
  1. 8개의 Snake 환경을 동시에 실행 (병렬 학습 → 데이터 수집 속도 8배)
  2. 512 스텝마다 수집한 경험으로 신경망 가중치 업데이트 (PPO)
  3. 10,000 스텝마다 별도 환경에서 현재 모델 평가 → 최고 모델 자동 저장
  4. 목표 점수 달성 시 목표를 5점씩 자동 상향 (5→10→15→ ... 무한 성장)
  5. Ctrl+C로 중단하면 models/ppo_snake_final.zip 저장

실행:
    python train.py

실시간 그래프 (다른 터미널):
    python plot.py
"""

import os
import sys
import csv
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from env.snake_env import SnakeEnv

# 저장 폴더 생성 (없으면 자동 생성)
os.makedirs("models/best", exist_ok=True)
os.makedirs("logs",        exist_ok=True)

# 병렬 환경 수: CPU 코어 수에 맞게 조정 가능 (많을수록 빠름)
N_ENVS = 8


# ══════════════════════════════════════════════
#  학습 진행 로그 콜백
#  - 매 롤아웃(데이터 수집 단위)이 끝날 때마다 호출됨
#  - 실제 게임 점수(먹은 먹이 수)를 CSV에 기록
#  - plot.py가 이 CSV를 읽어 실시간 그래프를 그림
# ══════════════════════════════════════════════
class TrainLogCallback(BaseCallback):
    def __init__(self, csv_path: str = "logs/train_log.csv"):
        super().__init__()
        self.csv_path    = csv_path
        self._ep_scores  = []   # 이번 롤아웃에서 끝난 에피소드들의 점수 누적

        # CSV 파일이 없으면 헤더 행 생성
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "total_steps", "avg_score", "max_score", "min_score", "episodes"]
                )

    def _on_step(self) -> bool:
        # 매 스텝마다 호출 → 에피소드가 끝난 환경의 info["score"] 수집
        dones = self.locals.get("dones", [])   # 각 환경의 done 여부 (길이=N_ENVS)
        infos = self.locals.get("infos", [])   # 각 환경의 info 딕셔너리
        for done, info in zip(dones, infos):
            if done:
                # done=True인 환경만 점수 저장 (에피소드 종료 시점)
                self._ep_scores.append(info.get("score", 0))
        return True   # False를 반환하면 학습 중단

    def on_rollout_end(self) -> None:
        # 롤아웃 1회(8환경 × 512스텝 = 4096 스텝) 완료 후 호출
        if len(self._ep_scores) == 0:
            return   # 이번 롤아웃에서 종료된 에피소드가 없으면 스킵

        scores = self._ep_scores
        avg = sum(scores) / len(scores)   # 평균 점수
        mx  = max(scores)                 # 최고 점수
        mn  = min(scores)                 # 최저 점수
        steps = self.num_timesteps        # 누적 학습 스텝 수

        # 터미널 출력
        print(f"  스텝 {steps:>10,} | 평균: {avg:5.1f} | 최고: {mx:4.0f} | 최저: {mn:4.0f} | 에피소드: {len(scores)}")

        # CSV에 한 줄 추가 (plot.py가 실시간으로 읽음)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                steps, round(avg, 2), mx, mn, len(scores)
            ])

        self._ep_scores = []   # 다음 롤아웃을 위해 초기화


# ══════════════════════════════════════════════
#  목표 점수 자동 상향 콜백
#  - 최근 20판 평균이 목표 점수 이상이면 목표를 5점 올림
#  - 5 → 10 → 15 → 20 → ... 무한히 높아짐
#  - 학습이 정체되지 않고 계속 더 높은 목표를 향해 학습하게 함
# ══════════════════════════════════════════════
class TargetUpCallback(BaseCallback):
    def __init__(self, start_target: float = 5.0, step: float = 5.0, check_freq: int = 10_000):
        super().__init__()
        self.target_score    = start_target   # 현재 목표 점수 (처음엔 5점)
        self.step            = step           # 목표 달성 시 올릴 점수 (5점씩)
        self.check_freq      = check_freq     # 몇 스텝마다 체크할지
        self._recent_scores  = []             # 최근 에피소드 점수 누적

    def _on_step(self) -> bool:
        # 에피소드 종료 시 실제 게임 점수 수집
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done:
                self._recent_scores.append(info.get("score", 0))

        # check_freq 스텝마다만 목표 달성 여부 확인 (매 스텝 체크는 비효율적)
        if self.num_timesteps % self.check_freq != 0:
            return True
        if len(self._recent_scores) < 20:
            return True   # 샘플이 20개 미만이면 아직 판단 불가

        # 최근 20판 평균으로 목표 달성 여부 판단
        avg = sum(self._recent_scores[-20:]) / 20
        if avg >= self.target_score:
            prev = self.target_score
            self.target_score += self.step   # 목표 상향!
            print(f"\n★ 목표 달성! 평균 {avg:.1f} ≥ {prev:.0f} → 새 목표: {self.target_score:.0f}")
        return True


# ══════════════════════════════════════════════
#  환경 설정
# ══════════════════════════════════════════════
print(f"병렬 환경 {N_ENVS}개 생성 중...")

# 학습용: 8개 환경 병렬 실행 (렌더링 없음 → 최대 속도)
train_env = make_vec_env(lambda: SnakeEnv(render_mode="none"), n_envs=N_ENVS)

# 평가용: 1개 환경 (학습과 별도로 성능 측정)
eval_env  = SnakeEnv(render_mode="none")

# EvalCallback: 일정 스텝마다 현재 모델을 평가하고 최고 성능 모델을 저장
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="models/best/",    # 최고 모델 저장 경로
    log_path="logs/eval/",                  # 평가 로그 저장 경로
    eval_freq=max(10_000 // N_ENVS, 1),     # 실제 스텝 기준 10,000 스텝마다 평가
    n_eval_episodes=10,                     # 10판 플레이해서 평균 점수 측정
    deterministic=True,                     # 평가 시엔 랜덤성 없이 최선의 행동만
    verbose=1,
)

log_cb       = TrainLogCallback()
earlystop_cb = TargetUpCallback(start_target=5.0, step=5.0)


# ══════════════════════════════════════════════
#  모델: 이어서 학습 or 새로 시작
#  - 프로그램을 껐다 켜도 이전 학습을 이어서 진행
# ══════════════════════════════════════════════
FINAL_PATH = "models/ppo_snake_final"

if os.path.exists(FINAL_PATH + ".zip"):
    # 저장된 모델이 있으면 가중치를 불러와서 이어서 학습
    print(f"기존 모델 발견 → 이어서 학습: {FINAL_PATH}.zip")
    model = PPO.load(FINAL_PATH, env=train_env)
else:
    # 처음 실행: 새 모델 생성
    print("새 모델로 학습 시작 (PPO + MlpPolicy 20차원 입력 + 8 envs)")
    model = PPO(
        policy="MlpPolicy",       # 벡터 입력용 MLP 신경망 정책
        env=train_env,
        learning_rate=3e-4,       # 학습률: 가중치 업데이트 크기 (너무 크면 불안정)
        n_steps=512,              # 환경당 512스텝 수집 후 한 번 업데이트
        batch_size=256,           # 미니배치 크기 (한 번에 256개 샘플로 학습)
        n_epochs=10,              # 같은 데이터로 10번 반복 학습
        gamma=0.99,               # 미래 보상 할인율 (1에 가까울수록 먼 미래 중시)
        gae_lambda=0.95,          # GAE 파라미터 (어드밴티지 추정 안정화)
        clip_range=0.2,           # PPO 클리핑 범위 (정책이 너무 급변하지 않도록)
        ent_coef=0.01,            # 엔트로피 계수 (탐험 장려, 너무 일찍 수렴 방지)
        policy_kwargs={"net_arch": [256, 128]},   # 신경망 구조: 256→128→행동 (입력 20차원으로 늘어서 첫 레이어 확장)
        verbose=0,
    )


# ══════════════════════════════════════════════
#  무한 학습 루프
#  - CHUNK 스텝씩 학습하고, 완료마다 모델 저장
#  - Ctrl+C로 언제든 중단 가능
# ══════════════════════════════════════════════
print("\n학습 시작! (Ctrl+C로 중단)")
print(f"실시간 그래프: 다른 터미널에서 python plot.py 실행\n")
print(f"{'스텝':>12} | {'평균점수':>6} | {'최고':>4} | {'최저':>4} | 에피소드")
print("-" * 55)

# 8환경 × 512스텝 × 12회 = 약 49,152 스텝씩 학습
CHUNK = N_ENVS * 512 * 12
total = 0   # 이번 실행에서의 총 학습 스텝 카운터

try:
    while True:
        total += CHUNK
        model.learn(
            total_timesteps=CHUNK,
            callback=[eval_cb, log_cb, earlystop_cb],
            log_interval=999999,       # SB3 기본 로그 출력 억제 (우리 콜백 사용)
            progress_bar=False,
            reset_num_timesteps=False, # 이어서 학습할 때 스텝 카운터 유지
        )
        model.save(FINAL_PATH)   # CHUNK마다 중간 저장 (비정상 종료 대비)

except KeyboardInterrupt:
    print("\n\nCtrl+C 감지 → 저장 중...")

finally:
    # 정상 종료든 강제 종료든 항상 저장
    model.save(FINAL_PATH)
    print(f"저장 완료: {FINAL_PATH}.zip")
    print(f"총 학습 스텝: {total:,}")
    print("플레이 확인: python play.py")
