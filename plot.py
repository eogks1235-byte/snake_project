"""
plot.py
───────
train.py 실행 중 학습 진행 상황을 실시간 그래프로 보여준다.

동작 방식:
  - logs/train_log.csv 파일을 5초마다 읽어서 그래프를 갱신
  - train.py와 별도 터미널에서 동시에 실행 가능
  - 창을 닫으면 종료 (train.py 학습은 계속 진행됨)

그래프 구성:
  위쪽: 점수 변화 (평균선 + 최저~최고 범위 음영)
  아래쪽: 이동 평균 (최근 10개 평균 → 학습 추세 파악)

실행 (train.py와 별도 터미널):
    python plot.py
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.font_manager as fm
except ImportError:
    print("matplotlib 설치 필요: pip install matplotlib")
    sys.exit(1)

# ── 한글 폰트 설정 (Windows) ─────────────────
# matplotlib 기본 폰트는 한글 미지원 → 깨짐 발생
# 설치된 폰트 중 한글 지원 폰트를 순서대로 찾아 적용
for font in ["Malgun Gothic", "NanumGothic", "AppleGothic", "MS Gothic"]:
    if any(f.name == font for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = font
        break
plt.rcParams["axes.unicode_minus"] = False   # 마이너스(-) 기호 깨짐 방지

CSV_PATH = "logs/train_log.csv"   # train.py가 기록하는 로그 파일 경로
INTERVAL = 5000                   # 그래프 갱신 주기 (밀리초, 5초마다 갱신)

# ── 그래프 초기 설정 ──────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
fig.suptitle("Snake RL 학습 진행", fontsize=14)


def read_csv():
    """
    train_log.csv를 읽어 학습 스텝, 평균/최고/최저 점수 리스트를 반환.
    파일이 없거나 읽기 실패 시 빈 리스트 반환.
    """
    if not os.path.exists(CSV_PATH):
        return [], [], [], []

    steps, avgs, maxes, mines = [], [], [], []
    try:
        with open(CSV_PATH, newline="") as f:
            for row in list(csv.DictReader(f)):
                steps.append(int(row["total_steps"]))
                avgs.append(float(row["avg_score"]))
                maxes.append(float(row["max_score"]))
                mines.append(float(row["min_score"]))
    except Exception:
        pass   # 파일을 동시에 쓰고 읽는 경우 오류 무시

    return steps, avgs, maxes, mines


def update(frame):
    """
    FuncAnimation이 INTERVAL마다 호출하는 갱신 함수.
    CSV를 다시 읽어 두 그래프를 새로 그린다.
    """
    steps, avgs, maxes, mines = read_csv()

    if not steps:
        return   # 데이터가 없으면 스킵

    # ── 위쪽 그래프: 점수 변화 ──────────────────
    ax1.clear()
    ax1.plot(steps, avgs, color="royalblue", linewidth=2, label="평균 점수")
    # fill_between: 최저~최고 범위를 반투명 음영으로 표시
    ax1.fill_between(steps, mines, maxes, alpha=0.15, color="royalblue", label="최저~최고")
    ax1.set_ylabel("점수")
    ax1.set_title(f"점수 변화  (최신: 평균 {avgs[-1]:.1f} / 최고 {maxes[-1]:.0f})")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # ── 아래쪽 그래프: 이동 평균 ────────────────
    # 이동 평균(Moving Average): 최근 N개의 값을 평균 → 노이즈 제거, 추세 파악
    ax2.clear()
    window = 10   # 최근 10개 데이터 포인트로 평균 계산
    if len(avgs) >= window:
        ma = [sum(avgs[i:i+window])/window for i in range(len(avgs)-window+1)]
        ma_steps = steps[window-1:]
        ax2.plot(ma_steps, ma, color="tomato", linewidth=2, label=f"이동평균({window})")
    ax2.plot(steps, avgs, color="royalblue", alpha=0.3, linewidth=1)   # 원본 데이터 (연하게)
    ax2.set_xlabel("학습 스텝")
    ax2.set_ylabel("점수")
    ax2.set_title("이동 평균 (학습 추세) — 우상향이면 학습 중!")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()   # 그래프 간 여백 자동 조정


print("실시간 그래프 시작! (창 닫으면 종료)")
print(f"CSV 경로: {CSV_PATH}")
print(f"갱신 주기: {INTERVAL//1000}초\n")

# FuncAnimation: update 함수를 INTERVAL마다 반복 호출
ani = animation.FuncAnimation(fig, update, interval=INTERVAL, cache_frame_data=False)
plt.show()
