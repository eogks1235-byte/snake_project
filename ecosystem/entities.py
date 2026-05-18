"""식물 / 초식 / 육식 — 3계층 entity 정의."""
import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ── 공통 파라미터 ──
WORLD_W = 1000
WORLD_H = 600


# ──────────────────────────────────────────────────────────────
# Plant
# ──────────────────────────────────────────────────────────────

PLANT_GROW_RATE = 0.85       # tick당 성장량
PLANT_MAX_GROWTH = 30.0      # 다 자라면 분열 가능
PLANT_SPAWN_PROB = 0.030     # 다 자란 식물이 자손 만들 확률 (tick당)
PLANT_SPAWN_RADIUS = 35.0    # 분열 반경


@dataclass
class Plant:
    x: float
    y: float
    growth: float = 0.0          # 0 ~ PLANT_MAX_GROWTH
    alive: bool = True

    def color(self) -> tuple:
        # 어린 식물은 노란-초록, 다 자라면 짙은 초록
        ratio = min(1.0, self.growth / PLANT_MAX_GROWTH)
        r = int(160 - ratio * 80)
        g = int(180 + ratio * 40)
        b = int(80 + ratio * 30)
        return (r, g, b)

    def radius(self) -> int:
        return max(2, int(2 + self.growth / 10))


# ──────────────────────────────────────────────────────────────
# Herbivore (초식)
# ──────────────────────────────────────────────────────────────

HERB_MAX_SPEED = 2.2
HERB_VISION = 70.0
HERB_SIZE = 5
HERB_METAB = 0.14
HERB_INIT_E = 70.0
HERB_REPRODUCE = 140.0
HERB_REPRODUCE_COST = 75.0
HERB_MAX_E = 200.0
HERB_PLANT_GAIN = 30.0       # 식물 먹을 때 회복

# boids 가중치
HERB_W_COHESION = 0.012
HERB_W_ALIGN = 0.05
HERB_W_SEPARATE = 0.18
HERB_W_FOOD = 0.32
HERB_W_FLEE = 0.7
HERB_W_NOISE = 0.18

HERB_COLOR = (110, 170, 230)
HERB_COLOR_DARK = (60, 100, 160)


@dataclass
class Herbivore:
    x: float
    y: float
    vx: float
    vy: float
    energy: float = HERB_INIT_E
    age: int = 0
    alive: bool = True
    id: int = 0
    parent_id: int = -1
    children: int = 0           # 살아 있는 동안 누적된 자식 수


# ──────────────────────────────────────────────────────────────
# Carnivore (육식)
# ──────────────────────────────────────────────────────────────

CARN_MAX_SPEED = 2.6
CARN_VISION = 110.0
CARN_SIZE = 7
CARN_METAB = 0.16
CARN_INIT_E = 90.0
CARN_REPRODUCE = 180.0
CARN_REPRODUCE_COST = 100.0
CARN_MAX_E = 250.0
CARN_HUNT_GAIN = 75.0        # 초식 1마리 잡을 때 회복
CARN_KILL_RADIUS = CARN_SIZE + HERB_SIZE + 1

CARN_W_COHESION = 0.005
CARN_W_ALIGN = 0.03
CARN_W_SEPARATE = 0.22
CARN_W_HUNT = 0.45
CARN_W_NOISE = 0.10

CARN_COLOR = (230, 100, 90)
CARN_COLOR_DARK = (160, 60, 50)


@dataclass
class Carnivore:
    x: float
    y: float
    vx: float
    vy: float
    energy: float = CARN_INIT_E
    age: int = 0
    alive: bool = True
    id: int = 0
    parent_id: int = -1
    children: int = 0


# ──────────────────────────────────────────────────────────────
# Carrion — 시체는 흙으로, 흙은 다시 생명으로
# ──────────────────────────────────────────────────────────────

CARRION_LIFESPAN = 140          # tick
CARRION_PLANT_PROB = 0.014      # 분해되며 그 자리에 새 식물이 돋아날 확률


@dataclass
class Carrion:
    x: float
    y: float
    species: str = 'herb'       # 'herb' / 'carn' — 색조 구분용
    age: int = 0
    lifespan: int = CARRION_LIFESPAN
    alive: bool = True


# ──────────────────────────────────────────────────────────────
# FearMark — 사냥당한 자리에 남는 공포의 흔적
# ──────────────────────────────────────────────────────────────

FEAR_LIFESPAN = 240
FEAR_RADIUS = 60.0
FEAR_FORCE = 0.55


@dataclass
class FearMark:
    x: float
    y: float
    age: int = 0
    lifespan: int = FEAR_LIFESPAN
    radius: float = FEAR_RADIUS
    alive: bool = True


# ──────────────────────────────────────────────────────────────
# Day/Night — 영원회귀의 사이클
# ──────────────────────────────────────────────────────────────

DAY_CYCLE_LEN = 900             # tick — 약 30초 @ 30fps
DAY_GROW_SPAN = 0.6             # 낮/밤 식물 성장 진폭 (light=0 → 0.4×, =1 → 1.6×)


# ──────────────────────────────────────────────────────────────
# Founder & Elder 표시 임계치
# ──────────────────────────────────────────────────────────────

FOUNDER_MIN_CHILDREN = 3        # 자손 3 이상이어야 시조로 인정
ELDER_MIN_AGE = 600             # 600 tick 이상 살아야 장로로 인정
