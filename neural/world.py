"""뉴럴 환경 — prey + predator 양쪽 뇌 진화, 벽, 음식 종류, 페로몬, 종 분화, 조상 추적."""
import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Iterable

import numpy as np

from .brain import (Brain, N_SECTORS, N_MEMORY, N_IN, N_OUT, N_HIDDEN,
                    PRED_N_IN, PRED_N_HIDDEN, PRED_N_OUT, OUT_MEM_START)


HEBBIAN_LR = 0.0015
HEBBIAN_REWARD_FLOOR = 0.5

# PPO 정책 사양 — 메모리 셀 없는 단순 reactive 정책 (vx, vy만 출력)
PPO_OBS_DIM = N_SECTORS * 4 + 3 + 1   # 4채널 8섹터 + energy + vx + vy + bias = 36
PPO_ACT_DIM = 2

# A: 포식자 PPO 사양 — prey 8섹터 + 동료 8섹터 + energy + vx + vy + bias = 19
PPO_PRED_OBS_DIM = N_SECTORS * 2 + 3 + 1
PPO_PRED_ACT_DIM = 2


WORLD_W = 1000
WORLD_H = 600

MAX_SPEED = 2.5
SIZE = 5
VISION = 90.0
METABOLISM = 0.18
EAT_RADIUS = SIZE + 4

# 음식 — 흔한 풀(값 1) + 희귀 열매(값 2.5) + K: 단백질(값 4, 매우 희귀)
INITIAL_FOOD = 220
FOOD_SPAWN_PER_TICK = 1.5
FOOD_ENERGY = 22.0
RARE_FOOD_PROB = 0.07
RARE_FOOD_VALUE = 2.5
COMMON_FOOD_VALUE = 1.0
PROTEIN_FOOD_PROB = 0.015        # K: 매우 희귀, 매우 영양 (multi-resource 모드)
PROTEIN_FOOD_VALUE = 4.0

# H: 커리큘럼 — 시작은 안전 환경, 점진적으로 어려움
CURRICULUM_PRED_ON_TICK = 600    # 이 틱부터 포식자 활성
CURRICULUM_WALL_ON_TICK = 200    # 이 틱부터 벽 활성

# J: 동적 환경 — 음식 계절성 / 포식자 웨이브
DYNAMIC_FOOD_CYCLE = 800         # 음식 풍년/흉년 주기 (tick)
DYNAMIC_PRED_WAVE_TICK = 1200    # 포식자 웨이브 주기

# 인구
INITIAL_POPULATION = 50
PPO_TARGET_POPULATION = 150     # PPO 모드 시 유지할 인구 (진화 모드 자연 평형치와 매칭)
INITIAL_ENERGY = 60.0
REPRODUCE_THRESHOLD = 110.0
REPRODUCE_COST = 60.0
SEXUAL_RANGE = 55.0
SEXUAL_BONUS_COST = 10.0
ASEXUAL_EXTRA = 20.0
MAX_ENERGY = 200.0
MAX_AGE = 1500

SPECIATION_THRESHOLD = 45.0      # random 평균 ~42, 시간 따라 cross-lineage 격리 형성

DIVERSITY_SAMPLE = 16
DIVERSITY_INTERVAL = 15

# 페로몬
PHEROMONE_LIFE = 80
PHEROMONE_MAX = 800

# 포식자
INITIAL_PREDATORS = 3
PREDATOR_MAX = 8
PREDATOR_SPAWN_INTERVAL = 600
PREDATOR_SIZE = 8
PREDATOR_SPEED = 2.1
PREDATOR_VISION = 140.0
PREDATOR_KILL_RADIUS = PREDATOR_SIZE + SIZE + 2
PREDATOR_KILL_ENERGY = 80.0
PREDATOR_METABOLISM = 0.32
PREDATOR_INITIAL_ENERGY = 120.0
PREDATOR_REPRODUCE_THRESHOLD = 240.0
PREDATOR_REPRODUCE_COST = 140.0
PREDATOR_MAX_ENERGY = 300.0

# 벽 — 정적 직사각형 장애물
INITIAL_WALLS = 5
WALL_MIN_W, WALL_MAX_W = 40, 140
WALL_MIN_H, WALL_MAX_H = 30, 100
WALL_VISION_SAMPLES = 6           # 한 벽당 가장자리 샘플 점 개수 (prey의 predator-채널에 주입)


@dataclass
class Food:
    x: float
    y: float
    value: float = COMMON_FOOD_VALUE


@dataclass
class Pheromone:
    x: float
    y: float
    life: int


@dataclass
class Wall:
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def sample_points(self, n: int) -> List[Tuple[float, float]]:
        """가장자리 위 n점 — 시야 입력에 주입용."""
        pts = []
        perim = 2 * (self.w + self.h)
        for i in range(n):
            t = (i + 0.5) / n * perim
            if t < self.w:
                pts.append((self.x + t, self.y))
            elif t < self.w + self.h:
                pts.append((self.x + self.w, self.y + (t - self.w)))
            elif t < 2 * self.w + self.h:
                pts.append((self.x + self.w - (t - self.w - self.h), self.y + self.h))
            else:
                pts.append((self.x, self.y + self.h - (t - 2 * self.w - self.h)))
        return pts


@dataclass
class Lineage:
    creature_id: int
    parent_ids: Tuple[int, ...]
    generation: int
    born_tick: int
    color: Tuple[int, int, int]
    kind: str = 'prey'           # 'prey' 또는 'pred'
    ancestor_id: int = -1
    died_tick: Optional[int] = None
    final_eaten: int = 0
    final_age: int = 0


@dataclass
class Creature:
    creature_id: int
    x: float
    y: float
    vx: float
    vy: float
    energy: float
    age: int
    brain: Brain
    generation: int
    parent_ids: Tuple[int, ...]
    ancestor_id: int
    memory: np.ndarray = field(
        default_factory=lambda: np.zeros(N_MEMORY, dtype=np.float32))
    lifetime_eaten: int = 0
    alive: bool = True
    last_inputs: Optional[np.ndarray] = None
    last_hidden: Optional[np.ndarray] = None
    last_outputs: Optional[np.ndarray] = None
    last_energy: float = 0.0
    # C: GRU PPO용 — 개체별 recurrent hidden state (None이면 zero)
    rnn_hidden: Optional[np.ndarray] = None

    def color(self):
        return self.brain.signature()


@dataclass
class Predator:
    creature_id: int
    x: float
    y: float
    vx: float
    vy: float
    energy: float
    age: int
    brain: Brain
    generation: int
    parent_ids: Tuple[int, ...]
    memory: np.ndarray = field(
        default_factory=lambda: np.zeros(N_MEMORY, dtype=np.float32))
    lifetime_kills: int = 0
    alive: bool = True
    # A: 포식자 PPO용
    last_inputs: Optional[np.ndarray] = None
    last_outputs: Optional[np.ndarray] = None
    last_energy: float = 0.0
    rnn_hidden: Optional[np.ndarray] = None

    def color(self):
        # 포식자 컬러: brain signature를 빨강 톤으로 시프트
        sig = self.brain.signature()
        return (min(255, sig[0] // 2 + 130), sig[1] // 3, sig[2] // 3)


class World:
    def __init__(self, seed: int, initial_brains: Optional[List[Brain]] = None,
                 prey_hidden: int = N_HIDDEN, pred_hidden: int = PRED_N_HIDDEN,
                 hebbian: bool = False,
                 curriculum: bool = False,
                 dynamic_env: bool = False,
                 multi_resource: bool = False):
        self.seed = seed
        self.rng = random.Random(seed)
        self.tick = 0
        self.births = 0
        self.deaths = 0
        self.kills = 0
        self.sexual_births = 0
        self.pred_births = 0
        self.history: List[dict] = []
        self.lineage_records: Dict[int, Lineage] = {}
        self._next_id = 0
        self._last_diversity = 0.0
        self._last_ancestor_counts: Dict[int, int] = {}

        self.prey_hidden = prey_hidden
        self.pred_hidden = pred_hidden
        self.hebbian = hebbian
        self.curriculum = curriculum
        self.dynamic_env = dynamic_env
        self.multi_resource = multi_resource

        # PPO 모드 — shared_policy가 None이 아니면 모든 prey가 이 정책으로 행동.
        # 반드시 disable_reproduction=True와 함께 — 진화/번식 안 함, 죽으면 새 spawn.
        self.shared_policy = None
        self.predator_policy = None
        self.disable_reproduction = False
        self.disable_pred_reproduction = False
        self._ppo_buffer: List[dict] = []
        self._pred_ppo_buffer: List[dict] = []
        self.ppo_buffer_max: Optional[int] = 256
        self.history_max: Optional[int] = 5000

        self.creatures: List[Creature] = []
        self.predators: List[Predator] = []
        self.foods: List[Food] = []
        self.pheromones: List[Pheromone] = []
        self.walls: List[Wall] = []
        self.injected_brain: Optional[Brain] = None
        self._wall_sense_xy: Optional[np.ndarray] = None
        self._initial_brains = initial_brains
        self._spawn_initial()
        self._refresh_wall_cache()

    # -------- ID + 라이프사이클 --------
    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _record_birth(self, c, kind: str = 'prey'):
        anc = getattr(c, 'ancestor_id', c.creature_id)
        self.lineage_records[c.creature_id] = Lineage(
            creature_id=c.creature_id,
            parent_ids=c.parent_ids,
            generation=c.generation,
            born_tick=self.tick,
            color=c.brain.signature(),
            kind=kind,
            ancestor_id=anc,
        )

    def _record_death(self, c):
        rec = self.lineage_records.get(c.creature_id)
        if rec is not None:
            rec.died_tick = self.tick
            rec.final_eaten = getattr(c, 'lifetime_eaten',
                                       getattr(c, 'lifetime_kills', 0))
            rec.final_age = c.age

    def inject_brain(self, brain: Brain) -> None:
        self.injected_brain = brain

    # -------- 초기 스폰 --------
    def _spawn_initial(self):
        for i in range(INITIAL_POPULATION):
            if self._initial_brains is not None and i < len(self._initial_brains):
                brain = self._initial_brains[i]
            else:
                brain = Brain.random(self.rng, N_IN, self.prey_hidden, N_OUT)
            cid = self._new_id()
            c = Creature(
                creature_id=cid,
                x=self.rng.uniform(0, WORLD_W),
                y=self.rng.uniform(0, WORLD_H),
                vx=self.rng.uniform(-1, 1),
                vy=self.rng.uniform(-1, 1),
                energy=INITIAL_ENERGY,
                age=0,
                brain=brain,
                generation=1,
                parent_ids=(),
                ancestor_id=cid,
            )
            # 벽 안에 스폰됐으면 옆으로 이동
            while self._in_any_wall(c.x, c.y):
                c.x = self.rng.uniform(0, WORLD_W)
                c.y = self.rng.uniform(0, WORLD_H)
            self.creatures.append(c)
            self._record_birth(c)
        for _ in range(INITIAL_FOOD):
            self._spawn_one_food()
        # H: 커리큘럼 — 벽/포식자는 처음엔 비활성
        if not self.curriculum:
            for _ in range(INITIAL_WALLS):
                self._spawn_wall()
            for _ in range(INITIAL_PREDATORS):
                self._spawn_predator()

    def _spawn_one_food(self):
        r = self.rng.random()
        # K: multi_resource 모드 — 단백질(매우 희귀) 가능
        if self.multi_resource and r < PROTEIN_FOOD_PROB:
            value = PROTEIN_FOOD_VALUE
        elif r < RARE_FOOD_PROB:
            value = RARE_FOOD_VALUE
        else:
            value = COMMON_FOOD_VALUE
        for _ in range(8):
            x = self.rng.uniform(0, WORLD_W)
            y = self.rng.uniform(0, WORLD_H)
            if not self._in_any_wall(x, y):
                break
        self.foods.append(Food(x=x, y=y, value=value))

    def _spawn_wall(self):
        for _ in range(20):
            w = self.rng.uniform(WALL_MIN_W, WALL_MAX_W)
            h = self.rng.uniform(WALL_MIN_H, WALL_MAX_H)
            x = self.rng.uniform(40, WORLD_W - 40 - w)
            y = self.rng.uniform(40, WORLD_H - 40 - h)
            wall = Wall(x, y, w, h)
            # 다른 벽과 겹치지 않게
            if not any(self._walls_overlap(wall, w2, pad=20) for w2 in self.walls):
                self.walls.append(wall)
                return

    def _walls_overlap(self, a: Wall, b: Wall, pad: float = 0) -> bool:
        return not (a.x + a.w + pad < b.x
                    or b.x + b.w + pad < a.x
                    or a.y + a.h + pad < b.y
                    or b.y + b.h + pad < a.y)

    def _in_any_wall(self, x: float, y: float) -> bool:
        for w in self.walls:
            if w.contains(x, y):
                return True
        return False

    def _safe_spawn_near(self, x: float, y: float,
                         fallback_x: float, fallback_y: float) -> Tuple[float, float]:
        """벽 안이면 부모 위치(fallback)로, 그것도 벽 안이면 무작위 반경 탐색."""
        if not self._in_any_wall(x, y):
            return x, y
        if not self._in_any_wall(fallback_x, fallback_y):
            return fallback_x, fallback_y
        for _ in range(8):
            r = self.rng.uniform(0, 25)
            a = self.rng.uniform(0, math.tau)
            tx = (fallback_x + math.cos(a) * r) % WORLD_W
            ty = (fallback_y + math.sin(a) * r) % WORLD_H
            if not self._in_any_wall(tx, ty):
                return tx, ty
        return fallback_x, fallback_y  # 포기

    def _refresh_wall_cache(self):
        if not self.walls:
            self._wall_sense_xy = np.empty((0, 2), dtype=np.float32)
            return
        pts = []
        for w in self.walls:
            pts.extend(w.sample_points(WALL_VISION_SAMPLES))
        self._wall_sense_xy = np.array(pts, dtype=np.float32)

    def _spawn_predator(self, brain: Optional[Brain] = None,
                        x: Optional[float] = None, y: Optional[float] = None):
        if brain is None:
            brain = Brain.random(self.rng, PRED_N_IN, self.pred_hidden, PRED_N_OUT)
        if x is None or y is None:
            for _ in range(10):
                x = self.rng.uniform(0, WORLD_W)
                y = self.rng.uniform(0, WORLD_H)
                if not self._in_any_wall(x, y):
                    break
        cid = self._new_id()
        p = Predator(
            creature_id=cid,
            x=x, y=y,
            vx=self.rng.uniform(-1, 1),
            vy=self.rng.uniform(-1, 1),
            energy=PREDATOR_INITIAL_ENERGY,
            age=0,
            brain=brain,
            generation=1,
            parent_ids=(),
        )
        self.predators.append(p)

    # -------- 외부 API (마우스 인터랙션) --------
    def add_food_at(self, x: float, y: float, value: float = COMMON_FOOD_VALUE):
        if not self._in_any_wall(x, y):
            self.foods.append(Food(x=x, y=y, value=value))

    def add_predator_at(self, x: float, y: float):
        if len(self.predators) < PREDATOR_MAX and not self._in_any_wall(x, y):
            self._spawn_predator(x=x, y=y)

    # -------- 메인 step --------
    def step(self):
        self.tick += 1
        # H: 커리큘럼 — 정해진 시점에 벽 일괄 등장
        if (self.curriculum and self.tick == CURRICULUM_WALL_ON_TICK
                and not self.walls):
            for _ in range(INITIAL_WALLS):
                self._spawn_wall()
            self._refresh_wall_cache()
        self._spawn_food()
        self._maybe_spawn_predator()
        if self.injected_brain is not None:
            self._inject_champion()
        self._decay_pheromones()
        self._think_and_move_creatures()
        self._think_and_move_predators()
        self._predator_kills()
        self._reproduce_and_die()
        if self.shared_policy is not None:
            self._finalize_ppo_rewards()
        if self.predator_policy is not None:
            self._finalize_pred_ppo_rewards()
        if self.disable_reproduction:
            self._maintain_population()
        if self.disable_pred_reproduction:
            self._maintain_predator_population()
        if self.tick % DIVERSITY_INTERVAL == 0:
            self._last_diversity = self._compute_diversity()
        self._update_ancestor_counts()
        self._record_history()

    def _inject_champion(self):
        cid = self._new_id()
        c = Creature(
            creature_id=cid,
            x=self.rng.uniform(0, WORLD_W),
            y=self.rng.uniform(0, WORLD_H),
            vx=0.0, vy=0.0,
            energy=MAX_ENERGY * 0.6,
            age=0,
            brain=self.injected_brain,
            generation=99,
            parent_ids=(),
            ancestor_id=cid,
        )
        while self._in_any_wall(c.x, c.y):
            c.x = self.rng.uniform(0, WORLD_W)
            c.y = self.rng.uniform(0, WORLD_H)
        self.creatures.append(c)
        self._record_birth(c)
        self.injected_brain = None

    def _spawn_food(self):
        rate = FOOD_SPAWN_PER_TICK
        # J: 동적 환경 — 음식 풍년/흉년 (사인파 0.3x ~ 1.7x)
        if self.dynamic_env:
            phase = (self.tick % DYNAMIC_FOOD_CYCLE) / DYNAMIC_FOOD_CYCLE
            rate *= 1.0 + 0.7 * math.sin(phase * math.tau)
        n_int = int(rate)
        frac = rate - n_int
        n = n_int + (1 if self.rng.random() < frac else 0)
        for _ in range(n):
            self._spawn_one_food()

    def _maybe_spawn_predator(self):
        # H: 커리큘럼 — 일정 시점 전에는 포식자 안 나옴
        if self.curriculum and self.tick < CURRICULUM_PRED_ON_TICK:
            return
        interval = PREDATOR_SPAWN_INTERVAL
        # J: 동적 환경 — 주기적으로 포식자 웨이브 (한 번에 3마리)
        if self.dynamic_env and self.tick > 0 and self.tick % DYNAMIC_PRED_WAVE_TICK == 0:
            for _ in range(3):
                if len(self.predators) < PREDATOR_MAX:
                    self._spawn_predator()
            return
        if (self.tick % interval == 0
                and len(self.predators) < PREDATOR_MAX):
            self._spawn_predator()

    def _decay_pheromones(self):
        keep = []
        for p in self.pheromones:
            p.life -= 1
            if p.life > 0:
                keep.append(p)
        self.pheromones = keep

    # -------- 시야 (벡터화) --------
    def _all_sectors(self, src_xy: np.ndarray, tgt_xy: np.ndarray,
                     vision: float, intensity: Optional[np.ndarray] = None,
                     exclude_self: bool = False) -> np.ndarray:
        M = src_xy.shape[0]
        N = tgt_xy.shape[0]
        out = np.zeros((M, N_SECTORS), dtype=np.float32)
        if M == 0 or N == 0:
            return out
        dx = tgt_xy[None, :, 0] - src_xy[:, None, 0]
        dy = tgt_xy[None, :, 1] - src_xy[:, None, 1]
        dx = (dx + WORLD_W / 2) % WORLD_W - WORLD_W / 2
        dy = (dy + WORLD_H / 2) % WORLD_H - WORLD_H / 2
        d2 = dx * dx + dy * dy
        valid = (d2 < vision * vision) & (d2 > 1e-6)
        if exclude_self and M == N:
            np.fill_diagonal(valid, False)
        closeness = (1.0 - np.sqrt(np.maximum(d2, 1e-12)) / vision).astype(np.float32)
        closeness[~valid] = 0.0
        if intensity is not None:
            closeness *= intensity[None, :]
        ang = np.mod(np.arctan2(dy, dx), math.tau)
        sectors = (ang / (math.tau / N_SECTORS)).astype(np.int32) % N_SECTORS
        for s in range(N_SECTORS):
            mask = sectors == s
            if mask.any():
                out[:, s] = np.where(mask, closeness, 0.0).max(axis=1)
        return out

    # -------- prey 사고 + 이동 --------
    def _think_and_move_creatures(self):
        if not self.creatures:
            return
        if self.shared_policy is not None:
            return self._policy_think_and_move()
        creature_xy = np.array([(c.x, c.y) for c in self.creatures],
                                dtype=np.float32)
        if self.foods:
            food_xy = np.array([(f.x, f.y) for f in self.foods],
                                dtype=np.float32)
            food_value = np.array([f.value for f in self.foods],
                                   dtype=np.float32) / RARE_FOOD_VALUE  # 0..1
        else:
            food_xy = np.empty((0, 2), dtype=np.float32)
            food_value = np.empty(0, dtype=np.float32)
        # predator + wall 샘플을 한 채널에 합침 (둘 다 "danger")
        if self.predators:
            pred_xy = np.array([(p.x, p.y) for p in self.predators],
                                dtype=np.float32)
        else:
            pred_xy = np.empty((0, 2), dtype=np.float32)
        if self._wall_sense_xy is None:
            self._refresh_wall_cache()
        wall_xy = self._wall_sense_xy
        danger_xy = np.concatenate([pred_xy, wall_xy], axis=0) \
                    if pred_xy.shape[0] + wall_xy.shape[0] > 0 \
                    else np.empty((0, 2), dtype=np.float32)
        if self.pheromones:
            ph_xy = np.array([(p.x, p.y) for p in self.pheromones],
                              dtype=np.float32)
            ph_intensity = np.array(
                [p.life / PHEROMONE_LIFE for p in self.pheromones],
                dtype=np.float32)
        else:
            ph_xy = np.empty((0, 2), dtype=np.float32)
            ph_intensity = np.empty(0, dtype=np.float32)

        food_all = self._all_sectors(creature_xy, food_xy, VISION,
                                      intensity=food_value)
        pred_all = self._all_sectors(creature_xy, danger_xy, VISION)
        kin_all = self._all_sectors(creature_xy, creature_xy, VISION,
                                     exclude_self=True)
        phero_all = self._all_sectors(creature_xy, ph_xy, VISION,
                                       intensity=ph_intensity)

        for idx, c in enumerate(self.creatures):
            if not c.alive:
                continue
            c.last_energy = c.energy   # 행동 전 스냅샷 (Hebbian 보상 계산용)
            inputs = np.empty(N_IN, dtype=np.float32)
            inputs[0:8] = food_all[idx]
            inputs[8:16] = pred_all[idx]
            inputs[16:24] = kin_all[idx]
            inputs[24:32] = phero_all[idx]
            inputs[32] = c.energy / MAX_ENERGY
            inputs[33] = c.vx / MAX_SPEED
            inputs[34] = c.vy / MAX_SPEED
            inputs[35:35 + N_MEMORY] = c.memory
            inputs[-1] = 1.0

            if self.hebbian:
                out, hidden = c.brain.forward_with_hidden(inputs)
                c.last_hidden = hidden
            else:
                out = c.brain.forward(inputs)
            c.last_inputs = inputs
            c.last_outputs = out
            c.vx = float(out[0]) * MAX_SPEED
            c.vy = float(out[1]) * MAX_SPEED
            c.memory = out[OUT_MEM_START:OUT_MEM_START + N_MEMORY].astype(np.float32).copy()

            self._apply_move_with_wall_slide(c, c.vx, c.vy)
            c.energy -= METABOLISM
            c.age += 1

        self._consume_food_batched()
        if self.hebbian:
            self._apply_hebbian_updates()

    def _apply_hebbian_updates(self):
        for c in self.creatures:
            if (not c.alive or c.last_inputs is None
                    or c.last_hidden is None or c.last_outputs is None):
                continue
            reward = c.energy - c.last_energy
            if abs(reward) < HEBBIAN_REWARD_FLOOR:
                continue
            c.brain.hebbian_update(c.last_inputs, c.last_hidden,
                                    c.last_outputs, reward=float(reward),
                                    lr=HEBBIAN_LR)

    def _apply_move_with_wall_slide(self, ent, vx: float, vy: float):
        """벽 충돌 시 x/y 한 축씩 시도 — 벽 따라 미끄러지듯 이동. 막힌 축의 속도는 0으로."""
        new_x = (ent.x + vx) % WORLD_W
        new_y = (ent.y + vy) % WORLD_H
        if not self._in_any_wall(new_x, new_y):
            ent.x, ent.y = new_x, new_y
            return
        # x만 이동
        if not self._in_any_wall(new_x, ent.y):
            ent.x = new_x
            ent.vy = 0.0
            return
        # y만 이동
        if not self._in_any_wall(ent.x, new_y):
            ent.y = new_y
            ent.vx = 0.0
            return
        # 둘 다 막힘 — 살짝 튕겨내기 (시각적 고착 방지)
        ent.vx = -vx * 0.3
        ent.vy = -vy * 0.3

    def _consume_food_batched(self):
        if not self.creatures or not self.foods:
            return
        creature_xy = np.array([(c.x, c.y) for c in self.creatures],
                                dtype=np.float32)
        food_xy = np.array([(f.x, f.y) for f in self.foods], dtype=np.float32)
        food_value = np.array([f.value for f in self.foods], dtype=np.float32)
        M = creature_xy.shape[0]
        alive_mask = np.array([c.alive for c in self.creatures], dtype=bool)

        dx = food_xy[None, :, 0] - creature_xy[:, None, 0]
        dy = food_xy[None, :, 1] - creature_xy[:, None, 1]
        dx = (dx + WORLD_W / 2) % WORLD_W - WORLD_W / 2
        dy = (dy + WORLD_H / 2) % WORLD_H - WORLD_H / 2
        d2 = dx * dx + dy * dy

        eat_r2 = EAT_RADIUS * EAT_RADIUS
        in_range = (d2 < eat_r2) & alive_mask[:, None]
        if not in_range.any():
            return
        d2_for_argmin = np.where(in_range, d2, np.inf)
        eater = np.argmin(d2_for_argmin, axis=0)
        food_eaten = in_range.any(axis=0)

        # 각 creature가 받은 총 에너지
        gained = np.zeros(M, dtype=np.float32)
        counts = np.zeros(M, dtype=np.int32)
        idx = np.arange(len(self.foods))
        if food_eaten.any():
            for j in idx[food_eaten]:
                e = int(eater[j])
                gained[e] += FOOD_ENERGY * float(food_value[j])
                counts[e] += 1

        for i, c in enumerate(self.creatures):
            if counts[i] == 0 or not c.alive:
                continue
            c.energy = min(MAX_ENERGY, c.energy + float(gained[i]))
            c.lifetime_eaten += int(counts[i])
            if len(self.pheromones) < PHEROMONE_MAX:
                self.pheromones.append(Pheromone(c.x, c.y, PHEROMONE_LIFE))

        self.foods = [f for j, f in enumerate(self.foods) if not food_eaten[j]]

    # -------- PPO 모드 prey 사고 + 이동 --------
    def _policy_think_and_move(self):
        """공유 정책(TorchBrain)이 모든 prey를 batched로 제어. PPO 학습용."""
        import torch
        creature_xy = np.array([(c.x, c.y) for c in self.creatures],
                                dtype=np.float32)
        if self.foods:
            food_xy = np.array([(f.x, f.y) for f in self.foods],
                                dtype=np.float32)
            food_value = np.array([f.value for f in self.foods],
                                   dtype=np.float32) / RARE_FOOD_VALUE
        else:
            food_xy = np.empty((0, 2), dtype=np.float32)
            food_value = np.empty(0, dtype=np.float32)
        if self.predators:
            pred_xy = np.array([(p.x, p.y) for p in self.predators],
                                dtype=np.float32)
        else:
            pred_xy = np.empty((0, 2), dtype=np.float32)
        if self._wall_sense_xy is None:
            self._refresh_wall_cache()
        wall_xy = self._wall_sense_xy
        danger_xy = np.concatenate([pred_xy, wall_xy], axis=0) \
                    if pred_xy.shape[0] + wall_xy.shape[0] > 0 \
                    else np.empty((0, 2), dtype=np.float32)
        if self.pheromones:
            ph_xy = np.array([(p.x, p.y) for p in self.pheromones],
                              dtype=np.float32)
            ph_intensity = np.array(
                [p.life / PHEROMONE_LIFE for p in self.pheromones],
                dtype=np.float32)
        else:
            ph_xy = np.empty((0, 2), dtype=np.float32)
            ph_intensity = np.empty(0, dtype=np.float32)

        food_all = self._all_sectors(creature_xy, food_xy, VISION,
                                      intensity=food_value)
        pred_all = self._all_sectors(creature_xy, danger_xy, VISION)
        kin_all = self._all_sectors(creature_xy, creature_xy, VISION,
                                     exclude_self=True)
        phero_all = self._all_sectors(creature_xy, ph_xy, VISION,
                                       intensity=ph_intensity)

        M = len(self.creatures)
        # PPO obs: 4×8 섹터 + energy + vx + vy + bias = 36 (메모리 없음)
        obs = np.zeros((M, PPO_OBS_DIM), dtype=np.float32)
        obs[:, 0:8] = food_all
        obs[:, 8:16] = pred_all
        obs[:, 16:24] = kin_all
        obs[:, 24:32] = phero_all
        for i, c in enumerate(self.creatures):
            obs[i, 32] = c.energy / MAX_ENERGY
            obs[i, 33] = c.vx / MAX_SPEED
            obs[i, 34] = c.vy / MAX_SPEED
        obs[:, -1] = 1.0

        obs_t = torch.from_numpy(obs)
        # C: GRU 정책일 경우 개체별 hidden state 묶어서 전달
        hidden_in = None
        is_gru = getattr(self.shared_policy, 'arch', 'mlp') == 'gru'
        if is_gru:
            h_dim = self.shared_policy.n_hidden
            hs = []
            for c in self.creatures:
                if c.rnn_hidden is None:
                    c.rnn_hidden = np.zeros(h_dim, dtype=np.float32)
                hs.append(c.rnn_hidden)
            hidden_in = torch.from_numpy(np.stack(hs))[None, :, :]  # (1, M, H)
        with torch.no_grad():
            actions, log_probs, values, hidden_out = self.shared_policy.act(
                obs_t, hidden_in)
        actions_np = actions.cpu().numpy()
        if is_gru and hidden_out is not None:
            new_hs = hidden_out.squeeze(0).cpu().numpy()
            for i, c in enumerate(self.creatures):
                c.rnn_hidden = new_hs[i].copy()
        log_probs_np = log_probs.cpu().numpy() if log_probs is not None else \
                       np.zeros(M, dtype=np.float32)
        values_np = values.cpu().numpy()

        energies_before = np.array([c.energy for c in self.creatures],
                                     dtype=np.float32)
        creature_ids = [c.creature_id for c in self.creatures]

        for i, c in enumerate(self.creatures):
            if not c.alive:
                continue
            c.last_energy = c.energy
            c.last_inputs = obs[i]
            c.last_outputs = actions_np[i]
            c.vx = float(actions_np[i, 0]) * MAX_SPEED
            c.vy = float(actions_np[i, 1]) * MAX_SPEED
            self._apply_move_with_wall_slide(c, c.vx, c.vy)
            c.energy -= METABOLISM
            c.age += 1

        self._consume_food_batched()

        # 이번 틱 transition을 버퍼에 (보상/done은 step() 끝에서 채움)
        self._ppo_buffer.append({
            'creature_ids': creature_ids,
            'obs': obs,
            'actions': actions_np,
            'log_probs': log_probs_np,
            'values': values_np,
            'energies_before': energies_before,
            'rewards': None,   # nullable until finalized
            'dones': None,
        })

    def _finalize_ppo_rewards(self):
        """step() 끝에서 호출 — 방금 저장된 transition의 reward/done을 채움."""
        if not self._ppo_buffer:
            return
        last = self._ppo_buffer[-1]
        if last['rewards'] is not None:
            return  # 이미 처리됨
        alive_now = {c.creature_id: c for c in self.creatures}
        n = len(last['creature_ids'])
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=bool)
        DEATH_PENALTY = -10.0
        for i, cid in enumerate(last['creature_ids']):
            c = alive_now.get(cid)
            if c is None:
                # creatures 리스트에서 제거됨 = 이번 틱 사망
                rewards[i] = DEATH_PENALTY
                dones[i] = True
            else:
                rewards[i] = float(c.energy - last['energies_before'][i])
                if not c.alive:
                    rewards[i] += DEATH_PENALTY
                    dones[i] = True
        last['rewards'] = rewards
        last['dones'] = dones
        # viz 모드: 버퍼 자라지 않게 잘라냄
        if self.ppo_buffer_max is not None and len(self._ppo_buffer) > self.ppo_buffer_max:
            self._ppo_buffer = self._ppo_buffer[-self.ppo_buffer_max:]

    def _policy_think_and_move_predators(self):
        """A: 포식자 공유 정책으로 모든 포식자 제어."""
        import torch
        pred_xy = np.array([(p.x, p.y) for p in self.predators],
                            dtype=np.float32)
        if self.creatures:
            prey_xy = np.array([(c.x, c.y) for c in self.creatures],
                                dtype=np.float32)
        else:
            prey_xy = np.empty((0, 2), dtype=np.float32)
        prey_all = self._all_sectors(pred_xy, prey_xy, PREDATOR_VISION)
        ally_all = self._all_sectors(pred_xy, pred_xy, PREDATOR_VISION,
                                      exclude_self=True)

        M = len(self.predators)
        obs = np.zeros((M, PPO_PRED_OBS_DIM), dtype=np.float32)
        obs[:, 0:8] = prey_all
        obs[:, 8:16] = ally_all
        for i, p in enumerate(self.predators):
            obs[i, 16] = p.energy / PREDATOR_MAX_ENERGY
            obs[i, 17] = p.vx / PREDATOR_SPEED
            obs[i, 18] = p.vy / PREDATOR_SPEED
        obs[:, -1] = 1.0

        obs_t = torch.from_numpy(obs)
        is_gru = getattr(self.predator_policy, 'arch', 'mlp') == 'gru'
        hidden_in = None
        if is_gru:
            h_dim = self.predator_policy.n_hidden
            hs = []
            for p in self.predators:
                if p.rnn_hidden is None:
                    p.rnn_hidden = np.zeros(h_dim, dtype=np.float32)
                hs.append(p.rnn_hidden)
            hidden_in = torch.from_numpy(np.stack(hs))[None, :, :]
        with torch.no_grad():
            actions, log_probs, values, hidden_out = self.predator_policy.act(
                obs_t, hidden_in)
        actions_np = actions.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy() if log_probs is not None else \
                       np.zeros(M, dtype=np.float32)
        values_np = values.cpu().numpy()
        if is_gru and hidden_out is not None:
            new_hs = hidden_out.squeeze(0).cpu().numpy()
            for i, p in enumerate(self.predators):
                p.rnn_hidden = new_hs[i].copy()

        energies_before = np.array([p.energy for p in self.predators],
                                     dtype=np.float32)
        pred_ids = [p.creature_id for p in self.predators]
        kills_before = self.kills   # 포식자 보상: 죽인 prey 수 증가

        for i, p in enumerate(self.predators):
            if not p.alive:
                continue
            p.last_energy = p.energy
            p.last_inputs = obs[i]
            p.last_outputs = actions_np[i]
            p.vx = float(actions_np[i, 0]) * PREDATOR_SPEED
            p.vy = float(actions_np[i, 1]) * PREDATOR_SPEED
            self._apply_move_with_wall_slide(p, p.vx, p.vy)
            p.energy -= PREDATOR_METABOLISM
            p.age += 1

        # 포식자 버퍼에 저장 (reward는 step 끝에서 채움)
        self._pred_ppo_buffer.append({
            'creature_ids': pred_ids,
            'obs': obs,
            'actions': actions_np,
            'log_probs': log_probs_np,
            'values': values_np,
            'energies_before': energies_before,
            'kills_before': kills_before,
            'rewards': None, 'dones': None,
        })

    def _finalize_pred_ppo_rewards(self):
        if not self._pred_ppo_buffer:
            return
        last = self._pred_ppo_buffer[-1]
        if last['rewards'] is not None:
            return
        alive_now = {p.creature_id: p for p in self.predators}
        n = len(last['creature_ids'])
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=bool)
        # 포식자 보상 = energy_delta + 작은 sparse bonus per kill
        kills_this_step = self.kills - last['kills_before']
        kill_bonus = 5.0 * kills_this_step / max(1, n)  # 분배
        for i, pid in enumerate(last['creature_ids']):
            p = alive_now.get(pid)
            if p is None:
                rewards[i] = -5.0
                dones[i] = True
            else:
                rewards[i] = float(p.energy - last['energies_before'][i])
                rewards[i] += kill_bonus
                if not p.alive:
                    rewards[i] -= 5.0
                    dones[i] = True
        last['rewards'] = rewards
        last['dones'] = dones
        if self.ppo_buffer_max is not None and len(self._pred_ppo_buffer) > self.ppo_buffer_max:
            self._pred_ppo_buffer = self._pred_ppo_buffer[-self.ppo_buffer_max:]

    def _maintain_predator_population(self, target: int = 6):
        while len(self.predators) < target:
            self._spawn_predator()

    def _maintain_population(self):
        """PPO 모드 — 죽은 자리에 새 prey spawn. 인구를 target에 유지."""
        target = PPO_TARGET_POPULATION
        while len(self.creatures) < target:
            for _ in range(10):
                x = self.rng.uniform(0, WORLD_W)
                y = self.rng.uniform(0, WORLD_H)
                if not self._in_any_wall(x, y):
                    break
            cid = self._new_id()
            # dummy brain — 실제로는 shared_policy 사용. signature()만 색용
            dummy = Brain.random(self.rng, N_IN, self.prey_hidden, N_OUT)
            c = Creature(
                creature_id=cid,
                x=x, y=y,
                vx=self.rng.uniform(-1, 1),
                vy=self.rng.uniform(-1, 1),
                energy=INITIAL_ENERGY,
                age=0,
                brain=dummy,
                generation=1,
                parent_ids=(),
                ancestor_id=cid,
            )
            self.creatures.append(c)
            self._record_birth(c)

    # -------- predator 사고 + 이동 (뇌 진화 또는 PPO) --------
    def _think_and_move_predators(self):
        if not self.predators:
            return
        if self.predator_policy is not None:
            return self._policy_think_and_move_predators()
        pred_xy = np.array([(p.x, p.y) for p in self.predators],
                            dtype=np.float32)
        if self.creatures:
            prey_xy = np.array([(c.x, c.y) for c in self.creatures],
                                dtype=np.float32)
        else:
            prey_xy = np.empty((0, 2), dtype=np.float32)

        prey_all = self._all_sectors(pred_xy, prey_xy, PREDATOR_VISION)
        ally_all = self._all_sectors(pred_xy, pred_xy, PREDATOR_VISION,
                                      exclude_self=True)

        for idx, p in enumerate(self.predators):
            if not p.alive:
                continue
            inputs = np.empty(PRED_N_IN, dtype=np.float32)
            inputs[0:8] = prey_all[idx]
            inputs[8:16] = ally_all[idx]
            inputs[16] = p.energy / PREDATOR_MAX_ENERGY
            inputs[17] = p.vx / PREDATOR_SPEED
            inputs[18] = p.vy / PREDATOR_SPEED
            inputs[19:19 + N_MEMORY] = p.memory
            inputs[-1] = 1.0

            out = p.brain.forward(inputs)
            p.vx = float(out[0]) * PREDATOR_SPEED
            p.vy = float(out[1]) * PREDATOR_SPEED
            p.memory = out[OUT_MEM_START:OUT_MEM_START + N_MEMORY].astype(np.float32).copy()

            self._apply_move_with_wall_slide(p, p.vx, p.vy)
            p.energy -= PREDATOR_METABOLISM
            p.age += 1

    def _predator_kills(self):
        kill_r2 = PREDATOR_KILL_RADIUS * PREDATOR_KILL_RADIUS
        for p in self.predators:
            if not p.alive:
                continue
            for c in self.creatures:
                if not c.alive:
                    continue
                dx = (c.x - p.x + WORLD_W / 2) % WORLD_W - WORLD_W / 2
                dy = (c.y - p.y + WORLD_H / 2) % WORLD_H - WORLD_H / 2
                if dx * dx + dy * dy < kill_r2:
                    c.alive = False
                    p.energy = min(PREDATOR_MAX_ENERGY,
                                   p.energy + PREDATOR_KILL_ENERGY)
                    p.lifetime_kills += 1
                    self.kills += 1
                    self.deaths += 1
                    self._record_death(c)
                    break

    # -------- 번식 + 죽음 --------
    def _reproduce_and_die(self):
        if self.disable_reproduction:
            # PPO 모드 — prey 번식 건너뛰고 죽음만 처리.
            for c in self.creatures:
                if not c.alive:
                    continue
                if c.energy <= 0 or c.age > MAX_AGE:
                    c.alive = False
                    self.deaths += 1
                    self._record_death(c)
            # 포식자는 그대로 진화/번식 (별도 dynamics)
            new_preds = []
            for p in self.predators:
                if not p.alive:
                    continue
                if p.energy <= 0:
                    p.alive = False
                    continue
                if (p.energy >= PREDATOR_REPRODUCE_THRESHOLD
                        and len(self.predators) + len(new_preds) < PREDATOR_MAX):
                    p.energy -= PREDATOR_REPRODUCE_COST
                    child_brain = p.brain.mutate(self.rng)
                    cid = self._new_id()
                    cx = (p.x + self.rng.uniform(-15, 15)) % WORLD_W
                    cy = (p.y + self.rng.uniform(-15, 15)) % WORLD_H
                    cx, cy = self._safe_spawn_near(cx, cy, p.x, p.y)
                    new_preds.append(Predator(
                        creature_id=cid, x=cx, y=cy,
                        vx=self.rng.uniform(-1, 1),
                        vy=self.rng.uniform(-1, 1),
                        energy=PREDATOR_INITIAL_ENERGY,
                        age=0,
                        brain=child_brain,
                        generation=p.generation + 1,
                        parent_ids=(p.creature_id,),
                    ))
                    self.pred_births += 1
            self.creatures = [c for c in self.creatures if c.alive]
            self.predators = [p for p in self.predators if p.alive] + new_preds
            self._prune_lineage()
            return

        new_kids: List[Creature] = []
        candidates = [c for c in self.creatures
                      if c.alive and c.energy >= REPRODUCE_THRESHOLD]
        partnered = set()

        sex_r2 = SEXUAL_RANGE * SEXUAL_RANGE
        for i, a in enumerate(candidates):
            if a.creature_id in partnered:
                continue
            for b in candidates[i + 1:]:
                if b.creature_id in partnered:
                    continue
                dx = (b.x - a.x + WORLD_W / 2) % WORLD_W - WORLD_W / 2
                dy = (b.y - a.y + WORLD_H / 2) % WORLD_H - WORLD_H / 2
                if dx * dx + dy * dy >= sex_r2:
                    continue
                if Brain.distance(a.brain, b.brain) > SPECIATION_THRESHOLD:
                    continue
                cost = REPRODUCE_COST + SEXUAL_BONUS_COST
                a.energy -= cost
                b.energy -= cost
                child_brain = Brain.crossover(a.brain, b.brain, self.rng)
                cid = self._new_id()
                offset_a = self.rng.uniform(0, math.tau)
                cx = ((a.x + b.x) / 2 + math.cos(offset_a) * (SIZE + 6)) % WORLD_W
                cy = ((a.y + b.y) / 2 + math.sin(offset_a) * (SIZE + 6)) % WORLD_H
                cx, cy = self._safe_spawn_near(cx, cy, (a.x + b.x) / 2, (a.y + b.y) / 2)
                kid = Creature(
                    creature_id=cid,
                    x=cx, y=cy,
                    vx=self.rng.uniform(-1, 1),
                    vy=self.rng.uniform(-1, 1),
                    energy=REPRODUCE_COST * 0.8,
                    age=0,
                    brain=child_brain,
                    generation=max(a.generation, b.generation) + 1,
                    parent_ids=(a.creature_id, b.creature_id),
                    ancestor_id=a.ancestor_id,
                )
                new_kids.append(kid)
                self._record_birth(kid)
                self.births += 1
                self.sexual_births += 1
                partnered.add(a.creature_id)
                partnered.add(b.creature_id)
                break

        for c in self.creatures:
            if not c.alive:
                continue
            if c.energy <= 0 or c.age > MAX_AGE:
                c.alive = False
                self.deaths += 1
                self._record_death(c)
                continue
            if (c.creature_id not in partnered
                    and c.energy >= REPRODUCE_THRESHOLD + ASEXUAL_EXTRA):
                c.energy -= REPRODUCE_COST
                child_brain = c.brain.mutate(self.rng)
                cid = self._new_id()
                offset_a = self.rng.uniform(0, math.tau)
                cx = (c.x + math.cos(offset_a) * (SIZE + 6)) % WORLD_W
                cy = (c.y + math.sin(offset_a) * (SIZE + 6)) % WORLD_H
                cx, cy = self._safe_spawn_near(cx, cy, c.x, c.y)
                kid = Creature(
                    creature_id=cid,
                    x=cx, y=cy,
                    vx=self.rng.uniform(-1, 1),
                    vy=self.rng.uniform(-1, 1),
                    energy=REPRODUCE_COST * 0.8,
                    age=0,
                    brain=child_brain,
                    generation=c.generation + 1,
                    parent_ids=(c.creature_id,),
                    ancestor_id=c.ancestor_id,
                )
                new_kids.append(kid)
                self._record_birth(kid)
                self.births += 1

        # 포식자 번식 + 죽음 — 뇌도 mutate
        new_preds: List[Predator] = []
        for p in self.predators:
            if not p.alive:
                continue
            if p.energy <= 0:
                p.alive = False
                continue
            if (p.energy >= PREDATOR_REPRODUCE_THRESHOLD
                    and len(self.predators) + len(new_preds) < PREDATOR_MAX):
                p.energy -= PREDATOR_REPRODUCE_COST
                child_brain = p.brain.mutate(self.rng)
                cid = self._new_id()
                cx = (p.x + self.rng.uniform(-15, 15)) % WORLD_W
                cy = (p.y + self.rng.uniform(-15, 15)) % WORLD_H
                cx, cy = self._safe_spawn_near(cx, cy, p.x, p.y)
                new_preds.append(Predator(
                    creature_id=cid,
                    x=cx, y=cy,
                    vx=self.rng.uniform(-1, 1),
                    vy=self.rng.uniform(-1, 1),
                    energy=PREDATOR_INITIAL_ENERGY,
                    age=0,
                    brain=child_brain,
                    generation=p.generation + 1,
                    parent_ids=(p.creature_id,),
                ))
                self.pred_births += 1

        self.creatures = [c for c in self.creatures if c.alive] + new_kids
        self.predators = [p for p in self.predators if p.alive] + new_preds
        self._prune_lineage()

    def _prune_lineage(self):
        if len(self.lineage_records) < 1500:
            return
        keep = set()
        frontier = [c.creature_id for c in self.creatures]
        depth = 0
        while frontier and depth < 20:
            next_layer = []
            for cid in frontier:
                if cid in keep:
                    continue
                keep.add(cid)
                rec = self.lineage_records.get(cid)
                if rec:
                    next_layer.extend(rec.parent_ids)
            frontier = next_layer
            depth += 1
        self.lineage_records = {k: v for k, v in self.lineage_records.items()
                                 if k in keep}

    def _compute_diversity(self) -> float:
        n = len(self.creatures)
        if n < 2:
            return 0.0
        k = min(DIVERSITY_SAMPLE, n)
        sample = self.rng.sample(self.creatures, k)
        total, pairs = 0.0, 0
        for i, a in enumerate(sample):
            for b in sample[i + 1:]:
                total += Brain.distance(a.brain, b.brain)
                pairs += 1
        return total / pairs if pairs else 0.0

    def _update_ancestor_counts(self):
        counts: Dict[int, int] = {}
        for c in self.creatures:
            counts[c.ancestor_id] = counts.get(c.ancestor_id, 0) + 1
        self._last_ancestor_counts = counts

    def _record_history(self):
        base = {
            'tick': self.tick,
            'count': self.population(),
            'avg_eaten': 0.0,
            'max_gen': 0,
            'avg_age': 0,
            'kills': self.kills,
            'predators': len(self.predators),
            'pheromones': len(self.pheromones),
            'diversity': self._last_diversity,
            'pred_max_gen': max((p.generation for p in self.predators),
                                 default=0),
            'pred_total_kills': sum(p.lifetime_kills for p in self.predators),
            'ancestor_counts': dict(self._last_ancestor_counts),
        }
        if self.creatures:
            n = len(self.creatures)
            base['avg_eaten'] = sum(c.lifetime_eaten for c in self.creatures) / n
            base['max_gen'] = max(c.generation for c in self.creatures)
            base['avg_age'] = sum(c.age for c in self.creatures) / n
        self.history.append(base)
        if self.history_max is not None and len(self.history) > self.history_max:
            # 오래된 절반 제거 — 그래프는 -200만 보므로 영향 없음
            self.history = self.history[-self.history_max:]

    def population(self) -> int:
        return len(self.creatures)

    def max_generation(self) -> int:
        return max((c.generation for c in self.creatures), default=0)

    def champion(self) -> Optional[Creature]:
        if not self.creatures:
            return None
        return max(self.creatures,
                   key=lambda c: (c.generation, c.lifetime_eaten, c.age))

    def find_at(self, world_x: float, world_y: float,
                radius: float = 12.0) -> Optional[Creature]:
        r2 = radius * radius
        best, best_d2 = None, r2
        for c in self.creatures:
            dx = (c.x - world_x + WORLD_W / 2) % WORLD_W - WORLD_W / 2
            dy = (c.y - world_y + WORLD_H / 2) % WORLD_H - WORLD_H / 2
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = c
        return best

    # -------- 내보내기 / 스냅샷 --------
    def write_history_csv(self, path: Path) -> int:
        """history를 CSV로 저장. ancestor_counts는 JSON 문자열로."""
        import json
        if not self.history:
            return 0
        fields = ['tick', 'count', 'avg_eaten', 'max_gen', 'avg_age',
                  'kills', 'predators', 'pheromones', 'diversity',
                  'pred_max_gen', 'pred_total_kills', 'ancestor_counts']
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for h in self.history:
                row = dict(h)
                row['ancestor_counts'] = json.dumps(h.get('ancestor_counts', {}))
                w.writerow(row)
        return len(self.history)
