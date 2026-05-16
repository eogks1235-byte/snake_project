"""환경 + 시뮬레이션 루프.

토러스 월드(가장자리 wrap). 음식이 무작위 위치에 자라고, 개체가 먹으면 에너지 회복.
에너지가 충분하면 번식, 0이면 사망.
"""
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

from .genome import Genome


# ── 환경 파라미터 ──
WORLD_W = 1000
WORLD_H = 600
INITIAL_FOOD = 250
FOOD_SPAWN_PER_TICK = 1.4    # 평균 tick당 새 음식 수
FOOD_ENERGY = 22.0
INITIAL_POPULATION = 60
INITIAL_ENERGY = 60.0
REPRODUCE_THRESHOLD = 110.0
REPRODUCE_COST = 60.0
MAX_ENERGY = 200.0
SIZE_EAT_BONUS = 1.0            # 큰 개체는 음식 1.x배
MAX_AGE = 1500


@dataclass
class Food:
    x: float
    y: float


@dataclass
class Creature:
    x: float
    y: float
    vx: float
    vy: float
    energy: float
    age: int
    genome: Genome
    generation: int
    alive: bool = True

    def color(self) -> tuple:
        return self.genome.color()


class World:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.tick = 0

        # 통계용
        self.births = 0
        self.deaths = 0
        self.history: List[dict] = []   # (tick, count, avg_speed, avg_vision, avg_size)

        # 초기 개체 + 음식
        self.creatures: List[Creature] = []
        self.foods: List[Food] = []
        self._spawn_initial()

    def _spawn_initial(self):
        for _ in range(INITIAL_POPULATION):
            g = Genome.random(self.rng)
            self.creatures.append(Creature(
                x=self.rng.uniform(0, WORLD_W),
                y=self.rng.uniform(0, WORLD_H),
                vx=self.rng.uniform(-1, 1),
                vy=self.rng.uniform(-1, 1),
                energy=INITIAL_ENERGY,
                age=0,
                genome=g,
                generation=1,
            ))
        for _ in range(INITIAL_FOOD):
            self.foods.append(Food(
                x=self.rng.uniform(0, WORLD_W),
                y=self.rng.uniform(0, WORLD_H),
            ))

    # ── 메인 루프 ─────────────────────────────────────────

    def step(self):
        self.tick += 1
        self._spawn_food()
        self._move_and_eat()
        self._reproduce_and_die()
        self._record_history()

    def _spawn_food(self):
        # FOOD_SPAWN_PER_TICK이 1.4면 평균 1.4개씩 추가 (정수 부분 + 확률)
        n_int = int(FOOD_SPAWN_PER_TICK)
        frac = FOOD_SPAWN_PER_TICK - n_int
        n = n_int + (1 if self.rng.random() < frac else 0)
        for _ in range(n):
            self.foods.append(Food(
                x=self.rng.uniform(0, WORLD_W),
                y=self.rng.uniform(0, WORLD_H),
            ))

    def _move_and_eat(self):
        # 빠른 음식 lookup (단순 O(N*M); 개체수 수백 정도면 충분)
        for c in self.creatures:
            if not c.alive:
                continue

            vision_r = c.genome.vision_px()
            speed = c.genome.speed_px()

            # 시야 내 가장 가까운 음식
            target = self._nearest_food(c.x, c.y, vision_r)
            if target is not None:
                dx = self._wrap_delta(target.x - c.x, WORLD_W)
                dy = self._wrap_delta(target.y - c.y, WORLD_H)
                d = math.hypot(dx, dy) or 1.0
                c.vx = (dx / d) * speed
                c.vy = (dy / d) * speed
            else:
                # 무작위 워크 — 진행 방향 약간 변경
                angle = math.atan2(c.vy, c.vx) + self.rng.uniform(-0.4, 0.4)
                c.vx = math.cos(angle) * speed
                c.vy = math.sin(angle) * speed

            c.x = (c.x + c.vx) % WORLD_W
            c.y = (c.y + c.vy) % WORLD_H

            # 에너지 소모
            c.energy -= c.genome.metabolism_per_tick()
            c.age += 1

            # 음식 충돌 (반지름 + 음식 작은 점)
            eat_r = c.genome.size_px() + 4
            self._consume_nearby_food(c, eat_r)

    def _wrap_delta(self, d: float, span: float) -> float:
        """토러스 거리 — 가까운 쪽 방향."""
        if d > span / 2:
            return d - span
        if d < -span / 2:
            return d + span
        return d

    def _nearest_food(self, x: float, y: float, radius: float) -> Optional[Food]:
        if not self.foods:
            return None
        best = None
        best_d2 = radius * radius
        for f in self.foods:
            dx = self._wrap_delta(f.x - x, WORLD_W)
            dy = self._wrap_delta(f.y - y, WORLD_H)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = f
        return best

    def _consume_nearby_food(self, c: Creature, eat_r: float):
        eat_r2 = eat_r * eat_r
        keep = []
        gained = 0.0
        for f in self.foods:
            dx = self._wrap_delta(f.x - c.x, WORLD_W)
            dy = self._wrap_delta(f.y - c.y, WORLD_H)
            if dx * dx + dy * dy < eat_r2:
                gained += FOOD_ENERGY * (1.0 + (c.genome.size - 0.5) * SIZE_EAT_BONUS * 0.4)
            else:
                keep.append(f)
        if gained > 0:
            c.energy = min(MAX_ENERGY, c.energy + gained)
        self.foods = keep

    def _reproduce_and_die(self):
        new_kids: List[Creature] = []
        for c in self.creatures:
            if not c.alive:
                continue
            if c.energy <= 0 or c.age > MAX_AGE:
                c.alive = False
                self.deaths += 1
                continue
            if c.energy >= REPRODUCE_THRESHOLD:
                c.energy -= REPRODUCE_COST
                child_genome = c.genome.mutate(self.rng)
                # 자식은 부모 위치 근처에서 약간 떨어진 곳에
                offset_a = self.rng.uniform(0, math.tau)
                offset_r = c.genome.size_px() + 6
                cx = (c.x + math.cos(offset_a) * offset_r) % WORLD_W
                cy = (c.y + math.sin(offset_a) * offset_r) % WORLD_H
                new_kids.append(Creature(
                    x=cx, y=cy,
                    vx=self.rng.uniform(-1, 1),
                    vy=self.rng.uniform(-1, 1),
                    energy=REPRODUCE_COST * 0.8,
                    age=0,
                    genome=child_genome,
                    generation=c.generation + 1,
                ))
                self.births += 1

        self.creatures = [c for c in self.creatures if c.alive] + new_kids

    # ── 통계 ─────────────────────────────────────────────

    def _record_history(self):
        if not self.creatures:
            self.history.append({
                'tick': self.tick, 'count': 0,
                'avg_speed': 0, 'avg_vision': 0, 'avg_size': 0, 'avg_gen': 0,
            })
            return
        n = len(self.creatures)
        self.history.append({
            'tick': self.tick,
            'count': n,
            'avg_speed': sum(c.genome.speed_px() for c in self.creatures) / n,
            'avg_vision': sum(c.genome.vision_px() for c in self.creatures) / n,
            'avg_size': sum(c.genome.size_px() for c in self.creatures) / n,
            'avg_gen': sum(c.generation for c in self.creatures) / n,
            'avg_metab': sum(c.genome.metabolism_per_tick() for c in self.creatures) / n,
        })

    def population(self) -> int:
        return len(self.creatures)

    def max_generation(self) -> int:
        return max((c.generation for c in self.creatures), default=0)
