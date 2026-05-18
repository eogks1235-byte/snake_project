"""환경 + 시뮬레이션 루프.

토러스 월드(가장자리 wrap). 음식이 무작위 위치에 자라고, 개체가 먹으면 에너지 회복.

neural/ 과 차별화되는 동학:
- **지역 생물군계(biome)** : 월드를 x축 기준으로 cold/temperate/warm 으로 나눠
  각각 다른 metabolism 배수 / food spawn 가중치. 종 분화(speciation)의 압력.
- **계절** : 시간에 따라 전체 food spawn 이 사인파로 진동. 풍년/흉년.
- **성생식** : M/F. 암컷이 시야 내 수컷 중 display_fitness 가장 높은 개체와 교배 → crossover.
- **hawk-dove** : 같은 종 내 만남에서 aggression 형질에 따라 hawk/dove 게임의 보수.

WorldConfig 로 모든 환경 파라미터를 외부에서 주입 가능. experiments/ 에서 sweep 용도.
"""
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from .genome import Genome


# ── 월드 크기(고정) ──
WORLD_W = 1000
WORLD_H = 600

# ── biome 카테고리(고정) ──
BIOME_COLD = 0
BIOME_TEMPERATE = 1
BIOME_WARM = 2

# ── 고정 상수 ──
MAX_ENERGY = 200.0
AGGR_HAWK_CUTOFF = 0.5

# ── 기본값(WorldConfig 에서 오버라이드 가능) ──
DEFAULT_INITIAL_FOOD = 250
DEFAULT_FOOD_SPAWN_PER_TICK = 1.6
DEFAULT_FOOD_ENERGY = 22.0
DEFAULT_INITIAL_POPULATION = 80
DEFAULT_INITIAL_ENERGY = 60.0
DEFAULT_REPRODUCE_COST = 55.0
DEFAULT_SIZE_EAT_BONUS = 1.0

DEFAULT_BIOME_METABOLISM_MULT = (1.35, 1.0, 0.85)
DEFAULT_BIOME_FOOD_WEIGHT = (0.6, 1.0, 1.55)

DEFAULT_SEASON_PERIOD = 1200
DEFAULT_SEASON_AMPLITUDE = 0.55

DEFAULT_MATE_RADIUS_FACTOR = 1.3
DEFAULT_MATE_COOLDOWN = 40

DEFAULT_ENCOUNTER_RADIUS = 14.0
DEFAULT_ENCOUNTER_COOLDOWN = 25
DEFAULT_ENCOUNTER_VALUE = 6.0
DEFAULT_HAWK_INJURY_COST = 8.0

DEFAULT_MIGRATION_COST = 0.0   # >0 이면 biome 경계 통과 시 에너지 손실


@dataclass
class WorldConfig:
    """모든 환경 파라미터. 기본값은 main.py 데모용 값. experiments/ 에서 sweep."""
    initial_food: int = DEFAULT_INITIAL_FOOD
    food_spawn_per_tick: float = DEFAULT_FOOD_SPAWN_PER_TICK
    food_energy: float = DEFAULT_FOOD_ENERGY
    initial_population: int = DEFAULT_INITIAL_POPULATION
    initial_energy: float = DEFAULT_INITIAL_ENERGY
    reproduce_cost: float = DEFAULT_REPRODUCE_COST
    size_eat_bonus: float = DEFAULT_SIZE_EAT_BONUS

    biome_metabolism_mult: Tuple[float, float, float] = DEFAULT_BIOME_METABOLISM_MULT
    biome_food_weight: Tuple[float, float, float] = DEFAULT_BIOME_FOOD_WEIGHT

    season_period: int = DEFAULT_SEASON_PERIOD
    season_amplitude: float = DEFAULT_SEASON_AMPLITUDE

    mate_radius_factor: float = DEFAULT_MATE_RADIUS_FACTOR
    mate_cooldown: int = DEFAULT_MATE_COOLDOWN

    encounter_radius: float = DEFAULT_ENCOUNTER_RADIUS
    encounter_cooldown: int = DEFAULT_ENCOUNTER_COOLDOWN
    encounter_value: float = DEFAULT_ENCOUNTER_VALUE
    hawk_injury_cost: float = DEFAULT_HAWK_INJURY_COST

    migration_cost: float = DEFAULT_MIGRATION_COST


@dataclass
class Food:
    x: float
    y: float
    biome: int = BIOME_TEMPERATE


class Sex(Enum):
    M = 'M'
    F = 'F'


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
    sex: Sex
    prev_biome: int = BIOME_TEMPERATE
    mate_cd: int = 0
    encounter_cd: int = 0
    last_role: str = ''
    alive: bool = True

    def color(self) -> tuple:
        return self.genome.color()

    def is_hawk(self) -> bool:
        return self.genome.aggression > AGGR_HAWK_CUTOFF


def biome_at(x: float) -> int:
    if x < WORLD_W / 3:
        return BIOME_COLD
    if x < WORLD_W * 2 / 3:
        return BIOME_TEMPERATE
    return BIOME_WARM


class World:
    def __init__(self, seed: int, config: Optional[WorldConfig] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.tick = 0
        self.cfg = config if config is not None else WorldConfig()

        # 통계용
        self.births = 0
        self.deaths = 0
        self.encounters = 0
        self.hh_encounters = 0
        self.hd_encounters = 0
        self.dd_encounters = 0
        self.migrations = 0
        self.history: List[dict] = []

        self.creatures: List[Creature] = []
        self.foods: List[Food] = []
        self._spawn_initial()

    def _spawn_initial(self):
        for _ in range(self.cfg.initial_population):
            g = Genome.random(self.rng)
            x = self.rng.uniform(0, WORLD_W)
            y = self.rng.uniform(0, WORLD_H)
            self.creatures.append(Creature(
                x=x, y=y,
                vx=self.rng.uniform(-1, 1),
                vy=self.rng.uniform(-1, 1),
                energy=self.cfg.initial_energy,
                age=0,
                genome=g,
                generation=1,
                sex=Sex.M if self.rng.random() < 0.5 else Sex.F,
                prev_biome=biome_at(x),
            ))
        for _ in range(self.cfg.initial_food):
            self._spawn_one_food()

    # ── 환경 ──

    def _spawn_one_food(self):
        weights = self.cfg.biome_food_weight
        total = sum(weights)
        pick = self.rng.uniform(0, total)
        acc = 0.0
        chosen = 0
        for i, w in enumerate(weights):
            acc += w
            if pick <= acc:
                chosen = i
                break
        x_min = chosen * WORLD_W / 3
        x_max = (chosen + 1) * WORLD_W / 3
        x = self.rng.uniform(x_min, x_max)
        y = self.rng.uniform(0, WORLD_H)
        self.foods.append(Food(x=x, y=y, biome=chosen))

    def season_factor(self) -> float:
        if self.cfg.season_period <= 0:
            return 1.0
        phase = (self.tick / self.cfg.season_period) * math.tau
        return 1.0 + math.sin(phase) * self.cfg.season_amplitude

    # ── 메인 루프 ──

    def step(self):
        self.tick += 1
        self._spawn_food()
        self._move_and_eat()
        self._encounter_phase()
        self._mate_and_die()
        self._record_history()

    def _spawn_food(self):
        rate = self.cfg.food_spawn_per_tick * max(0.0, self.season_factor())
        n_int = int(rate)
        frac = rate - n_int
        n = n_int + (1 if self.rng.random() < frac else 0)
        for _ in range(n):
            self._spawn_one_food()

    def _move_and_eat(self):
        for c in self.creatures:
            if not c.alive:
                continue

            vision_r = c.genome.vision_px()
            speed = c.genome.speed_px()

            target = self._nearest_food(c.x, c.y, vision_r)
            if target is not None:
                dx = self._wrap_delta(target.x - c.x, WORLD_W)
                dy = self._wrap_delta(target.y - c.y, WORLD_H)
                d = math.hypot(dx, dy) or 1.0
                c.vx = (dx / d) * speed
                c.vy = (dy / d) * speed
            else:
                angle = math.atan2(c.vy, c.vx) + self.rng.uniform(-0.4, 0.4)
                c.vx = math.cos(angle) * speed
                c.vy = math.sin(angle) * speed

            c.x = (c.x + c.vx) % WORLD_W
            c.y = (c.y + c.vy) % WORLD_H

            # biome 별 대사 비용
            new_biome = biome_at(c.x)
            bm = self.cfg.biome_metabolism_mult[new_biome]
            c.energy -= c.genome.metabolism_per_tick() * bm

            # 이주 비용 — biome 경계를 새로 넘으면 추가 에너지 손실
            if new_biome != c.prev_biome:
                c.energy -= self.cfg.migration_cost
                c.prev_biome = new_biome
                self.migrations += 1

            c.age += 1
            if c.mate_cd > 0:
                c.mate_cd -= 1
            if c.encounter_cd > 0:
                c.encounter_cd -= 1

            eat_r = c.genome.size_px() + 4
            self._consume_nearby_food(c, eat_r)

    def _wrap_delta(self, d: float, span: float) -> float:
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
                gained += self.cfg.food_energy * (1.0 + (c.genome.size - 0.5) *
                                                  self.cfg.size_eat_bonus * 0.4)
            else:
                keep.append(f)
        if gained > 0:
            c.energy = min(MAX_ENERGY, c.energy + gained)
        self.foods = keep

    # ── hawk / dove ──

    def _encounter_phase(self):
        creatures = [c for c in self.creatures if c.alive]
        r2 = self.cfg.encounter_radius * self.cfg.encounter_radius
        n = len(creatures)
        for i in range(n):
            a = creatures[i]
            if a.encounter_cd > 0:
                continue
            for j in range(i + 1, n):
                b = creatures[j]
                if b.encounter_cd > 0:
                    continue
                dx = self._wrap_delta(b.x - a.x, WORLD_W)
                dy = self._wrap_delta(b.y - a.y, WORLD_H)
                if dx * dx + dy * dy < r2:
                    self._resolve_hawk_dove(a, b)
                    a.encounter_cd = self.cfg.encounter_cooldown
                    b.encounter_cd = self.cfg.encounter_cooldown
                    break

    def _resolve_hawk_dove(self, a: Creature, b: Creature):
        self.encounters += 1
        ah, bh = a.is_hawk(), b.is_hawk()
        a.last_role = 'H' if ah else 'D'
        b.last_role = 'H' if bh else 'D'
        V = self.cfg.encounter_value
        C = self.cfg.hawk_injury_cost

        if ah and bh:
            self.hh_encounters += 1
            a.energy -= C
            b.energy -= C
        elif ah and not bh:
            self.hd_encounters += 1
            a.energy = min(MAX_ENERGY, a.energy + V)
        elif bh and not ah:
            self.hd_encounters += 1
            b.energy = min(MAX_ENERGY, b.energy + V)
        else:
            self.dd_encounters += 1
            half = V * 0.5
            a.energy = min(MAX_ENERGY, a.energy + half)
            b.energy = min(MAX_ENERGY, b.energy + half)

    # ── 성생식 + 사망 ──

    def _mate_and_die(self):
        for c in self.creatures:
            if not c.alive:
                continue
            if c.energy <= 0 or c.age > c.genome.lifespan_ticks():
                c.alive = False
                self.deaths += 1

        new_kids: List[Creature] = []
        alive = [c for c in self.creatures if c.alive]
        for female in alive:
            if female.sex != Sex.F:
                continue
            if female.mate_cd > 0:
                continue
            if female.energy < female.genome.repro_threshold_energy():
                continue

            mate = self._find_best_mate(female, alive)
            if mate is None:
                continue

            female.energy -= self.cfg.reproduce_cost
            mate.energy -= self.cfg.reproduce_cost * 0.3
            female.mate_cd = self.cfg.mate_cooldown
            mate.mate_cd = self.cfg.mate_cooldown

            child_genome = Genome.crossover(female.genome, mate.genome, self.rng).mutate(self.rng)
            offset_a = self.rng.uniform(0, math.tau)
            offset_r = female.genome.size_px() + 6
            cx = (female.x + math.cos(offset_a) * offset_r) % WORLD_W
            cy = (female.y + math.sin(offset_a) * offset_r) % WORLD_H
            new_kids.append(Creature(
                x=cx, y=cy,
                vx=self.rng.uniform(-1, 1),
                vy=self.rng.uniform(-1, 1),
                energy=self.cfg.reproduce_cost * 0.85,
                age=0,
                genome=child_genome,
                generation=max(female.generation, mate.generation) + 1,
                sex=Sex.M if self.rng.random() < 0.5 else Sex.F,
                prev_biome=biome_at(cx),
            ))
            self.births += 1

        self.creatures = [c for c in self.creatures if c.alive] + new_kids

    def _find_best_mate(self, female: Creature, candidates: List[Creature]) -> Optional[Creature]:
        radius = female.genome.vision_px() * self.cfg.mate_radius_factor
        r2 = radius * radius
        best = None
        best_score = -1.0
        for m in candidates:
            if m is female or m.sex != Sex.M or m.mate_cd > 0:
                continue
            if m.energy < m.genome.repro_threshold_energy() * 0.6:
                continue
            dx = self._wrap_delta(m.x - female.x, WORLD_W)
            dy = self._wrap_delta(m.y - female.y, WORLD_H)
            if dx * dx + dy * dy > r2:
                continue
            score = m.genome.display_fitness()
            if score > best_score:
                best_score = score
                best = m
        return best

    # ── 통계 ──

    def _record_history(self):
        n = len(self.creatures)
        if n == 0:
            self.history.append({
                'tick': self.tick, 'count': 0,
                'avg_speed': 0, 'avg_vision': 0, 'avg_size': 0, 'avg_gen': 0,
                'avg_metab': 0, 'avg_aggr': 0, 'avg_pattern': 0, 'avg_lifespan': 0,
                'hawk_ratio': 0, 'season': self.season_factor(),
            })
            return
        avg_aggr = sum(c.genome.aggression for c in self.creatures) / n
        hawks = sum(1 for c in self.creatures if c.is_hawk())
        self.history.append({
            'tick': self.tick,
            'count': n,
            'avg_speed': sum(c.genome.speed_px() for c in self.creatures) / n,
            'avg_vision': sum(c.genome.vision_px() for c in self.creatures) / n,
            'avg_size': sum(c.genome.size_px() for c in self.creatures) / n,
            'avg_gen': sum(c.generation for c in self.creatures) / n,
            'avg_metab': sum(c.genome.metabolism_per_tick() for c in self.creatures) / n,
            'avg_aggr': avg_aggr,
            'avg_pattern': sum(c.genome.pattern for c in self.creatures) / n,
            'avg_lifespan': sum(c.genome.lifespan_ticks() for c in self.creatures) / n,
            'hawk_ratio': hawks / n,
            'season': self.season_factor(),
        })

    def population(self) -> int:
        return len(self.creatures)

    def max_generation(self) -> int:
        return max((c.generation for c in self.creatures), default=0)

    def biome_populations(self) -> Tuple[int, int, int]:
        cold = temp = warm = 0
        for c in self.creatures:
            b = biome_at(c.x)
            if b == BIOME_COLD:
                cold += 1
            elif b == BIOME_TEMPERATE:
                temp += 1
            else:
                warm += 1
        return cold, temp, warm

    def trait_means_per_biome(self) -> dict:
        """biome 별 trait 평균 — 종 분화 측정용."""
        groups = {BIOME_COLD: [], BIOME_TEMPERATE: [], BIOME_WARM: []}
        for c in self.creatures:
            groups[biome_at(c.x)].append(c.genome)
        out = {}
        for b, gs in groups.items():
            if not gs:
                out[b] = None
                continue
            out[b] = {
                'speed': sum(g.speed for g in gs) / len(gs),
                'vision': sum(g.vision for g in gs) / len(gs),
                'size': sum(g.size for g in gs) / len(gs),
                'metabolism': sum(g.metabolism for g in gs) / len(gs),
                'aggression': sum(g.aggression for g in gs) / len(gs),
                'pattern': sum(g.pattern for g in gs) / len(gs),
                'lifespan': sum(g.lifespan for g in gs) / len(gs),
                'count': len(gs),
            }
        return out
