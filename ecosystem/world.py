"""3계층 생태계 시뮬레이션.

Plant → Herbivore → Carnivore. 동물은 boids 규칙 + 추격/도망.
모든 좌표는 토러스 (가장자리 wrap).
"""
import math
import random
from typing import List, Optional, Tuple

from .entities import (
    WORLD_W, WORLD_H,
    Plant, PLANT_GROW_RATE, PLANT_MAX_GROWTH, PLANT_SPAWN_PROB, PLANT_SPAWN_RADIUS,
    Herbivore, HERB_MAX_SPEED, HERB_VISION, HERB_SIZE, HERB_METAB,
    HERB_INIT_E, HERB_REPRODUCE, HERB_REPRODUCE_COST, HERB_MAX_E, HERB_PLANT_GAIN,
    HERB_W_COHESION, HERB_W_ALIGN, HERB_W_SEPARATE, HERB_W_FOOD, HERB_W_FLEE, HERB_W_NOISE,
    Carnivore, CARN_MAX_SPEED, CARN_VISION, CARN_SIZE, CARN_METAB,
    CARN_INIT_E, CARN_REPRODUCE, CARN_REPRODUCE_COST, CARN_MAX_E, CARN_HUNT_GAIN,
    CARN_KILL_RADIUS,
    CARN_W_COHESION, CARN_W_ALIGN, CARN_W_SEPARATE, CARN_W_HUNT, CARN_W_NOISE,
)
from .spatial import SpatialHash


INITIAL_PLANTS = 300
INITIAL_HERB = 50
INITIAL_CARN = 10

# 인구 cap — boids는 O(N²)라 폭발하면 시뮬이 느려진다. 안정 영역 강제.
MAX_PLANTS = 700
MAX_HERB = 280
MAX_CARN = 80
PLANT_FLOOR_REGEN = 2        # 식물이 너무 적으면 자연 재생 (사이클 회복)
PLANT_FLOOR_THRESHOLD = 80


def _wrap_delta(d: float, span: float) -> float:
    if d > span / 2: return d - span
    if d < -span / 2: return d + span
    return d


class World:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.tick = 0

        self.plants: List[Plant] = []
        self.herbivores: List[Herbivore] = []
        self.carnivores: List[Carnivore] = []

        # 공간 해시 — 매 tick rebuild
        self.plant_grid = SpatialHash()
        self.herb_grid = SpatialHash()
        self.carn_grid = SpatialHash()

        self.history: List[dict] = []
        self.births = {'herb': 0, 'carn': 0, 'plant': 0}
        self.deaths = {'herb_old': 0, 'herb_starve': 0, 'herb_killed': 0,
                       'carn_old': 0, 'carn_starve': 0}

        self._spawn_initial()

    def _spawn_initial(self):
        for _ in range(INITIAL_PLANTS):
            self.plants.append(Plant(
                x=self.rng.uniform(0, WORLD_W),
                y=self.rng.uniform(0, WORLD_H),
                growth=self.rng.uniform(10, PLANT_MAX_GROWTH),
            ))
        for _ in range(INITIAL_HERB):
            self.herbivores.append(Herbivore(
                x=self.rng.uniform(0, WORLD_W),
                y=self.rng.uniform(0, WORLD_H),
                vx=self.rng.uniform(-1, 1),
                vy=self.rng.uniform(-1, 1),
            ))
        for _ in range(INITIAL_CARN):
            self.carnivores.append(Carnivore(
                x=self.rng.uniform(0, WORLD_W),
                y=self.rng.uniform(0, WORLD_H),
                vx=self.rng.uniform(-1, 1),
                vy=self.rng.uniform(-1, 1),
            ))

    # ── 메인 루프 ─────────────────────────────────────────

    def step(self):
        self.tick += 1
        # 매 tick: 해시 rebuild (모두 위치가 바뀐 후 재구축)
        self.plant_grid.build(self.plants)
        self.herb_grid.build(self.herbivores)
        self.carn_grid.build(self.carnivores)

        self._update_plants()
        # 식물 목록 변경되었으니 plant_grid rebuild
        self.plant_grid.build(self.plants)

        self._update_herbivores()
        self.herb_grid.build(self.herbivores)

        self._update_carnivores()
        self._record_history()

    # ── 식물 ────────────────────────────────────────────

    def _update_plants(self):
        new_plants = []
        for p in self.plants:
            if not p.alive:
                continue
            p.growth = min(PLANT_MAX_GROWTH, p.growth + PLANT_GROW_RATE)
            if (p.growth >= PLANT_MAX_GROWTH * 0.8
                    and len(self.plants) + len(new_plants) < MAX_PLANTS
                    and self.rng.random() < PLANT_SPAWN_PROB):
                a = self.rng.uniform(0, math.tau)
                r = self.rng.uniform(8, PLANT_SPAWN_RADIUS)
                nx = (p.x + math.cos(a) * r) % WORLD_W
                ny = (p.y + math.sin(a) * r) % WORLD_H
                new_plants.append(Plant(x=nx, y=ny, growth=0.0))
                self.births['plant'] += 1

        # 식물이 너무 적으면 자연 재생 — 사이클이 한쪽으로 무너지지 않도록
        total = len(self.plants) + len(new_plants)
        if total < PLANT_FLOOR_THRESHOLD:
            for _ in range(PLANT_FLOOR_REGEN):
                new_plants.append(Plant(
                    x=self.rng.uniform(0, WORLD_W),
                    y=self.rng.uniform(0, WORLD_H),
                    growth=0.0,
                ))
                self.births['plant'] += 1

        self.plants = [p for p in self.plants if p.alive] + new_plants

    # ── 초식 ────────────────────────────────────────────

    def _update_herbivores(self):
        new_kids: List[Herbivore] = []
        for h in self.herbivores:
            if not h.alive:
                continue

            ax = ay = 0.0  # 가속도 누적

            # 도망 — 가장 가까운 carnivore (강한 가중)
            danger = self._nearest_in(self.carn_grid, h.x, h.y, HERB_VISION * 0.9)
            if danger is not None:
                dx = _wrap_delta(h.x - danger.x, WORLD_W)
                dy = _wrap_delta(h.y - danger.y, WORLD_H)
                d = math.hypot(dx, dy) or 1.0
                ax += (dx / d) * HERB_W_FLEE
                ay += (dy / d) * HERB_W_FLEE

            # 식물 — 가장 가까운 자란 식물
            target_plant = self._nearest_plant(h.x, h.y, HERB_VISION)
            if target_plant is not None:
                dx = _wrap_delta(target_plant.x - h.x, WORLD_W)
                dy = _wrap_delta(target_plant.y - h.y, WORLD_H)
                d = math.hypot(dx, dy) or 1.0
                ax += (dx / d) * HERB_W_FOOD
                ay += (dy / d) * HERB_W_FOOD

            # boids — 같은 종 응집/정렬/분리
            cx, cy, vx, vy, n = self._neighbors_in(self.herb_grid, h, HERB_VISION)
            if n > 0:
                # cohesion
                cdx = _wrap_delta(cx - h.x, WORLD_W)
                cdy = _wrap_delta(cy - h.y, WORLD_H)
                ax += cdx * HERB_W_COHESION / max(1, n)
                ay += cdy * HERB_W_COHESION / max(1, n)
                # alignment
                ax += (vx / n - h.vx) * HERB_W_ALIGN
                ay += (vy / n - h.vy) * HERB_W_ALIGN

            # separation — 너무 가까운 같은 종에서 밀어냄
            sep_x, sep_y = self._separation_in(self.herb_grid, h, HERB_SIZE * 3)
            ax += sep_x * HERB_W_SEPARATE
            ay += sep_y * HERB_W_SEPARATE

            # 노이즈
            ax += self.rng.uniform(-1, 1) * HERB_W_NOISE
            ay += self.rng.uniform(-1, 1) * HERB_W_NOISE

            # 적용
            h.vx += ax
            h.vy += ay
            speed = math.hypot(h.vx, h.vy) or 1.0
            if speed > HERB_MAX_SPEED:
                h.vx *= HERB_MAX_SPEED / speed
                h.vy *= HERB_MAX_SPEED / speed

            h.x = (h.x + h.vx) % WORLD_W
            h.y = (h.y + h.vy) % WORLD_H
            h.energy -= HERB_METAB
            h.age += 1

            # 식물 섭취
            self._herb_eat(h)

            # 죽음 / 번식
            if h.energy <= 0:
                h.alive = False
                self.deaths['herb_starve'] += 1
            elif h.age > 1500:
                h.alive = False
                self.deaths['herb_old'] += 1
            elif (h.energy >= HERB_REPRODUCE
                    and len(self.herbivores) + len(new_kids) < MAX_HERB):
                h.energy -= HERB_REPRODUCE_COST
                a = self.rng.uniform(0, math.tau)
                ox = (h.x + math.cos(a) * (HERB_SIZE + 4)) % WORLD_W
                oy = (h.y + math.sin(a) * (HERB_SIZE + 4)) % WORLD_H
                new_kids.append(Herbivore(
                    x=ox, y=oy,
                    vx=self.rng.uniform(-1, 1),
                    vy=self.rng.uniform(-1, 1),
                    energy=HERB_REPRODUCE_COST * 0.8,
                ))
                self.births['herb'] += 1

        self.herbivores = [h for h in self.herbivores if h.alive] + new_kids

    def _herb_eat(self, h: Herbivore):
        eat_r = HERB_SIZE + 4
        eat_r2 = eat_r * eat_r
        for p in self.plant_grid.query(h.x, h.y, eat_r):
            if not p.alive or p.growth < 8:
                continue
            dx = _wrap_delta(p.x - h.x, WORLD_W)
            dy = _wrap_delta(p.y - h.y, WORLD_H)
            if dx * dx + dy * dy < eat_r2:
                gain = HERB_PLANT_GAIN * (p.growth / PLANT_MAX_GROWTH)
                h.energy = min(HERB_MAX_E, h.energy + gain)
                p.alive = False
                return  # 한 tick에 한 식물만

    # ── 육식 ────────────────────────────────────────────

    def _update_carnivores(self):
        new_kids: List[Carnivore] = []
        for c in self.carnivores:
            if not c.alive:
                continue

            ax = ay = 0.0

            # 추격 — 가장 가까운 초식
            prey = self._nearest_in(self.herb_grid, c.x, c.y, CARN_VISION)
            if prey is not None:
                dx = _wrap_delta(prey.x - c.x, WORLD_W)
                dy = _wrap_delta(prey.y - c.y, WORLD_H)
                d = math.hypot(dx, dy) or 1.0
                ax += (dx / d) * CARN_W_HUNT
                ay += (dy / d) * CARN_W_HUNT

            # boids
            cx, cy, vx, vy, n = self._neighbors_in(self.carn_grid, c, CARN_VISION)
            if n > 0:
                cdx = _wrap_delta(cx - c.x, WORLD_W)
                cdy = _wrap_delta(cy - c.y, WORLD_H)
                ax += cdx * CARN_W_COHESION / max(1, n)
                ay += cdy * CARN_W_COHESION / max(1, n)
                ax += (vx / n - c.vx) * CARN_W_ALIGN
                ay += (vy / n - c.vy) * CARN_W_ALIGN

            sep_x, sep_y = self._separation_in(self.carn_grid, c, CARN_SIZE * 3)
            ax += sep_x * CARN_W_SEPARATE
            ay += sep_y * CARN_W_SEPARATE

            ax += self.rng.uniform(-1, 1) * CARN_W_NOISE
            ay += self.rng.uniform(-1, 1) * CARN_W_NOISE

            c.vx += ax
            c.vy += ay
            speed = math.hypot(c.vx, c.vy) or 1.0
            if speed > CARN_MAX_SPEED:
                c.vx *= CARN_MAX_SPEED / speed
                c.vy *= CARN_MAX_SPEED / speed

            c.x = (c.x + c.vx) % WORLD_W
            c.y = (c.y + c.vy) % WORLD_H
            c.energy -= CARN_METAB
            c.age += 1

            self._carn_hunt(c)

            if c.energy <= 0:
                c.alive = False
                self.deaths['carn_starve'] += 1
            elif c.age > 1800:
                c.alive = False
                self.deaths['carn_old'] += 1
            elif (c.energy >= CARN_REPRODUCE
                    and len(self.carnivores) + len(new_kids) < MAX_CARN):
                c.energy -= CARN_REPRODUCE_COST
                a = self.rng.uniform(0, math.tau)
                ox = (c.x + math.cos(a) * (CARN_SIZE + 4)) % WORLD_W
                oy = (c.y + math.sin(a) * (CARN_SIZE + 4)) % WORLD_H
                new_kids.append(Carnivore(
                    x=ox, y=oy,
                    vx=self.rng.uniform(-1, 1),
                    vy=self.rng.uniform(-1, 1),
                    energy=CARN_REPRODUCE_COST * 0.8,
                ))
                self.births['carn'] += 1

        self.carnivores = [c for c in self.carnivores if c.alive] + new_kids

    def _carn_hunt(self, c: Carnivore):
        kr2 = CARN_KILL_RADIUS * CARN_KILL_RADIUS
        for h in self.herb_grid.query(c.x, c.y, CARN_KILL_RADIUS):
            if not h.alive:
                continue
            dx = _wrap_delta(h.x - c.x, WORLD_W)
            dy = _wrap_delta(h.y - c.y, WORLD_H)
            if dx * dx + dy * dy < kr2:
                c.energy = min(CARN_MAX_E, c.energy + CARN_HUNT_GAIN)
                h.alive = False
                self.deaths['herb_killed'] += 1
                return

    # ── 공용 유틸 ───────────────────────────────────────

    def _nearest_in(self, grid: SpatialHash, x: float, y: float, radius: float):
        best = None
        best_d2 = radius * radius
        for o in grid.query(x, y, radius):
            if not o.alive:
                continue
            dx = _wrap_delta(o.x - x, WORLD_W)
            dy = _wrap_delta(o.y - y, WORLD_H)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = o
        return best

    def _neighbors_in(self, grid: SpatialHash, self_obj, radius: float):
        cx = cy = vx = vy = 0.0
        n = 0
        r2 = radius * radius
        for o in grid.query(self_obj.x, self_obj.y, radius):
            if o is self_obj or not o.alive:
                continue
            dx = _wrap_delta(o.x - self_obj.x, WORLD_W)
            dy = _wrap_delta(o.y - self_obj.y, WORLD_H)
            if dx * dx + dy * dy > r2:
                continue
            cx += self_obj.x + dx
            cy += self_obj.y + dy
            vx += o.vx
            vy += o.vy
            n += 1
        if n == 0:
            return (self_obj.x, self_obj.y, 0.0, 0.0, 0)
        return (cx / n, cy / n, vx, vy, n)

    def _separation_in(self, grid: SpatialHash, self_obj, radius: float):
        sx = sy = 0.0
        r2 = radius * radius
        for o in grid.query(self_obj.x, self_obj.y, radius):
            if o is self_obj or not o.alive:
                continue
            dx = _wrap_delta(self_obj.x - o.x, WORLD_W)
            dy = _wrap_delta(self_obj.y - o.y, WORLD_H)
            d2 = dx * dx + dy * dy
            if d2 < r2 and d2 > 0:
                inv = 1.0 / d2
                sx += dx * inv
                sy += dy * inv
        return sx, sy

    def _nearest(self, lst, x: float, y: float, radius: float):
        best = None
        best_d2 = radius * radius
        for o in lst:
            if not o.alive:
                continue
            dx = _wrap_delta(o.x - x, WORLD_W)
            dy = _wrap_delta(o.y - y, WORLD_H)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = o
        return best

    def _nearest_plant(self, x: float, y: float, radius: float) -> Optional[Plant]:
        best = None
        best_d2 = radius * radius
        for p in self.plant_grid.query(x, y, radius):
            if not p.alive or p.growth < 8:
                continue
            dx = _wrap_delta(p.x - x, WORLD_W)
            dy = _wrap_delta(p.y - y, WORLD_H)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = p
        return best

    def _neighbors_summary(self, lst, self_obj, radius: float
                           ) -> Tuple[float, float, float, float, int]:
        """boids용 — (cx, cy, vx_sum, vy_sum, n) 반환."""
        cx = cy = vx = vy = 0.0
        n = 0
        r2 = radius * radius
        for o in lst:
            if o is self_obj or not o.alive:
                continue
            dx = _wrap_delta(o.x - self_obj.x, WORLD_W)
            dy = _wrap_delta(o.y - self_obj.y, WORLD_H)
            if dx * dx + dy * dy > r2:
                continue
            cx += self_obj.x + dx
            cy += self_obj.y + dy
            vx += o.vx
            vy += o.vy
            n += 1
        if n == 0:
            return (self_obj.x, self_obj.y, 0.0, 0.0, 0)
        return (cx / n, cy / n, vx, vy, n)

    def _separation(self, lst, self_obj, radius: float) -> Tuple[float, float]:
        sx = sy = 0.0
        r2 = radius * radius
        for o in lst:
            if o is self_obj or not o.alive:
                continue
            dx = _wrap_delta(self_obj.x - o.x, WORLD_W)
            dy = _wrap_delta(self_obj.y - o.y, WORLD_H)
            d2 = dx * dx + dy * dy
            if d2 < r2 and d2 > 0:
                inv = 1.0 / d2
                sx += dx * inv
                sy += dy * inv
        return sx, sy

    # ── 통계 ─────────────────────────────────────────────

    def _record_history(self):
        self.history.append({
            'tick': self.tick,
            'plants': len(self.plants),
            'herb': len(self.herbivores),
            'carn': len(self.carnivores),
        })

    def populations(self) -> dict:
        return {
            'plants': len(self.plants),
            'herb': len(self.herbivores),
            'carn': len(self.carnivores),
        }

    def is_extinct(self) -> bool:
        return not self.herbivores and not self.carnivores
