"""게임 루프 + 에이전트 컨텍스트 (영토 확장 + 아이템)"""
import math
import random
from dataclasses import dataclass
from typing import List, Optional

from .grid import Grid
from .agents import Agent, AgentState, DIRECTIONS
from .items import Item, EffectFlash, ALL_EFFECTS, EFFECT_META, apply_effect


ITEM_TARGET = 4
FLASH_DURATION = 36


@dataclass
class GameContext:
    agent_states: dict
    areas: dict
    grid: Grid
    rng: random.Random

    def get_ranking(self) -> list:
        return sorted(self.areas.items(), key=lambda x: -x[1])

    def get_rank_state(self, rank: int, exclude_id: int = None) -> Optional[AgentState]:
        ranking = [(aid, a) for aid, a in self.get_ranking() if aid != exclude_id]
        if rank - 1 >= len(ranking):
            return None
        return self.agent_states[ranking[rank - 1][0]]

    def get_rank_position(self, rank: int, exclude_id: int = None):
        s = self.get_rank_state(rank, exclude_id)
        return (s.x, s.y) if s else None

    def get_last_state(self, exclude_id: int = None) -> Optional[AgentState]:
        ranking = [(aid, a) for aid, a in self.get_ranking() if aid != exclude_id]
        if not ranking:
            return None
        return self.agent_states[ranking[-1][0]]


class Simulation:
    def __init__(self, agents: List[Agent], width: int = 60, height: int = 60,
                 seed: int = 42, randomize_start: bool = True):
        self.grid = Grid(width, height)
        self.agents = agents
        self.tick = 0
        self.rng = random.Random(seed)

        if randomize_start:
            starts = self._random_quadrant_starts(len(agents), margin=4)
        else:
            starts = [
                (width // 4, height // 4),
                (3 * width // 4, height // 4),
                (width // 4, 3 * height // 4),
                (3 * width // 4, 3 * height // 4),
            ][:len(agents)]

        self.states = {}
        for agent, (sx, sy) in zip(agents, starts):
            self.states[agent.id] = AgentState(
                agent_id=agent.id,
                x=sx, y=sy,
                color=agent.color,
                name=agent.name,
            )
            self.grid.claim(sx, sy, agent.id)

        # 아이템 시스템: 매 판 효과 풀에서 4개 활성화
        all_keys = [e[0] for e in ALL_EFFECTS]
        self.active_effects = self.rng.sample(all_keys, k=min(4, len(all_keys)))
        self.items: List[Item] = []
        self.effect_flashes: List[EffectFlash] = []
        self.freeze_ticks: dict = {a.id: 0 for a in agents}
        self.rage_ticks: dict = {a.id: 0 for a in agents}    # 매 step에 2번 행동
        self.echo_ticks: dict = {a.id: 0 for a in agents}    # 점령 시 인접 빈 칸도
        self._spawn_items(ITEM_TARGET)

    def _random_quadrant_starts(self, n_agents: int, margin: int = 4):
        """격자를 cols×rows 셀로 분할, 셀을 무작위로 섞어 한 명씩 배치."""
        w, h = self.grid.width, self.grid.height
        cols = max(1, math.ceil(math.sqrt(n_agents)))
        rows = max(1, math.ceil(n_agents / cols))
        cell_w = w // cols
        cell_h = h // rows

        cells = []
        for r in range(rows):
            for c in range(cols):
                x0 = c * cell_w + margin
                y0 = r * cell_h + margin
                x1 = (c + 1) * cell_w - margin
                y1 = (r + 1) * cell_h - margin
                if x1 > x0 and y1 > y0:
                    cells.append((x0, y0, x1, y1))

        self.rng.shuffle(cells)
        starts = []
        for cell in cells[:n_agents]:
            x = self.rng.randint(cell[0], cell[2] - 1)
            y = self.rng.randint(cell[1], cell[3] - 1)
            starts.append((x, y))
        return starts

    def step(self):
        # RAGE 활성 에이전트는 1 step에 두 번 행동
        self._do_actions_phase()
        rage_active = [aid for aid, t in self.rage_ticks.items() if t > 0]
        if rage_active:
            self._do_actions_phase(only_ids=set(rage_active))

        # buff/debuff 카운터 일괄 감소
        for aid in list(self.freeze_ticks):
            if self.freeze_ticks[aid] > 0:
                self.freeze_ticks[aid] -= 1
        for aid in list(self.rage_ticks):
            if self.rage_ticks[aid] > 0:
                self.rage_ticks[aid] -= 1
        for aid in list(self.echo_ticks):
            if self.echo_ticks[aid] > 0:
                self.echo_ticks[aid] -= 1

        # 효과 플래시 ticks_left 감소
        for f in self.effect_flashes:
            f.ticks_left -= 1
        self.effect_flashes = [f for f in self.effect_flashes if f.ticks_left > 0]

        self.tick += 1

    def _do_actions_phase(self, only_ids: Optional[set] = None):
        """행동 결정 + 이동 1회. only_ids 지정 시 그 에이전트만."""
        ctx = GameContext(
            agent_states={**self.states},
            areas=self.grid.get_areas(len(self.agents)),
            grid=self.grid,
            rng=self.rng,
        )

        actions = {}
        for agent in self.agents:
            if only_ids is not None and agent.id not in only_ids:
                continue
            if self.freeze_ticks.get(agent.id, 0) > 0:
                actions[agent.id] = None
            else:
                actions[agent.id] = agent.act(self.states[agent.id], ctx)

        for agent in self.agents:
            if agent.id not in actions:
                continue
            state = self.states[agent.id]
            action = actions[agent.id]
            if action is None:
                continue

            dx, dy = DIRECTIONS[action]
            nx, ny = state.x + dx, state.y + dy

            if not self._can_enter(nx, ny, agent.id):
                chosen = self._find_alternative(state.x, state.y, agent.id)
                if chosen is None:
                    continue
                nx, ny, action = chosen

            state.x, state.y = nx, ny
            state.last_action = action
            if int(self.grid.cells[ny, nx]) == 0:
                self.grid.claim(nx, ny, agent.id)
                # ECHO 활성 시 인접 빈 칸도 같이 점령
                if self.echo_ticks.get(agent.id, 0) > 0:
                    for ddx, ddy in DIRECTIONS:
                        ax, ay = nx + ddx, ny + ddy
                        if (self.grid.in_bounds(ax, ay)
                                and int(self.grid.cells[ay, ax]) == 0):
                            self.grid.claim(ax, ay, agent.id)

            # 아이템 수집 (1회용)
            self._try_collect_item(agent.id)

    def _try_collect_item(self, agent_id: int):
        state = self.states[agent_id]
        for it in list(self.items):
            if it.x == state.x and it.y == state.y:
                self.items.remove(it)
                effect_key = self.rng.choice(self.active_effects)
                label, color = apply_effect(self, agent_id, effect_key)
                # 플래시 위치는 효과 받은 에이전트 머리 (teleport 후에도 갱신 위치 사용)
                ax, ay = self.states[agent_id].x, self.states[agent_id].y
                self.effect_flashes.append(EffectFlash(
                    x=ax, y=ay, text=label, color=color,
                    ticks_left=FLASH_DURATION, initial_ticks=FLASH_DURATION,
                ))
                break  # 한 칸에 한 아이템만 수집

    def _spawn_items(self, target: int):
        if len(self.items) >= target:
            return
        existing = {(it.x, it.y) for it in self.items}
        # 에이전트 머리 위치도 제외 (수집 즉시 trigger 방지)
        for s in self.states.values():
            existing.add((s.x, s.y))

        ys, xs = (self.grid.cells == 0).nonzero()
        if len(xs) == 0:
            return

        candidates = [(int(x), int(y)) for x, y in zip(xs, ys)
                      if (int(x), int(y)) not in existing]
        if not candidates:
            return

        self.rng.shuffle(candidates)
        for x, y in candidates:
            if len(self.items) >= target:
                break
            self.items.append(Item(x=x, y=y))

    def _can_enter(self, x: int, y: int, agent_id: int) -> bool:
        if not self.grid.in_bounds(x, y):
            return False
        cell = int(self.grid.cells[y, x])
        return cell == 0 or cell == agent_id

    def _find_alternative(self, x: int, y: int, agent_id: int):
        """빈 칸 우선, 자기 영토 차선. (nx, ny, action) 반환 or None"""
        order = list(range(4))
        self.rng.shuffle(order)
        # 1순위: 빈 칸
        for a in order:
            ddx, ddy = DIRECTIONS[a]
            ex, ey = x + ddx, y + ddy
            if self.grid.in_bounds(ex, ey) and int(self.grid.cells[ey, ex]) == 0:
                return ex, ey, a
        # 2순위: 자기 영토
        for a in order:
            ddx, ddy = DIRECTIONS[a]
            ex, ey = x + ddx, y + ddy
            if self.grid.in_bounds(ex, ey) and int(self.grid.cells[ey, ex]) == agent_id:
                return ex, ey, a
        return None

    def get_areas(self) -> dict:
        return self.grid.get_areas(len(self.agents))

    def get_percentages(self) -> dict:
        areas = self.get_areas()
        total = self.grid.total_cells()
        return {aid: 100.0 * a / total for aid, a in areas.items()}
