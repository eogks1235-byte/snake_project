"""게임 루프 + 에이전트 컨텍스트 (영토 확장 + 아이템)"""
import math
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.ndimage import binary_fill_holes

from .grid import Grid, WALL
from .agents import Agent, AgentState, DIRECTIONS
from .items import Item, EffectFlash, ALL_EFFECTS, EFFECT_META, apply_effect
from .maps import resolve_map


ITEM_TARGET = 4
FLASH_DURATION = 36
ELIMINATION_FLASH_DURATION = 120
BETRAYER_REVEAL_TICK = 600     # 20초 @ 30 FPS — Betrayer 가 팀에서 분리되는 시점

# 카테고리별 속도 modifier (1.0 = 정상)
# DEF 는 약간 느리지만 head 자리 인클로저 면역 + 영토 회복력 + 적 영토 침식.
ROLE_SPEED = {
    'ATK': 1.25,   # 빠른 확장 (1.30 → 1.25 가벼운 너프)
    'DEF': 0.95,   # 살짝 느림 + 인클로저 면역 + (E) 침식
    'JKR': 1.00,
    'EXP': 1.05,   # 약간 빠름 (1.10 → 1.05 가벼운 너프)
    'TM':  1.00,
}


@dataclass
class GameContext:
    agent_states: dict
    areas: dict
    grid: Grid
    rng: random.Random
    items: list  # 현재 맵 위 아이템 (Item 객체 리스트)
    effect_flashes: list = None   # 에이전트 능력 시각화용 (EffectFlash 리스트)
    tick: int = 0
    teams: dict = None            # agent_id -> team_id. free-for-all 에선 각자 별도 팀.
    team_mode: bool = False
    sticky_cells: dict = None     # TrapperAgent 가 함정 깔 때 직접 추가 (sim 의 dict)
    freeze_ticks: dict = None     # 에이전트가 다른 에이전트를 freeze 시킬 때 사용
    dead: dict = None             # AnarchistAgent 가 살아있는 에이전트 swap 용
    wildcard_pending: list = None # WildcardAgent 가 효과 발동 요청 시 사용
    role_by_id: dict = None       # DEF 50% 저항 체크용 (agents 측에서 사용)

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
                 seed: int = 42, randomize_start: bool = True,
                 map_preset: str = None, teams: Optional[dict] = None,
                 team_mode: Optional[bool] = None):
        self.grid = Grid(width, height)
        self.agents = agents
        self.tick = 0
        self.rng = random.Random(seed)
        # teams: {agent_id: team_id} 매핑. None이면 free-for-all (각자 1팀).
        # team_mode: 명시 전달 시 그 값 사용. 자동 추론은 fallback.
        if teams is None:
            self.teams = {a.id: a.id for a in agents}
            self.team_mode = False if team_mode is None else team_mode
        else:
            self.teams = dict(teams)
            if team_mode is None:
                self.team_mode = len(set(self.teams.values())) < len(agents)
            else:
                self.team_mode = team_mode

        # 맵 프리셋 — 이름 지정 시 그것, 아니면 랜덤
        self.map_name, build_map = resolve_map(map_preset, self.rng)
        self.grid.add_walls(build_map(width, height, self.rng))

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
        self.haste_ticks: dict = {a.id: 0 for a in agents}   # x2.5 속도
        self.haste_carry: dict = {a.id: 0.0 for a in agents} # haste 누적 잔여 행동
        # 팀모드 전용: 같은 팀 영토 위에 머무를 때 +0.5/step 누적 → 평균 x1.5 속도
        self.team_speed_carry: dict = {a.id: 0.0 for a in agents}
        # SLOW (sticky 트랩 등) — 활성 중엔 매 phase 50% 확률로 행동 skip → x0.5 속도
        self.slow_ticks: dict = {a.id: 0 for a in agents}
        self.dead: dict = {a.id: False for a in agents}
        self.death_positions: dict = {}    # agent_id -> (x, y) at moment of death
        self.death_ticks: dict = {}        # agent_id -> tick when died
        # 끈적이 함정 — dict (x, y) -> owner_team_id. 밟으면 1회 트리거 후 제거.
        # 팀모드에선 같은 팀이 밟아도 트리거 안 됨 (팀 보호). free-for-all에선
        # owner_team 이 발동자 id 그 자체라 자기 함정도 자기를 잡음 (무차별).
        self.sticky_cells: dict = {}
        self.sticky_slow_ticks = 90        # 3초 @ 30fps
        # Betrayer 특권: 모든 영토 통과 자유 (자기 영토 / 옛 팀 / 적 팀 무관).
        # _can_enter 가 매 step 빈번 호출되므로 캐시 매핑.
        self.is_betrayer: dict = {
            a.id: a.__class__.__name__ == 'BetrayerAgent' for a in agents
        }
        # Betrayer reveal 시점 — main.py 가 build_simulation 에서 설정. 도달 시 솔로 팀 분리.
        self.betrayer_reveal_at: dict = {}      # agent_id -> tick
        self.betrayer_solo_team: dict = {}      # agent_id -> 솔로 팀 id (reveal 시 사용)
        # Trapper head 자리는 인클로저 면역 — 가만히 있는 동안 영토 0 되어 죽지 않게.
        self.is_trapper: dict = {
            a.id: a.__class__.__name__ == 'TrapperAgent' for a in agents
        }
        self.is_inferno: dict = {
            a.id: a.__class__.__name__ == 'InfernoAgent' for a in agents
        }
        # WildcardAgent 가 발동 예약한 (agent_id, effect_key) 리스트
        self.wildcard_pending: list = []
        # 카테고리 속도 modifier 매핑 — agent.id -> 'ATK'/'DEF'/...
        from .agents import ROLE_GLYPHS
        self.role_by_id: dict = {a.id: ROLE_GLYPHS.get(a.name, '') for a in agents}
        # Revenant 시스템 — RevenantAgent 는 1회 즉시 부활 (목숨 +1). 능력 없음.
        self.is_revenant_class: dict = {
            a.id: a.__class__.__name__ == 'RevenantAgent' for a in agents
        }
        self.revive_used: dict = {a.id: False for a in agents}   # 부활 1회 사용 여부
        # 영토 % 시간별 추이 (sparkline 용) — 최근 50틱
        from collections import deque
        self.area_history: dict = {a.id: deque(maxlen=50) for a in agents}
        # 시간 제한 모드 (Blitz) — None 이면 무제한, int 면 그 tick 도달 시 종료
        self.time_limit: int = None
        self._spawn_items(ITEM_TARGET)

    def _random_quadrant_starts(self, n_agents: int, margin: int = 4):
        """격자를 cols×rows 셀로 분할, 셀을 무작위로 섞어 한 명씩 배치.

        벽 위 또는 이미 사용된 위치는 피해서 빈 칸을 고른다.
        """
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
        used = set()
        starts = []
        for cell in cells[:n_agents]:
            pos = self._pick_empty_in_region(cell, used) or self._pick_any_empty(used)
            if pos is None:
                pos = (cell[0], cell[1])  # 최후의 보루
            used.add(pos)
            starts.append(pos)

        # Trapper 재배치 — 다른 에이전트 시작 위치에서 6~10칸 거리에
        # (5×5 trigger 범위 밖이라 즉시 발동은 안 함, 자연 진행 중 발동 유도)
        for i, agent in enumerate(self.agents[:n_agents]):
            if agent.__class__.__name__ != 'TrapperAgent':
                continue
            targets = [p for j, p in enumerate(starts) if j != i]
            if not targets:
                continue
            target = self.rng.choice(targets)
            new_pos = None
            for _ in range(60):
                angle = self.rng.uniform(0, 2 * math.pi)
                dist = self.rng.uniform(6, 10)
                nx = target[0] + int(math.cos(angle) * dist)
                ny = target[1] + int(math.sin(angle) * dist)
                if (self.grid.in_bounds(nx, ny)
                        and int(self.grid.cells[ny, nx]) != WALL
                        and (nx, ny) not in used):
                    new_pos = (nx, ny)
                    break
            if new_pos:
                used.discard(starts[i])
                starts[i] = new_pos
                used.add(new_pos)
        return starts

    def _pick_empty_in_region(self, region, used):
        x0, y0, x1, y1 = region
        for _ in range(60):
            x = self.rng.randint(x0, x1 - 1)
            y = self.rng.randint(y0, y1 - 1)
            if (x, y) not in used and int(self.grid.cells[y, x]) == 0:
                return (x, y)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if (xx, yy) not in used and int(self.grid.cells[yy, xx]) == 0:
                    return (xx, yy)
        return None

    def _pick_any_empty(self, used):
        ys, xs = (self.grid.cells == 0).nonzero()
        options = [(int(x), int(y)) for x, y in zip(xs, ys)
                   if (int(x), int(y)) not in used]
        if not options:
            return None
        return self.rng.choice(options)

    def step(self):
        # Betrayer reveal — 시점 도달한 Betrayer 를 솔로 팀으로 전환.
        # 옛 팀원이 Betrayer 영토 위에 있으면 갇힘 방지를 위해 그 셀을 자기 영토로
        # 강제 변환 (작은 island 1칸). Betrayer 는 그 island 를 enclosure 로 다시
        # 흡수 가능 — 게임 진행에 자연스럽게 통합.
        if self.betrayer_reveal_at:
            for aid in [a for a, t in self.betrayer_reveal_at.items()
                        if self.tick >= t]:
                old_team = self.teams[aid]
                for other in self.agents:
                    if other.id == aid or self.dead.get(other.id, False):
                        continue
                    if self.teams.get(other.id) != old_team:
                        continue
                    o = self.states[other.id]
                    if int(self.grid.cells[o.y, o.x]) == aid:
                        self.grid.cells[o.y, o.x] = other.id
                self.teams[aid] = self.betrayer_solo_team[aid]
                state = self.states[aid]
                self.effect_flashes.append(EffectFlash(
                    x=state.x, y=state.y,
                    text=f'BETRAYED!',
                    color=(220, 40, 60),
                    ticks_left=ELIMINATION_FLASH_DURATION,
                    initial_ticks=ELIMINATION_FLASH_DURATION,
                ))
                del self.betrayer_reveal_at[aid]

        # RAGE 활성 에이전트는 1 step에 두 번 행동
        self._do_actions_phase()
        rage_active = [aid for aid, t in self.rage_ticks.items() if t > 0
                       and not self.dead.get(aid, False)]
        if rage_active:
            self._do_actions_phase(only_ids=set(rage_active))

        # 팀 역할 modifier 효과 — 캐릭터 행동에 더해 매 N틱 추가 효과
        if self.team_mode and self.tick % 25 == 0:
            for agent in self.agents:
                if self.dead.get(agent.id, False):
                    continue
                role = getattr(agent, 'team_role', None)
                if role is None:
                    continue
                my_team = self.teams.get(agent.id)
                own = self.grid.cells == agent.id
                ys, xs = own.nonzero()
                if role == 'attacker':
                    # 자기 영토 옆 적 팀 영토 1칸 침식
                    enemy_edges = []
                    for x, y in zip(xs.tolist(), ys.tolist()):
                        for ddx, ddy in DIRECTIONS:
                            nx, ny = x + ddx, y + ddy
                            if not self.grid.in_bounds(nx, ny):
                                continue
                            cell = int(self.grid.cells[ny, nx])
                            if (cell > 0 and cell != agent.id
                                    and self.teams.get(cell) != my_team):
                                enemy_edges.append((nx, ny))
                    if enemy_edges:
                        nx, ny = self.rng.choice(enemy_edges)
                        self.grid.cells[ny, nx] = agent.id
                elif role == 'supporter':
                    # 팀원 영토 옆 빈 칸 1칸 점령 (팀 지원)
                    team_cells = np.zeros_like(self.grid.cells, dtype=bool)
                    for a2 in self.agents:
                        if self.teams.get(a2.id) == my_team:
                            team_cells |= (self.grid.cells == a2.id)
                    tys, txs = team_cells.nonzero()
                    empty_edges = []
                    for x, y in zip(txs.tolist(), tys.tolist()):
                        for ddx, ddy in DIRECTIONS:
                            nx, ny = x + ddx, y + ddy
                            if (self.grid.in_bounds(nx, ny)
                                    and int(self.grid.cells[ny, nx]) == 0):
                                empty_edges.append((nx, ny))
                    if empty_edges:
                        nx, ny = self.rng.choice(empty_edges)
                        self.grid.cells[ny, nx] = agent.id

        # (E) DEF 카테고리 적 영토 침식 — 20틱마다 자기 영토 옆 적 영토 1칸 변환.
        # 차단 정체성 강화 + 약간의 능동 확장. 같은 팀 영토는 침식 대상 아님.
        if self.tick % 20 == 0:
            for agent in self.agents:
                if self.dead.get(agent.id, False):
                    continue
                if self.role_by_id.get(agent.id, '') != 'DEF':
                    continue
                own = self.grid.cells == agent.id
                ys, xs = own.nonzero()
                enemy_edges = []
                for x, y in zip(xs.tolist(), ys.tolist()):
                    for ddx, ddy in DIRECTIONS:
                        nx, ny = x + ddx, y + ddy
                        if not self.grid.in_bounds(nx, ny):
                            continue
                        cell = int(self.grid.cells[ny, nx])
                        if cell > 0 and cell != agent.id:
                            if (self.team_mode and
                                    self.teams.get(cell) == self.teams.get(agent.id)):
                                continue
                            enemy_edges.append((nx, ny))
                if enemy_edges:
                    nx, ny = self.rng.choice(enemy_edges)
                    self.grid.cells[ny, nx] = agent.id

        # 카테고리 속도 — ATK/EXP 는 speed>1.0 차이만큼 확률로 추가 phase
        bonus_ids = set()
        for a in self.agents:
            if self.dead.get(a.id, False):
                continue
            role = self.role_by_id.get(a.id, '')
            speed = ROLE_SPEED.get(role, 1.0)
            if speed > 1.0 and self.rng.random() < (speed - 1.0):
                bonus_ids.add(a.id)
        if bonus_ids:
            self._do_actions_phase(only_ids=bonus_ids)

        # HASTE: x2.5 — carry += 1.5 후, 1.0 이상이면 추가 phase 진행
        haste_active = [aid for aid, t in self.haste_ticks.items() if t > 0
                        and not self.dead.get(aid, False)]
        if haste_active:
            for aid in haste_active:
                self.haste_carry[aid] = self.haste_carry.get(aid, 0.0) + 1.5
            for _ in range(2):  # 최대 2번 추가 phase (carry 누적 한계)
                ids = {aid for aid in haste_active
                       if self.haste_carry.get(aid, 0.0) >= 1.0}
                if not ids:
                    break
                self._do_actions_phase(only_ids=ids)
                for aid in ids:
                    self.haste_carry[aid] -= 1.0

        # 팀모드 보너스: 같은 팀 영토 위에 있는 에이전트는 +0.5/step carry,
        # 1.0 이상이면 추가 phase → 평균 x1.5 속도. 팀 영토 밖이면 천천히 감쇠.
        # 단, Haste(x2.5) 활성 중인 에이전트는 더 빠른 효과가 우선 → team 보너스 미적용.
        if self.team_mode:
            on_team = set()
            for agent in self.agents:
                if self.dead.get(agent.id, False):
                    continue
                if self.haste_ticks.get(agent.id, 0) > 0:
                    continue   # haste 가 우선
                state = self.states[agent.id]
                cell = int(self.grid.cells[state.y, state.x])
                if cell > 0 and self.teams.get(cell) == self.teams.get(agent.id):
                    on_team.add(agent.id)
            for aid in on_team:
                self.team_speed_carry[aid] = (
                    self.team_speed_carry.get(aid, 0.0) + 0.5
                )
            for aid in list(self.team_speed_carry):
                if aid not in on_team:
                    self.team_speed_carry[aid] = max(
                        0.0, self.team_speed_carry[aid] - 0.2
                    )
            ids = {aid for aid, c in self.team_speed_carry.items()
                   if c >= 1.0 and not self.dead.get(aid, False)}
            if ids:
                self._do_actions_phase(only_ids=ids)
                for aid in ids:
                    self.team_speed_carry[aid] -= 1.0

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
        for aid in list(self.haste_ticks):
            if self.haste_ticks[aid] > 0:
                self.haste_ticks[aid] -= 1
            else:
                # haste 비활성이면 carry 초기화 (누적 방지)
                self.haste_carry[aid] = 0.0
        for aid in list(self.slow_ticks):
            if self.slow_ticks[aid] > 0:
                self.slow_ticks[aid] -= 1

        # Wildcard 예약 효과 처리 — agent 가 예약한 효과를 자기 자신에게 발동
        if self.wildcard_pending:
            for aid, key in self.wildcard_pending:
                if self.dead.get(aid, False):
                    continue
                apply_effect(self, aid, key)
            self.wildcard_pending.clear()

        # 효과 플래시 ticks_left 감소
        for f in self.effect_flashes:
            f.ticks_left -= 1
        self.effect_flashes = [f for f in self.effect_flashes if f.ticks_left > 0]

        # 인클로저(둘러싸기) 해소 → 0칸으로 떨어진 에이전트 사망 처리
        self._resolve_enclosures()
        self._check_deaths()

        # 끈적이 트리거: 에이전트가 sticky 셀 위에 있으면 slow + sticky 제거
        self._trigger_sticky()
        # 영토가 칠해진 sticky 위치는 자동 정리 (빈 칸에만 남는다)
        if self.sticky_cells:
            self.sticky_cells = {pos: owner for pos, owner in self.sticky_cells.items()
                                 if int(self.grid.cells[pos[1], pos[0]]) == 0}

        # 영토 시간별 추이 기록 (sparkline 용)
        areas = self.grid.get_areas(len(self.agents))
        total = self.grid.total_cells()
        for a in self.agents:
            pct = (100.0 * areas.get(a.id, 0) / total) if total else 0.0
            self.area_history[a.id].append(pct)

        self.tick += 1

    def _resolve_enclosures(self):
        """벽 + (자기/같은 팀) 영토로 완전히 둘러싸인 영역을 흡수.

        팀 인식: 같은 팀원의 영토도 barrier로 인정 → 팀 단위로 적 영토를 가둠.
        흡수된 영역은 둘러싼 팀 내에서 행동한 에이전트(=barrier에 참여한 first
        encloser)에게 귀속.

        - 한 팀만 둘러싼 영역 → 그 팀의 대표 에이전트가 흡수
        - 두 팀 이상이 동시에 둘러싼 영역 → 분쟁, 변동 없음
        - 맵 경계까지 닿은 영역 → 외부 → 흡수 대상 아님
        """
        cells = self.grid.cells
        # encloser_code: 0=없음, >0=(팀id+1), -1=분쟁.
        # team id 가 0일 수 있으므로 1-shift 인코딩을 쓴다.
        encloser_code = np.zeros(cells.shape, dtype=np.int16)
        # 팀 대표 에이전트 (그 팀에서 살아있는 가장 작은 id)
        team_rep: dict = {}
        for agent in self.agents:
            if self.dead.get(agent.id, False):
                continue
            tid = self.teams[agent.id]
            if tid not in team_rep or agent.id < team_rep[tid]:
                team_rep[tid] = agent.id

        any_enclosed = False
        for tid, rep_id in team_rep.items():
            code = tid + 1
            team_member_ids = [a.id for a in self.agents
                               if self.teams[a.id] == tid
                               and not self.dead.get(a.id, False)]
            barrier = np.zeros_like(cells, dtype=bool)
            for mid in team_member_ids:
                barrier |= (cells == mid)
            barrier |= (cells == WALL)

            filled = binary_fill_holes(barrier)
            enclosed = filled & ~barrier
            if not enclosed.any():
                continue
            any_enclosed = True
            new_solo = enclosed & (encloser_code == 0)
            encloser_code[new_solo] = code
            contested = enclosed & (encloser_code > 0) & (encloser_code != code)
            encloser_code[contested] = -1

        if not any_enclosed:
            return

        # Trapper head + DEF 카테고리 head 자리는 인클로저 흡수 면역
        # → 발동 대기 / 후반 안정성 보장
        protected = set()
        for agent in self.agents:
            if self.dead.get(agent.id, False):
                continue
            is_trap = self.is_trapper.get(agent.id, False)
            is_def = self.role_by_id.get(agent.id, '') == 'DEF'
            if is_trap or is_def:
                s = self.states[agent.id]
                protected.add((s.x, s.y))

        # DEF 카테고리 셀 마스크 — 인클로저 흡수 시 빈 칸으로만 변환 (적 색 X)
        def_cell_mask = np.zeros_like(cells, dtype=bool)
        for aid, role in self.role_by_id.items():
            if role == 'DEF' and not self.dead.get(aid, False):
                def_cell_mask |= (cells == aid)

        for tid, rep_id in team_rep.items():
            mask = encloser_code == (tid + 1)
            if not mask.any():
                continue
            for (px, py) in protected:
                mask[py, px] = False
            # DEF 영토는 빈 칸으로 회복, 나머지는 적 색으로 흡수
            def_in_mask = mask & def_cell_mask
            cells[def_in_mask] = 0
            cells[mask & ~def_in_mask] = rep_id

            # 인클로저 안 아이템 자동 수집 + 효과 발동
            if self.items:
                captured = [it for it in self.items if mask[it.y, it.x]]
                for it in captured:
                    self.items.remove(it)
                    effect_key = self.rng.choice(self.active_effects)
                    label, color = apply_effect(self, rep_id, effect_key)
                    self.effect_flashes.append(EffectFlash(
                        x=it.x, y=it.y, text=label, color=color,
                        ticks_left=FLASH_DURATION,
                        initial_ticks=FLASH_DURATION,
                    ))

    def def_resists(self, cell_value: int) -> bool:
        """DEF 카테고리 셀이면 50% 확률로 저항 (변경 무효)."""
        if cell_value <= 0:
            return False
        if self.role_by_id.get(cell_value, '') != 'DEF':
            return False
        return self.rng.random() < 0.5

    def _trigger_sticky(self):
        """에이전트 머리가 끈적이 위에 있으면 SLOW (x0.5 속도) 부여 + 함정 제거.

        팀모드: 같은 팀이 깐 함정은 트리거 안 함. free-for-all: 모두 트리거.
        """
        if not self.sticky_cells:
            return
        for agent in self.agents:
            if self.dead.get(agent.id, False):
                continue
            state = self.states[agent.id]
            pos = (state.x, state.y)
            if pos in self.sticky_cells:
                owner_team = self.sticky_cells[pos]
                if self.team_mode and self.teams[agent.id] == owner_team:
                    continue
                del self.sticky_cells[pos]
                self.slow_ticks[agent.id] = (
                    self.slow_ticks.get(agent.id, 0) + self.sticky_slow_ticks
                )
                self.effect_flashes.append(EffectFlash(
                    x=pos[0], y=pos[1], text='STICKY!',
                    color=(170, 100, 180),
                    ticks_left=FLASH_DURATION,
                    initial_ticks=FLASH_DURATION,
                ))

    def _check_deaths(self):
        """영토 0칸으로 떨어진 에이전트를 사망 처리.

        RevenantAgent 는 첫 사망 시 즉시 무작위 빈 칸에서 부활 (목숨 +1).
        두 번째 사망부터는 일반 처리.
        """
        areas = self.grid.get_areas(len(self.agents))
        for agent in self.agents:
            if self.dead.get(agent.id, False):
                continue
            if areas.get(agent.id, 0) > 0:
                continue
            state = self.states[agent.id]

            # 부활 자격: RevenantAgent + 아직 부활 안 씀
            if (self.is_revenant_class.get(agent.id, False)
                    and not self.revive_used.get(agent.id, False)):
                if self._revive_agent(agent.id):
                    self.revive_used[agent.id] = True
                    continue   # 부활 성공 → 사망 처리 건너뜀

            # 일반 사망
            self.dead[agent.id] = True
            self.death_positions[agent.id] = (state.x, state.y)
            self.death_ticks[agent.id] = self.tick
            # Inferno 폭사: 사망 시 주변 7×7 모두 빈 칸으로 (벽 제외, DEF 50% 저항)
            if self.is_inferno.get(agent.id, False):
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        nx, ny = state.x + dx, state.y + dy
                        if not self.grid.in_bounds(nx, ny):
                            continue
                        cell = int(self.grid.cells[ny, nx])
                        if cell == WALL:
                            continue
                        if self.def_resists(cell):
                            continue
                        self.grid.cells[ny, nx] = 0
                self.effect_flashes.append(EffectFlash(
                    x=state.x, y=state.y, text='INFERNO!',
                    color=(255, 100, 30),
                    ticks_left=ELIMINATION_FLASH_DURATION,
                    initial_ticks=ELIMINATION_FLASH_DURATION,
                ))
            else:
                self.effect_flashes.append(EffectFlash(
                    x=state.x, y=state.y,
                    text=f'X {state.name}',
                    color=(220, 60, 60),
                    ticks_left=ELIMINATION_FLASH_DURATION,
                    initial_ticks=ELIMINATION_FLASH_DURATION,
                ))

    def _revive_agent(self, agent_id: int) -> bool:
        """RevenantAgent 즉시 부활 — 무작위 빈 칸으로 텔레포트 + 3×3 강제 점령.

        부활 위치를 중심으로 3×3 셀을 무조건 자기 색으로 변환 (적 영토, 팀 영토,
        벽 모두 덮어씀). 능력/속도 보너스 없음. 빈 칸 없으면 부활 실패.
        벽이 덮인 만큼 grid._wall_count 도 갱신해서 총 셀 수 일관성 유지.
        반환: 부활 성공 여부.
        """
        cells = self.grid.cells
        ys, xs = (cells == 0).nonzero()
        if len(xs) == 0:
            return False
        i = self.rng.randint(0, len(xs) - 1)
        x, y = int(xs[i]), int(ys[i])
        state = self.states[agent_id]
        state.x, state.y = x, y

        # 3×3 강제 점령 — 격자 밖은 무시. 벽도 자기 색으로 변환.
        walls_overwritten = 0
        h, w = cells.shape
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if int(cells[ny, nx]) == WALL:
                        walls_overwritten += 1
                    cells[ny, nx] = agent_id
        if walls_overwritten:
            self.grid._wall_count -= walls_overwritten

        self.freeze_ticks[agent_id] = 0
        self.effect_flashes.append(EffectFlash(
            x=x, y=y, text='REVIVE!',
            color=(255, 80, 90),
            ticks_left=ELIMINATION_FLASH_DURATION,
            initial_ticks=ELIMINATION_FLASH_DURATION,
        ))
        return True

    def _do_actions_phase(self, only_ids: Optional[set] = None):
        """행동 결정 + 이동 1회. only_ids 지정 시 그 에이전트만."""
        ctx = GameContext(
            agent_states={**self.states},
            areas=self.grid.get_areas(len(self.agents)),
            grid=self.grid,
            rng=self.rng,
            items=list(self.items),
            effect_flashes=self.effect_flashes,
            tick=self.tick,
            teams=self.teams,
            team_mode=self.team_mode,
            sticky_cells=self.sticky_cells,
            freeze_ticks=self.freeze_ticks,
            dead=self.dead,
            wildcard_pending=self.wildcard_pending,
            role_by_id=self.role_by_id,
        )

        actions = {}
        for agent in self.agents:
            if only_ids is not None and agent.id not in only_ids:
                continue
            if self.dead.get(agent.id, False):
                continue
            if self.freeze_ticks.get(agent.id, 0) > 0:
                actions[agent.id] = None
                continue
            # SLOW — 매 phase 50% 확률로 skip → 평균 x0.5 속도
            if (self.slow_ticks.get(agent.id, 0) > 0
                    and self.rng.random() < 0.5):
                actions[agent.id] = None
                continue
            # 카테고리 속도 modifier — DEF 는 15% 확률 skip (느림)
            role = self.role_by_id.get(agent.id, '')
            speed = ROLE_SPEED.get(role, 1.0)
            if speed < 1.0 and self.rng.random() < (1.0 - speed):
                actions[agent.id] = None
                continue
            actions[agent.id] = agent.act(self.states[agent.id], ctx)

        for agent in self.agents:
            if agent.id not in actions:
                continue
            if self.dead.get(agent.id, False):
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
        if cell == WALL:
            return False
        if cell == 0 or cell == agent_id:
            return True
        # 팀모드: 같은 팀 영토 통과 허용 (점령은 안 함 — 그저 통행)
        if (self.team_mode and cell > 0
                and self.teams.get(cell) == self.teams.get(agent_id)):
            return True
        # Betrayer 특권: 옛 팀 / 적 팀 영토 모두 자유 통과 (점령은 안 함, 빈 칸만)
        if self.is_betrayer.get(agent_id, False):
            return True
        return False

    def _is_team_ground(self, cell: int, agent_id: int) -> bool:
        if cell <= 0:
            return False
        if cell == agent_id:
            return True
        if (self.team_mode
                and self.teams.get(cell) == self.teams.get(agent_id)):
            return True
        # Betrayer 특권 — 모든 영토를 "통행 가능" 으로 인정 (fallback 경로용)
        if self.is_betrayer.get(agent_id, False):
            return True
        return False

    def _find_alternative(self, x: int, y: int, agent_id: int):
        """빈 칸 우선, 자기/팀 영토 차선. (nx, ny, action) 반환 or None"""
        order = list(range(4))
        self.rng.shuffle(order)
        # 1순위: 빈 칸
        for a in order:
            ddx, ddy = DIRECTIONS[a]
            ex, ey = x + ddx, y + ddy
            if self.grid.in_bounds(ex, ey) and int(self.grid.cells[ey, ex]) == 0:
                return ex, ey, a
        # 2순위: 자기 영토 또는 같은 팀 영토 (통행)
        for a in order:
            ddx, ddy = DIRECTIONS[a]
            ex, ey = x + ddx, y + ddy
            if not self.grid.in_bounds(ex, ey):
                continue
            if self._is_team_ground(int(self.grid.cells[ey, ex]), agent_id):
                return ex, ey, a
        return None

    def get_areas(self) -> dict:
        return self.grid.get_areas(len(self.agents))

    def get_percentages(self) -> dict:
        areas = self.get_areas()
        total = self.grid.total_cells()
        return {aid: 100.0 * a / total for aid, a in areas.items()}
