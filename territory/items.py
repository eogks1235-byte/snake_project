"""아이템 시스템 — 효과 풀 15종에서 매 판 4개 활성화."""
from dataclasses import dataclass

from .grid import WALL

DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


# 효과 카탈로그 (key, label, color, sign)
ALL_EFFECTS = [
    ('bomb',       'BOMB!',       (251, 191, 36),  '+'),
    ('teleport',   'TELEPORT!',   (56, 189, 248),  '+'),
    ('steal',      'STEAL!',      (74, 222, 128),  '+'),
    ('storm',      'STORM!',      (147, 197, 253), '+'),
    ('rage',       'RAGE!',       (239, 68, 68),   '+'),
    ('earthquake', 'EARTHQUAKE!', (180, 83, 9),    '+'),
    ('curse',      'CURSED!',     (124, 58, 237),  '+'),
    ('echo',       'ECHO!',       (52, 211, 153),  '+'),
    ('pulse',      'PULSE!',      (255, 255, 255), '+'),
    ('erase',      'ERASE!',      (244, 114, 182), '-'),
    ('freeze',     'FROZEN!',     (148, 163, 184), '-'),
    ('shrink',     'SHRINK!',     (236, 72, 153),  '-'),
    ('scatter',    'SCATTER!',    (251, 113, 133), '-'),
    ('blackhole',  'BLACKHOLE!',  (15, 15, 15),    '-'),
    ('swap',       'SWAP!',       (250, 204, 21),  '?'),
    ('sticky',     'STICKY!',     (170, 100, 180), '?'),
    ('haste',      'HASTE!',      (250, 220, 50),  '+'),
]
EFFECT_META = {key: (label, color, sign) for key, label, color, sign in ALL_EFFECTS}


@dataclass
class Item:
    x: int
    y: int


@dataclass
class EffectFlash:
    x: int
    y: int
    text: str
    color: tuple
    ticks_left: int
    initial_ticks: int


def _sample_indices(rng, length: int, n: int):
    n = min(n, length)
    return rng.sample(range(length), n) if n > 0 else []


def _same_team(sim, a_id: int, b_id: int) -> bool:
    """두 에이전트가 같은 팀인지. free-for-all에선 본인일 때만 True."""
    return sim.teams[a_id] == sim.teams[b_id]


def _team_members(sim, agent_id: int) -> list:
    """같은 팀의 살아있는 에이전트 id 리스트 (자신 포함)."""
    return [a.id for a in sim.agents
            if sim.teams[a.id] == sim.teams[agent_id]
            and not sim.dead.get(a.id, False)]


def apply_effect(sim, agent_id: int, effect_key: str):
    """효과 적용. (label, color) 반환."""
    state = sim.states[agent_id]
    grid = sim.grid
    rng = sim.rng
    head = (state.x, state.y)

    if effect_key == 'bomb':
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = state.x + dx, state.y + dy
                if not grid.in_bounds(nx, ny):
                    continue
                cell = int(grid.cells[ny, nx])
                if cell == WALL:
                    continue
                # 팀원 영토는 보호 (자기 자신은 아래 통일된 색칠로 처리됨)
                if cell != 0 and cell != agent_id and _same_team(sim, agent_id, cell):
                    continue
                # DEF 50% 저항
                if sim.def_resists(cell):
                    continue
                grid.cells[ny, nx] = agent_id

    elif effect_key == 'teleport':
        ys, xs = (grid.cells == 0).nonzero()
        if len(xs) > 0:
            i = rng.randint(0, len(xs) - 1)
            state.x, state.y = int(xs[i]), int(ys[i])
            grid.cells[state.y, state.x] = agent_id

    elif effect_key == 'steal':
        areas = grid.get_areas(len(sim.agents))
        # 팀모드: 다른 팀에서만 훔침. free-for-all: 자기 외 누구든.
        rivals = {aid: a for aid, a in areas.items()
                  if not _same_team(sim, agent_id, aid) and a > 0}
        if rivals:
            biggest = max(rivals, key=rivals.get)
            ys, xs = (grid.cells == biggest).nonzero()
            for i in _sample_indices(rng, len(xs), max(1, int(len(xs) * 0.05))):
                grid.cells[ys[i], xs[i]] = agent_id

    elif effect_key == 'storm':
        # 버프: 빈 칸 50개를 팀원들에게 라운드로빈 분담 (자유전 = 발동자만)
        ys, xs = (grid.cells == 0).nonzero()
        if len(xs) > 0:
            team = _team_members(sim, agent_id) or [agent_id]
            idxs = _sample_indices(rng, len(xs), min(50, len(xs)))
            for k, i in enumerate(idxs):
                owner = team[k % len(team)]
                grid.cells[ys[i], xs[i]] = owner

    elif effect_key == 'rage':
        # 버프: 팀 전원에게 적용
        for tid in _team_members(sim, agent_id):
            sim.rage_ticks[tid] = 80

    elif effect_key == 'earthquake':
        # 공격: 다른 팀의 영토만 5% 삭제
        for a in sim.agents:
            if _same_team(sim, agent_id, a.id):
                continue
            ys, xs = (grid.cells == a.id).nonzero()
            if len(xs) <= 1:
                continue
            other_head = (sim.states[a.id].x, sim.states[a.id].y)
            n = max(1, int(len(xs) * 0.05))
            for i in _sample_indices(rng, len(xs), n):
                if (int(xs[i]), int(ys[i])) == other_head:
                    continue
                grid.cells[ys[i], xs[i]] = 0

    elif effect_key == 'curse':
        # 공격: 다른 팀 전원 freeze
        for a in sim.agents:
            if _same_team(sim, agent_id, a.id):
                continue
            sim.freeze_ticks[a.id] = sim.freeze_ticks.get(a.id, 0) + 25

    elif effect_key == 'echo':
        # 버프: 팀 전원
        for tid in _team_members(sim, agent_id):
            sim.echo_ticks[tid] = 60

    elif effect_key == 'pulse':
        # 버프: 팀 전원의 영토 인접 빈 칸을 각자 색으로 점령
        team = _team_members(sim, agent_id) or [agent_id]
        claimed: dict = {}  # (x, y) -> owner (먼저 도달한 팀원이 가져감)
        for owner in team:
            own = (grid.cells == owner)
            ys, xs = own.nonzero()
            for x, y in zip(xs, ys):
                for dx, dy in DIRECTIONS:
                    nx, ny = int(x) + dx, int(y) + dy
                    if (grid.in_bounds(nx, ny)
                            and int(grid.cells[ny, nx]) == 0
                            and (nx, ny) not in claimed):
                        claimed[(nx, ny)] = owner
        for (nx, ny), owner in claimed.items():
            grid.cells[ny, nx] = owner

    elif effect_key == 'erase':
        ys, xs = (grid.cells == agent_id).nonzero()
        if len(xs) > 1:
            n = max(1, int(len(xs) * 0.08))
            for i in _sample_indices(rng, len(xs), n):
                if (int(xs[i]), int(ys[i])) == head:
                    continue
                grid.cells[ys[i], xs[i]] = 0

    elif effect_key == 'freeze':
        sim.freeze_ticks[agent_id] = sim.freeze_ticks.get(agent_id, 0) + 50

    elif effect_key == 'shrink':
        ys, xs = (grid.cells == agent_id).nonzero()
        if len(xs) > 1:
            n = int(len(xs) * 0.30)
            for i in _sample_indices(rng, len(xs), n):
                if (int(xs[i]), int(ys[i])) == head:
                    continue
                grid.cells[ys[i], xs[i]] = 0

    elif effect_key == 'scatter':
        ys, xs = (grid.cells == agent_id).nonzero()
        if len(xs) > 5:
            n = min(10, len(xs) - 1)
            removed = []
            for i in _sample_indices(rng, len(xs), n):
                if (int(xs[i]), int(ys[i])) == head:
                    continue
                grid.cells[ys[i], xs[i]] = 0
                removed.append(i)
            ys2, xs2 = (grid.cells == 0).nonzero()
            if len(xs2) > 0:
                for i in _sample_indices(rng, len(xs2), len(removed)):
                    grid.cells[ys2[i], xs2[i]] = agent_id

    elif effect_key == 'blackhole':
        # 공격: 주변 7×7을 빈 칸으로 (팀원 영토 보호, DEF 50% 저항)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx, ny = state.x + dx, state.y + dy
                if not grid.in_bounds(nx, ny) or (nx, ny) == head:
                    continue
                cell = int(grid.cells[ny, nx])
                if cell == WALL:
                    continue
                if cell != 0 and _same_team(sim, agent_id, cell):
                    continue
                if sim.def_resists(cell):
                    continue
                grid.cells[ny, nx] = 0

    elif effect_key == 'haste':
        # 버프: 팀 전원에게 x2.5 — 150틱 (= 5초 @ 30fps).
        # 팀 영토 보너스(x1.5)는 위치 기반이라 별개로 stack 가능.
        for tid in _team_members(sim, agent_id):
            sim.haste_ticks[tid] = max(sim.haste_ticks.get(tid, 0), 150)

    elif effect_key == 'sticky':
        # 공격: 빈 칸의 1%를 끈적이 함정. 함정마다 owner team 기록.
        # 밟으면 x0.5 속도 (slow) — 멈춤 아님. 같은 팀이 깐 건 팀모드에서 면제.
        ys, xs = (grid.cells == 0).nonzero()
        if len(xs) > 0:
            owner_team = sim.teams[agent_id]
            n = max(1, int(len(xs) * 0.01))
            for i in _sample_indices(rng, len(xs), n):
                sim.sticky_cells[(int(xs[i]), int(ys[i]))] = owner_team

    elif effect_key == 'swap':
        # 공격: 다른 팀 에이전트와 영토/위치 교환
        rivals = [a.id for a in sim.agents
                  if not _same_team(sim, agent_id, a.id)]
        if rivals:
            other_id = rng.choice(rivals)
            my_mask = grid.cells == agent_id
            their_mask = grid.cells == other_id
            grid.cells[my_mask] = other_id
            grid.cells[their_mask] = agent_id
            my_state = sim.states[agent_id]
            their_state = sim.states[other_id]
            my_state.x, their_state.x = their_state.x, my_state.x
            my_state.y, their_state.y = their_state.y, my_state.y
            grid.cells[my_state.y, my_state.x] = agent_id
            grid.cells[their_state.y, their_state.x] = other_id

    label, color, _sign = EFFECT_META[effect_key]
    return label, color
