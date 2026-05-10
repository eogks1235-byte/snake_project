"""아이템 시스템 — 효과 풀 15종에서 매 판 4개 활성화."""
from dataclasses import dataclass

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
                if grid.in_bounds(nx, ny):
                    grid.cells[ny, nx] = agent_id

    elif effect_key == 'teleport':
        ys, xs = (grid.cells == 0).nonzero()
        if len(xs) > 0:
            i = rng.randint(0, len(xs) - 1)
            state.x, state.y = int(xs[i]), int(ys[i])
            grid.cells[state.y, state.x] = agent_id

    elif effect_key == 'steal':
        areas = grid.get_areas(len(sim.agents))
        rivals = {aid: a for aid, a in areas.items() if aid != agent_id and a > 0}
        if rivals:
            biggest = max(rivals, key=rivals.get)
            ys, xs = (grid.cells == biggest).nonzero()
            for i in _sample_indices(rng, len(xs), max(1, int(len(xs) * 0.05))):
                grid.cells[ys[i], xs[i]] = agent_id

    elif effect_key == 'storm':
        ys, xs = (grid.cells == 0).nonzero()
        if len(xs) > 0:
            for i in _sample_indices(rng, len(xs), 50):
                grid.cells[ys[i], xs[i]] = agent_id

    elif effect_key == 'rage':
        sim.rage_ticks[agent_id] = 80

    elif effect_key == 'earthquake':
        for a in sim.agents:
            if a.id == agent_id:
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
        for a in sim.agents:
            if a.id != agent_id:
                sim.freeze_ticks[a.id] = sim.freeze_ticks.get(a.id, 0) + 25

    elif effect_key == 'echo':
        sim.echo_ticks[agent_id] = 60

    elif effect_key == 'pulse':
        own = (grid.cells == agent_id)
        ys, xs = own.nonzero()
        converted = set()
        for x, y in zip(xs, ys):
            for dx, dy in DIRECTIONS:
                nx, ny = int(x) + dx, int(y) + dy
                if grid.in_bounds(nx, ny) and int(grid.cells[ny, nx]) == 0:
                    converted.add((nx, ny))
        for nx, ny in converted:
            grid.cells[ny, nx] = agent_id

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
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx, ny = state.x + dx, state.y + dy
                if grid.in_bounds(nx, ny) and (nx, ny) != head:
                    grid.cells[ny, nx] = 0

    elif effect_key == 'swap':
        rivals = [a.id for a in sim.agents if a.id != agent_id]
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
