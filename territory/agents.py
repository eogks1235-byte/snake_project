"""규칙 기반 에이전트 — 캐릭터별 점수 함수로 차별화"""
import random
from collections import deque
from dataclasses import dataclass

from .grid import WALL

# 방향: 0=상, 1=우, 2=하, 3=좌
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


@dataclass
class AgentState:
    agent_id: int
    x: int
    y: int
    last_action: int = 0
    color: tuple = (255, 255, 255)
    name: str = ''


# ──────────────────────────────────────────────────────────────
# 공통 탐색 유틸
# ──────────────────────────────────────────────────────────────

def find_step_to_nearest_empty(grid, start_x: int, start_y: int, agent_id: int):
    """BFS — 자기 영토 통과해 가장 가까운 빈 칸으로 가는 첫 걸음 방향."""
    queue = deque()
    queue.append((start_x, start_y, None))
    visited = {(start_x, start_y)}

    while queue:
        x, y, first_dir = queue.popleft()
        for d, (dx, dy) in enumerate(DIRECTIONS):
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited or not grid.in_bounds(nx, ny):
                continue
            cell = int(grid.cells[ny, nx])
            if cell == 0:
                return first_dir if first_dir is not None else d
            if cell == agent_id:
                visited.add((nx, ny))
                queue.append((nx, ny, first_dir if first_dir is not None else d))
    return None


def adjacent_owners(grid, x: int, y: int) -> set:
    """(x,y)의 4방향 이웃 셀 owner 집합 (빈 칸 0 포함)."""
    owners = set()
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if grid.in_bounds(nx, ny):
            owners.add(int(grid.cells[ny, nx]))
    return owners


def pick_direction(grid, state: AgentState, rng: random.Random, score_fn):
    """인접 빈 칸 중 score_fn 최댓값 방향. 없으면 BFS로 자기 영토 통과.

    score_fn(d, nx, ny, state) -> int (높을수록 선호)
    도달 가능한 빈 칸이 전혀 없으면 None을 반환 → simulation이 정지로 해석.
    """
    candidates = []
    for d, (dx, dy) in enumerate(DIRECTIONS):
        nx, ny = state.x + dx, state.y + dy
        if not grid.in_bounds(nx, ny):
            continue
        cell = int(grid.cells[ny, nx])
        if cell != 0:
            continue
        score = score_fn(d, nx, ny, state)
        candidates.append((score, d))

    if candidates:
        max_score = max(c[0] for c in candidates)
        best = [d for s, d in candidates if s == max_score]
        return rng.choice(best)

    bfs = find_step_to_nearest_empty(grid, state.x, state.y, state.agent_id)
    if bfs is not None:
        return bfs

    return None  # 도달 가능한 빈 칸 없음 → 정지


# ──────────────────────────────────────────────────────────────
# 에이전트
# ──────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, agent_id: int, name: str, color: tuple, description: str = ''):
        self.id = agent_id
        self.name = name
        self.color = color
        self.description = description

    def act(self, state: AgentState, context) -> int:
        raise NotImplementedError


class TrollAgent(Agent):
    """2등 영토 경계의 빈 칸을 우선 점령 (2등 침공)."""

    def __init__(self, agent_id: int, color: tuple = (255, 122, 69)):
        super().__init__(agent_id, 'Troll', color, 'Hunts 2nd place')

    def act(self, state, context):
        target = context.get_rank_state(2, exclude_id=self.id)
        target_id = target.agent_id if target else None

        def score(d, nx, ny, _s):
            s = 1
            if target_id is not None and target_id in adjacent_owners(context.grid, nx, ny):
                s += 10
            return s

        return pick_direction(context.grid, state, context.rng, score)


class MirrorAgent(Agent):
    """1등 영토 경계의 빈 칸을 우선 점령 (1등 침공)."""

    def __init__(self, agent_id: int, color: tuple = (185, 103, 255)):
        super().__init__(agent_id, 'Mirror', color, 'Chases the leader')

    def act(self, state, context):
        leader = context.get_rank_state(1, exclude_id=self.id)
        leader_id = leader.agent_id if leader else None

        def score(d, nx, ny, _s):
            s = 1
            if leader_id is not None and leader_id in adjacent_owners(context.grid, nx, ny):
                s += 10
            return s

        return pick_direction(context.grid, state, context.rng, score)


class BoomerangAgent(Agent):
    """시계방향 관성 — 같은 방향 유지 > 시계방향 회전."""

    def __init__(self, agent_id: int, color: tuple = (74, 222, 128)):
        super().__init__(agent_id, 'Boomerang', color, 'Spirals clockwise')

    def act(self, state, context):
        last = state.last_action
        clockwise = (last + 1) % 4
        counter_cw = (last + 3) % 4
        opposite = (last + 2) % 4
        priority = {last: 4, clockwise: 3, counter_cw: 2, opposite: 1}

        def score(d, nx, ny, _s):
            return priority.get(d, 0)

        return pick_direction(context.grid, state, context.rng, score)


class AntiMimicAgent(Agent):
    """꼴등 영토 경계의 빈 칸을 우선 점령 (약자 봉쇄)."""

    def __init__(self, agent_id: int, color: tuple = (56, 189, 248)):
        super().__init__(agent_id, 'Anti-Mimic', color, 'Targets last place')

    def act(self, state, context):
        last_place = context.get_last_state(exclude_id=self.id)
        target_id = last_place.agent_id if last_place else None

        def score(d, nx, ny, _s):
            s = 1
            if target_id is not None and target_id in adjacent_owners(context.grid, nx, ny):
                s += 10
            return s

        return pick_direction(context.grid, state, context.rng, score)


# ──────────────────────────────────────────────────────────────
# 추가 에이전트 (변수성 풀)
# ──────────────────────────────────────────────────────────────

class RandomAgent(Agent):
    """무작위 행동 — 베이스라인."""

    def __init__(self, agent_id: int, color: tuple = (148, 163, 184)):
        super().__init__(agent_id, 'Random', color, 'Pure chaos')

    def act(self, state, context):
        return pick_direction(context.grid, state, context.rng, lambda *a: 1)


class GreedyAgent(Agent):
    """가장 많은 빈 칸을 가진 방향 — 단순 영토 욕심쟁이."""

    def __init__(self, agent_id: int, color: tuple = (251, 191, 36)):
        super().__init__(agent_id, 'Greedy', color, 'Most open space')

    def act(self, state, context):
        def score(d, nx, ny, _s):
            count = 0
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if context.grid.in_bounds(ax, ay) and int(context.grid.cells[ay, ax]) == 0:
                    count += 1
            return count

        return pick_direction(context.grid, state, context.rng, score)


class DefenderAgent(Agent):
    """자기 영토와 인접한 빈 칸 우선 — 단단한 덩어리 형성."""

    def __init__(self, agent_id: int, color: tuple = (244, 114, 182)):
        super().__init__(agent_id, 'Defender', color, 'Builds a fortress')

    def act(self, state, context):
        def score(d, nx, ny, _s):
            count = 0
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if (context.grid.in_bounds(ax, ay)
                        and int(context.grid.cells[ay, ax]) == self.id):
                    count += 1
            return count

        return pick_direction(context.grid, state, context.rng, score)


class AggressorAgent(Agent):
    """아무 적 영토와 인접한 빈 칸 우선 — 무차별 침공."""

    def __init__(self, agent_id: int, color: tuple = (239, 68, 68)):
        super().__init__(agent_id, 'Aggressor', color, 'Picks any fight')

    def act(self, state, context):
        def score(d, nx, ny, _s):
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if not context.grid.in_bounds(ax, ay):
                    continue
                cell = int(context.grid.cells[ay, ax])
                if cell != 0 and cell != self.id:
                    return 11
            return 1

        return pick_direction(context.grid, state, context.rng, score)


class HermitAgent(Agent):
    """모든 적 영토와 멀리 떨어진 빈 칸 우선 — 회피주의."""

    def __init__(self, agent_id: int, color: tuple = (45, 212, 191)):
        super().__init__(agent_id, 'Hermit', color, 'Avoids all rivals')

    def act(self, state, context):
        def score(d, nx, ny, _s):
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if not context.grid.in_bounds(ax, ay):
                    continue
                cell = int(context.grid.cells[ay, ax])
                if cell != 0 and cell != self.id:
                    return 0
            return 5

        return pick_direction(context.grid, state, context.rng, score)


class ScoutAgent(Agent):
    """벽 가장자리 인접 빈 칸 우선 — 격자 둘레를 먼저 차지."""

    def __init__(self, agent_id: int, color: tuple = (250, 204, 21)):
        super().__init__(agent_id, 'Scout', color, 'Hugs the walls')

    def act(self, state, context):
        w, h = context.grid.width, context.grid.height

        def score(d, nx, ny, _s):
            s = 1
            if nx == 0 or nx == w - 1 or ny == 0 or ny == h - 1:
                s += 8
            return s

        return pick_direction(context.grid, state, context.rng, score)


class PioneerAgent(Agent):
    """모든 적 헤드와 가장 멀리 — Hermit의 헤드 거리 버전."""

    def __init__(self, agent_id: int, color: tuple = (147, 197, 253)):
        super().__init__(agent_id, 'Pioneer', color, 'Walks the frontier')

    def act(self, state, context):
        rivals = [a for a in context.agent_states.values() if a.agent_id != self.id]

        def score(d, nx, ny, _s):
            if not rivals:
                return 1
            min_d = min(abs(a.x - nx) + abs(a.y - ny) for a in rivals)
            return min_d

        return pick_direction(context.grid, state, context.rng, score)


class SpiralAgent(Agent):
    """시작점 중심 나선형 확장 — 반지름이 tick에 따라 점점 커짐."""

    def __init__(self, agent_id: int, color: tuple = (192, 132, 252)):
        super().__init__(agent_id, 'Spiral', color, 'Expands outward')
        self.start = None
        self.tick = 0

    def act(self, state, context):
        if self.start is None:
            self.start = (state.x, state.y)
        self.tick += 1
        target = (self.tick // 5) + 1

        def score(d, nx, ny, _s):
            r = abs(nx - self.start[0]) + abs(ny - self.start[1])
            return -abs(r - target)  # 목표 반지름에 가까울수록 우선

        return pick_direction(context.grid, state, context.rng, score)


class CompassAgent(Agent):
    """매 인스턴스마다 정해진 한 방향으로 직진."""

    def __init__(self, agent_id: int, color: tuple = (250, 250, 250)):
        super().__init__(agent_id, 'Compass', color, 'Marches one way')
        self.preferred = None

    def act(self, state, context):
        if self.preferred is None:
            self.preferred = context.rng.randint(0, 3)
        opp = (self.preferred + 2) % 4

        def score(d, nx, ny, _s):
            if d == self.preferred:
                return 5
            if d == opp:
                return 0
            return 2

        return pick_direction(context.grid, state, context.rng, score)


class ChargerAgent(Agent):
    """가장 가까운 적 헤드를 추적 — 영토 무관하게 머리만 노림."""

    def __init__(self, agent_id: int, color: tuple = (244, 63, 94)):
        super().__init__(agent_id, 'Charger', color, 'Rams the closest head')

    def act(self, state, context):
        rivals = [a for a in context.agent_states.values() if a.agent_id != self.id]
        if not rivals:
            return pick_direction(context.grid, state, context.rng, lambda *a: 1)
        target = min(rivals, key=lambda a: abs(a.x - state.x) + abs(a.y - state.y))

        def score(d, nx, ny, _s):
            return -(abs(target.x - nx) + abs(target.y - ny))

        return pick_direction(context.grid, state, context.rng, score)


class PatriotAgent(Agent):
    """시작점(home) 가까이의 빈 칸 우선 — 베이스 캠프 컨셉."""

    def __init__(self, agent_id: int, color: tuple = (251, 146, 60)):
        super().__init__(agent_id, 'Patriot', color, 'Stays close to home')
        self.home = None

    def act(self, state, context):
        if self.home is None:
            self.home = (state.x, state.y)

        def score(d, nx, ny, _s):
            return -(abs(nx - self.home[0]) + abs(ny - self.home[1]))

        return pick_direction(context.grid, state, context.rng, score)


# ──────────────────────────────────────────────────────────────
# 아이템 인식 에이전트
# ──────────────────────────────────────────────────────────────

class HoarderAgent(Agent):
    """가장 가까운 아이템(맨해튼 거리)으로 직진."""

    def __init__(self, agent_id: int, color: tuple = (255, 215, 90)):
        super().__init__(agent_id, 'Hoarder', color, 'Bee-lines for items')

    def act(self, state, context):
        items = context.items
        if not items:
            return pick_direction(context.grid, state, context.rng, lambda *a: 1)
        target = min(items, key=lambda it: abs(it.x - state.x) + abs(it.y - state.y))

        def score(d, nx, ny, _s):
            return -(abs(target.x - nx) + abs(target.y - ny))

        return pick_direction(context.grid, state, context.rng, score)


class SaboteurAgent(Agent):
    """우리가 가장 유리한 아이템(상대보다 가까운)으로 가서 가로채기."""

    def __init__(self, agent_id: int, color: tuple = (168, 85, 247)):
        super().__init__(agent_id, 'Saboteur', color, "Steals enemy items")

    def act(self, state, context):
        items = context.items
        rivals = [a for a in context.agent_states.values() if a.agent_id != self.id]
        if not items:
            return pick_direction(context.grid, state, context.rng, lambda *a: 1)

        best_item = items[0]
        best_margin = None
        for it in items:
            our_d = abs(it.x - state.x) + abs(it.y - state.y)
            their_d = (min(abs(r.x - it.x) + abs(r.y - it.y) for r in rivals)
                       if rivals else 9999)
            margin = their_d - our_d  # 양수일수록 우리가 유리
            if best_margin is None or margin > best_margin:
                best_margin = margin
                best_item = it

        target = best_item

        def score(d, nx, ny, _s):
            return -(abs(target.x - nx) + abs(target.y - ny))

        return pick_direction(context.grid, state, context.rng, score)


# ──────────────────────────────────────────────────────────────
# 맵 인식 에이전트
# ──────────────────────────────────────────────────────────────

class CartographerAgent(Agent):
    """벽 인접 + 통로 (인접 빈 칸 ≤ 2) 우선 — 좁은 길목 점거."""

    def __init__(self, agent_id: int, color: tuple = (217, 119, 6)):
        super().__init__(agent_id, 'Cartographer', color, 'Camps choke points')

    def act(self, state, context):
        grid = context.grid

        def score(d, nx, ny, _s):
            wall_count = 0
            empty_count = 0
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if not grid.in_bounds(ax, ay):
                    wall_count += 1  # 격자 밖도 벽 취급
                    continue
                cell = int(grid.cells[ay, ax])
                if cell == WALL:
                    wall_count += 1
                elif cell == 0:
                    empty_count += 1
            s = wall_count * 3
            if empty_count <= 2:
                s += 5  # 좁은 통로 보너스
            return s

        return pick_direction(context.grid, state, context.rng, score)


# ──────────────────────────────────────────────────────────────
# 상태 머신 에이전트
# ──────────────────────────────────────────────────────────────

class PhoenixAgent(Agent):
    """영토 < 10%면 광폭(Aggressor), > 30%면 농성(Defender), 사이는 탐욕(Greedy)."""

    def __init__(self, agent_id: int, color: tuple = (244, 63, 94)):
        super().__init__(agent_id, 'Phoenix', color, 'Berserk → calm by area')

    def act(self, state, context):
        my_area = context.areas.get(self.id, 0)
        total = context.grid.total_cells()
        pct = my_area / total if total else 0
        grid = context.grid

        if pct < 0.10:
            def score(d, nx, ny, _s):
                for ddx, ddy in DIRECTIONS:
                    ax, ay = nx + ddx, ny + ddy
                    if not grid.in_bounds(ax, ay):
                        continue
                    cell = int(grid.cells[ay, ax])
                    if cell != 0 and cell != self.id and cell != WALL:
                        return 11
                return 1
        elif pct > 0.30:
            def score(d, nx, ny, _s):
                count = 0
                for ddx, ddy in DIRECTIONS:
                    ax, ay = nx + ddx, ny + ddy
                    if (grid.in_bounds(ax, ay)
                            and int(grid.cells[ay, ax]) == self.id):
                        count += 1
                return count
        else:
            def score(d, nx, ny, _s):
                count = 0
                for ddx, ddy in DIRECTIONS:
                    ax, ay = nx + ddx, ny + ddy
                    if (grid.in_bounds(ax, ay)
                            and int(grid.cells[ay, ax]) == 0):
                        count += 1
                return count

        return pick_direction(context.grid, state, context.rng, score)


class SleeperAgent(Agent):
    """첫 SLEEP_TICKS는 정지, 이후 가장 가까운 적 헤드 추적 (Charger)."""

    SLEEP_TICKS = 60

    def __init__(self, agent_id: int, color: tuple = (99, 102, 241)):
        super().__init__(agent_id, 'Sleeper', color, 'Naps then attacks')
        self.elapsed = 0

    def act(self, state, context):
        self.elapsed += 1
        if self.elapsed <= self.SLEEP_TICKS:
            return None  # zzz — 시뮬레이션이 정지로 처리

        rivals = [a for a in context.agent_states.values() if a.agent_id != self.id]
        if not rivals:
            return pick_direction(context.grid, state, context.rng, lambda *a: 1)
        target = min(rivals, key=lambda a: abs(a.x - state.x) + abs(a.y - state.y))

        def score(d, nx, ny, _s):
            return -(abs(target.x - nx) + abs(target.y - ny))

        return pick_direction(context.grid, state, context.rng, score)


# ──────────────────────────────────────────────────────────────
# 행동 모방 에이전트
# ──────────────────────────────────────────────────────────────

class MimicAgent(Agent):
    """매 REFRESH tick마다 1등의 last_action을 복사 — 행동 모방."""

    REFRESH = 5

    def __init__(self, agent_id: int, color: tuple = (45, 212, 191)):
        super().__init__(agent_id, 'Mimic', color, 'Copies leader actions')
        self.preferred = None
        self.tick = 0

    def act(self, state, context):
        self.tick += 1
        if self.preferred is None or self.tick % self.REFRESH == 0:
            leader = context.get_rank_state(1, exclude_id=self.id)
            if leader is not None:
                self.preferred = leader.last_action
        opp = (self.preferred + 2) % 4 if self.preferred is not None else None

        def score(d, nx, ny, _s):
            if d == self.preferred:
                return 5
            if d == opp:
                return 0
            return 2

        return pick_direction(context.grid, state, context.rng, score)


# 매 판 랜덤 픽용 풀 (21종)
ALGORITHM_POOL = [
    TrollAgent, MirrorAgent, BoomerangAgent, AntiMimicAgent,
    RandomAgent, GreedyAgent, DefenderAgent, AggressorAgent, HermitAgent,
    ScoutAgent, PioneerAgent, SpiralAgent, CompassAgent, ChargerAgent, PatriotAgent,
    HoarderAgent, SaboteurAgent, CartographerAgent,
    PhoenixAgent, SleeperAgent, MimicAgent,
]
