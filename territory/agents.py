"""규칙 기반 에이전트 — 캐릭터별 점수 함수로 차별화"""
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

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

def find_step_to_nearest_empty(grid, start_x: int, start_y: int, agent_id: int,
                                friend_ids: set = None):
    """BFS — 자기/팀 영토 통과해 가장 가까운 빈 칸으로 가는 첫 걸음 방향.

    friend_ids: 통과 허용 셀 id 집합. None 이면 {agent_id} 만.
    """
    friends = friend_ids if friend_ids is not None else {agent_id}
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
            if cell in friends:
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


def _friends(context, agent_id: int):
    """팀모드 friend_ids 자동 계산 — BFS 가 팀 영토 통과해 빈 칸으로 가게."""
    teams = getattr(context, 'teams', None)
    if not teams:
        return None
    my_team = teams.get(agent_id)
    if my_team is None:
        return None
    return {aid for aid, t in teams.items() if t == my_team}


def pick_direction(grid, state: AgentState, rng: random.Random, score_fn,
                    friend_ids: set = None):
    """인접 빈 칸 중 score_fn 최댓값 방향. 없으면 BFS로 자기/팀 영토 통과.

    score_fn(d, nx, ny, state) -> int (높을수록 선호)
    friend_ids: BFS 통과 허용 셀. None 이면 {agent_id} 만.
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

    bfs = find_step_to_nearest_empty(grid, state.x, state.y, state.agent_id,
                                       friend_ids=friend_ids)
    if bfs is not None:
        return bfs

    return None  # 도달 가능한 빈 칸 없음 → 정지


# ──────────────────────────────────────────────────────────────
# 에이전트
# ──────────────────────────────────────────────────────────────

class Agent:
    # 공격적 능력 보유 에이전트가 게임 시작 후 능력을 봉인당하는 틱 수.
    # 0이면 디버프 없음. 서브클래스에서 오버라이드.
    STARTUP_DEBUFF = 0

    def __init__(self, agent_id: int, name: str, color: tuple, description: str = ''):
        self.id = agent_id
        self.name = name
        self.color = color
        self.description = description
        self.startup_remaining = self.STARTUP_DEBUFF
        # 팀 모드일 때 main.py 가 설정. None / 'attacker' / 'supporter' / 'flex' / 'raider'
        self.team_role = None

    def act(self, state: AgentState, context) -> int:
        raise NotImplementedError

    def _tick_startup(self):
        if self.startup_remaining > 0:
            self.startup_remaining -= 1

    def is_debuffed(self) -> bool:
        return self.startup_remaining > 0

    def _weak_act(self, state, context):
        """Startup 디버프 중 약화된 행동 — 단순 빈 칸 욕심."""
        def score(d, nx, ny, _s):
            count = 0
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if (context.grid.in_bounds(ax, ay)
                        and int(context.grid.cells[ay, ax]) == 0):
                    count += 1
            return count
        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class TrollAgent(Agent):
    """2등 영토 깊숙이 침투 — 인접 카운트^2 가중으로 영토 내부에 가시처럼 박힘."""

    def __init__(self, agent_id: int, color: tuple = (255, 122, 69)):
        super().__init__(agent_id, 'Driller', color, 'Drills into 2nd place')

    def act(self, state, context):
        target = context.get_rank_state(2, exclude_id=self.id)
        if target is None:
            return pick_direction(context.grid, state, context.rng, lambda *a: 1,
                                  friend_ids=_friends(context, self.id))
        target_id = target.agent_id

        def score(d, nx, ny, _s):
            t = 0
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if (context.grid.in_bounds(ax, ay)
                        and int(context.grid.cells[ay, ax]) == target_id):
                    t += 1
            # 1면→3, 2면→12, 3면→27, 4면→48 (영토 내부일수록 강하게)
            return 1 + t * t * 3

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class MirrorAgent(Agent):
    """Antipode — 1등 좌표의 거울 위치로 이동. 1등 반대편으로 펼친다."""

    def __init__(self, agent_id: int, color: tuple = (185, 103, 255)):
        super().__init__(agent_id, 'Antipode', color, 'Mirrors leader position')

    def act(self, state, context):
        leader = context.get_rank_state(1, exclude_id=self.id)
        if leader is None:
            return pick_direction(context.grid, state, context.rng, lambda *a: 1,
                                  friend_ids=_friends(context, self.id))
        w, h = context.grid.width, context.grid.height
        mx, my = w - 1 - leader.x, h - 1 - leader.y

        def score(d, nx, ny, _s):
            return -(abs(mx - nx) + abs(my - ny))

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


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

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class AntiMimicAgent(Agent):
    """Vulture — 꼴등 head 를 직접 추격 (Charger의 꼴등 한정 버전)."""

    def __init__(self, agent_id: int, color: tuple = (56, 189, 248)):
        super().__init__(agent_id, 'Vulture', color, 'Hunts the weakest')

    def act(self, state, context):
        last_place = context.get_last_state(exclude_id=self.id)
        if last_place is None:
            return pick_direction(context.grid, state, context.rng, lambda *a: 1,
                                  friend_ids=_friends(context, self.id))

        def score(d, nx, ny, _s):
            return -(abs(last_place.x - nx) + abs(last_place.y - ny))

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


# ──────────────────────────────────────────────────────────────
# 추가 에이전트 (변수성 풀)
# ──────────────────────────────────────────────────────────────

class RandomAgent(Agent):
    """무작위 행동 — 베이스라인."""

    def __init__(self, agent_id: int, color: tuple = (148, 163, 184)):
        super().__init__(agent_id, 'Random', color, 'Pure chaos')

    def act(self, state, context):
        return pick_direction(context.grid, state, context.rng, lambda *a: 1,
                                  friend_ids=_friends(context, self.id))


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

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class DefenderAgent(Agent):
    """자기 영토 옆 (단단) + 적 영토 옆 가산 (차단). 견고 + 적극 방어."""

    def __init__(self, agent_id: int, color: tuple = (244, 114, 182)):
        super().__init__(agent_id, 'Defender', color, 'Fortress + blocks foes')

    def act(self, state, context):
        def score(d, nx, ny, _s):
            own = 0
            enemy_adj = False
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if not context.grid.in_bounds(ax, ay):
                    continue
                cell = int(context.grid.cells[ay, ax])
                if cell == self.id:
                    own += 1
                elif cell != 0 and cell != WALL:
                    enemy_adj = True
            s = own * 2          # 자기 영토 옆 강화
            if enemy_adj:
                s += 4           # 적 옆 차단 보너스 (밸런스: ATK +11 보다 약함)
            return s

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class AggressorAgent(Agent):
    """Brawler — 여러 적 종류와 동시 접하는 격전지 빈 칸 우선.

    적 종류 수^2 으로 가중치 — 분기점/3-way 교전지로 모임.
    초반 80틱 디버프.
    """
    STARTUP_DEBUFF = 80

    def __init__(self, agent_id: int, color: tuple = (239, 68, 68)):
        super().__init__(agent_id, 'Brawler', color, 'Seeks chokepoint fights')

    def act(self, state, context):
        self._tick_startup()
        if self.is_debuffed():
            return self._weak_act(state, context)

        def score(d, nx, ny, _s):
            kinds = set()
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if not context.grid.in_bounds(ax, ay):
                    continue
                cell = int(context.grid.cells[ay, ax])
                if cell > 0 and cell != self.id:
                    kinds.add(cell)
            return 1 + len(kinds) * len(kinds) * 5  # 1→6, 2→21, 3→46

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


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

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


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

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


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

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class SquarerAgent(Agent):
    """사각형 그리며 점점 커지는 형태로 이동.

    시작점 (sx, sy) → 우(size) → 하(size) → 좌(size) → 상(size) = 한 바퀴.
    한 바퀴 끝나면 size += 1 → 더 큰 사각형. 영토가 동심 사각형 모양.
    """
    LEG_SEQ = [1, 2, 3, 0]   # right, down, left, up

    def __init__(self, agent_id: int, color: tuple = (120, 80, 200)):
        super().__init__(agent_id, 'Squarer', color, 'Draws expanding squares')
        self.start = None
        self.size = 1
        self.leg_idx = 0
        self.leg_step = 0

    def act(self, state, context):
        if self.start is None:
            self.start = (state.x, state.y)
        want_dir = self.LEG_SEQ[self.leg_idx]

        def score(d, nx, ny, _s):
            return 10 if d == want_dir else 1

        action = pick_direction(context.grid, state, context.rng, score,
                                friend_ids=_friends(context, self.id))
        if action == want_dir:
            self.leg_step += 1
            if self.leg_step >= self.size:
                self.leg_step = 0
                self.leg_idx = (self.leg_idx + 1) % 4
                if self.leg_idx == 0:
                    self.size += 1
        return action


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

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


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

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class ChargerAgent(Agent):
    """가장 가까운 적 헤드를 추적 — 영토 무관하게 머리만 노림.

    초반 80틱은 디버프 — 헤드 추적 봉인.
    """
    STARTUP_DEBUFF = 80

    def __init__(self, agent_id: int, color: tuple = (244, 63, 94)):
        super().__init__(agent_id, 'Charger', color, 'Rams the closest head')

    def act(self, state, context):
        self._tick_startup()
        if self.is_debuffed():
            return self._weak_act(state, context)

        rivals = [a for a in context.agent_states.values() if a.agent_id != self.id]
        if not rivals:
            return pick_direction(context.grid, state, context.rng, lambda *a: 1,
                                  friend_ids=_friends(context, self.id))
        target = min(rivals, key=lambda a: abs(a.x - state.x) + abs(a.y - state.y))

        def score(d, nx, ny, _s):
            return -(abs(target.x - nx) + abs(target.y - ny))

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class PatriotAgent(Agent):
    """홈 근처 + 적 head 인접 시 차단. 베이스 방어 + 접근자 격퇴."""

    def __init__(self, agent_id: int, color: tuple = (251, 146, 60)):
        super().__init__(agent_id, 'Patriot', color, 'Home guard + repels')
        self.home = None

    def act(self, state, context):
        if self.home is None:
            self.home = (state.x, state.y)
        rivals = [s for s in context.agent_states.values()
                  if s.agent_id != self.id]
        closest = (min(rivals,
                       key=lambda a: abs(a.x - state.x) + abs(a.y - state.y))
                   if rivals else None)

        def score(d, nx, ny, _s):
            s = -(abs(nx - self.home[0]) + abs(ny - self.home[1]))
            # 적 head 1칸 옆이면 차단 보너스
            if closest and abs(nx - closest.x) + abs(ny - closest.y) == 1:
                s += 4
            return s

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


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
            return pick_direction(context.grid, state, context.rng, lambda *a: 1,
                                  friend_ids=_friends(context, self.id))
        target = min(items, key=lambda it: abs(it.x - state.x) + abs(it.y - state.y))

        def score(d, nx, ny, _s):
            return -(abs(target.x - nx) + abs(target.y - ny))

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class SaboteurAgent(Agent):
    """우리가 가장 유리한 아이템(상대보다 가까운)으로 가서 가로채기.

    팀모드: 'rivals' 는 다른 팀의 head 들만. 같은 팀이 가까이 있어도 양보.
    """

    def __init__(self, agent_id: int, color: tuple = (168, 85, 247)):
        super().__init__(agent_id, 'Saboteur', color, "Steals enemy items")

    def act(self, state, context):
        items = context.items
        teams = context.teams
        my_team = teams[self.id] if teams else None

        if teams is not None:
            rivals = [a for aid, a in context.agent_states.items()
                      if teams.get(aid, aid) != my_team]
        else:
            rivals = [a for a in context.agent_states.values() if a.agent_id != self.id]

        if not items:
            return pick_direction(context.grid, state, context.rng, lambda *a: 1,
                                  friend_ids=_friends(context, self.id))

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

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


# ──────────────────────────────────────────────────────────────
# 맵 인식 에이전트
# ──────────────────────────────────────────────────────────────

class BlockerAgent(Agent):
    """가장 가까운 적의 진행 방향 1~2칸 앞을 점거 — 적의 다음 발걸음 차단.

    적이 자기 영토 못 들어와서 우회 강제. 적의 진로 적극 방해.
    """

    def __init__(self, agent_id: int, color: tuple = (80, 130, 200)):
        super().__init__(agent_id, 'Blocker', color, 'Blocks enemy paths')

    def act(self, state, context):
        rivals = [s for s in context.agent_states.values()
                  if s.agent_id != self.id]
        if not rivals:
            return pick_direction(context.grid, state, context.rng,
                                  lambda *a: 1,
                                  friend_ids=_friends(context, self.id))
        target = min(rivals,
                     key=lambda a: abs(a.x - state.x) + abs(a.y - state.y))
        # 적 진행 방향 2칸 앞 예측
        last = target.last_action
        dx, dy = DIRECTIONS[last] if last is not None else (0, 0)
        block_x, block_y = target.x + dx * 2, target.y + dy * 2

        def score(d, nx, ny, _s):
            return -(abs(block_x - nx) + abs(block_y - ny))
        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class GatekeeperAgent(Agent):
    """좁은 통로 + 적 영토 옆 동시 점거 — 통로 봉쇄.

    벽 인접 + 적 영토 인접 시 강한 보너스 → 적의 영토 확장 진입로 차단.
    """

    def __init__(self, agent_id: int, color: tuple = (100, 150, 180)):
        super().__init__(agent_id, 'Gatekeeper', color, 'Seals chokepoints')

    def act(self, state, context):
        grid = context.grid

        def score(d, nx, ny, _s):
            wall_count = 0
            enemy_adj = False
            empty_count = 0
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if not grid.in_bounds(ax, ay):
                    wall_count += 1
                    continue
                cell = int(grid.cells[ay, ax])
                if cell == WALL:
                    wall_count += 1
                elif cell == 0:
                    empty_count += 1
                elif cell != self.id:
                    enemy_adj = True
            s = wall_count * 3
            if enemy_adj:
                s += 8       # 적 옆 차단 보너스
            if empty_count <= 2:
                s += 5       # 좁은 통로
            return s

        return pick_direction(grid, state, context.rng, score)


class WallAgent(Agent):
    """일관된 한 방향 직진 + 자기 영토 옆 — 직선 장벽 형성.

    적의 영토 확장 경로를 가로지르는 긴 일자 영토. Compass + Defender 융합.
    """

    def __init__(self, agent_id: int, color: tuple = (140, 100, 160)):
        super().__init__(agent_id, 'Wall', color, 'Forms long walls')
        self.preferred = None

    def act(self, state, context):
        if self.preferred is None:
            self.preferred = context.rng.randint(0, 3)
        grid = context.grid

        def score(d, nx, ny, _s):
            s = 1
            if d == self.preferred:
                s += 6
            own = 0
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if (grid.in_bounds(ax, ay)
                        and int(grid.cells[ay, ax]) == self.id):
                    own += 1
            s += own * 2
            return s

        return pick_direction(grid, state, context.rng, score)


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

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


# ──────────────────────────────────────────────────────────────
# 상태 머신 에이전트
# ──────────────────────────────────────────────────────────────

class PhoenixAgent(Agent):
    """영토 < 10%면 광폭(Aggressor), > 30%면 농성(Defender), 사이는 탐욕(Greedy).

    초반 80틱은 광폭 모드 봉인 (시작 직후 모두가 10% 미만이므로).
    """
    STARTUP_DEBUFF = 80

    def __init__(self, agent_id: int, color: tuple = (244, 63, 94)):
        super().__init__(agent_id, 'Phoenix', color, 'Berserk → calm by area')

    def act(self, state, context):
        self._tick_startup()
        if self.is_debuffed():
            return self._weak_act(state, context)

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

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class SleeperAgent(Agent):
    """Mine — 자다가 적 head 가 3×3 이내 접근 시 5×5 강제 흡수 후 깨어남.

    발동 후 cooldown 80틱. 깨어난 뒤엔 Charger 행동 (가장 가까운 적 추격).
    """

    SLEEP_TICKS = 60
    TRIGGER_RADIUS = 1   # 3×3 = 반경 1
    BURST_RADIUS = 2     # 5×5 = 반경 2
    BURST_COOLDOWN = 80

    def __init__(self, agent_id: int, color: tuple = (99, 102, 241)):
        super().__init__(agent_id, 'Mine', color, 'Detonates when approached')
        self.elapsed = 0
        self.burst_cooldown = 0

    def act(self, state, context):
        self.elapsed += 1
        self.burst_cooldown = max(0, self.burst_cooldown - 1)
        grid = context.grid

        # 적 head 가 3×3 안에 들어왔는지
        enemy_near = False
        for s in context.agent_states.values():
            if s.agent_id == self.id:
                continue
            if (abs(s.x - state.x) <= self.TRIGGER_RADIUS
                    and abs(s.y - state.y) <= self.TRIGGER_RADIUS):
                enemy_near = True
                break

        # 발동 — 5×5 강제 흡수 (벽 빼고)
        if enemy_near and self.burst_cooldown == 0:
            for dy in range(-self.BURST_RADIUS, self.BURST_RADIUS + 1):
                for dx in range(-self.BURST_RADIUS, self.BURST_RADIUS + 1):
                    nx, ny = state.x + dx, state.y + dy
                    if (grid.in_bounds(nx, ny)
                            and int(grid.cells[ny, nx]) != WALL):
                        grid.cells[ny, nx] = self.id
            self.burst_cooldown = self.BURST_COOLDOWN
            self.elapsed = self.SLEEP_TICKS + 1   # 깨어남
            if context.effect_flashes is not None:
                from .items import EffectFlash
                context.effect_flashes.append(EffectFlash(
                    x=state.x, y=state.y, text='BOOM!',
                    color=(255, 80, 60),
                    ticks_left=36, initial_ticks=36,
                ))
            return None

        # 자는 동안 정지
        if self.elapsed <= self.SLEEP_TICKS:
            return None

        # 깨어났으면 Charger 행동
        rivals = [a for a in context.agent_states.values()
                  if a.agent_id != self.id]
        if not rivals:
            return pick_direction(grid, state, context.rng, lambda *a: 1)
        target = min(rivals, key=lambda a: abs(a.x - state.x) + abs(a.y - state.y))

        def score(d, nx, ny, _s):
            return -(abs(target.x - nx) + abs(target.y - ny))

        return pick_direction(grid, state, context.rng, score)


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

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


# ──────────────────────────────────────────────────────────────
# 능력 보유 에이전트 (텔레포트/돌파)
# ──────────────────────────────────────────────────────────────

class EscapistAgent(Agent):
    """위험할 때 빈 칸으로 텔레포트하는 방어형. 디버프 없음.

    위험 = 4방향 이웃 중 3개 이상이 적 영토/벽으로 막혀있을 때.
    텔레포트 후 80틱 쿨다운. 평소엔 Defender처럼 자기 영토와 인접 빈 칸 우선.
    """
    COOLDOWN = 50           # 보강: 80 → 50 (더 자주 텔레포트)
    DANGER_THRESHOLD = 3

    def __init__(self, agent_id: int, color: tuple = (96, 165, 250)):
        super().__init__(agent_id, 'Escapist', color, 'Teleports out of traps')
        self.cooldown = 0

    def act(self, state, context):
        self.cooldown = max(0, self.cooldown - 1)
        grid = context.grid

        # 위험도 측정
        danger = 0
        for dx, dy in DIRECTIONS:
            nx, ny = state.x + dx, state.y + dy
            if not grid.in_bounds(nx, ny):
                danger += 1
                continue
            cell = int(grid.cells[ny, nx])
            if cell == WALL:
                danger += 1
            elif cell != 0 and cell != self.id:
                danger += 1

        # 능력 발동 — 빈 칸으로 텔레포트
        if danger >= self.DANGER_THRESHOLD and self.cooldown == 0:
            ys, xs = (grid.cells == 0).nonzero()
            if len(xs) > 0:
                i = context.rng.randint(0, len(xs) - 1)
                old_pos = (state.x, state.y)
                state.x, state.y = int(xs[i]), int(ys[i])
                grid.cells[state.y, state.x] = self.id
                self.cooldown = self.COOLDOWN
                # 시각 플래시
                if context.effect_flashes is not None:
                    from .items import EffectFlash
                    context.effect_flashes.append(EffectFlash(
                        x=old_pos[0], y=old_pos[1], text='POOF',
                        color=(96, 165, 250),
                        ticks_left=24, initial_ticks=24,
                    ))
                    context.effect_flashes.append(EffectFlash(
                        x=state.x, y=state.y, text='WARP',
                        color=(96, 165, 250),
                        ticks_left=24, initial_ticks=24,
                    ))
                return None

        # 평소: 자기 영토 인접 빈 칸 우선 (Defender 비슷)
        def score(d, nx, ny, _s):
            count = 0
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if (grid.in_bounds(ax, ay)
                        and int(grid.cells[ay, ax]) == self.id):
                    count += 1
            return count

        return pick_direction(grid, state, context.rng, score)


class BreacherAgent(Agent):
    """인접한 적 영토를 한 칸 뚫고 들어가 변환하는 공격형.

    능력 — 인접한 적 셀로 직접 이동하고 그 칸을 자기 색으로 변환.
    쿨다운 40틱. 초반 120틱은 디버프 — 능력 봉인.
    능력 비활성 시엔 Aggressor처럼 적 인접 빈 칸 우선.
    """
    STARTUP_DEBUFF = 120
    COOLDOWN = 40

    def __init__(self, agent_id: int, color: tuple = (220, 38, 38)):
        super().__init__(agent_id, 'Breacher', color, 'Pierces enemy lines')
        self.cooldown = 0

    def act(self, state, context):
        self._tick_startup()
        self.cooldown = max(0, self.cooldown - 1)
        grid = context.grid

        # 디버프 중엔 약화된 행동만
        if self.is_debuffed():
            return self._weak_act(state, context)

        # 능력 — 인접한 적 셀로 돌파
        if self.cooldown == 0:
            enemy_dirs = []
            for d, (dx, dy) in enumerate(DIRECTIONS):
                nx, ny = state.x + dx, state.y + dy
                if not grid.in_bounds(nx, ny):
                    continue
                cell = int(grid.cells[ny, nx])
                if cell != 0 and cell != self.id and cell != WALL:
                    enemy_dirs.append((d, nx, ny))
            if enemy_dirs:
                d, nx, ny = context.rng.choice(enemy_dirs)
                state.x, state.y = nx, ny
                state.last_action = d
                grid.cells[ny, nx] = self.id
                self.cooldown = self.COOLDOWN
                if context.effect_flashes is not None:
                    from .items import EffectFlash
                    context.effect_flashes.append(EffectFlash(
                        x=nx, y=ny, text='BREACH',
                        color=(220, 38, 38),
                        ticks_left=24, initial_ticks=24,
                    ))
                return None

        # 평소: 적 인접 빈 칸 우선 (Aggressor 비슷)
        def score(d, nx, ny, _s):
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if not grid.in_bounds(ax, ay):
                    continue
                cell = int(grid.cells[ay, ax])
                if cell != 0 and cell != self.id and cell != WALL:
                    return 11
            return 1

        return pick_direction(grid, state, context.rng, score)


# ──────────────────────────────────────────────────────────────
# 팀 역할 에이전트 (팀 모드 강제 배치용)
# ──────────────────────────────────────────────────────────────

def _enemy_team_ids(context, my_id: int) -> set:
    """팀모드 기준 적 팀 agent_id 집합. free-for-all 에선 본인 외 전부."""
    teams = context.teams
    if teams is None:
        return {aid for aid in context.agent_states if aid != my_id}
    my_t = teams[my_id]
    return {aid for aid, t in teams.items() if t != my_t}


def _team_friend_ids(context, my_id: int) -> set:
    """팀모드 기준 같은 팀 agent_id 집합 (자신 포함)."""
    teams = context.teams
    if teams is None:
        return {my_id}
    my_t = teams[my_id]
    return {aid for aid, t in teams.items() if t == my_t}


def _attacker_score(my_id, context, d, nx, ny):
    """팀 공격수 점수 — 적 팀 영토 인접 + 적 head 길 막기."""
    grid = context.grid
    enemy_ids = _enemy_team_ids(context, my_id)
    best = 1
    for ddx, ddy in DIRECTIONS:
        ax, ay = nx + ddx, ny + ddy
        if not grid.in_bounds(ax, ay):
            continue
        cell = int(grid.cells[ay, ax])
        if cell != 0 and cell != WALL and cell in enemy_ids:
            best = max(best, 12)
    # 적 head 인접: 길 막기 보너스
    for aid in enemy_ids:
        if aid not in context.agent_states:
            continue
        s = context.agent_states[aid]
        if abs(s.x - nx) + abs(s.y - ny) == 1:
            best = max(best, 8)
    return best


def _supporter_score(my_id, context, d, nx, ny):
    """팀 서포터 점수 — 팀원 영토 인접 + 아이템 우선."""
    grid = context.grid
    friends = _team_friend_ids(context, my_id)
    base = 0
    for ddx, ddy in DIRECTIONS:
        ax, ay = nx + ddx, ny + ddy
        if not grid.in_bounds(ax, ay):
            continue
        cell = int(grid.cells[ay, ax])
        if cell in friends:
            base += 2
    # 아이템 근접 가산
    for it in context.items:
        if abs(it.x - nx) + abs(it.y - ny) <= 2:
            base += 6
    return max(1, base)


class TeamAttackerAgent(Agent):
    """팀 공격수 — 적 팀 영토 침공 + 적 머리 길 막기."""

    def __init__(self, agent_id: int, color: tuple = (220, 80, 80)):
        super().__init__(agent_id, 'Attacker', color, 'Strikes enemy team')

    def act(self, state, context):
        friends = _team_friend_ids(context, self.id)
        def score(d, nx, ny, _s):
            return _attacker_score(self.id, context, d, nx, ny)
        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=friends)


class TeamSupporterAgent(Agent):
    """팀 서포터 — 팀원 영토 보강 + 아이템 우선."""

    def __init__(self, agent_id: int, color: tuple = (120, 170, 230)):
        super().__init__(agent_id, 'Supporter', color, 'Backs the team')

    def act(self, state, context):
        friends = _team_friend_ids(context, self.id)
        # 아이템이 있으면 가장 가까운 아이템 직진 우선
        if context.items:
            target = min(context.items,
                         key=lambda it: abs(it.x - state.x) + abs(it.y - state.y))
            def to_item(d, nx, ny, _s):
                return -(abs(target.x - nx) + abs(target.y - ny))
            return pick_direction(context.grid, state, context.rng, to_item,
                                  friend_ids=friends)

        def score(d, nx, ny, _s):
            return _supporter_score(self.id, context, d, nx, ny)
        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=friends)


class TeamFlexAgent(Agent):
    """팀 유동 — 우리 팀 합산 영토가 적 팀 이하면 공격, 더 크면 서포터."""

    def __init__(self, agent_id: int, color: tuple = (200, 160, 80)):
        super().__init__(agent_id, 'Flex', color, 'Adapts to team state')

    def act(self, state, context):
        teams = context.teams
        friends = _team_friend_ids(context, self.id)
        if teams is None:
            def score(d, nx, ny, _s):
                return _attacker_score(self.id, context, d, nx, ny)
            return pick_direction(context.grid, state, context.rng, score,
                                  friend_ids=friends)

        my_team = teams[self.id]
        team_total: dict = {}
        for aid, tid in teams.items():
            team_total[tid] = team_total.get(tid, 0) + context.areas.get(aid, 0)
        my_total = team_total.get(my_team, 0)
        others = [t for tid, t in team_total.items() if tid != my_team]
        other_max = max(others) if others else 0

        if my_total <= other_max:
            def score(d, nx, ny, _s):
                return _attacker_score(self.id, context, d, nx, ny)
        else:
            def score(d, nx, ny, _s):
                return _supporter_score(self.id, context, d, nx, ny)
        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=friends)


class HealerAgent(Agent):
    """Mason — 자기/팀 영토에 3면 이상 둘러싸인 빈 칸(구멍) 우선 메우기.

    Defender(외각 확장) 와 정반대 — 내부 구멍을 채워 단단한 덩어리 완성.
    팀모드에선 팀 영토 안의 구멍도 메움.
    """

    def __init__(self, agent_id: int, color: tuple = (120, 220, 180)):
        super().__init__(agent_id, 'Mason', color, 'Fills inner gaps')

    def act(self, state, context):
        teams = context.teams
        if teams is not None:
            my_t = teams[self.id]
            friends = {aid for aid, t in teams.items() if t == my_t}
        else:
            friends = {self.id}

        def score(d, nx, ny, _s):
            count = 0
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if (context.grid.in_bounds(ax, ay)
                        and int(context.grid.cells[ay, ax]) in friends):
                    count += 1
            # 3+면 구멍이라고 보고 강한 보너스 (1→1, 2→2, 3→15, 4→20)
            if count >= 3:
                return 12 + count
            return count

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=friends)


class TrapperAgent(Agent):
    """Trapper — 가만히 있다가 적 head/영토가 5×5 이내 접근 시 13×13 강제 흡수.

    발동 후 cooldown 동안엔 Greedy 식으로 빈 칸 확장 (먹으면서 이동).
    Mine 보다 더 멀리서 발동(5×5 vs 3×3) + 훨씬 큰 폭발(169칸 ≈ 5% vs 25칸).
    발동 안 했을 땐 자기 자리에 정지 — 함정처럼 깔려있음.
    head 자리는 인클로저 면역 (시뮬레이션 측에서 처리).
    """
    TRIGGER_RADIUS = 2          # 5×5 이내 적 감지
    BURST_SIDE = 13             # 13×13 = 169칸 ≈ 5% 영토 강제 흡수
    BURST_COOLDOWN = 100

    def __init__(self, agent_id: int, color: tuple = (190, 130, 220)):
        super().__init__(agent_id, 'Trapper', color, 'Erupts on intrusion')
        self.burst_cooldown = 0

    def act(self, state, context):
        self.burst_cooldown = max(0, self.burst_cooldown - 1)
        grid = context.grid

        # 5×5 이내에 head 또는 영토(자기 제외) 있는가 — 팀 무관 무차별 발동
        enemy_near = False
        for s in context.agent_states.values():
            if s.agent_id == self.id:
                continue
            if (abs(s.x - state.x) <= self.TRIGGER_RADIUS
                    and abs(s.y - state.y) <= self.TRIGGER_RADIUS):
                enemy_near = True
                break
        if not enemy_near:
            for dy in range(-self.TRIGGER_RADIUS, self.TRIGGER_RADIUS + 1):
                for dx in range(-self.TRIGGER_RADIUS, self.TRIGGER_RADIUS + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = state.x + dx, state.y + dy
                    if not grid.in_bounds(nx, ny):
                        continue
                    cell = int(grid.cells[ny, nx])
                    if cell <= 0 or cell == self.id:
                        continue
                    enemy_near = True
                    break
                if enemy_near:
                    break

        # 발동 — BURST_SIDE × BURST_SIDE 강제 흡수 (적/팀/벽 덮음, DEF 50% 저항)
        if enemy_near and self.burst_cooldown == 0:
            walls_overwritten = 0
            half = self.BURST_SIDE // 2
            role_by_id = context.role_by_id or {}
            for dy in range(-half, self.BURST_SIDE - half):
                for dx in range(-half, self.BURST_SIDE - half):
                    nx, ny = state.x + dx, state.y + dy
                    if not grid.in_bounds(nx, ny):
                        continue
                    cell = int(grid.cells[ny, nx])
                    # DEF 50% 저항
                    if (cell > 0 and cell != self.id
                            and role_by_id.get(cell, '') == 'DEF'
                            and context.rng.random() < 0.5):
                        continue
                    if cell == WALL:
                        walls_overwritten += 1
                    grid.cells[ny, nx] = self.id
            if walls_overwritten:
                grid._wall_count -= walls_overwritten
            self.burst_cooldown = self.BURST_COOLDOWN
            # 폭발 안에 있던 head 는 모두 3초 freeze (팀원도 예외 없음)
            if context.freeze_ticks is not None:
                for s in context.agent_states.values():
                    if s.agent_id == self.id:
                        continue
                    if (abs(s.x - state.x) <= self.TRIGGER_RADIUS
                            and abs(s.y - state.y) <= self.TRIGGER_RADIUS):
                        context.freeze_ticks[s.agent_id] = (
                            context.freeze_ticks.get(s.agent_id, 0) + 90
                        )
            if context.effect_flashes is not None:
                from .items import EffectFlash
                context.effect_flashes.append(EffectFlash(
                    x=state.x, y=state.y, text='ERUPT!',
                    color=(220, 100, 240),
                    ticks_left=36, initial_ticks=36,
                ))
            return None

        # 첫 발동 전엔 가만히 있음 (함정 대기)
        if self.burst_cooldown == 0:
            return None

        # 발동 후 cooldown 중 — 가장 가까운 적 head 향해 이동 (적 진영 파고들기).
        # cooldown 끝나는 시점에 적 옆이면 즉시 또 발동.
        my_team = context.teams.get(self.id) if context.teams else None
        enemies = []
        for s in context.agent_states.values():
            if s.agent_id == self.id:
                continue
            if context.teams and context.teams.get(s.agent_id) == my_team:
                continue
            enemies.append(s)
        if not enemies:
            return pick_direction(grid, state, context.rng, lambda *a: 1)

        target = min(enemies,
                     key=lambda a: abs(a.x - state.x) + abs(a.y - state.y))

        def score(d, nx, ny, _s):
            return -(abs(target.x - nx) + abs(target.y - ny))

        return pick_direction(grid, state, context.rng, score)


class RevenantAgent(Agent):
    """부활자 — 한 번 죽으면 즉시 무작위 빈 칸에서 부활 (목숨 +1).

    능력은 부활 1회뿐. 부활 후 두 번째 사망 시엔 일반 사망 처리.
    부활 위치는 무작위 빈 칸 (없으면 비-벽 셀). 능력/속도 보너스 없음.
    """

    def __init__(self, agent_id: int, color: tuple = (130, 40, 50)):
        super().__init__(agent_id, 'Revenant', color, 'One extra life')

    def act(self, state, context):
        # 평소 행동 — Greedy 비슷 (빈 칸 많은 방향). 부활 후엔 simulation 측이
        # 점령 / 속도를 처리하므로 별도 분기 불필요.
        def score(d, nx, ny, _s):
            count = 0
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if context.grid.in_bounds(ax, ay) and int(context.grid.cells[ay, ax]) == 0:
                    count += 1
            return count

        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class WildcardAgent(Agent):
    """30틱 쿨다운으로 무작위 효과 1개를 자기 자신에게 발동. 좋을지 나쁠지 모름."""
    COOLDOWN = 30

    def __init__(self, agent_id: int, color: tuple = (240, 90, 230)):
        super().__init__(agent_id, 'Wildcard', color, 'Random effect roulette')
        self.cooldown = 0

    def act(self, state, context):
        self.cooldown = max(0, self.cooldown - 1)
        if self.cooldown == 0 and context.wildcard_pending is not None:
            from .items import ALL_EFFECTS
            key = context.rng.choice([e[0] for e in ALL_EFFECTS])
            context.wildcard_pending.append((self.id, key))
            self.cooldown = self.COOLDOWN
            if context.effect_flashes is not None:
                from .items import EffectFlash
                context.effect_flashes.append(EffectFlash(
                    x=state.x, y=state.y, text='WILDCARD!',
                    color=(240, 90, 230),
                    ticks_left=24, initial_ticks=24,
                ))

        def score(d, nx, ny, _s):
            count = 0
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if (context.grid.in_bounds(ax, ay)
                        and int(context.grid.cells[ay, ax]) == 0):
                    count += 1
            return count
        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class AnarchistAgent(Agent):
    """50틱 쿨다운으로 살아있는 두 에이전트 head 위치 swap (자기 포함). 카오스."""
    COOLDOWN = 50

    def __init__(self, agent_id: int, color: tuple = (200, 50, 120)):
        super().__init__(agent_id, 'Anarchist', color, 'Swaps random heads')
        self.cooldown = 0

    def act(self, state, context):
        self.cooldown = max(0, self.cooldown - 1)
        if self.cooldown == 0:
            dead = context.dead or {}
            alive = [s for s in context.agent_states.values()
                     if not dead.get(s.agent_id, False)]
            if len(alive) >= 2:
                a, b = context.rng.sample(alive, 2)
                a.x, b.x = b.x, a.x
                a.y, b.y = b.y, a.y
                self.cooldown = self.COOLDOWN
                if context.effect_flashes is not None:
                    from .items import EffectFlash
                    context.effect_flashes.append(EffectFlash(
                        x=a.x, y=a.y, text='ANARCHY!',
                        color=(200, 50, 120),
                        ticks_left=24, initial_ticks=24,
                    ))

        def score(d, nx, ny, _s):
            count = 0
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if (context.grid.in_bounds(ax, ay)
                        and int(context.grid.cells[ay, ax]) == 0):
                    count += 1
            return count
        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class JesterAgent(Agent):
    """매 행동마다 5개 알고리즘 (Greedy/Aggressor/Hermit/Random/Defender) 중 무작위 선택."""

    def __init__(self, agent_id: int, color: tuple = (240, 200, 60)):
        super().__init__(agent_id, 'Jester', color, 'Random algorithm per tick')

    def act(self, state, context):
        mode = context.rng.randint(0, 4)
        grid = context.grid

        if mode == 0:  # Greedy
            def score(d, nx, ny, _s):
                count = 0
                for ddx, ddy in DIRECTIONS:
                    ax, ay = nx + ddx, ny + ddy
                    if grid.in_bounds(ax, ay) and int(grid.cells[ay, ax]) == 0:
                        count += 1
                return count
        elif mode == 1:  # Aggressor
            def score(d, nx, ny, _s):
                for ddx, ddy in DIRECTIONS:
                    ax, ay = nx + ddx, ny + ddy
                    if not grid.in_bounds(ax, ay):
                        continue
                    cell = int(grid.cells[ay, ax])
                    if cell != 0 and cell != self.id and cell != WALL:
                        return 11
                return 1
        elif mode == 2:  # Hermit
            def score(d, nx, ny, _s):
                for ddx, ddy in DIRECTIONS:
                    ax, ay = nx + ddx, ny + ddy
                    if not grid.in_bounds(ax, ay):
                        continue
                    cell = int(grid.cells[ay, ax])
                    if cell != 0 and cell != self.id:
                        return 0
                return 5
        elif mode == 3:  # Random
            def score(d, nx, ny, _s):
                return 1
        else:  # Defender
            def score(d, nx, ny, _s):
                count = 0
                for ddx, ddy in DIRECTIONS:
                    ax, ay = nx + ddx, ny + ddy
                    if grid.in_bounds(ax, ay) and int(grid.cells[ay, ax]) == self.id:
                        count += 1
                return count

        return pick_direction(grid, state, context.rng, score)


class RouletteAgent(Agent):
    """라운드당 1회 한정으로 1위와 영토 + 위치 swap. 도박 한 방.

    초반 300틱(=10초)은 잠재기 — 1위가 충분히 커진 후 swap 시도.
    """
    INITIAL_DELAY = 300
    USED_LIMIT = 1

    def __init__(self, agent_id: int, color: tuple = (210, 40, 70)):
        super().__init__(agent_id, 'Roulette', color, 'One-shot swap with #1')
        self.cooldown = self.INITIAL_DELAY
        self.uses_left = self.USED_LIMIT

    def act(self, state, context):
        self.cooldown = max(0, self.cooldown - 1)
        grid = context.grid

        if self.cooldown == 0 and self.uses_left > 0:
            best, best_area = None, -1
            for aid, area in context.areas.items():
                if aid == self.id:
                    continue
                if area > best_area:
                    best_area = area
                    best = aid
            if best is not None and best_area > 0 and best in context.agent_states:
                my_mask = grid.cells == self.id
                their_mask = grid.cells == best
                grid.cells[my_mask] = best
                grid.cells[their_mask] = self.id
                other = context.agent_states[best]
                state.x, other.x = other.x, state.x
                state.y, other.y = other.y, state.y
                grid.cells[state.y, state.x] = self.id
                grid.cells[other.y, other.x] = best
                self.uses_left -= 1     # 1회 한정 — 다시 발동 안 됨
                if context.effect_flashes is not None:
                    from .items import EffectFlash
                    context.effect_flashes.append(EffectFlash(
                        x=state.x, y=state.y, text='ROULETTE!',
                        color=(210, 40, 70),
                        ticks_left=24, initial_ticks=24,
                    ))
                return None

        def score(d, nx, ny, _s):
            count = 0
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if grid.in_bounds(ax, ay) and int(grid.cells[ay, ax]) == 0:
                    count += 1
            return count
        return pick_direction(grid, state, context.rng, score)


class InfernoAgent(Agent):
    """일반 행동은 Aggressor 비슷. 사망 시 sim 측이 주변 7×7 을 빈 칸으로 폭사."""

    def __init__(self, agent_id: int, color: tuple = (255, 110, 40)):
        super().__init__(agent_id, 'Inferno', color, 'Explodes on death')

    def act(self, state, context):
        def score(d, nx, ny, _s):
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if not context.grid.in_bounds(ax, ay):
                    continue
                cell = int(context.grid.cells[ay, ax])
                if cell != 0 and cell != self.id and cell != WALL:
                    return 11
            return 1
        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=_friends(context, self.id))


class TeamRaiderAgent(Agent):
    """팀 침투수 — 팀 영토 무게중심에서 가장 먼 빈 칸 우선.

    팀이 한쪽에 몰리지 않고 펼쳐지게 + 자연스레 적 영토 옆으로 가서 침투.
    BFS 는 팀 영토 통과 가능.
    """

    def __init__(self, agent_id: int, color: tuple = (220, 150, 100)):
        super().__init__(agent_id, 'Raider', color, 'Splits from team')

    def act(self, state, context):
        teams = context.teams
        if teams is not None:
            my_team = teams[self.id]
            friends = {aid for aid, t in teams.items() if t == my_team}
        else:
            friends = {self.id}

        grid = context.grid
        # 팀 영토 무게중심
        mask = np.isin(grid.cells, list(friends))
        ys, xs = mask.nonzero()
        if len(xs) > 0:
            cx = float(xs.mean())
            cy = float(ys.mean())
        else:
            cx, cy = float(state.x), float(state.y)

        def score(d, nx, ny, _s):
            return int(abs(nx - cx) + abs(ny - cy))

        return pick_direction(grid, state, context.rng, score,
                              friend_ids=friends)


class BetrayerAgent(Agent):
    """배신자 — 첫 20초(600틱)는 팀원처럼 행동, 이후 sim 측에서 솔로 팀으로 분리됨.

    Reveal 전엔 sim.teams[id] 가 원래 팀이라 같은 팀 영토를 적으로 보지 않음.
    Reveal 후엔 sim.teams[id] 가 솔로 팀이 되어 모든 영토가 적이 됨 — 자동 전환.
    행동은 Aggressor 식 (다른 팀 영토 옆 빈 칸 우선).
    BFS 는 항상 모든 영토 통과 가능 (배신 후 어디든 갈 수 있게).
    """

    def __init__(self, agent_id: int, color: tuple = (180, 60, 60)):
        super().__init__(agent_id, 'Betrayer', color, 'Turns at 20s')

    def act(self, state, context):
        my_team = context.teams.get(self.id) if context.teams else None

        def score(d, nx, ny, _s):
            for ddx, ddy in DIRECTIONS:
                ax, ay = nx + ddx, ny + ddy
                if not context.grid.in_bounds(ax, ay):
                    continue
                cell = int(context.grid.cells[ay, ax])
                if cell <= 0 or cell == self.id or cell == WALL:
                    continue
                if context.teams and context.teams.get(cell) == my_team:
                    continue   # 같은 팀(아직 reveal 전이면 옛 팀원) 은 공격 안 함
                return 11
            return 1

        # 모든 에이전트 id 를 friend 로 — BFS 가 어떤 영토든 통과 가능
        all_ids = set(context.agent_states.keys())
        all_ids.add(self.id)
        return pick_direction(context.grid, state, context.rng, score,
                              friend_ids=all_ids)


# 매 판 랜덤 픽용 풀 (35종 — BetrayerAgent 는 팀 모드 전용이라 별도 처리)
ALGORITHM_POOL = [
    TrollAgent, MirrorAgent, BoomerangAgent, AntiMimicAgent,
    RandomAgent, GreedyAgent, DefenderAgent, AggressorAgent, HermitAgent,
    ScoutAgent, PioneerAgent, SpiralAgent, SquarerAgent,
    CompassAgent, ChargerAgent, PatriotAgent,
    HoarderAgent, SaboteurAgent, CartographerAgent,
    PhoenixAgent, SleeperAgent, MimicAgent,
    EscapistAgent, BreacherAgent, RevenantAgent,
    HealerAgent, TrapperAgent,
    WildcardAgent, AnarchistAgent, JesterAgent, RouletteAgent, InfernoAgent,
    BlockerAgent, GatekeeperAgent, WallAgent,
]


# ──────────────────────────────────────────────────────────────
# 카테고리 심볼 매핑 — 사이드바 이름 뒤에 표시
# ──────────────────────────────────────────────────────────────
ROLE_GLYPHS = {
    # 공격
    'Brawler': 'ATK', 'Charger': 'ATK', 'Driller': 'ATK', 'Vulture': 'ATK',
    'Antipode': 'ATK', 'Breacher': 'ATK', 'Mine': 'ATK', 'Trapper': 'ATK',
    'Phoenix': 'ATK', 'Inferno': 'ATK', 'Saboteur': 'ATK',
    # 방어 (Boomerang 은 회전으로 인클로저 형성 → 너무 강해 EXP 로 이동)
    'Defender': 'DEF', 'Mason': 'DEF', 'Escapist': 'DEF', 'Patriot': 'DEF',
    'Hermit': 'DEF', 'Cartographer': 'DEF', 'Revenant': 'DEF',
    'Blocker': 'DEF', 'Gatekeeper': 'DEF', 'Wall': 'DEF',
    # 조커
    'Wildcard': 'JKR', 'Anarchist': 'JKR', 'Jester': 'JKR', 'Roulette': 'JKR',
    'Random': 'JKR', 'Mimic': 'JKR',
    # 욕심/확장
    'Greedy': 'EXP', 'Spiral': 'EXP', 'Squarer': 'EXP', 'Compass': 'EXP',
    'Scout': 'EXP', 'Pioneer': 'EXP', 'Sleeper': 'EXP', 'Hoarder': 'EXP',
    'Boomerang': 'EXP',
    # 팀
    'Attacker': 'TM', 'Supporter': 'TM', 'Flex': 'TM', 'Raider': 'TM',
    'Betrayer': 'TM',
}
