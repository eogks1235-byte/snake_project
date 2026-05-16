"""팀 능력치 — FIFA 랭킹 포인트 기반 자동 생성.

각 팀은 4개 능력치 (0~100):
  - attack:  공격력 (슈팅, 파이널서드 위협)
  - defense: 수비력 (수비 압박, 라인 견고함)
  - midfield: 미드 점유 (점유율, 패스 정확도)
  - keeper:  골키퍼 (실점 방어)

FIFA 포인트 → 종합 OVR → 4개 능력치로 jitter 분산.
컨페더레이션 + style_tag 보정으로 색깔 부여.
"""
import random
from dataclasses import dataclass, field

from .data import TeamData, SQUADS, PlayerData


# FIFA 포인트 정규화 범위 (실제 분포 근사)
MIN_POINTS = 1180.0
MAX_POINTS = 1900.0


@dataclass
class Team:
    data: TeamData
    attack: int
    defense: int
    midfield: int
    keeper: int

    color: tuple = (200, 200, 200)
    secondary: tuple = (40, 40, 40)
    formation: str = '4-2-3-1'
    secondary_formation: str = '4-4-2'
    style_tag: str = 'balanced'
    wc_titles: int = 0
    squad: list = field(default_factory=list)   # PlayerData 리스트 (16명)

    # 누적 토너먼트 통계
    goals_for: int = 0
    goals_against: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points: int = 0

    @property
    def name(self) -> str:
        return self.data.name

    @property
    def code(self) -> str:
        return self.data.code

    @property
    def iso2(self) -> str:
        return self.data.iso2

    @property
    def altitude_native(self) -> bool:
        return self.data.altitude_native

    @property
    def heat_native(self) -> bool:
        return self.data.heat_native

    @property
    def overall(self) -> int:
        return round((self.attack + self.defense + self.midfield + self.keeper) / 4)

    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    def reset_stats(self):
        self.goals_for = self.goals_against = 0
        self.wins = self.draws = self.losses = self.points = 0


def _confederation_bias(conf: str) -> tuple:
    """(att, def, mid, gk) 보정값."""
    return {
        'UEFA':     (0,  +2, +2,  0),
        'CONMEBOL': (+3, -1, +1, -1),
        'AFC':      (-1, +1,  0,  0),
        'CAF':      (+2, -1,  0, -1),
        'CONCACAF': (0,  -1,  0,  0),
        'OFC':      (-2, -1, -1, -1),
    }.get(conf, (0, 0, 0, 0))


def overall_from_points(points: float) -> float:
    """FIFA 포인트를 60~92 범위로 매핑."""
    t = (points - MIN_POINTS) / (MAX_POINTS - MIN_POINTS)
    t = max(0.0, min(1.0, t))
    return 60.0 + t * 32.0


def build_team(td: TeamData, color: tuple, secondary: tuple,
               rng: random.Random) -> Team:
    ovr = overall_from_points(td.fifa_points)
    bias = _confederation_bias(td.confederation)

    # 각 스탯은 OVR 기준 ±jitter, 컨페더레이션 보정 추가
    def stat(b: int) -> int:
        v = ovr + b + rng.uniform(-4.5, 4.5)
        return int(max(40, min(99, round(v))))

    return Team(
        data=td,
        attack=stat(bias[0]),
        defense=stat(bias[1]),
        midfield=stat(bias[2]),
        keeper=stat(bias[3]),
        color=color,
        secondary=secondary,
        formation=td.formation,
        secondary_formation=(td.secondary_formation or td.formation),
        style_tag=td.style_tag,
        wc_titles=td.wc_titles,
        squad=SQUADS.get(td.code, []),
    )


# 컨페더레이션별 대표 색 — JSON에 kit_primary가 없을 때 fallback.
CONFEDERATION_COLORS = {
    'UEFA':     (90, 140, 220),
    'CONMEBOL': (240, 200, 60),
    'AFC':      (220, 90, 90),
    'CAF':      (90, 200, 120),
    'CONCACAF': (220, 130, 60),
    'OFC':      (180, 120, 220),
}


def palette_for(td: TeamData, rng: random.Random) -> tuple:
    """JSON에 정의된 실제 유니폼 색 사용. 없으면 컨페더레이션 베이스로 fallback.

    너무 어두운 primary는 시각화에서 잔디 위에 안 보일 수 있어 살짝 밝게 보정.
    """
    primary = td.kit_primary
    secondary = td.kit_secondary

    # primary가 거의 회색/흑색에 가까우면 컨페더레이션 베이스로 대체
    luma = 0.299 * primary[0] + 0.587 * primary[1] + 0.114 * primary[2]
    if luma < 50:
        primary = CONFEDERATION_COLORS.get(td.confederation, (180, 180, 180))

    return primary, secondary
