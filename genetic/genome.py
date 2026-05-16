"""유전자 정의 + 돌연변이 + 색 매핑.

각 유전자는 0~1 정규화된 실수. 표현형 변환 시 의미 있는 범위로 매핑.
"""
import random
from dataclasses import dataclass, replace


# 표현형 매핑 범위
SPEED_RANGE = (0.4, 3.5)         # px/tick
VISION_RANGE = (10.0, 90.0)      # 감지 반경 px
SIZE_RANGE = (3.0, 9.0)          # 반지름 px (큰 개체는 음식 효율↑, 에너지 소모↑)
METABOLISM_RANGE = (0.05, 0.30)  # tick당 기본 에너지 소모

MUTATION_STD = 0.08              # 돌연변이 표준편차 (정규화 공간)
MUTATION_RATE = 0.85             # 한 자식에서 평균 85% 유전자가 살짝 변형됨


@dataclass(frozen=True)
class Genome:
    speed: float          # 0~1
    vision: float         # 0~1
    size: float           # 0~1
    metabolism: float     # 0~1 (낮을수록 효율 좋음)
    aggression: float     # 0~1 (다른 개체와 마주칠 때 반응 — 미래 확장용)

    @staticmethod
    def random(rng: random.Random) -> 'Genome':
        return Genome(
            speed=rng.random(),
            vision=rng.random(),
            size=rng.random(),
            metabolism=rng.random(),
            aggression=rng.random(),
        )

    def mutate(self, rng: random.Random) -> 'Genome':
        def jitter(v: float) -> float:
            if rng.random() > MUTATION_RATE:
                return v
            return max(0.0, min(1.0, v + rng.gauss(0.0, MUTATION_STD)))
        return replace(
            self,
            speed=jitter(self.speed),
            vision=jitter(self.vision),
            size=jitter(self.size),
            metabolism=jitter(self.metabolism),
            aggression=jitter(self.aggression),
        )

    # ── 표현형 ──

    def speed_px(self) -> float:
        a, b = SPEED_RANGE
        return a + self.speed * (b - a)

    def vision_px(self) -> float:
        a, b = VISION_RANGE
        return a + self.vision * (b - a)

    def size_px(self) -> float:
        a, b = SIZE_RANGE
        return a + self.size * (b - a)

    def metabolism_per_tick(self) -> float:
        """기본 소모 + 크기·속도 비례 보정."""
        a, b = METABOLISM_RANGE
        base = a + self.metabolism * (b - a)
        size_factor = (self.size_px() / 6.0) ** 2 * 0.04
        speed_factor = (self.speed_px() / 2.0) ** 2 * 0.05
        return base + size_factor + speed_factor

    def color(self) -> tuple:
        """유전자 → RGB. 시야는 R, 속도는 G, 크기는 B 채널."""
        r = int(80 + self.vision * 175)
        g = int(80 + self.speed * 175)
        b = int(80 + self.size * 175)
        return (r, g, b)

    def summary(self) -> str:
        return (f'spd={self.speed_px():.1f}  vis={self.vision_px():.0f}  '
                f'sz={self.size_px():.1f}  met={self.metabolism_per_tick():.3f}')
