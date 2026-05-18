"""유전자 정의 + 돌연변이 + 교차(crossover) + 색·무늬 표현형.

각 유전자는 0~1 정규화된 실수. 표현형 변환 시 의미 있는 범위로 매핑.

이 폴더(genetic)는 neural/ 과 명확히 분리:
- neural = 신경망 가중치/메모리/PPO를 진화
- genetic = "스칼라 형질" 만 진화, 행동은 하드코딩 규칙
  형질 다양성·성선택·hawk-dove 게임이론에 집중.
"""
import random
from dataclasses import dataclass, replace


# 표현형 매핑 범위
SPEED_RANGE = (0.4, 3.5)          # px/tick
VISION_RANGE = (10.0, 90.0)       # 감지 반경 px
SIZE_RANGE = (3.0, 9.0)           # 반지름 px
METABOLISM_RANGE = (0.05, 0.30)   # tick당 기본 에너지 소모
LIFESPAN_RANGE = (700, 2400)      # 최대 수명 tick
REPRO_THRESHOLD_RANGE = (80.0, 160.0)  # 번식 임계 에너지
PATTERN_RANGE = (0, 5)            # 줄무늬 개수 (성선택 시그널, 비용 있음)

MUTATION_STD = 0.07
MUTATION_RATE = 0.85
CROSSOVER_BLEND = 0.15            # 0=단일점 선택, 1=완전 평균. 중간이면 부분 블렌딩


@dataclass(frozen=True)
class Genome:
    speed: float
    vision: float
    size: float
    metabolism: float
    aggression: float       # hawk-dove: 0=dove, 1=hawk
    lifespan: float         # 최대 수명
    repro_threshold: float  # 번식 시작 임계
    pattern: float          # 줄무늬(성선택 ornament — 비용 있음)

    @staticmethod
    def random(rng: random.Random) -> 'Genome':
        return Genome(
            speed=rng.random(),
            vision=rng.random(),
            size=rng.random(),
            metabolism=rng.random(),
            aggression=rng.random(),
            lifespan=rng.random(),
            repro_threshold=rng.random(),
            pattern=rng.random(),
        )

    # ── 돌연변이 & 교차 ──

    def mutate(self, rng: random.Random) -> 'Genome':
        def jitter(v: float) -> float:
            if rng.random() > MUTATION_RATE:
                return v
            return max(0.0, min(1.0, v + rng.gauss(0.0, MUTATION_STD)))
        return Genome(
            speed=jitter(self.speed),
            vision=jitter(self.vision),
            size=jitter(self.size),
            metabolism=jitter(self.metabolism),
            aggression=jitter(self.aggression),
            lifespan=jitter(self.lifespan),
            repro_threshold=jitter(self.repro_threshold),
            pattern=jitter(self.pattern),
        )

    @staticmethod
    def crossover(a: 'Genome', b: 'Genome', rng: random.Random) -> 'Genome':
        """부모 두 유전체의 자식을 만든다. 유전자별 50/50 선택 + 약간의 블렌딩."""
        def mix(va: float, vb: float) -> float:
            pick = va if rng.random() < 0.5 else vb
            blended = pick * (1 - CROSSOVER_BLEND) + ((va + vb) * 0.5) * CROSSOVER_BLEND
            return max(0.0, min(1.0, blended))
        return Genome(
            speed=mix(a.speed, b.speed),
            vision=mix(a.vision, b.vision),
            size=mix(a.size, b.size),
            metabolism=mix(a.metabolism, b.metabolism),
            aggression=mix(a.aggression, b.aggression),
            lifespan=mix(a.lifespan, b.lifespan),
            repro_threshold=mix(a.repro_threshold, b.repro_threshold),
            pattern=mix(a.pattern, b.pattern),
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

    def lifespan_ticks(self) -> int:
        a, b = LIFESPAN_RANGE
        return int(a + self.lifespan * (b - a))

    def repro_threshold_energy(self) -> float:
        a, b = REPRO_THRESHOLD_RANGE
        return a + self.repro_threshold * (b - a)

    def pattern_count(self) -> int:
        a, b = PATTERN_RANGE
        return int(round(a + self.pattern * (b - a)))

    def metabolism_per_tick(self) -> float:
        """기본 소모 + 크기·속도·무늬(ornament) 비례 보정."""
        a, b = METABOLISM_RANGE
        base = a + self.metabolism * (b - a)
        size_factor = (self.size_px() / 6.0) ** 2 * 0.04
        speed_factor = (self.speed_px() / 2.0) ** 2 * 0.05
        pattern_factor = self.pattern * 0.025   # 화려한 ornament 비용
        return base + size_factor + speed_factor + pattern_factor

    def display_fitness(self) -> float:
        """성선택 시그널 — 암컷이 보는 수컷의 매력도.
        size + vision + pattern의 가중합. 모두 비용 있는 ornament."""
        return (self.size * 0.4 + self.vision * 0.2 +
                self.pattern * 0.4)

    def color(self) -> tuple:
        """유전자 → RGB. 시야는 R, 속도는 G, 크기는 B 채널."""
        r = int(80 + self.vision * 175)
        g = int(80 + self.speed * 175)
        b = int(80 + self.size * 175)
        return (r, g, b)

    def summary(self) -> str:
        return (f'spd={self.speed_px():.1f}  vis={self.vision_px():.0f}  '
                f'sz={self.size_px():.1f}  met={self.metabolism_per_tick():.3f}  '
                f'agg={self.aggression:.2f}  life={self.lifespan_ticks()}')
