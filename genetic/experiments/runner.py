"""헤드리스 배치 러너 — 여러 시드를 독립 실행 후 결과 집계."""
import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from ..world import World, WorldConfig


@dataclass
class RunResult:
    seed: int
    final_tick: int
    final_pop: int
    max_generation: int
    history: List[dict]            # tick별 통계
    final_trait_means: dict        # biome별 trait 평균
    final_hawk_ratio: float
    avg_pattern_late: float        # 마지막 25% 구간 평균 (수렴 추정치)
    avg_hawk_ratio_late: float
    biome_pop_late: tuple
    runtime_sec: float


def run_one(seed: int, cfg: WorldConfig, ticks: int) -> RunResult:
    t0 = time.perf_counter()
    w = World(seed=seed, config=cfg)
    for _ in range(ticks):
        if w.population() == 0:
            break
        w.step()
    runtime = time.perf_counter() - t0

    hist = w.history
    late_start = max(0, len(hist) - len(hist) // 4)
    late = hist[late_start:] if hist else []
    avg_pattern_late = (
        sum(d['avg_pattern'] for d in late) / max(1, len(late)) if late else 0.0
    )
    avg_hawk_ratio_late = (
        sum(d['hawk_ratio'] for d in late) / max(1, len(late)) if late else 0.0
    )

    return RunResult(
        seed=seed,
        final_tick=w.tick,
        final_pop=w.population(),
        max_generation=w.max_generation(),
        history=hist,
        final_trait_means=w.trait_means_per_biome(),
        final_hawk_ratio=hist[-1]['hawk_ratio'] if hist else 0.0,
        avg_pattern_late=avg_pattern_late,
        avg_hawk_ratio_late=avg_hawk_ratio_late,
        biome_pop_late=w.biome_populations(),
        runtime_sec=runtime,
    )


def batch_run(cfg: WorldConfig, seeds: Sequence[int], ticks: int,
              label: str = '') -> List[RunResult]:
    results = []
    for i, s in enumerate(seeds):
        r = run_one(s, cfg, ticks)
        results.append(r)
        print(f'  [{label}] seed {s} ({i+1}/{len(seeds)}) — '
              f'pop={r.final_pop} gen={r.max_generation} '
              f'pattern_late={r.avg_pattern_late:.3f} '
              f'hawk_late={r.avg_hawk_ratio_late:.3f} '
              f'({r.runtime_sec:.1f}s)')
    return results


def mean_std(values: Sequence[float]) -> tuple:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) <= 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)
