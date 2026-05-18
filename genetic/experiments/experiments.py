"""실험 정의 — 4개. 각 실험은 (1) sweep 정의, (2) batch_run, (3) plot+CSV."""
import csv
import math
from dataclasses import replace
from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use('Agg')  # 헤드리스
import matplotlib.pyplot as plt

from ..world import (WorldConfig, BIOME_COLD, BIOME_TEMPERATE, BIOME_WARM)
from .runner import batch_run, mean_std, RunResult


RESULTS_DIR = Path(__file__).resolve().parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Experiment B — 계절 주기 vs 성선택 ornament
# 가설: 긴 안정기일수록 성선택 ornament(pattern)이 폭주, 짧은 변동성에선 비용 부담으로 억제.
# ──────────────────────────────────────────────────────────────

def experiment_B(seeds: Sequence[int], ticks: int) -> dict:
    print('\n=== Experiment B: season period vs sexual ornament ===')
    season_periods = [200, 600, 1200, 2400, 4800]
    rows = []
    aggregated = {}

    for period in season_periods:
        cfg = WorldConfig(season_period=period)
        results = batch_run(cfg, seeds, ticks, label=f'B/period={period}')
        patterns = [r.avg_pattern_late for r in results]
        pops = [r.final_pop for r in results]
        gens = [r.max_generation for r in results]
        m, s = mean_std(patterns)
        pm, ps = mean_std(pops)
        aggregated[period] = (m, s, patterns)
        rows.append({
            'season_period': period,
            'pattern_mean': m, 'pattern_std': s,
            'pop_mean': pm, 'pop_std': ps,
            'gen_mean': mean_std(gens)[0],
            'n_seeds': len(seeds),
        })
        print(f'  → period={period}: pattern={m:.3f} ± {s:.3f} (pop={pm:.0f})')

    _write_csv(rows, RESULTS_DIR / 'exp_B_seasons.csv')

    # 플롯
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = sorted(aggregated.keys())
    means = [aggregated[x][0] for x in xs]
    stds = [aggregated[x][1] for x in xs]
    ax.errorbar(xs, means, yerr=stds, marker='o', capsize=4,
                color='#c45a8c', linewidth=1.5)
    for x, vals in aggregated.items():
        ax.scatter([x] * len(vals[2]), vals[2], color='#c45a8c',
                   alpha=0.25, s=18)
    ax.set_xscale('log')
    ax.set_xlabel('season period (ticks, log scale)')
    ax.set_ylabel('avg pattern (late 25% mean)')
    ax.set_title('Experiment B — Sexual ornament under varying season periods')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'exp_B_seasons.png', dpi=130)
    plt.close(fig)
    return {'rows': rows, 'aggregated': aggregated}


# ──────────────────────────────────────────────────────────────
# Experiment C — 이주 비용 vs 종 분화 (biome 간 trait 분리)
# 가설: migration_cost가 클수록 biome 별 trait 평균이 갈라진다.
#       between-biome variance / within-biome variance 비율로 측정.
# ──────────────────────────────────────────────────────────────

def experiment_C(seeds: Sequence[int], ticks: int) -> dict:
    print('\n=== Experiment C: migration cost vs speciation ===')
    costs = [0.0, 0.5, 2.0, 5.0, 12.0, 30.0]
    rows = []
    aggregated = {}

    for cost in costs:
        cfg = WorldConfig(migration_cost=cost)
        results = batch_run(cfg, seeds, ticks, label=f'C/cost={cost}')
        ratios = [_speciation_ratio(r) for r in results]
        m, s = mean_std(ratios)
        aggregated[cost] = (m, s, ratios)
        rows.append({
            'migration_cost': cost,
            'speciation_ratio_mean': m,
            'speciation_ratio_std': s,
            'n_seeds': len(seeds),
        })
        print(f'  → cost={cost}: speciation_ratio={m:.3f} ± {s:.3f}')

    _write_csv(rows, RESULTS_DIR / 'exp_C_migration.csv')

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = sorted(aggregated.keys())
    means = [aggregated[x][0] for x in xs]
    stds = [aggregated[x][1] for x in xs]
    ax.errorbar(xs, means, yerr=stds, marker='s', capsize=4,
                color='#5a9cc4', linewidth=1.5)
    for x, vals in aggregated.items():
        ax.scatter([x] * len(vals[2]), vals[2], color='#5a9cc4',
                   alpha=0.25, s=18)
    ax.set_xlabel('migration cost (energy per biome crossing)')
    ax.set_ylabel('between-biome / within-biome trait variance')
    ax.set_title('Experiment C — Speciation pressure from migration cost')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'exp_C_migration.png', dpi=130)
    plt.close(fig)
    return {'rows': rows, 'aggregated': aggregated}


def _speciation_ratio(r: RunResult) -> float:
    """biome별 trait 평균의 분산(between) ÷ 풀(pooled) 평균 분산.
    값이 클수록 biome 간 trait가 갈라진 것 = speciation."""
    traits = ['speed', 'vision', 'size', 'metabolism', 'aggression']
    means_by_biome = r.final_trait_means
    used = [b for b, m in means_by_biome.items() if m is not None and m['count'] >= 3]
    if len(used) < 2:
        return 0.0

    totals = []
    for t in traits:
        vals = [means_by_biome[b][t] for b in used]
        mu = sum(vals) / len(vals)
        between = sum((v - mu) ** 2 for v in vals) / len(vals)
        # within-biome variance proxy: trait는 0~1 정규화이므로 분산 상한 0.25
        # 작은 within이면 ratio 폭발 → 안전한 floor 0.01
        within = 0.01
        totals.append(between / within)
    return sum(totals) / len(totals)


# ──────────────────────────────────────────────────────────────
# Experiment D — 음식 비대칭 vs Hawk 비율 ESS 이동
# 가설: biome 간 음식 분배가 비대칭일수록 자원 경쟁 격화 → 매 비율 증가.
# ──────────────────────────────────────────────────────────────

def experiment_D(seeds: Sequence[int], ticks: int) -> dict:
    print('\n=== Experiment D: food asymmetry vs hawk ratio ===')
    # 비대칭 정도 — 한 biome에 집중되는 정도. asym ∈ [0, 1] (0=대칭, 1=극단)
    asyms = [0.0, 0.25, 0.5, 0.75, 0.9]
    rows = []
    aggregated = {}

    for asym in asyms:
        # asym=0: (1,1,1). asym=1: (~0, ~0, ~3). 합은 동일 유지.
        warm = 1.0 + 2.0 * asym
        cold = max(0.05, 1.0 - asym)
        temp = max(0.05, 1.0 - asym * 0.5)
        # 정규화 (합 = 3)
        total = warm + cold + temp
        weights = (cold * 3 / total, temp * 3 / total, warm * 3 / total)

        cfg = WorldConfig(biome_food_weight=weights)
        results = batch_run(cfg, seeds, ticks, label=f'D/asym={asym}')
        hawks = [r.avg_hawk_ratio_late for r in results]
        m, s = mean_std(hawks)
        aggregated[asym] = (m, s, hawks, weights)
        rows.append({
            'asymmetry': asym,
            'weights_cold': weights[0],
            'weights_temp': weights[1],
            'weights_warm': weights[2],
            'hawk_ratio_mean': m,
            'hawk_ratio_std': s,
            'n_seeds': len(seeds),
        })
        print(f'  → asym={asym} weights={weights}: hawk={m:.3f} ± {s:.3f}')

    _write_csv(rows, RESULTS_DIR / 'exp_D_asymmetry.csv')

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = sorted(aggregated.keys())
    means = [aggregated[x][0] for x in xs]
    stds = [aggregated[x][1] for x in xs]
    ax.errorbar(xs, means, yerr=stds, marker='D', capsize=4,
                color='#c4795a', linewidth=1.5, label='observed')
    for x, vals in aggregated.items():
        ax.scatter([x] * len(vals[2]), vals[2], color='#c4795a',
                   alpha=0.25, s=18)
    # ESS 이론치 (V=6, C=8): p_hawk = V/C = 0.75. 대칭 환경 기준.
    ax.axhline(0.75, ls='--', color='gray', alpha=0.6, label='ESS V/C = 0.75')
    ax.axhline(0.5, ls=':', color='lightgray', alpha=0.5)
    ax.set_xlabel('food asymmetry (0 = uniform, 1 = concentrated in warm)')
    ax.set_ylabel('hawk ratio (late 25% mean)')
    ax.set_title('Experiment D — Hawk-Dove ESS shift under resource asymmetry')
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'exp_D_asymmetry.png', dpi=130)
    plt.close(fig)
    return {'rows': rows, 'aggregated': aggregated}


# ──────────────────────────────────────────────────────────────
# Experiment A — Trait vs Policy 적응 속도 비교 (genetic vs neural)
# 가설: 단순 환경에서는 trait 진화(genetic)가 더 빨리 안정 인구를 회복,
#       복잡 환경에서는 policy 진화(neural)가 더 좋은 적응.
# 공통 metric: 인구 회복 속도 + max_generation 증가율.
# ──────────────────────────────────────────────────────────────

def experiment_A(seeds: Sequence[int], ticks: int) -> dict:
    print('\n=== Experiment A: trait vs policy adaptation speed ===')

    # genetic: 기본 환경
    print('  -- genetic --')
    genetic_results = batch_run(WorldConfig(), seeds, ticks, label='A/genetic')

    # neural: import 시도. 실패하면 skip.
    try:
        from neural.world import World as NeuralWorld
        neural_available = True
    except Exception as e:
        print(f'  [WARN] neural module import 실패: {e}. neural 부분 skip.')
        neural_available = False

    neural_traj = []
    if neural_available:
        print('  -- neural (curriculum=False, dynamic_env=False) --')
        for i, s in enumerate(seeds):
            try:
                nw = NeuralWorld(seed=s, curriculum=False, dynamic_env=False)
            except TypeError:
                # 인자 시그니처가 다르면 최소 인자만
                nw = NeuralWorld(seed=s)
            traj = []
            for t in range(ticks):
                if nw.population() == 0:
                    break
                nw.step()
                if t % 10 == 0:
                    traj.append({
                        'tick': nw.tick,
                        'pop': nw.population(),
                        'max_gen': nw.max_generation(),
                    })
            neural_traj.append(traj)
            print(f'  [A/neural] seed {s} ({i+1}/{len(seeds)}) — '
                  f'pop={nw.population()} gen={nw.max_generation()}')

    # genetic trajectory를 같은 간격으로 다운샘플
    genetic_traj = []
    for r in genetic_results:
        traj = [{'tick': d['tick'], 'pop': d['count'], 'max_gen': int(d['avg_gen'])}
                for i, d in enumerate(r.history) if i % 10 == 0]
        genetic_traj.append(traj)

    # 평균 trajectory 계산
    def avg_trajectory(trajs, key):
        if not trajs:
            return [], [], []
        max_len = min(len(t) for t in trajs)
        xs, means, stds = [], [], []
        for i in range(max_len):
            xs.append(trajs[0][i]['tick'])
            vs = [t[i][key] for t in trajs]
            m, s = mean_std(vs)
            means.append(m)
            stds.append(s)
        return xs, means, stds

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    gxs, gpm, gps = avg_trajectory(genetic_traj, 'pop')
    ax1.plot(gxs, gpm, color='#c45a8c', label='genetic (trait evolution)', linewidth=2)
    ax1.fill_between(gxs, [m - s for m, s in zip(gpm, gps)],
                     [m + s for m, s in zip(gpm, gps)],
                     color='#c45a8c', alpha=0.2)
    if neural_traj:
        nxs, npm, nps = avg_trajectory(neural_traj, 'pop')
        ax1.plot(nxs, npm, color='#5a9cc4', label='neural (policy evolution)', linewidth=2)
        ax1.fill_between(nxs, [m - s for m, s in zip(npm, nps)],
                         [m + s for m, s in zip(npm, nps)],
                         color='#5a9cc4', alpha=0.2)
    ax1.set_xlabel('tick')
    ax1.set_ylabel('population (mean ± std)')
    ax1.set_title('Population trajectory')
    ax1.legend(); ax1.grid(alpha=0.3)

    gxs2, ggm, ggs = avg_trajectory(genetic_traj, 'max_gen')
    ax2.plot(gxs2, ggm, color='#c45a8c', label='genetic', linewidth=2)
    ax2.fill_between(gxs2, [m - s for m, s in zip(ggm, ggs)],
                     [m + s for m, s in zip(ggm, ggs)],
                     color='#c45a8c', alpha=0.2)
    if neural_traj:
        nxs2, ngm, ngs = avg_trajectory(neural_traj, 'max_gen')
        ax2.plot(nxs2, ngm, color='#5a9cc4', label='neural', linewidth=2)
        ax2.fill_between(nxs2, [m - s for m, s in zip(ngm, ngs)],
                         [m + s for m, s in zip(ngm, ngs)],
                         color='#5a9cc4', alpha=0.2)
    ax2.set_xlabel('tick')
    ax2.set_ylabel('avg generation (mean ± std)')
    ax2.set_title('Generation turnover (adaptation speed proxy)')
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle('Experiment A — Trait evolution vs Policy evolution', fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'exp_A_compare.png', dpi=130)
    plt.close(fig)

    # CSV 요약
    rows = []
    rows.append({
        'system': 'genetic',
        'final_pop_mean': mean_std([r.final_pop for r in genetic_results])[0],
        'max_gen_mean': mean_std([r.max_generation for r in genetic_results])[0],
        'n_seeds': len(seeds),
    })
    if neural_traj:
        final_pops = [t[-1]['pop'] for t in neural_traj if t]
        final_gens = [t[-1]['max_gen'] for t in neural_traj if t]
        rows.append({
            'system': 'neural',
            'final_pop_mean': mean_std(final_pops)[0],
            'max_gen_mean': mean_std(final_gens)[0],
            'n_seeds': len(seeds),
        })
    _write_csv(rows, RESULTS_DIR / 'exp_A_compare.csv')
    return {'rows': rows, 'genetic_traj': genetic_traj, 'neural_traj': neural_traj}


# ──────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────

def _write_csv(rows: List[dict], path: Path):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'  [CSV] {path.name}')


ALL_EXPERIMENTS = {
    'a': experiment_A,
    'b': experiment_B,
    'c': experiment_C,
    'd': experiment_D,
}
