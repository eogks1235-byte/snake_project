"""CLI: python -m genetic.experiments [--only a|b|c|d|...] [--seeds N] [--ticks N]"""
import argparse
import sys
import time
from pathlib import Path

# Windows 콘솔 cp949 회피 — UTF-8 강제
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# 직접 실행 지원
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from genetic.experiments.experiments import ALL_EXPERIMENTS, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description='Genetic evolution research-mode experiments.')
    parser.add_argument('--only', type=str, default='abcd',
                        help='실행할 실험 선택 (예: ab, c, abcd 기본 = 모두)')
    parser.add_argument('--seeds', type=int, default=8,
                        help='각 조건당 시드 수 (기본 8)')
    parser.add_argument('--ticks', type=int, default=1500,
                        help='시뮬 길이 (기본 1500)')
    parser.add_argument('--seed-base', type=int, default=1000,
                        help='시드 시작 값')
    args = parser.parse_args()

    selected = [c for c in args.only.lower() if c in ALL_EXPERIMENTS]
    if not selected:
        print(f'no valid experiment in --only={args.only}. 유효: a,b,c,d')
        sys.exit(1)

    seeds = list(range(args.seed_base, args.seed_base + args.seeds))
    print(f'== experiments={selected}, seeds={args.seeds}, ticks={args.ticks} ==')
    print(f'== results dir: {RESULTS_DIR} ==')

    t0 = time.perf_counter()
    summary = {}
    for key in selected:
        summary[key] = ALL_EXPERIMENTS[key](seeds, args.ticks)
    elapsed = time.perf_counter() - t0
    print(f'\n== all done in {elapsed:.1f}s ==')
    print(f'== outputs (csv + png) in {RESULTS_DIR} ==')


if __name__ == '__main__':
    main()
