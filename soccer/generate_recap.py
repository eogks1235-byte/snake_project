"""저장된 MC 결과의 selected_seed로 토너먼트를 재시뮬해서
득점왕/도움왕 라인차트 영상만 mp4로 추출.

사용:
  python soccer/generate_recap.py recordings/mc_20260512_064118_n5000_thriller
  python soccer/generate_recap.py --seed 900715
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from soccer.tournament import Tournament
    from soccer.match_engine import play_full_fast
    from soccer.monte_carlo import match_seed_offset
    from soccer.render import Renderer
    from soccer.recorder import VideoRecorder
else:
    from .tournament import Tournament
    from .match_engine import play_full_fast
    from .monte_carlo import match_seed_offset
    from .render import Renderer
    from .recorder import VideoRecorder


RECAP_BUILD_FRAMES = 270
RECAP_HOLD_FRAMES = 90
RECORD_FPS = 30


def simulate_tournament(seed: int) -> Tournament:
    t = Tournament(seed)
    while t.has_next_match():
        mid, h, a = t.peek_next()
        rng = random.Random(seed * 7919 + match_seed_offset(mid))
        cond = t.conditions_of(mid)
        res = play_full_fast(
            h, a, knockout=t.is_knockout(mid), rng=rng,
            altitude=cond['altitude'], hot=cond['hot'],
            last_round_push=cond['last_round_push'],
            home_familiar=t.familiar(h.code, cond.get('venue', '')),
            away_familiar=t.familiar(a.code, cond.get('venue', '')),
            home_starting_stamina=t.starting_stamina_for(h.code),
            away_starting_stamina=t.starting_stamina_for(a.code),
        )
        t.record_result(mid, res)
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mc_dir', nargs='?', default=None,
                        help='recordings/mc_... 폴더 경로 (summary.json 읽음)')
    parser.add_argument('--seed', type=int, default=None,
                        help='시드 직접 지정 (mc_dir 무시)')
    parser.add_argument('--out', type=str, default=None,
                        help='출력 mp4 경로 (기본: mc_dir/recap_topscorers.mp4)')
    args = parser.parse_args()

    if args.seed is not None:
        seed = args.seed
        out_path = args.out or f'recordings/recap_seed{seed}.mp4'
    elif args.mc_dir:
        summary = json.loads((Path(args.mc_dir) / 'summary.json').read_text(encoding='utf-8'))
        seed = summary['meta']['selected_seed']
        out_path = args.out or str(Path(args.mc_dir) / 'recap_topscorers.mp4')
    else:
        print('mc_dir 또는 --seed 중 하나 필요')
        sys.exit(1)

    print(f'[RECAP] seed={seed} → simulating tournament...')
    t = simulate_tournament(seed)
    print(f'[RECAP] simulated {len(t.stat_history)} matches')

    import pygame
    renderer = Renderer()
    recorder = VideoRecorder(out_path, fps=RECORD_FPS)
    print(f'[RECAP] recording → {out_path}')

    total = RECAP_BUILD_FRAMES + RECAP_HOLD_FRAMES
    for frame in range(1, total + 1):
        # pygame 이벤트 펌프 (창 비응답 방지)
        for _ in pygame.event.get():
            pass
        progress = min(1.0, frame / RECAP_BUILD_FRAMES)
        renderer.draw_stats_recap(t, progress)
        recorder.capture(renderer.screen)

    recorder.close()
    print(f'[RECAP] saved {recorder.frame_count} frames '
          f'({recorder.duration_sec:.1f}s) → {out_path}')
    renderer.quit()


if __name__ == '__main__':
    main()
