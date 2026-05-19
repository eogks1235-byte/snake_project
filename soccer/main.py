"""실행 진입점 — 2026 월드컵 시뮬.

조작:
  SPACE  : Fast forward
  N      : 현재 경기 즉시 종결 → 다음 경기
  R      : 새 토너먼트 (시드 새로 뽑음)
  V      : 녹화 토글
  ESC    : 종료

CLI:
  python soccer/main.py                                # 모든 경기 시각화
  python soccer/main.py --seed 1234                    # 시드 고정
  python soccer/main.py --watch groupA                 # A조 6경기만 시각화
  python soccer/main.py --watch r32                    # 32강 16경기만
  python soccer/main.py --watch r16                    # 16강 8경기만
  python soccer/main.py --watch final                  # 결승만
  python soccer/main.py --real soccer/real_results.json
                                                       # 실제 결과 주입
  python soccer/main.py --mc 5000                      # 몬테카를로 5000회
  python soccer/main.py --mc --real ... --watch r16    # 종합 사용
  python soccer/main.py --mc 5000 --select realistic --seed 2026  --record
"""
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from soccer.tournament import Tournament
    from soccer.match_engine import Match, play_full_fast
    from soccer.render import Renderer, ShortsRenderer
    from soccer.recorder import VideoRecorder
    from soccer.monte_carlo import (
        run_montecarlo, save_selected_tournament, save_summary, match_seed_offset,
    )
else:
    from .tournament import Tournament
    from .match_engine import Match, play_full_fast
    from .render import Renderer, ShortsRenderer
    from .recorder import VideoRecorder
    from .monte_carlo import (
        run_montecarlo, save_selected_tournament, save_summary, match_seed_offset,
    )

import pygame


FPS_NORMAL = 30
FPS_FAST = 240
RECORD_FPS = 30           # 시뮬 fps와 일치 (이전 60: 2배속 재생 버그)
HOLD_AFTER_MATCH = 60      # 한 경기 끝난 뒤 결과 보여주는 프레임
HOLD_AFTER_FINAL = 240     # 결승 끝난 뒤 우승 화면 유지
RECAP_BUILD_FRAMES = 270   # 라인차트 build-up 프레임 (~9초)
RECAP_HOLD_FRAMES = 90     # 완성된 차트 정지 프레임 (~3초)
BESTXI_BUILD_FRAMES = 240  # Best XI 11명 순차 등장 (~8초)
BESTXI_HOLD_FRAMES = 120   # 완성된 Best XI 정지 (~4초)


def should_visualize(match_id: str, watch: str) -> bool:
    """--watch 필터 매칭. None/all/'' → 모두 시각화."""
    if not watch or watch == 'all':
        return True
    f = watch.lower()
    if f == 'groups':
        return match_id.startswith('G')
    if f.startswith('group') and len(f) == 6:    # groupA ~ groupL
        letter = f[5].upper()
        return match_id.startswith('G') and match_id[1] == letter
    if not match_id.startswith('M'):
        return False
    n = int(match_id[1:])
    ranges = {
        'r32':      (73, 88),
        'r16':      (89, 96),
        'qf':       (97, 100),
        'sf':       (101, 102),
        'bronze':   (103, 103),
        'final':    (104, 104),
        'knockout': (73, 104),
    }
    rng = ranges.get(f)
    if rng:
        return rng[0] <= n <= rng[1]
    print(f'[WARN] unknown --watch filter: {watch}  (showing all)')
    return True


def load_real_results(path_str: str) -> dict:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        print(f'[WARN] --real file not found: {path}')
        return {}
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('results', {})


def make_session_dir() -> Path:
    """매치별 mp4를 모을 세션 폴더 — recordings/soccer_<timestamp>/"""
    base = Path(__file__).resolve().parent.parent / 'recordings'
    base.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = base / f'soccer_{stamp}'
    out.mkdir(exist_ok=True)
    return out


def announce_tournament(t: Tournament):
    print(f'[TOURNAMENT] seed={t.seed}')
    for letter, teams in t.group_teams.items():
        line = ' / '.join(f'{tm.code}({tm.formation})' for tm in teams)
        print(f'  Group {letter}:  {line}')
    pre_recorded = [k for k in t.results if k]
    print(f'  total matches: {len(t.match_queue)}  '
          f'(pre-recorded from --real: {len(pre_recorded)})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--record', action='store_true',
                        help='매치마다 mp4 자동 저장 (territory 스타일)')
    parser.add_argument('--mc', type=int, nargs='?', const=5000, default=None,
                        help='몬테카를로 N회 (default 5000)')
    parser.add_argument('--select', type=str, default='drama',
                        help='MC 선정 기준: drama (기본) / thriller / '
                             'realistic / team:CODE (예: team:KOR)')
    parser.add_argument('--no-visualize', action='store_true',
                        help='--mc와 함께 사용 시 시각화 건너뛰고 저장만')
    parser.add_argument('--watch', type=str, default=None,
                        help='시각화 필터: groupA~groupL / groups / r32 / r16 / qf / sf / bronze / final / knockout')
    parser.add_argument('--real', type=str, default=None,
                        help='실제 결과 JSON 경로 (예: soccer/real_results.json)')
    parser.add_argument('--shorts', action='store_true',
                        help='YouTube Shorts 9:16 세로 모드 (1080×1920)')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)
    real_results = load_real_results(args.real)
    if real_results:
        print(f'[REAL] loaded {len(real_results)} real match results from {args.real}')

    # ── 몬테카를로 모드 ────────────────────────────────
    if args.mc:
        print(f'[MC] running {args.mc} headless tournaments  '
              f'(base_seed={seed}, real={len(real_results)}, select={args.select})')
        mc_result = run_montecarlo(seed, args.mc, real_results=real_results,
                                    verbose=True, strategy=args.select)

        # 폴더 이름에 strategy 표시
        sel_tag = mc_result['strategy'].replace(':', '_')
        out_dir = (Path(__file__).resolve().parent.parent / 'recordings'
                   / f'mc_{datetime.now().strftime("%Y%m%d_%H%M%S")}_n{args.mc}_{sel_tag}')
        n_saved = save_selected_tournament(mc_result['selected_tournament'], out_dir)
        save_summary(mc_result, out_dir)

        sel_t = mc_result['selected_tournament']
        print(f'[MC] selected seed={mc_result["selected_seed"]}  '
              f'{mc_result["strategy"]}={mc_result["selected_score"]:.1f}')
        print(f'[MC]   champion: {sel_t.champion().name}  ({sel_t.champion().formation})')
        print(f'[MC]   runner-up: {sel_t.runner_up().name}')
        print(f'[MC]   3rd:       {sel_t.third().name}')
        print(f'[MC] saved {n_saved} match files + summary.json to {out_dir}')

        if args.no_visualize:
            sys.exit(0)

        seed = mc_result['selected_seed']

    tournament = Tournament(seed, real_results=real_results)
    announce_tournament(tournament)

    renderer = ShortsRenderer() if args.shorts else Renderer()

    # ── 녹화 세션 ────────────────────────────────────
    record_each = args.record  # V 키로 토글
    session_dir: Path = make_session_dir() if record_each else None
    if record_each:
        print(f'[REC] per-match recordings -> {session_dir}')
    recorder: VideoRecorder | None = None  # 현재 진행 매치 recorder

    fast_forward = False
    running = True

    current_match = None
    current_match_id = None
    hold = 0

    # ── Recap 상태 (토너먼트 종료 후 득점왕/어시왕 → Best XI) ──
    recap_started = False
    recap_frame = 0
    recap_recorder: VideoRecorder | None = None
    recap_done = False
    # Best XI 단계 — 득점왕/어시왕 끝난 뒤 자동 진입
    bestxi_started = False
    bestxi_frame = 0
    bestxi_recorder: VideoRecorder | None = None
    bestxi_done = False

    def open_recorder_for(match_id: str):
        """현재 매치용 새 recorder 시작."""
        nonlocal recorder
        path = session_dir / f'{match_id}.mp4'
        recorder = VideoRecorder(str(path), fps=RECORD_FPS)

    def close_recorder():
        nonlocal recorder
        if recorder is not None:
            print(f'[REC] saved {recorder.output_path.name}  '
                  f'({recorder.frame_count}f, {recorder.duration_sec:.1f}s)')
            recorder.close()
            recorder = None

    def start_next_match():
        """큐에서 다음 매치 — 필터 안 맞으면 헤드리스로 즉시 처리하고 다음으로."""
        nonlocal current_match, current_match_id
        while True:
            if not tournament.has_next_match():
                current_match = None
                current_match_id = None
                return
            match_id, home, away = tournament.peek_next()
            knockout = tournament.is_knockout(match_id)
            rng = random.Random(seed * 7919 + match_seed_offset(match_id))

            cond = tournament.conditions_of(match_id)
            venue = cond.get('venue', '')
            h_fam = tournament.familiar(home.code, venue)
            a_fam = tournament.familiar(away.code, venue)
            h_stam = tournament.starting_stamina_for(home.code)
            a_stam = tournament.starting_stamina_for(away.code)
            if should_visualize(match_id, args.watch):
                current_match_id = match_id
                current_match = Match(home, away, knockout=knockout, rng=rng,
                                       altitude=cond['altitude'],
                                       hot=cond['hot'],
                                       last_round_push=cond['last_round_push'],
                                       home_familiar=h_fam,
                                       away_familiar=a_fam,
                                       home_starting_stamina=h_stam,
                                       away_starting_stamina=a_stam)
                tags = []
                if cond['altitude'] == 'high': tags.append('HIGH-ALT')
                elif cond['altitude'] == 'mid': tags.append('mid-alt')
                if cond['hot']: tags.append('HOT')
                if cond['last_round_push']: tags.append('FINAL-RD')
                tag_str = ('  [' + ','.join(tags) + ']') if tags else ''
                print(f'[MATCH] {tournament.stage_of(match_id):<10} {match_id}: '
                      f'{home.code}({home.formation}) vs '
                      f'{away.code}({away.formation})  '
                      f'(OVR {home.overall} vs {away.overall}){tag_str}')
                if record_each and session_dir is not None:
                    open_recorder_for(match_id)
                return
            # filter mismatch — 헤드리스로 빠르게 처리
            res = play_full_fast(home, away, knockout=knockout, rng=rng,
                                  altitude=cond['altitude'],
                                  hot=cond['hot'],
                                  last_round_push=cond['last_round_push'],
                                  home_familiar=h_fam,
                                  away_familiar=a_fam,
                                  home_starting_stamina=h_stam,
                                  away_starting_stamina=a_stam)
            tournament.record_result(match_id, res)

    def reset_tournament():
        nonlocal seed, tournament, current_match, current_match_id, hold
        close_recorder()
        seed = random.randint(0, 1_000_000)
        tournament = Tournament(seed, real_results=real_results)
        announce_tournament(tournament)
        current_match = None
        current_match_id = None
        hold = 0
        start_next_match()

    start_next_match()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    fast_forward = not fast_forward
                elif event.key == pygame.K_n:
                    # 현재 경기를 내부적으로 끝까지 진행
                    if current_match and not current_match.finished:
                        while not current_match.finished:
                            current_match.tick()
                elif event.key == pygame.K_r:
                    reset_tournament()
                elif event.key == pygame.K_v:
                    # V — 매치별 녹화 토글
                    record_each = not record_each
                    if record_each:
                        if session_dir is None:
                            session_dir = make_session_dir()
                        print(f'[REC] per-match recording ON -> {session_dir}')
                        if current_match is not None and current_match_id is not None:
                            open_recorder_for(current_match_id)
                    else:
                        print('[REC] per-match recording OFF')
                        close_recorder()
                elif event.key == pygame.K_l:
                    # L — 라인차트 recap 재생 (수동 트리거)
                    if tournament.stat_history:
                        recap_started = True
                        recap_frame = 0
                        recap_done = False
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # 매치 진행
        if current_match is not None:
            if not current_match.finished:
                current_match.tick()
            else:
                # 결과 등록 + 잠시 holding 후 다음 매치
                if current_match_id and current_match_id not in tournament.results:
                    tournament.record_result(current_match_id, current_match.result)
                    print(f'  -> {current_match.result.score_str()}  '
                          f'winner: '
                          f'{current_match.result.winner.code if current_match.result.winner else "DRAW"}')
                    hold = HOLD_AFTER_MATCH
                if hold > 0:
                    hold -= 1
                else:
                    # 매치 끝 — 녹화 닫고 다음 매치 시작 (다음 매치는 새 recorder)
                    close_recorder()
                    if tournament.has_next_match():
                        start_next_match()
                    else:
                        current_match_id = None  # 종료 화면 유지
                        current_match = None      # recap 트리거 위해 None 표시

        # ── Recap 모드 (토너 종료 후 자동, 또는 L 키로 수동) ──
        # 토너 종료 + 매치 없음 + 아직 recap 시작 안 함 → 자동 진입
        if (tournament.completed and current_match is None
                and not recap_started and not recap_done):
            recap_started = True
            recap_frame = 0
            if record_each and session_dir is not None:
                recap_path = session_dir / 'recap_topscorers.mp4'
                recap_recorder = VideoRecorder(str(recap_path), fps=RECORD_FPS)
                print(f'[REC] recording recap -> {recap_path.name}')

        if recap_started and not recap_done:
            recap_frame += 1
            progress = min(1.0, recap_frame / RECAP_BUILD_FRAMES)
            renderer.draw_stats_recap(tournament, progress)
            if recap_recorder is not None:
                recap_recorder.capture(renderer.screen)
            if recap_frame >= RECAP_BUILD_FRAMES + RECAP_HOLD_FRAMES:
                recap_done = True
                if recap_recorder is not None:
                    print(f'[REC] saved {recap_recorder.output_path.name} '
                          f'({recap_recorder.frame_count}f, '
                          f'{recap_recorder.duration_sec:.1f}s)')
                    recap_recorder.close()
                    recap_recorder = None
        elif recap_done and not bestxi_done:
            # Best XI 단계 — recap_done 후 자동 진입
            if not bestxi_started:
                bestxi_started = True
                bestxi_frame = 0
                if record_each and session_dir is not None:
                    bestxi_path = session_dir / 'recap_bestxi.mp4'
                    bestxi_recorder = VideoRecorder(str(bestxi_path), fps=RECORD_FPS)
                    print(f'[REC] recording best XI -> {bestxi_path.name}')
            bestxi_frame += 1
            xi_progress = min(1.0, bestxi_frame / BESTXI_BUILD_FRAMES)
            renderer.draw_best_xi_recap(tournament, xi_progress)
            if bestxi_recorder is not None:
                bestxi_recorder.capture(renderer.screen)
            if bestxi_frame >= BESTXI_BUILD_FRAMES + BESTXI_HOLD_FRAMES:
                bestxi_done = True
                if bestxi_recorder is not None:
                    print(f'[REC] saved {bestxi_recorder.output_path.name} '
                          f'({bestxi_recorder.frame_count}f, '
                          f'{bestxi_recorder.duration_sec:.1f}s)')
                    bestxi_recorder.close()
                    bestxi_recorder = None
        else:
            renderer.draw(tournament, current_match, current_match_id, fast_forward)
            if recorder is not None:
                recorder.capture(renderer.screen)

        # 결승 끝 + recap 끝 + 시간 지나면 자동 종료 (--record 시)
        if tournament.completed and current_match is None and record_each:
            if bestxi_done:
                hold -= 1
                if hold <= -HOLD_AFTER_FINAL:
                    running = False

        renderer.tick_clock(FPS_FAST if fast_forward else FPS_NORMAL)

    close_recorder()
    renderer.quit()
    sys.exit(0)


if __name__ == '__main__':
    main()
