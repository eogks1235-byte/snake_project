"""축구 매치 시각화 — 다크 톤 + 픽셀 경기장 + 점 선수 + 이벤트 피드."""
import pygame
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional

from .match_engine import Match, MatchEvent
from .teams import Team
from .tournament import Tournament


# ── 국기 캐시 (flagcdn.com) ───────────────────────────────
FLAG_DIR = Path(__file__).resolve().parent / 'assets' / 'flags'


def _resolve_flag_code(iso2: str, team_code: str) -> str:
    """ISO2 → flagcdn 파일 슬러그. GB는 ENG/SCO/WAL 서브디비전."""
    if iso2 == 'GB':
        if team_code == 'ENG':
            return 'gb-eng'
        if team_code == 'SCO':
            return 'gb-sct'
        if team_code == 'WAL':
            return 'gb-wls'
    return iso2.lower()


def _ensure_flag_png(flag_code: str) -> Optional[Path]:
    """국기 PNG 캐시 — 없으면 flagcdn에서 다운로드."""
    FLAG_DIR.mkdir(parents=True, exist_ok=True)
    path = FLAG_DIR / f'{flag_code}.png'
    if path.exists():
        return path
    url = f'https://flagcdn.com/w320/{flag_code}.png'
    try:
        req = urllib.request.Request(
            url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = r.read()
        path.write_bytes(data)
        print(f'[flag] downloaded {flag_code}')
        return path
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f'[flag] failed to download {flag_code}: {e}')
        return None


# territory와 동일 톤
BG_COLOR = (15, 17, 21)
PANEL_COLOR = (22, 25, 31)
PITCH_GREEN_DARK = (18, 50, 28)
PITCH_GREEN_LIGHT = (24, 64, 36)
PITCH_LINE = (210, 215, 220)
TEXT_PRIMARY = (235, 235, 240)
TEXT_SECONDARY = (140, 145, 160)
ACCENT_GOLD = (255, 215, 90)
PANEL_HEADER = (28, 32, 40)


def compute_rating_labels(match: Match) -> dict:
    """팀별 라이브 평점 max/min 선수에게 라벨 매핑 (팀당 2명, 총 4명).
    반환: {id(player): (text, color)}. max=min이면 제외 (킥오프 직후 등)."""
    labels = {}
    ratings = getattr(match, 'player_ratings', None)
    if not ratings:
        return labels
    for team_idx in (0, 1):
        on = [p for p in match.players
              if p.on_pitch and p.team_idx == team_idx and p.name]
        if len(on) < 2:
            continue
        scored = [(p, ratings.get((p.team_idx, p.name), 6.0)) for p in on]
        hi_p, hi_r = max(scored, key=lambda x: x[1])
        lo_p, lo_r = min(scored, key=lambda x: x[1])
        if hi_r <= lo_r:
            continue
        hi_name = hi_p.name.split()[-1]
        lo_name = lo_p.name.split()[-1]
        labels[id(hi_p)] = (f'★ {hi_name} {hi_r:.1f}', ACCENT_GOLD)
        labels[id(lo_p)] = (f'▼ {lo_name} {lo_r:.1f}', (235, 100, 90))
    return labels

# 2026 월드컵 호스트 3국 컬러 (프레임 액센트)
WC_CAN_RED = (230, 29, 37)
WC_MEX_GREEN = (60, 172, 59)
WC_USA_BLUE = (42, 57, 141)
WC_GOLD = (212, 175, 55)
WC_BAR_H = 8


class Renderer:
    """현재 진행 중인 매치 1개와 토너먼트 진행 상황을 그린다."""

    PITCH_PX_W = 720
    PITCH_PX_H = 450
    HEADER_H = 90
    SIDEBAR_W = 320
    EVENTS_H = 130
    BRACKET_H = 100
    MARGIN = 28

    def __init__(self):
        pygame.init()
        self.window_w = self.PITCH_PX_W + self.SIDEBAR_W + self.MARGIN * 3
        self.window_h = (self.HEADER_H + self.PITCH_PX_H + self.EVENTS_H
                         + self.BRACKET_H + self.MARGIN * 4)
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption('World Cup 2026 — Sim')
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont('malgungothic,arial', 28, bold=True)
        self.font_sub = pygame.font.SysFont('malgungothic,arial', 14)
        self.font_score = pygame.font.SysFont('malgungothic,arial', 38, bold=True)
        self.font_team = pygame.font.SysFont('malgungothic,arial', 18, bold=True)
        self.font_min = pygame.font.SysFont('malgungothic,arial', 13, bold=True)
        self.font_event = pygame.font.SysFont('malgungothic,arial', 13)
        self.font_pin = pygame.font.SysFont('malgungothic,arial', 12, bold=True)
        self.font_stat = pygame.font.SysFont('malgungothic,arial', 12)

        self.pitch_x = self.MARGIN
        self.pitch_y = self.HEADER_H + self.MARGIN
        self.bracket_y = (self.pitch_y + self.PITCH_PX_H + self.MARGIN
                           + self.EVENTS_H + self.MARGIN)

    # ── 외부 진입점 ─────────────────────────────────────────

    def draw(self, tournament: Tournament, match: Optional[Match],
             match_id: Optional[str], fast_forward: bool):
        self.screen.fill(BG_COLOR)
        self._draw_header(tournament, match_id, fast_forward)

        if match is not None:
            self._draw_pitch_panel(match)
            self._draw_scoreboard(match, tournament, match_id)
            self._draw_event_feed(match)
            self._draw_sidebar(tournament, match)
        else:
            self._draw_completion(tournament)

        self._draw_mini_bracket(tournament)
        self._draw_wc_frame()
        pygame.display.flip()

    def _draw_wc_frame(self):
        """2026 월드컵 호스트 3국 컬러 프레임 (상하 3색 바 + 좌우 골드).
        모든 패널 위에 덮어 그려서 항상 보이게."""
        w, h = self.window_w, self.window_h
        third = w // 3
        # 상단: 캐나다 빨강 / 멕시코 초록 / 미국 파랑
        pygame.draw.rect(self.screen, WC_CAN_RED, (0, 0, third, WC_BAR_H))
        pygame.draw.rect(self.screen, WC_MEX_GREEN,
                         (third, 0, third, WC_BAR_H))
        pygame.draw.rect(self.screen, WC_USA_BLUE,
                         (third * 2, 0, w - third * 2, WC_BAR_H))
        # 하단: 좌우 반전
        pygame.draw.rect(self.screen, WC_USA_BLUE,
                         (0, h - WC_BAR_H, third, WC_BAR_H))
        pygame.draw.rect(self.screen, WC_MEX_GREEN,
                         (third, h - WC_BAR_H, third, WC_BAR_H))
        pygame.draw.rect(self.screen, WC_CAN_RED,
                         (third * 2, h - WC_BAR_H, w - third * 2, WC_BAR_H))
        # 좌/우 골드 세로 바
        pygame.draw.rect(self.screen, WC_GOLD,
                         (0, WC_BAR_H, 3, h - WC_BAR_H * 2))
        pygame.draw.rect(self.screen, WC_GOLD,
                         (w - 3, WC_BAR_H, 3, h - WC_BAR_H * 2))

    # ── 득점왕/어시왕 라인차트 (매치별 누적) ─────────────────

    def draw_stats_recap(self, t: Tournament, progress: float):
        """매치 진행에 따른 득점왕/어시왕 라인차트 애니메이션.
        progress: 0.0 ~ 1.0 (0=경기 0개 시점, 1=마지막 경기까지)."""
        self.screen.fill(BG_COLOR)
        hist = getattr(t, 'stat_history', [])
        if not hist:
            txt = self.font_title.render('NO STATS YET', True, TEXT_SECONDARY)
            self.screen.blit(txt, (self.window_w // 2 - txt.get_width() // 2,
                                    self.window_h // 2))
            pygame.display.flip()
            return

        total = len(hist)
        # 현재 progress에서 보여줄 매치 수 (최소 1)
        cur_idx = max(1, int(round(progress * total)))
        cur_idx = min(total, cur_idx)

        # 최종 통계 기준 top N 선수 선정
        final_snap = hist[-1][1]
        top_scorers = sorted(
            final_snap.items(),
            key=lambda kv: (-kv[1]['goals'], -kv[1]['assists'])
        )[:8]
        top_assists = sorted(
            final_snap.items(),
            key=lambda kv: (-kv[1]['assists'], -kv[1]['goals'])
        )[:8]

        # 제목
        title = self.font_title.render(
            'GOLDEN BOOT & ASSIST KING RACE', True, ACCENT_GOLD)
        self.screen.blit(title,
            (self.window_w // 2 - title.get_width() // 2, 14))

        # 진행 라벨 — 현재 매치 단계 표시
        cur_mid = hist[cur_idx - 1][0]
        stage = self._stage_label_for(cur_mid)
        sub = self.font_sub.render(
            f'after {cur_mid}  —  {stage}  '
            f'({cur_idx}/{total} matches)',
            True, TEXT_SECONDARY)
        self.screen.blit(sub,
            (self.window_w // 2 - sub.get_width() // 2, 48))

        # 두 패널 — 좌: 득점, 우: 어시
        chart_top = 80
        chart_h = self.window_h - chart_top - 28
        panel_w = (self.window_w - self.MARGIN * 3) // 2
        left_x = self.MARGIN
        right_x = self.MARGIN * 2 + panel_w

        self._draw_race_panel(
            left_x, chart_top, panel_w, chart_h,
            top_scorers, hist, cur_idx, total, 'goals', t, 'TOP SCORERS')
        self._draw_race_panel(
            right_x, chart_top, panel_w, chart_h,
            top_assists, hist, cur_idx, total, 'assists', t, 'TOP ASSISTS')

        self._draw_wc_frame()
        pygame.display.flip()

    # ── Best XI 화면 (recap 2단계) ────────────────────────────
    def draw_best_xi_recap(self, t: Tournament, progress: float):
        """포메이션 4-3-3 시각화 + 평균 평점 기반 베스트 11.
        progress 0.0~1.0 — 0~1 사이 선수가 순차 등장 (GK → DEF → MID → FWD)."""
        self.screen.fill(BG_COLOR)
        awards = t.compute_awards() if hasattr(t, 'compute_awards') else None
        if not awards or not awards.get('best_xi'):
            txt = self.font_title.render('NO BEST XI YET', True, TEXT_SECONDARY)
            self.screen.blit(txt, (self.window_w // 2 - txt.get_width() // 2,
                                    self.window_h // 2))
            self._draw_wc_frame()
            pygame.display.flip()
            return
        best_xi = awards['best_xi']

        # 타이틀
        title = self.font_title.render(
            'BEST XI  —  TOURNAMENT TEAM', True, ACCENT_GOLD)
        self.screen.blit(title,
            (self.window_w // 2 - title.get_width() // 2, 14))
        sub = self.font_sub.render(
            'selected by average match rating  (4-3-3, min 3 matches)',
            True, TEXT_SECONDARY)
        self.screen.blit(sub,
            (self.window_w // 2 - sub.get_width() // 2, 48))

        # 피치 — 세로형 (공격 위쪽)
        pitch_top = 80
        pitch_h = self.window_h - pitch_top - 28
        pitch_w = int(self.window_w * 0.60)
        pitch_x = (self.window_w - pitch_w) // 2 - 130
        pitch_y = pitch_top
        self._draw_best_xi_pitch(pitch_x, pitch_y, pitch_w, pitch_h)

        # 세부 슬롯별 좌표 (sx_ratio, sy_ratio).
        # compute_awards 의 slots 순서와 일치해야 idx 매칭됨.
        slot_coords = {
            'GK':  (0.50, 0.92),
            'CB1': (0.38, 0.76),   # 좌측 CB
            'CB2': (0.62, 0.76),   # 우측 CB
            'LB':  (0.13, 0.72),
            'RB':  (0.87, 0.72),
            'DM':  (0.50, 0.58),
            'CM':  (0.30, 0.48),
            'AM':  (0.50, 0.36),
            'ST':  (0.50, 0.14),
            'LW':  (0.18, 0.20),
            'RW':  (0.82, 0.20),
        }
        # best_xi 순서 (compute_awards):
        #   [GK, CB×2, LB, RB, DM, CM, AM, ST, LW, RW]
        slot_keys = ['GK', 'CB1', 'CB2', 'LB', 'RB',
                      'DM', 'CM', 'AM', 'ST', 'LW', 'RW']
        # 등장 시점 — 0~1 진행에 11개를 균등 배치
        appear_threshold = [(i + 1) / 11 for i in range(11)]

        for idx, p in enumerate(best_xi):
            if idx >= len(slot_keys):
                break
            slot = slot_keys[idx]
            sx_ratio, sy_ratio = slot_coords[slot]
            if progress < appear_threshold[idx] - 0.05:
                continue
            local = min(1.0, max(0.0,
                                  (progress - (appear_threshold[idx] - 0.05)) / 0.05))
            cx = pitch_x + int(pitch_w * sx_ratio)
            cy = pitch_y + int(pitch_h * sy_ratio)
            # CB1/CB2 → 화면 라벨은 'CB' 로
            label_pos = slot.rstrip('12') if slot.startswith('CB') else slot
            self._draw_xi_player(cx, cy, p, local, t, label_pos)

        # 우측 리더보드 (rank 1~11)
        list_x = pitch_x + pitch_w + 32
        list_y = pitch_top + 30
        list_w = self.window_w - list_x - self.MARGIN
        list_bottom = self._draw_xi_list(list_x, list_y, list_w, best_xi, progress)

        # G 픽스: Tournament MVP + Golden Boot 카드 (ROSTER 아래)
        if progress > 0.95:
            self._draw_awards_card(list_x, list_bottom + 14, list_w, awards, t)

        self._draw_wc_frame()
        pygame.display.flip()

    def _draw_awards_card(self, x: int, y: int, w: int, awards: dict, t):
        """Tournament MVP + Golden Boot 컴팩트 카드."""
        h = 110
        pygame.draw.rect(self.screen, PANEL_COLOR, (x, y, w, h),
                          border_radius=10)
        hdr = self.font_team.render('AWARDS', True, ACCENT_GOLD)
        self.screen.blit(hdr, (x + 14, y + 8))
        gb = awards.get('golden_boot')
        mvp = awards.get('mvp')
        cur_y = y + 36
        if gb is not None:
            lbl = self.font_min.render('GOLDEN BOOT', True, TEXT_SECONDARY)
            self.screen.blit(lbl, (x + 14, cur_y))
            team = t.teams.get(gb.country) if hasattr(t, 'teams') else None
            color = team.color if team else (180, 180, 180)
            pygame.draw.circle(self.screen, color, (x + 28, cur_y + 24), 5)
            name = f'{gb.name.split()[-1]} ({gb.country})'
            ntxt = self.font_event.render(name, True, TEXT_PRIMARY)
            self.screen.blit(ntxt, (x + 40, cur_y + 16))
            val = self.font_min.render(f'{gb.goals}G {gb.assists}A',
                                        True, ACCENT_GOLD)
            self.screen.blit(val, (x + w - val.get_width() - 14, cur_y + 16))
            cur_y += 34
        if mvp is not None:
            lbl = self.font_min.render('TOURNAMENT MVP', True, TEXT_SECONDARY)
            self.screen.blit(lbl, (x + 14, cur_y))
            team = t.teams.get(mvp.country) if hasattr(t, 'teams') else None
            color = team.color if team else (180, 180, 180)
            pygame.draw.circle(self.screen, color, (x + 28, cur_y + 24), 5)
            name = f'{mvp.name.split()[-1]} ({mvp.country})'
            ntxt = self.font_event.render(name, True, TEXT_PRIMARY)
            self.screen.blit(ntxt, (x + 40, cur_y + 16))
            val = self.font_min.render(
                f'avg {mvp.avg_rating:.2f}', True, ACCENT_GOLD)
            self.screen.blit(val, (x + w - val.get_width() - 14, cur_y + 16))

    def _draw_best_xi_pitch(self, x: int, y: int, w: int, h: int):
        """베스트 11용 단순 세로 피치 (스트라이프 + 라인)."""
        # 잔디 스트라이프 (수평)
        stripes = 8
        sh = h / stripes
        for i in range(stripes):
            c = PITCH_GREEN_DARK if i % 2 == 0 else PITCH_GREEN_LIGHT
            pygame.draw.rect(self.screen, c,
                              (x, int(y + i * sh), w, int(sh + 1)))
        line = PITCH_LINE
        pygame.draw.rect(self.screen, line, (x, y, w, h), 2)
        # 중앙선 (수평)
        pygame.draw.line(self.screen, line, (x, y + h // 2), (x + w, y + h // 2), 2)
        # 중앙 원
        pygame.draw.circle(self.screen, line,
                            (x + w // 2, y + h // 2), 38, 2)
        # 박스
        pb_w = int(w * 0.55)
        pb_h = int(h * 0.13)
        pb_x = x + (w - pb_w) // 2
        pygame.draw.rect(self.screen, line, (pb_x, y, pb_w, pb_h), 2)
        pygame.draw.rect(self.screen, line,
                          (pb_x, y + h - pb_h, pb_w, pb_h), 2)

    def _draw_xi_player(self, cx: int, cy: int, p, local: float, t,
                          label_pos: str = ''):
        """피치 위 한 선수: 팀 컬러 원 + 이름 + 평점 + 포지션 라벨."""
        team = t.teams.get(p.country) if hasattr(t, 'teams') else None
        color = team.color if team else (180, 180, 180)
        outline = team.secondary if team else (240, 240, 240)
        # 등장 애니메이션 — 반경 0→14
        r = int(14 * local)
        if r <= 0:
            return
        pygame.draw.circle(self.screen, outline, (cx, cy), r + 2)
        pygame.draw.circle(self.screen, color, (cx, cy), r)
        # 포지션 라벨 — 선수 실제 position 우선 (슬롯에 폴백된 경우 진짜 포지션 표시)
        display_pos = (p.position if getattr(p, 'position', '') else label_pos)
        if display_pos and local > 0.7:
            pos_s = self.font_pin.render(display_pos, True, (255, 255, 255))
            self.screen.blit(pos_s, (cx - pos_s.get_width() // 2,
                                       cy - pos_s.get_height() // 2))
        # 이름 (성만)
        last = p.name.split()[-1]
        surf = self.font_min.render(last, True, TEXT_PRIMARY)
        bg = self.font_min.render(last, True, (0, 0, 0))
        lx = cx - surf.get_width() // 2
        ly = cy + r + 4
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            self.screen.blit(bg, (lx + dx, ly + dy))
        self.screen.blit(surf, (lx, ly))
        # 평점 (골드)
        rating_txt = f'{p.avg_rating:.2f}'
        rs = self.font_min.render(rating_txt, True, ACCENT_GOLD)
        rbg = self.font_min.render(rating_txt, True, (0, 0, 0))
        rx = cx - rs.get_width() // 2
        ry = ly + surf.get_height() + 1
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            self.screen.blit(rbg, (rx + dx, ry + dy))
        self.screen.blit(rs, (rx, ry))

    def _draw_xi_list(self, x: int, y: int, w: int, best_xi: list,
                       progress: float) -> int:
        """우측 패널 — 11명 리스트. 반환: 패널 바닥 y (awards 카드 위치용)."""
        panel_h = len(best_xi) * 24 + 50
        pygame.draw.rect(self.screen, PANEL_COLOR,
                          (x, y - 12, w, panel_h),
                          border_radius=10)
        hdr = self.font_team.render('ROSTER', True, TEXT_PRIMARY)
        self.screen.blit(hdr, (x + 14, y - 4))
        sub = self.font_min.render('Pos  Player              AVG    G/A',
                                    True, TEXT_SECONDARY)
        self.screen.blit(sub, (x + 14, y + 22))

        appear_threshold = [(i + 1) / 11 for i in range(11)]
        row_y = y + 42
        for idx, p in enumerate(best_xi):
            if progress < appear_threshold[idx] - 0.05:
                row_y += 22
                continue
            pos_label = p.position or p.role
            pos = self.font_min.render(pos_label, True, TEXT_SECONDARY)
            self.screen.blit(pos, (x + 14, row_y))
            name = f'{p.name.split()[-1]} ({p.country})'
            nm = self.font_event.render(name, True, TEXT_PRIMARY)
            self.screen.blit(nm, (x + 48, row_y))
            avg_s = self.font_min.render(f'{p.avg_rating:.2f}',
                                          True, ACCENT_GOLD)
            self.screen.blit(avg_s, (x + w - 90, row_y))
            ga_s = self.font_min.render(f'{p.goals}/{p.assists}',
                                         True, TEXT_PRIMARY)
            self.screen.blit(ga_s, (x + w - 36, row_y))
            row_y += 22
        return y - 12 + panel_h

    def _stage_label_for(self, match_id: str) -> str:
        if match_id.startswith('G'):
            return f'Group Stage  ·  Group {match_id[1]}'
        n = int(match_id[1:])
        if 73 <= n <= 88: return 'Round of 32'
        if 89 <= n <= 96: return 'Round of 16'
        if 97 <= n <= 100: return 'Quarter-final'
        if 101 <= n <= 102: return 'Semi-final'
        if n == 103: return '3rd-place'
        if n == 104: return 'FINAL'
        return ''

    def _draw_race_panel(self, x: int, y: int, w: int, h: int,
                          top_players: list, hist: list, cur_idx: int,
                          total: int, key: str, t, panel_title: str):
        """단일 차트 패널: 라인차트 + 리더보드 표.
        카메라가 cur_idx / max_v 에 맞춰 우측+상단으로 늘어남."""
        pygame.draw.rect(self.screen, PANEL_COLOR, (x, y, w, h),
                          border_radius=10)
        head = self.font_team.render(panel_title, True, TEXT_PRIMARY)
        self.screen.blit(head, (x + 18, y + 14))

        inner_x = x + 18
        chart_y = y + 48
        chart_w = w - 36
        board_h = 18 * (len(top_players) + 1) + 12
        chart_h = h - 48 - board_h - 16
        board_y = chart_y + chart_h + 12

        # ── 카메라 X 범위 — 현재까지 진행된 매치 + 약간의 미래 패딩 ──
        # 시청자가 라인이 자라는 걸 느낄 수 있게 살짝 앞을 비워둠
        view_x_min = 0
        view_x_max = max(4, int(cur_idx * 1.15))   # 현재의 115%까지 view
        # 토너먼트 거의 끝나면 full range
        if cur_idx >= total * 0.9:
            view_x_max = total

        # ── 카메라 Y 범위 — 현재 최댓값 + 여유 ──
        cur_max_v = 1
        for name, _info in top_players:
            v = hist[cur_idx - 1][1].get(name, {}).get(key, 0)
            if v > cur_max_v:
                cur_max_v = v
        # Y 범위는 현재 max + 1 정도 (애니메이션 헤드룸)
        max_v = max(3, cur_max_v + 1)

        # 격자선 (Y축)
        for i in range(0, max_v + 1):
            gy = chart_y + chart_h - int(chart_h * i / max_v)
            color = (50, 55, 65) if i > 0 else (90, 95, 110)
            pygame.draw.line(self.screen, color,
                              (inner_x, gy), (inner_x + chart_w, gy), 1)
            if i % max(1, max_v // 5) == 0:
                lbl = self.font_min.render(str(i), True, TEXT_SECONDARY)
                self.screen.blit(lbl,
                    (inner_x - lbl.get_width() - 4, gy - lbl.get_height() // 2))

        # X축 마커 — 현재 view 안에 들어온 단계만
        marker_indices = self._chart_x_markers(hist)
        view_span = max(1, view_x_max - view_x_min)
        for mi, label in marker_indices:
            if mi < view_x_min or mi > view_x_max:
                continue
            mx = inner_x + int(chart_w * (mi - view_x_min) / view_span)
            pygame.draw.line(self.screen, (60, 65, 75),
                              (mx, chart_y), (mx, chart_y + chart_h), 1)
            lbl = self.font_min.render(label, True, TEXT_SECONDARY)
            self.screen.blit(lbl,
                (mx - lbl.get_width() // 2, chart_y + chart_h + 2))

        # 각 top 선수 라인
        for rank, (name, _info) in enumerate(top_players):
            country = _info.get('country', '')
            team = t.teams.get(country) if hasattr(t, 'teams') else None
            color = team.color if team else (180, 180, 180)
            pts = []
            for i in range(cur_idx):
                snap = hist[i][1]
                v = snap.get(name, {}).get(key, 0)
                px = inner_x + int(chart_w * (i + 1 - view_x_min) / view_span)
                py = chart_y + chart_h - int(chart_h * v / max_v)
                pts.append((px, py))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, color, False, pts, 2)
            elif pts:
                pygame.draw.circle(self.screen, color, pts[0], 2)
            # 끝점 마커 + 이름 (top 5)
            if pts and rank < 5:
                end_x, end_y = pts[-1]
                pygame.draw.circle(self.screen, color, (end_x, end_y), 3)
                cur_val = hist[cur_idx - 1][1].get(name, {}).get(key, 0)
                lbl = self.font_min.render(
                    f'{name} {cur_val}', True, color)
                lx = min(inner_x + chart_w - lbl.get_width() - 4, end_x + 6)
                ly = max(chart_y + 2,
                          min(chart_y + chart_h - lbl.get_height() - 2,
                              end_y - lbl.get_height() // 2))
                self.screen.blit(lbl, (lx, ly))

        # 리더보드 — 현재 시점 기준 내림차순 정렬 (보조 키: 다른 스탯)
        head_y = board_y
        hdr = self.font_min.render(
            f'#  PLAYER (TEAM)         {key.upper()}', True, TEXT_SECONDARY)
        self.screen.blit(hdr, (inner_x, head_y))
        cur_snap = hist[cur_idx - 1][1]
        other = 'assists' if key == 'goals' else 'goals'
        # 전체 선수에서 현재값 기준 top N 재선정 (live ranking)
        live_top = sorted(
            cur_snap.items(),
            key=lambda kv: (-kv[1].get(key, 0), -kv[1].get(other, 0))
        )[:len(top_players)]
        for rank, (name, info) in enumerate(live_top):
            row_y = head_y + 18 + rank * 18
            country = info.get('country', '')
            team = t.teams.get(country) if hasattr(t, 'teams') else None
            color = team.color if team else (180, 180, 180)
            cur_val = info.get(key, 0)
            rank_s = self.font_min.render(f'{rank+1}.', True, TEXT_SECONDARY)
            self.screen.blit(rank_s, (inner_x, row_y))
            pygame.draw.circle(self.screen, color,
                                (inner_x + 26, row_y + 7), 4)
            nm = f'{name} ({country})'
            ntxt = self.font_event.render(nm, True, TEXT_PRIMARY)
            self.screen.blit(ntxt, (inner_x + 36, row_y))
            v_txt = self.font_min.render(str(cur_val), True, ACCENT_GOLD)
            self.screen.blit(v_txt,
                (inner_x + chart_w - v_txt.get_width(), row_y))

    def _chart_x_markers(self, hist: list) -> list:
        """차트 X축에 표시할 (index, label) 리스트."""
        markers = []
        # 조별 마지막 (GL6 위치)
        group_end = 0
        for i, (mid, _) in enumerate(hist):
            if mid.startswith('G'):
                group_end = i + 1
            else:
                break
        if group_end > 0:
            markers.append((group_end, 'Groups'))
        # 토너먼트 라운드별
        stage_breaks = {
            88: 'R32', 96: 'R16', 100: 'QF', 102: 'SF', 104: 'F',
        }
        for i, (mid, _) in enumerate(hist):
            if mid.startswith('M'):
                n = int(mid[1:])
                if n in stage_breaks:
                    markers.append((i + 1, stage_breaks[n]))
        return markers

    # ── 미니 브래킷 ─────────────────────────────────────────

    def _draw_mini_bracket(self, t: Tournament):
        x = self.MARGIN
        y = self.bracket_y
        w = self.PITCH_PX_W
        h = self.BRACKET_H
        pygame.draw.rect(self.screen, PANEL_COLOR, (x, y, w, h),
                          border_radius=10)
        title = self.font_min.render('TOURNAMENT BRACKET', True, TEXT_SECONDARY)
        self.screen.blit(title, (x + 14, y + 10))

        stages = [
            ('GROUPS', 72, lambda mid: mid.startswith('G')),
            ('R32',    16, lambda mid: mid.startswith('M') and 73 <= int(mid[1:]) <= 88),
            ('R16',     8, lambda mid: mid.startswith('M') and 89 <= int(mid[1:]) <= 96),
            ('QF',      4, lambda mid: mid.startswith('M') and 97 <= int(mid[1:]) <= 100),
            ('SF',      2, lambda mid: mid in ('M101', 'M102')),
            ('FINAL',   1, lambda mid: mid == 'M104'),
        ]
        stage_w = (w - 28) // 6
        sx = x + 14
        sy = y + 32
        for label, total, pred in stages:
            played = sum(1 for k in t.results if pred(k))
            lab = self.font_pin.render(label, True, TEXT_SECONDARY)
            self.screen.blit(lab, (sx + (stage_w - lab.get_width()) // 2, sy))
            bar_y = sy + 18
            bar_w = stage_w - 14
            pygame.draw.rect(self.screen, (40, 44, 54),
                              (sx + 7, bar_y, bar_w, 7), border_radius=3)
            fill = int(bar_w * played / total) if total else 0
            if fill > 0:
                color = (ACCENT_GOLD if played == total
                          else (140, 200, 240))
                pygame.draw.rect(self.screen, color,
                                  (sx + 7, bar_y, fill, 7), border_radius=3)
            ct = self.font_stat.render(f'{played}/{total}', True,
                                        TEXT_PRIMARY)
            self.screen.blit(ct, (sx + (stage_w - ct.get_width()) // 2,
                                   bar_y + 12))
            sx += stage_w

        # 챔피언 표시 (있을 때)
        champ = t.champion()
        if champ is not None:
            big = pygame.font.SysFont('malgungothic,arial', 18, bold=True)
            txt = big.render(f'★  {champ.code}  WORLD CHAMPION  ★',
                              True, ACCENT_GOLD)
            self.screen.blit(txt, (x + (w - txt.get_width()) // 2,
                                    y + h - 28))

    # ── 헤더 ───────────────────────────────────────────────

    def _draw_header(self, t: Tournament, match_id: Optional[str], fast: bool):
        title = self.font_title.render('WORLD CUP 2026', True, TEXT_PRIMARY)
        self.screen.blit(title, (self.MARGIN, 22))

        if match_id:
            stage = t.stage_of(match_id)
            played = len(t.results)
            total = 104
            cond = t.conditions_of(match_id) if hasattr(t, 'conditions_of') else {}
            venue = cond.get('venue', '')
            venue_str = f'  @ {venue}' if venue and venue != '?' else ''
            sub = (f'{stage}  ·  {match_id}{venue_str}'
                   f'  ·  match {played + 1}/{total}'
                   f'  ·  seed {t.seed}')
        else:
            sub = f'completed  ·  seed {t.seed}'
        sub_surf = self.font_sub.render(sub, True, TEXT_SECONDARY)
        self.screen.blit(sub_surf, (self.MARGIN, 56))

        # 우측: FAST FORWARD + 매치 조건 칩
        chip_x = self.window_w - self.MARGIN
        chip_y = 32

        if fast:
            label = self.font_pin.render('FAST FORWARD', True, BG_COLOR)
            pad_x, pad_y = 10, 6
            w = label.get_width() + pad_x * 2
            rect = pygame.Rect(chip_x - w, chip_y, w,
                                label.get_height() + pad_y * 2)
            pygame.draw.rect(self.screen, ACCENT_GOLD, rect, border_radius=10)
            self.screen.blit(label, (rect.x + pad_x, rect.y + pad_y))
            chip_x = rect.x - 8

        # 매치 조건 칩 (HIGH-ALT / HOT / FINAL-RD)
        if match_id and hasattr(t, 'conditions_of'):
            cond = t.conditions_of(match_id)
            chips = []
            if cond.get('altitude') == 'high':
                chips.append(('HIGH-ALT', (180, 130, 230)))
            elif cond.get('altitude') == 'mid':
                chips.append(('mid-alt', (140, 110, 180)))
            if cond.get('hot'):
                chips.append(('HOT', (235, 110, 80)))
            if cond.get('last_round_push'):
                chips.append(('FINAL-RD', (90, 200, 140)))
            for text, color in reversed(chips):
                lab = self.font_pin.render(text, True, BG_COLOR)
                pad_x, pad_y = 8, 5
                w = lab.get_width() + pad_x * 2
                rect = pygame.Rect(chip_x - w, chip_y, w,
                                    lab.get_height() + pad_y * 2)
                pygame.draw.rect(self.screen, color, rect, border_radius=10)
                self.screen.blit(lab, (rect.x + pad_x, rect.y + pad_y))
                chip_x = rect.x - 6

        # Match counter — 좌측 sub line 우측에 진행도 미니 바
        if match_id:
            played = len(t.results)
            bar_w = 180
            bar_h = 4
            bar_x = self.MARGIN
            bar_y = 78
            pygame.draw.rect(self.screen, (40, 44, 54),
                              (bar_x, bar_y, bar_w, bar_h), border_radius=2)
            fill_w = int(bar_w * played / 104)
            if fill_w > 0:
                pygame.draw.rect(self.screen, ACCENT_GOLD,
                                  (bar_x, bar_y, fill_w, bar_h), border_radius=2)

    # ── 경기장 ─────────────────────────────────────────────

    def _draw_pitch_panel(self, match: Match):
        # 패널 배경
        pad = 8
        panel = pygame.Rect(
            self.pitch_x - pad, self.pitch_y - pad,
            self.PITCH_PX_W + pad * 2, self.PITCH_PX_H + pad * 2,
        )
        pygame.draw.rect(self.screen, PANEL_COLOR, panel, border_radius=10)

        # 잔디 줄무늬
        stripes = 8
        stripe_w = self.PITCH_PX_W / stripes
        for i in range(stripes):
            color = PITCH_GREEN_DARK if i % 2 == 0 else PITCH_GREEN_LIGHT
            r = pygame.Rect(self.pitch_x + i * stripe_w, self.pitch_y,
                            stripe_w + 1, self.PITCH_PX_H)
            pygame.draw.rect(self.screen, color, r)

        # 경기장 라인
        self._draw_pitch_lines()

        # 골 플래시 — 골 넣은 팀 골대 쪽 강조
        if match.goal_flash > 0:
            ratio = match.goal_flash / 24.0
            alpha = int(150 * ratio)
            overlay = pygame.Surface((self.PITCH_PX_W, self.PITCH_PX_H),
                                     pygame.SRCALPHA)
            color = match.home.color if match.goal_flash_team == 0 else match.away.color
            band_x = 0 if match.goal_flash_team == 0 else self.PITCH_PX_W * 0.7
            band_w = self.PITCH_PX_W * 0.3
            pygame.draw.rect(overlay, (*color, alpha),
                             (band_x, 0, band_w, self.PITCH_PX_H))
            self.screen.blit(overlay, (self.pitch_x, self.pitch_y))

        # 선수 점
        sx = self.PITCH_PX_W / Match.PITCH_W
        sy = self.PITCH_PX_H / Match.PITCH_H
        # 키트 충돌 시 어웨이는 secondary 사용
        away_color = (match.away.secondary if match.away_use_secondary
                       else match.away.color)
        away_outline = (match.away.color if match.away_use_secondary
                         else match.away.secondary)

        # 각 팀 라이브 평점 max/min 선수 찾기 (4명만 라벨 표시)
        label_for = compute_rating_labels(match)

        for p in match.players:
            if not p.on_pitch:
                continue
            cx = self.pitch_x + p.x * sx
            cy = self.pitch_y + p.y * sy
            if p.team_idx == 0:
                color = match.home.color
                outline = match.home.secondary
            else:
                color = away_color
                outline = away_outline
            r = 7 if p.role == 'GK' else 6
            pygame.draw.circle(self.screen, outline, (int(cx), int(cy)), r + 1)
            pygame.draw.circle(self.screen, color, (int(cx), int(cy)), r)

            # 평점 라벨 (★ 팀 MAX / ▼ 팀 MIN)
            lbl = label_for.get(id(p))
            if lbl is not None:
                text, txt_color = lbl
                surf = self.font_pin.render(text, True, txt_color)
                # 검은 외곽으로 가독성 ↑
                bg = self.font_pin.render(text, True, (0, 0, 0))
                lx = int(cx - surf.get_width() / 2)
                ly = int(cy - r - 14)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    self.screen.blit(bg, (lx + dx, ly + dy))
                self.screen.blit(surf, (lx, ly))

        # 공
        bx = self.pitch_x + match.ball_x * sx
        by = self.pitch_y + match.ball_y * sy
        pygame.draw.circle(self.screen, (15, 15, 18), (int(bx), int(by)), 5)
        pygame.draw.circle(self.screen, (250, 250, 250), (int(bx), int(by)), 4)

    def _draw_pitch_lines(self):
        x, y, w, h = self.pitch_x, self.pitch_y, self.PITCH_PX_W, self.PITCH_PX_H
        line = PITCH_LINE
        # 외곽
        pygame.draw.rect(self.screen, line, (x, y, w, h), 2)
        # 중앙선
        pygame.draw.line(self.screen, line, (x + w // 2, y), (x + w // 2, y + h), 2)
        # 중앙 원
        pygame.draw.circle(self.screen, line, (x + w // 2, y + h // 2), 50, 2)
        pygame.draw.circle(self.screen, line, (x + w // 2, y + h // 2), 3)
        # 페널티 박스
        pb_w, pb_h = 90, 200
        pygame.draw.rect(self.screen, line, (x, y + (h - pb_h) // 2, pb_w, pb_h), 2)
        pygame.draw.rect(self.screen, line,
                         (x + w - pb_w, y + (h - pb_h) // 2, pb_w, pb_h), 2)
        # 골에리어
        ga_w, ga_h = 40, 100
        pygame.draw.rect(self.screen, line, (x, y + (h - ga_h) // 2, ga_w, ga_h), 2)
        pygame.draw.rect(self.screen, line,
                         (x + w - ga_w, y + (h - ga_h) // 2, ga_w, ga_h), 2)

    # ── 스코어보드 (경기장 위 헤더 부분에 살짝) ────────

    def _draw_scoreboard(self, match: Match, t: Tournament, match_id: Optional[str]):
        # 사이드바 위쪽에 빅 스코어보드 그릴 거라 여기선 패스
        pass

    # ── 사이드바: 빅 스코어 + 팀 스탯 ─────────────────

    def _draw_sidebar(self, t: Tournament, match: Match):
        x = self.pitch_x + self.PITCH_PX_W + self.MARGIN
        y = self.pitch_y - 8
        w = self.SIDEBAR_W
        h = self.PITCH_PX_H + 16
        pygame.draw.rect(self.screen, PANEL_COLOR, (x, y, w, h), border_radius=10)

        inner_x = x + 18
        cur_y = y + 16

        # 단계 라벨
        stage_text = '— in progress —' if not match.finished else '— full time —'
        if match.in_pk and not match.finished:
            stage_text = '— penalty shootout —'
        stage_surf = self.font_sub.render(stage_text, True, TEXT_SECONDARY)
        self.screen.blit(stage_surf, (inner_x, cur_y))
        cur_y += 24

        # 홈팀
        cur_y = self._draw_team_block(
            inner_x, cur_y, match.home, match.result.home_goals,
            match.result.home_pk if match.in_pk or match.result.went_to_pk else None,
            is_winner=(match.finished and match.result.winner is match.home),
            used_formation=getattr(match, 'home_formation_used', None),
            used_style=getattr(match, 'home_style_effective', None),
        )
        cur_y += 12
        # vs
        vs = self.font_sub.render('VS', True, TEXT_SECONDARY)
        self.screen.blit(vs, (inner_x + (w - 36 - vs.get_width()) // 2, cur_y))
        cur_y += 24
        # 원정팀
        cur_y = self._draw_team_block(
            inner_x, cur_y, match.away, match.result.away_goals,
            match.result.away_pk if match.in_pk or match.result.went_to_pk else None,
            is_winner=(match.finished and match.result.winner is match.away),
            used_formation=getattr(match, 'away_formation_used', None),
            used_style=getattr(match, 'away_style_effective', None),
        )

        # 시간
        cur_y += 12
        if match.in_pk:
            time_text = f"PK round {match.pk_round}"
        else:
            time_text = f"{min(match.minute, 90)}'"
        big = pygame.font.SysFont('malgungothic,arial', 32, bold=True)
        time_surf = big.render(time_text, True, ACCENT_GOLD)
        self.screen.blit(time_surf, (inner_x + (w - 36 - time_surf.get_width()) // 2, cur_y))
        cur_y += 50

        # 능력치 표
        self._draw_stat_row(inner_x, cur_y, 'ATT',
                            match.home.attack, match.away.attack); cur_y += 20
        self._draw_stat_row(inner_x, cur_y, 'MID',
                            match.home.midfield, match.away.midfield); cur_y += 20
        self._draw_stat_row(inner_x, cur_y, 'DEF',
                            match.home.defense, match.away.defense); cur_y += 20
        self._draw_stat_row(inner_x, cur_y, 'GK',
                            match.home.keeper, match.away.keeper); cur_y += 26

        # ── MATCH STATS ───────────────────────────────────
        section = self.font_min.render('— MATCH STATS —', True, TEXT_SECONDARY)
        self.screen.blit(section, (inner_x + (w - 36 - section.get_width()) // 2, cur_y))
        cur_y += 18

        # 점유율 막대 (color-coded both teams)
        h_pos, a_pos = match.possession_pct()
        self._draw_possession_bar(inner_x, cur_y, w - 36,
                                   match.home.color, match.away.color,
                                   h_pos, a_pos)
        cur_y += 22

        # 슛 (유효슛)
        h_shots, a_shots = match.shot_counts()
        self._draw_kv_row(inner_x, cur_y, w - 36, 'shots',
                          f'{h_shots[0]} ({h_shots[1]})', f'{a_shots[0]} ({a_shots[1]})')
        cur_y += 18

        # xG
        h_xg, a_xg = match.xg()
        self._draw_kv_row(inner_x, cur_y, w - 36, 'xG',
                          f'{h_xg:.2f}', f'{a_xg:.2f}')
        cur_y += 18

        # 세트피스 (코너 / 프리킥 / 페널티) — Malgun Gothic 호환 텍스트 라벨
        h_sp, a_sp = match.setpiece_counts()
        h_sp_s = f'C{h_sp[0]} FK{h_sp[1]} PK{h_sp[2]}'
        a_sp_s = f'C{a_sp[0]} FK{a_sp[1]} PK{a_sp[2]}'
        self._draw_kv_row(inner_x, cur_y, w - 36, 'set-piece goals',
                          h_sp_s, a_sp_s)
        cur_y += 22

        # ── 골 기록 (스코어러 + 어시스트 + 분) — 최근 5골 ──
        goals_log = getattr(match, 'goals_log', [])
        if goals_log:
            hdr = self.font_min.render('— GOALS —', True, TEXT_SECONDARY)
            self.screen.blit(hdr,
                (inner_x + (w - 36 - hdr.get_width()) // 2, cur_y))
            cur_y += 16
        for g in goals_log[-5:]:
            team_color = (match.home.color if g['team_idx'] == 0
                          else match.away.color)
            minute = g.get('minute', 0)
            scorer = g.get('scorer', '') or '???'
            assist = g.get('assist', '')
            gtype = g.get('type', 'open_play')
            tag = ''
            if gtype == 'penalty':
                tag = ' (PK)'
            elif gtype == 'corner':
                tag = ' (C)'
            elif gtype == 'free_kick':
                tag = ' (FK)'
            min_surf = self.font_min.render(f"{minute}'", True, team_color)
            self.screen.blit(min_surf, (inner_x, cur_y))
            rest = f"  {scorer}{tag}"
            if assist:
                rest += f"  · {assist}"
            rest_surf = self.font_event.render(rest, True, TEXT_PRIMARY)
            self.screen.blit(rest_surf,
                (inner_x + min_surf.get_width(), cur_y))
            cur_y += 15

        # ── MAN OF THE MATCH (매치 종료 시) ───────────────
        if match.finished and getattr(match.result, 'mvp', None):
            cur_y += 6
            hdr = self.font_min.render(
                '— MAN OF THE MATCH —', True, ACCENT_GOLD)
            self.screen.blit(hdr,
                (inner_x + (w - 36 - hdr.get_width()) // 2, cur_y))
            cur_y += 18
            team_idx, mvp_name, mvp_rating = match.result.mvp
            mvp_team = match.home if team_idx == 0 else match.away
            mvp_color = mvp_team.color
            # 팀 색 원 + 이름 + 평점
            pygame.draw.circle(self.screen, mvp_color,
                                (inner_x + 8, cur_y + 8), 6)
            name_surf = self.font_team.render(
                mvp_name, True, TEXT_PRIMARY)
            self.screen.blit(name_surf, (inner_x + 22, cur_y))
            sub_surf = self.font_min.render(
                f'{mvp_team.code}', True, TEXT_SECONDARY)
            self.screen.blit(sub_surf,
                (inner_x + 22, cur_y + name_surf.get_height()))
            rating_surf = self.font_score.render(
                f'{mvp_rating:.1f}', True, ACCENT_GOLD)
            rx = inner_x + (w - 36) - rating_surf.get_width()
            self.screen.blit(rating_surf, (rx, cur_y - 4))

    def _draw_possession_bar(self, x: int, y: int, w: int,
                              h_color: tuple, a_color: tuple,
                              h_pct: int, a_pct: int):
        """홈/어웨이 색으로 분할된 점유율 막대 + %텍스트."""
        bar_h = 12
        h_w = int(w * h_pct / 100)
        # 좌(홈) | 우(원정)
        pygame.draw.rect(self.screen, h_color, (x, y, h_w, bar_h), border_radius=4)
        pygame.draw.rect(self.screen, a_color, (x + h_w, y, w - h_w, bar_h),
                         border_radius=4)
        # 퍼센트 텍스트 — 홈은 좌, 어웨이는 우
        h_t = self.font_min.render(f'{h_pct}%', True, BG_COLOR)
        a_t = self.font_min.render(f'{a_pct}%', True, BG_COLOR)
        self.screen.blit(h_t, (x + 4, y - 1))
        self.screen.blit(a_t, (x + w - a_t.get_width() - 4, y - 1))

    def _draw_kv_row(self, x: int, y: int, w: int, label: str,
                     h_val: str, a_val: str):
        """홈값 | 라벨 | 어웨이값 형식 행."""
        h_t = self.font_stat.render(h_val, True, TEXT_PRIMARY)
        a_t = self.font_stat.render(a_val, True, TEXT_PRIMARY)
        l_t = self.font_label.render(label, True, TEXT_SECONDARY) if hasattr(
            self, 'font_label') else self.font_min.render(label, True, TEXT_SECONDARY)
        center = x + w // 2
        self.screen.blit(l_t, (center - l_t.get_width() // 2, y))
        self.screen.blit(h_t, (x, y))
        self.screen.blit(a_t, (x + w - a_t.get_width(), y))

    def _draw_team_block(self, x: int, y: int, team: Team, goals: int,
                         pk: Optional[int], is_winner: bool,
                         used_formation: str = None,
                         used_style: str = None) -> int:
        # 색 블록 (primary + secondary 두 줄 띠)
        pygame.draw.rect(self.screen, team.color, (x, y, 14, 18), border_radius=3)
        pygame.draw.rect(self.screen, team.secondary, (x, y + 18, 14, 14), border_radius=3)
        # 이름 (+우승 횟수 — Arial/MalgunGothic에서 안전한 글리프)
        title_text = team.name
        if team.wc_titles > 0:
            title_text = f'{team.name}  WC×{team.wc_titles}'
        name_surf = self.font_team.render(title_text, True, TEXT_PRIMARY)
        self.screen.blit(name_surf, (x + 22, y + 2))
        # 보조 라인: 코드 · OVR · 사용 스타일 · 사용 포메이션 (★표시: secondary)
        eff_form = used_formation or team.formation
        eff_style = used_style or team.style_tag
        is_secondary = (eff_form != team.formation)
        form_str = f'{eff_form}*' if is_secondary else eff_form
        sub_text = (f'{team.code}({team.iso2})  OVR {team.overall}  '
                    f'{eff_style}  {form_str}')
        code_surf = self.font_stat.render(sub_text, True, TEXT_SECONDARY)
        self.screen.blit(code_surf, (x + 22, y + 22))
        # 우측 스코어
        score_text = str(goals)
        score_surf = self.font_score.render(score_text, True,
                                            ACCENT_GOLD if is_winner else TEXT_PRIMARY)
        sx = x + self.SIDEBAR_W - 36 - score_surf.get_width() - 8
        self.screen.blit(score_surf, (sx, y - 6))
        if pk is not None:
            pk_surf = self.font_min.render(f'PK {pk}', True, ACCENT_GOLD)
            self.screen.blit(pk_surf, (sx - pk_surf.get_width() - 8, y + 6))
        return y + 36

    def _draw_stat_row(self, x: int, y: int, label: str, h_val: int, a_val: int):
        # 좌(홈) | 라벨 | 우(원정) — 막대 양쪽
        w = self.SIDEBAR_W - 36
        center = x + w // 2
        # 라벨 가운데
        lab = self.font_min.render(label, True, TEXT_SECONDARY)
        self.screen.blit(lab, (center - lab.get_width() // 2, y))
        # 좌 막대 (홈)
        bar_max = (w - 60) // 2
        h_w = int(bar_max * h_val / 100)
        pygame.draw.rect(self.screen, (40, 44, 54),
                         (center - 24 - bar_max, y + 4, bar_max, 8), border_radius=4)
        pygame.draw.rect(self.screen, (180, 200, 220),
                         (center - 24 - h_w, y + 4, h_w, 8), border_radius=4)
        # 우 막대
        a_w = int(bar_max * a_val / 100)
        pygame.draw.rect(self.screen, (40, 44, 54),
                         (center + 24, y + 4, bar_max, 8), border_radius=4)
        pygame.draw.rect(self.screen, (180, 200, 220),
                         (center + 24, y + 4, a_w, 8), border_radius=4)
        # 숫자
        h_num = self.font_stat.render(str(h_val), True, TEXT_PRIMARY)
        a_num = self.font_stat.render(str(a_val), True, TEXT_PRIMARY)
        self.screen.blit(h_num, (center - 22 - h_num.get_width(), y))
        self.screen.blit(a_num, (center + 22, y))

    # ── 이벤트 피드 ────────────────────────────────────────

    def _draw_event_feed(self, match: Match):
        x = self.MARGIN
        y = self.pitch_y + self.PITCH_PX_H + self.MARGIN
        w = self.PITCH_PX_W
        h = self.EVENTS_H
        pygame.draw.rect(self.screen, PANEL_COLOR, (x, y, w, h), border_radius=10)

        title = self.font_min.render('MATCH EVENTS', True, TEXT_SECONDARY)
        self.screen.blit(title, (x + 14, y + 10))

        # 최근 이벤트 6개 표시 (최신이 위)
        events = match.events[-6:][::-1]
        cur_y = y + 32
        for ev in events:
            self._draw_event_row(x + 14, cur_y, ev, match)
            cur_y += 16

    def _draw_event_row(self, x: int, y: int, ev: MatchEvent, match: Match):
        # 분 라벨
        if ev.kind == 'pk':
            min_text = 'PK'
        else:
            min_text = f"{ev.minute}'"
        min_surf = self.font_min.render(min_text, True, TEXT_SECONDARY)
        self.screen.blit(min_surf, (x, y))

        # 색 점
        if ev.team_idx == 0:
            color = match.home.color
        elif ev.team_idx == 1:
            color = match.away.color
        else:
            color = TEXT_SECONDARY
        pygame.draw.circle(self.screen, color, (x + 38, y + 7), 4)

        # 텍스트 — 종류별 색
        if ev.kind == 'goal':
            text_color = ACCENT_GOLD
        elif ev.kind == 'tactical':
            text_color = (140, 200, 240)    # 청색 — 전술 변경
        elif ev.kind == 'sub':
            text_color = (90, 200, 140)     # 녹색 — 교체
        else:
            text_color = TEXT_PRIMARY
        text_surf = self.font_event.render(ev.text, True, text_color)
        self.screen.blit(text_surf, (x + 50, y))

    # ── 토너먼트 종료 화면 ───────────────────────────────

    def _draw_completion(self, t: Tournament):
        cx = self.window_w // 2
        cy = self.HEADER_H + 20

        champ = t.champion()
        runner = t.runner_up()
        third = t.third()

        if champ:
            big = pygame.font.SysFont('malgungothic,arial', 38, bold=True)
            txt = big.render('WORLD CHAMPIONS', True, ACCENT_GOLD)
            self.screen.blit(txt, (cx - txt.get_width() // 2, cy))
            cy += 44
            name = pygame.font.SysFont('malgungothic,arial', 48, bold=True)
            ntxt = name.render(champ.name, True, champ.color)
            self.screen.blit(ntxt, (cx - ntxt.get_width() // 2, cy))
            cy += 60

        # Runner-up / 3rd 한 줄로
        ru_3rd_line = ''
        if runner is not None:
            ru_3rd_line += f'Runner-up: {runner.name}'
        if third is not None:
            ru_3rd_line += (' | ' if ru_3rd_line else '') + f'3rd: {third.name}'
        if ru_3rd_line:
            line = self.font_team.render(ru_3rd_line, True, TEXT_PRIMARY)
            self.screen.blit(line, (cx - line.get_width() // 2, cy))
            cy += 40

        # ── Awards ──────────────────────────────────────────
        awards = t.compute_awards() if hasattr(t, 'compute_awards') else None
        if awards:
            cy = self._draw_award_section(cx, cy, '🏆 GOLDEN BOOT',
                                           awards['golden_boot'], ACCENT_GOLD)
            cy = self._draw_award_section(cx, cy, '★ MVP',
                                           awards['mvp'], (140, 200, 240))

            # Best XI title
            cy += 6
            xi_label = self.font_min.render('— BEST XI (4-3-3) —',
                                              True, TEXT_SECONDARY)
            self.screen.blit(xi_label, (cx - xi_label.get_width() // 2, cy))
            cy += 22

            # 4-3-3 피라미드: FWD(3) → MID(3) → DEF(4) → GK(1) (위에서 아래로 공격→수비)
            xi = awards['best_xi']
            rows = [
                ('FWD', [p for p in xi if p.role == 'FWD']),
                ('MID', [p for p in xi if p.role == 'MID']),
                ('DEF', [p for p in xi if p.role == 'DEF']),
                ('GK',  [p for p in xi if p.role == 'GK']),
            ]
            for role_label, players in rows:
                if not players:
                    continue
                slot_w = 200
                total_w = slot_w * len(players)
                x_start = cx - total_w // 2
                for i, p in enumerate(players):
                    self._draw_xi_card(x_start + i * slot_w, cy, p)
                cy += 38

        cy += 16
        hint = self.font_sub.render('press R for new tournament  ·  ESC to quit',
                                     True, TEXT_SECONDARY)
        self.screen.blit(hint, (cx - hint.get_width() // 2, cy))

    def _draw_award_section(self, cx: int, cy: int, title: str,
                              player, accent: tuple) -> int:
        """Golden Boot / MVP 섹션 — 작은 라벨 + 한 줄 결과."""
        lab = self.font_min.render(title, True, TEXT_SECONDARY)
        self.screen.blit(lab, (cx - lab.get_width() // 2, cy))
        cy += 18
        body = pygame.font.SysFont('malgungothic,arial', 22, bold=True)
        text = (f'{player.name}  ·  {player.country}  ·  {player.role}  '
                f'·  {player.goals}G')
        bs = body.render(text, True, accent)
        self.screen.blit(bs, (cx - bs.get_width() // 2, cy))
        cy += 32
        return cy

    def _draw_xi_card(self, x: int, y: int, p):
        """Best XI 한 명 카드 (이름 + 국가 + 골 수)."""
        star = ' ★' if p.is_star else ''
        name_text = f'{p.name}{star}'
        sub_text = f'{p.country}  ·  {p.goals}G'
        nt = self.font_stat.render(name_text, True, TEXT_PRIMARY)
        st = self.font_min.render(sub_text, True, TEXT_SECONDARY)
        # 살짝 가운데 정렬
        slot_w = 200
        self.screen.blit(nt, (x + (slot_w - nt.get_width()) // 2, y))
        self.screen.blit(st, (x + (slot_w - st.get_width()) // 2, y + 16))

    # ── 외부에서 호출 ─────────────────────────────────────

    def tick_clock(self, fps: int):
        self.clock.tick(fps)

    def quit(self):
        pygame.quit()


# ════════════════════════════════════════════════════════════════
# ShortsRenderer — 1080×1920 세로 모드 (YouTube Shorts용)
# 피치를 90도 회전: 홈 골대 TOP, 어웨이 골대 BOTTOM
# ════════════════════════════════════════════════════════════════

class ShortsRenderer:
    """세로 9:16 (1080×1920) 레이아웃. 헤더 + 큰 수직 피치 + 이벤트."""

    W = 1080
    H = 1920
    HEADER_H = 300
    PITCH_TOP = 310
    PITCH_LEFT = 90
    PITCH_W = 900           # screen width (maps sim_y 0..50)
    PITCH_H = 1420          # screen height (maps sim_x 0..80)
    EVENTS_TOP = 1750
    EVENTS_H = 152
    PAD = 10

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption('World Cup 2026 — Shorts')
        self.clock = pygame.time.Clock()

        self.font_stage = pygame.font.SysFont('malgungothic,arial', 28, bold=True)
        self.font_team_big = pygame.font.SysFont('malgungothic,arial', 56, bold=True)
        self.font_team_code = pygame.font.SysFont('malgungothic,arial', 26, bold=True)
        self.font_team_sub = pygame.font.SysFont('malgungothic,arial', 22)
        self.font_score = pygame.font.SysFont('malgungothic,arial', 140, bold=True)
        self.font_minute = pygame.font.SysFont('malgungothic,arial', 44, bold=True)
        self.font_event = pygame.font.SysFont('malgungothic,arial', 24, bold=True)
        self.font_event_sub = pygame.font.SysFont('malgungothic,arial', 20)
        self.font_small = pygame.font.SysFont('malgungothic,arial', 18)
        self.font_recap_title = pygame.font.SysFont('malgungothic,arial', 56, bold=True)
        self.font_recap = pygame.font.SysFont('malgungothic,arial', 32, bold=True)
        self.font_recap_sub = pygame.font.SysFont('malgungothic,arial', 24)

        # 국기 surface 캐시: (flag_code, w, h) → Surface
        self._flag_cache: dict = {}

    # ── 외부 API ──────────────────────────────────────────
    def tick_clock(self, fps: int):
        self.clock.tick(fps)

    def quit(self):
        pygame.quit()

    def draw(self, tournament: Tournament, match: Optional[Match],
             match_id: Optional[str], fast_forward: bool):
        self.screen.fill(BG_COLOR)
        if match is not None:
            self._draw_header(match, tournament, match_id, fast_forward)
            self._draw_pitch(match)
            self._draw_events(match)
        else:
            self._draw_completion(tournament)
        self._draw_wc_frame()
        pygame.display.flip()

    def draw_best_xi_recap(self, t: Tournament, progress: float):
        """Shorts용 Best XI — 세로 레이아웃, 4-3-3 피치 + 11명 순차 등장."""
        self.screen.fill(BG_COLOR)
        awards = t.compute_awards() if hasattr(t, 'compute_awards') else None
        if not awards or not awards.get('best_xi'):
            self._draw_wc_frame()
            pygame.display.flip()
            return
        best_xi = awards['best_xi']

        # 타이틀
        title = self.font_recap_title.render(
            'BEST XI', True, ACCENT_GOLD)
        self.screen.blit(title, (self.W // 2 - title.get_width() // 2, 60))
        sub = self.font_recap_sub.render(
            'TOURNAMENT TEAM  —  avg match rating',
            True, TEXT_SECONDARY)
        self.screen.blit(sub, (self.W // 2 - sub.get_width() // 2, 130))

        # 피치 — 세로 9:16. 공격 위로
        pitch_x = 90
        pitch_y = 200
        pitch_w = 900
        pitch_h = 1400
        self._draw_xi_pitch(pitch_x, pitch_y, pitch_w, pitch_h)

        # 슬롯 좌표 — Renderer 와 동일 비율
        slot_coords = {
            'GK':  (0.50, 0.93),
            'CB1': (0.36, 0.78),
            'CB2': (0.64, 0.78),
            'LB':  (0.12, 0.74),
            'RB':  (0.88, 0.74),
            'DM':  (0.50, 0.60),
            'CM':  (0.30, 0.50),
            'AM':  (0.50, 0.38),
            'ST':  (0.50, 0.13),
            'LW':  (0.18, 0.20),
            'RW':  (0.82, 0.20),
        }
        slot_keys = ['GK', 'CB1', 'CB2', 'LB', 'RB',
                      'DM', 'CM', 'AM', 'ST', 'LW', 'RW']
        appear_threshold = [(i + 1) / 11 for i in range(11)]

        for idx, p in enumerate(best_xi):
            if idx >= len(slot_keys):
                break
            slot = slot_keys[idx]
            sx_ratio, sy_ratio = slot_coords[slot]
            if progress < appear_threshold[idx] - 0.05:
                continue
            local = min(1.0, max(0.0,
                                  (progress - (appear_threshold[idx] - 0.05)) / 0.05))
            cx = pitch_x + int(pitch_w * sx_ratio)
            cy = pitch_y + int(pitch_h * sy_ratio)
            label_pos = slot.rstrip('12') if slot.startswith('CB') else slot
            self._draw_xi_player_shorts(cx, cy, p, local, t, label_pos)

        self._draw_wc_frame()
        pygame.display.flip()

    def _draw_xi_pitch(self, x: int, y: int, w: int, h: int):
        """Shorts Best XI 피치 (세로)."""
        stripes = 10
        sh = h / stripes
        for i in range(stripes):
            c = PITCH_GREEN_DARK if i % 2 == 0 else PITCH_GREEN_LIGHT
            pygame.draw.rect(self.screen, c,
                              (x, int(y + i * sh), w, int(sh + 1)))
        line = PITCH_LINE
        pygame.draw.rect(self.screen, line, (x, y, w, h), 3)
        pygame.draw.line(self.screen, line, (x, y + h // 2), (x + w, y + h // 2), 3)
        pygame.draw.circle(self.screen, line,
                            (x + w // 2, y + h // 2), 70, 3)
        pb_w = int(w * 0.55)
        pb_h = int(h * 0.13)
        pb_x = x + (w - pb_w) // 2
        pygame.draw.rect(self.screen, line, (pb_x, y, pb_w, pb_h), 3)
        pygame.draw.rect(self.screen, line,
                          (pb_x, y + h - pb_h, pb_w, pb_h), 3)

    def _draw_xi_player_shorts(self, cx: int, cy: int, p, local: float,
                                 t, label_pos: str = ''):
        """Shorts Best XI 한 선수."""
        team = t.teams.get(p.country) if hasattr(t, 'teams') else None
        color = team.color if team else (180, 180, 180)
        outline = team.secondary if team else (240, 240, 240)
        r = int(28 * local)
        if r <= 0:
            return
        pygame.draw.circle(self.screen, outline, (cx, cy), r + 3)
        pygame.draw.circle(self.screen, color, (cx, cy), r)
        # 라벨 — F 픽스: 선수 실제 position 우선, 없으면 슬롯
        display_pos = (p.position if getattr(p, 'position', '') else label_pos)
        if display_pos and local > 0.7:
            pos_s = self.font_recap_sub.render(display_pos, True, (255, 255, 255))
            self.screen.blit(pos_s, (cx - pos_s.get_width() // 2,
                                       cy - pos_s.get_height() // 2))
        last = p.name.split()[-1]
        surf = self.font_recap_sub.render(last, True, TEXT_PRIMARY)
        bg = self.font_recap_sub.render(last, True, (0, 0, 0))
        lx = cx - surf.get_width() // 2
        ly = cy + r + 6
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            self.screen.blit(bg, (lx + dx, ly + dy))
        self.screen.blit(surf, (lx, ly))
        rating_txt = f'{p.avg_rating:.2f}'
        rs = self.font_recap.render(rating_txt, True, ACCENT_GOLD)
        rbg = self.font_recap.render(rating_txt, True, (0, 0, 0))
        rx = cx - rs.get_width() // 2
        ry = ly + surf.get_height() + 2
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            self.screen.blit(rbg, (rx + dx, ry + dy))
        self.screen.blit(rs, (rx, ry))

    def draw_stats_recap(self, t: Tournament, progress: float):
        """Shorts용 간소화 recap — Top scorers / assists 리더보드."""
        self.screen.fill(BG_COLOR)
        hist = getattr(t, 'stat_history', [])
        if not hist:
            self._draw_wc_frame()
            pygame.display.flip()
            return

        cur_idx = max(1, min(len(hist), int(round(progress * len(hist)))))
        snap = hist[cur_idx - 1][1]
        cur_mid = hist[cur_idx - 1][0]

        # 제목
        title = self.font_recap_title.render('GOLDEN BOOT', True, ACCENT_GOLD)
        self.screen.blit(title, (self.W // 2 - title.get_width() // 2, 60))
        sub = self.font_recap_sub.render(
            f'after {cur_mid}  —  {cur_idx}/{len(hist)} matches',
            True, TEXT_SECONDARY)
        self.screen.blit(sub, (self.W // 2 - sub.get_width() // 2, 130))

        # Top scorers
        top_scorers = sorted(
            snap.items(),
            key=lambda kv: (-kv[1].get('goals', 0), -kv[1].get('assists', 0))
        )[:8]
        self._draw_recap_list(170, 'TOP SCORERS', top_scorers, 'goals', t)

        # Top assists
        title2 = self.font_recap_title.render('ASSIST KING', True, (140, 200, 240))
        self.screen.blit(title2, (self.W // 2 - title2.get_width() // 2, 1010))
        top_assists = sorted(
            snap.items(),
            key=lambda kv: (-kv[1].get('assists', 0), -kv[1].get('goals', 0))
        )[:8]
        self._draw_recap_list(1090, 'TOP ASSISTS', top_assists, 'assists', t)

        self._draw_wc_frame()
        pygame.display.flip()

    def _draw_recap_list(self, top_y: int, header: str, players: list,
                         key: str, t: Tournament):
        x = 60
        w = self.W - 120
        row_h = 80
        for rank, (name, info) in enumerate(players):
            y = top_y + rank * row_h
            country = info.get('country', '')
            team = t.teams.get(country) if hasattr(t, 'teams') else None
            color = team.color if team else (180, 180, 180)
            # 색 사이드 바
            pygame.draw.rect(self.screen, color, (x, y, 8, row_h - 12),
                              border_radius=4)
            # 순위
            rank_s = self.font_recap.render(f'{rank + 1}.', True, TEXT_SECONDARY)
            self.screen.blit(rank_s, (x + 24, y + 10))
            # 이름
            name_s = self.font_recap.render(name, True, TEXT_PRIMARY)
            self.screen.blit(name_s, (x + 90, y + 10))
            # 국가
            country_s = self.font_recap_sub.render(country, True, TEXT_SECONDARY)
            self.screen.blit(country_s, (x + 90, y + 46))
            # 값
            val = info.get(key, 0)
            val_s = self.font_recap.render(str(val), True, ACCENT_GOLD)
            self.screen.blit(val_s,
                (x + w - val_s.get_width() - 16, y + 18))

    # ── 헤더 (스코어 + 팀) ────────────────────────────────
    def _draw_header(self, match: Match, t: Tournament,
                      match_id: Optional[str], fast: bool):
        # 단계
        if match_id:
            stage = t.stage_of(match_id)
            stage_s = self.font_stage.render(stage.upper(), True, ACCENT_GOLD)
            self.screen.blit(stage_s,
                (self.W // 2 - stage_s.get_width() // 2, 20))

        # 빅 스코어보드 — 좌 홈 / 중앙 스코어+분 / 우 어웨이
        home_x_center = 175
        away_x_center = self.W - 175
        score_y = 70

        # 직사각 국기 뱃지 (스테이지 라벨과 안 겹치게 살짝 아래)
        self._draw_team_badge(match.home, home_x_center, score_y + 60)
        self._draw_team_badge(match.away, away_x_center, score_y + 60)

        # 스코어 (큰 숫자)
        h_g, a_g = match.result.home_goals, match.result.away_goals
        h_score_s = self.font_score.render(str(h_g), True, TEXT_PRIMARY)
        a_score_s = self.font_score.render(str(a_g), True, TEXT_PRIMARY)
        # 가운데 정렬: 양 옆 팀 사이의 빈 공간 가운데
        score_cx = self.W // 2
        dash_s = self.font_score.render('-', True, TEXT_SECONDARY)
        self.screen.blit(h_score_s,
            (score_cx - 80 - h_score_s.get_width(), score_y))
        self.screen.blit(dash_s,
            (score_cx - dash_s.get_width() // 2, score_y))
        self.screen.blit(a_score_s, (score_cx + 80, score_y))

        # PK
        if match.in_pk or match.result.went_to_pk:
            h_pk = match.result.home_pk
            a_pk = match.result.away_pk
            pk_s = self.font_event.render(
                f'PK  {h_pk} - {a_pk}', True, ACCENT_GOLD)
            self.screen.blit(pk_s,
                (score_cx - pk_s.get_width() // 2, score_y + 130))

        # 분 (스코어 아래)
        if match.in_pk:
            minute_text = f'PK {match.pk_round}'
        elif match.finished:
            minute_text = 'FULL TIME'
        else:
            minute_text = f"{min(match.minute, 120)}'"
        min_s = self.font_minute.render(minute_text, True, ACCENT_GOLD)
        self.screen.blit(min_s,
            (score_cx - min_s.get_width() // 2, score_y + 175))

        # FAST FORWARD 칩
        if fast:
            ff = self.font_small.render('  FAST  ', True, BG_COLOR)
            rect = pygame.Rect(self.W - 100, 25,
                                ff.get_width() + 10, ff.get_height() + 6)
            pygame.draw.rect(self.screen, ACCENT_GOLD, rect, border_radius=8)
            self.screen.blit(ff, (rect.x + 5, rect.y + 3))

    def _get_flag_surface(self, team: Team, w: int, h: int) -> Optional[pygame.Surface]:
        flag_code = _resolve_flag_code(team.iso2, team.code)
        key = (flag_code, w, h)
        cached = self._flag_cache.get(key)
        if cached is not None:
            return cached
        path = _ensure_flag_png(flag_code)
        if path is None:
            return None
        try:
            raw = pygame.image.load(str(path)).convert_alpha()
            scaled = pygame.transform.smoothscale(raw, (w, h))
            mask = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255),
                              (0, 0, w, h), border_radius=12)
            out = pygame.Surface((w, h), pygame.SRCALPHA)
            out.blit(scaled, (0, 0))
            out.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            self._flag_cache[key] = out
            return out
        except (pygame.error, OSError) as e:
            print(f'[flag] render error {flag_code}: {e}')
            return None

    def _draw_team_badge(self, team: Team, cx: int, cy: int):
        # 직사각 국기 (둥근 모서리)
        rect_w, rect_h = 210, 130
        flag = self._get_flag_surface(team, rect_w, rect_h)
        rx = cx - rect_w // 2
        ry = cy - rect_h // 2
        if flag is not None:
            self.screen.blit(flag, (rx, ry))
            # 외곽 라인 (살짝 입체감)
            pygame.draw.rect(self.screen, (235, 235, 240),
                              (rx, ry, rect_w, rect_h), 2, border_radius=12)
        else:
            # 다운로드 실패 시 fallback: 색 동그라미 + 코드
            pygame.draw.circle(self.screen, team.color, (cx, cy), 56)
            pygame.draw.circle(self.screen, team.secondary, (cx, cy), 56, 6)
            code_in = self.font_team_big.render(team.code, True, (250, 250, 250))
            self.screen.blit(code_in,
                (cx - code_in.get_width() // 2,
                 cy - code_in.get_height() // 2 + 2))
        # 팀 코드 (국기 아래)
        code_s = self.font_team_code.render(team.code, True, ACCENT_GOLD)
        self.screen.blit(code_s,
            (cx - code_s.get_width() // 2, ry + rect_h + 6))

    # ── 피치 (세로) ───────────────────────────────────────
    def _draw_pitch(self, match: Match):
        # 패널
        panel = pygame.Rect(
            self.PITCH_LEFT - self.PAD, self.PITCH_TOP - self.PAD,
            self.PITCH_W + self.PAD * 2, self.PITCH_H + self.PAD * 2)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel, border_radius=12)

        # 잔디 스트라이프 (수평 스트라이프 — 세로 피치에서)
        stripes = 12
        stripe_h = self.PITCH_H / stripes
        for i in range(stripes):
            c = PITCH_GREEN_DARK if i % 2 == 0 else PITCH_GREEN_LIGHT
            r = pygame.Rect(self.PITCH_LEFT,
                            int(self.PITCH_TOP + i * stripe_h),
                            self.PITCH_W,
                            int(stripe_h + 1))
            pygame.draw.rect(self.screen, c, r)

        self._draw_pitch_lines()

        # 골 플래시 (위쪽=홈, 아래쪽=어웨이)
        if match.goal_flash > 0:
            ratio = match.goal_flash / 24.0
            alpha = int(150 * ratio)
            overlay = pygame.Surface((self.PITCH_W, self.PITCH_H),
                                      pygame.SRCALPHA)
            color = (match.home.color if match.goal_flash_team == 0
                      else match.away.color)
            band_h = int(self.PITCH_H * 0.3)
            band_y = 0 if match.goal_flash_team == 0 else self.PITCH_H - band_h
            pygame.draw.rect(overlay, (*color, alpha),
                              (0, band_y, self.PITCH_W, band_h))
            self.screen.blit(overlay, (self.PITCH_LEFT, self.PITCH_TOP))

        # 좌표 변환 스케일
        sx_scale = self.PITCH_W / Match.PITCH_H  # sim_y → screen_x
        sy_scale = self.PITCH_H / Match.PITCH_W  # sim_x → screen_y

        # 선수
        away_color = (match.away.secondary if match.away_use_secondary
                       else match.away.color)
        away_outline = (match.away.color if match.away_use_secondary
                         else match.away.secondary)

        # 팀별 max/min 평점 라벨 (4명만)
        label_for = compute_rating_labels(match)

        for p in match.players:
            if not p.on_pitch:
                continue
            cx = self.PITCH_LEFT + p.y * sx_scale
            cy = self.PITCH_TOP + p.x * sy_scale
            if p.team_idx == 0:
                color = match.home.color
                outline = match.home.secondary
            else:
                color = away_color
                outline = away_outline
            r = 14 if p.role == 'GK' else 12
            pygame.draw.circle(self.screen, outline, (int(cx), int(cy)), r + 2)
            pygame.draw.circle(self.screen, color, (int(cx), int(cy)), r)

            lbl = label_for.get(id(p))
            if lbl is not None:
                text, txt_color = lbl
                surf = self.font_pin.render(text, True, txt_color)
                bg = self.font_pin.render(text, True, (0, 0, 0))
                lx = int(cx - surf.get_width() / 2)
                ly = int(cy - r - 20)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    self.screen.blit(bg, (lx + dx, ly + dy))
                self.screen.blit(surf, (lx, ly))

        # 공
        bx = self.PITCH_LEFT + match.ball_y * sx_scale
        by = self.PITCH_TOP + match.ball_x * sy_scale
        pygame.draw.circle(self.screen, (15, 15, 18), (int(bx), int(by)), 10)
        pygame.draw.circle(self.screen, (250, 250, 250), (int(bx), int(by)), 8)

    def _draw_pitch_lines(self):
        x, y, w, h = self.PITCH_LEFT, self.PITCH_TOP, self.PITCH_W, self.PITCH_H
        line = PITCH_LINE
        # 외곽
        pygame.draw.rect(self.screen, line, (x, y, w, h), 3)
        # 중앙선 (수평, 세로 피치이므로)
        pygame.draw.line(self.screen, line,
                          (x, y + h // 2), (x + w, y + h // 2), 3)
        # 중앙 원
        pygame.draw.circle(self.screen, line,
                            (x + w // 2, y + h // 2), 100, 3)
        pygame.draw.circle(self.screen, line,
                            (x + w // 2, y + h // 2), 6)
        # 페널티 박스 (위/아래)
        pb_w = int(w * 0.55)   # 가로폭
        pb_h = int(h * 0.13)   # 세로 깊이
        pb_x = x + (w - pb_w) // 2
        pygame.draw.rect(self.screen, line, (pb_x, y, pb_w, pb_h), 3)
        pygame.draw.rect(self.screen, line,
                          (pb_x, y + h - pb_h, pb_w, pb_h), 3)
        # 골에리어
        ga_w = int(w * 0.28)
        ga_h = int(h * 0.05)
        ga_x = x + (w - ga_w) // 2
        pygame.draw.rect(self.screen, line, (ga_x, y, ga_w, ga_h), 3)
        pygame.draw.rect(self.screen, line,
                          (ga_x, y + h - ga_h, ga_w, ga_h), 3)
        # 페널티 점
        pygame.draw.circle(self.screen, line,
                            (x + w // 2, y + int(h * 0.105)), 4)
        pygame.draw.circle(self.screen, line,
                            (x + w // 2, y + h - int(h * 0.105)), 4)

    # ── 이벤트 ────────────────────────────────────────────
    def _draw_events(self, match: Match):
        x = 40
        y = self.EVENTS_TOP
        w = self.W - 80
        h = self.EVENTS_H
        pygame.draw.rect(self.screen, PANEL_COLOR, (x, y, w, h),
                          border_radius=10)
        # 최근 3개
        events = [e for e in match.events
                  if e.kind in ('goal', 'own_goal', 'pk', 'half', 'fulltime',
                                 'kickoff')][-3:][::-1]
        cur_y = y + 14
        for ev in events:
            if ev.kind == 'pk':
                min_text = 'PK'
            elif ev.kind in ('half', 'fulltime', 'kickoff'):
                min_text = ev.text[:14]
            else:
                min_text = f"{ev.minute}'"
            min_s = self.font_event.render(min_text, True, TEXT_SECONDARY)
            self.screen.blit(min_s, (x + 18, cur_y))
            # 색 점
            if ev.team_idx == 0:
                color = match.home.color
            elif ev.team_idx == 1:
                color = match.away.color
            else:
                color = TEXT_SECONDARY
            pygame.draw.circle(self.screen, color,
                                (x + 130, cur_y + 14), 7)
            # 텍스트
            text_color = (ACCENT_GOLD if ev.kind in ('goal', 'own_goal', 'pk')
                           else TEXT_PRIMARY)
            text_s = self.font_event.render(ev.text[:50], True, text_color)
            self.screen.blit(text_s, (x + 150, cur_y))
            cur_y += 42

    # ── 종료 화면 ─────────────────────────────────────────
    def _draw_completion(self, t: Tournament):
        champ = t.champion()
        runner = t.runner_up()
        third = t.third()
        cx = self.W // 2
        cy = 200

        title = self.font_recap_title.render('WORLD CHAMPIONS', True, ACCENT_GOLD)
        self.screen.blit(title, (cx - title.get_width() // 2, cy))
        cy += 100

        if champ:
            # 큰 팀 배지
            pygame.draw.circle(self.screen, champ.color, (cx, cy + 80), 110)
            pygame.draw.circle(self.screen, champ.secondary, (cx, cy + 80), 110, 8)
            code_s = self.font_score.render(champ.code, True, (250, 250, 250))
            self.screen.blit(code_s,
                (cx - code_s.get_width() // 2,
                 cy + 80 - code_s.get_height() // 2 + 8))
            cy += 220
            name_big = pygame.font.SysFont(
                'malgungothic,arial', 64, bold=True)
            n_s = name_big.render(champ.name, True, champ.color)
            self.screen.blit(n_s, (cx - n_s.get_width() // 2, cy))
            cy += 90

        # 준우승 / 3위
        if runner:
            r_s = self.font_recap.render(
                f'Runner-up:  {runner.name}', True, TEXT_PRIMARY)
            self.screen.blit(r_s, (cx - r_s.get_width() // 2, cy))
            cy += 50
        if third:
            t_s = self.font_recap.render(
                f'3rd Place:  {third.name}', True, TEXT_SECONDARY)
            self.screen.blit(t_s, (cx - t_s.get_width() // 2, cy))
            cy += 60

        # Awards
        awards = t.compute_awards() if hasattr(t, 'compute_awards') else None
        if awards:
            cy += 30
            gb = awards.get('golden_boot')
            mvp = awards.get('mvp')
            if gb:
                lbl = self.font_recap_sub.render(
                    'GOLDEN BOOT', True, ACCENT_GOLD)
                self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy))
                cy += 40
                txt = self.font_recap.render(
                    f'{gb.name} ({gb.country})  {gb.goals}G',
                    True, TEXT_PRIMARY)
                self.screen.blit(txt, (cx - txt.get_width() // 2, cy))
                cy += 70
            if mvp:
                lbl = self.font_recap_sub.render(
                    'MVP', True, (140, 200, 240))
                self.screen.blit(lbl, (cx - lbl.get_width() // 2, cy))
                cy += 40
                txt = self.font_recap.render(
                    f'{mvp.name} ({mvp.country})',
                    True, TEXT_PRIMARY)
                self.screen.blit(txt, (cx - txt.get_width() // 2, cy))

    # ── WC 프레임 (3국 컬러) ──────────────────────────────
    def _draw_wc_frame(self):
        w, h = self.W, self.H
        third = w // 3
        pygame.draw.rect(self.screen, WC_CAN_RED, (0, 0, third, WC_BAR_H))
        pygame.draw.rect(self.screen, WC_MEX_GREEN,
                          (third, 0, third, WC_BAR_H))
        pygame.draw.rect(self.screen, WC_USA_BLUE,
                          (third * 2, 0, w - third * 2, WC_BAR_H))
        pygame.draw.rect(self.screen, WC_USA_BLUE,
                          (0, h - WC_BAR_H, third, WC_BAR_H))
        pygame.draw.rect(self.screen, WC_MEX_GREEN,
                          (third, h - WC_BAR_H, third, WC_BAR_H))
        pygame.draw.rect(self.screen, WC_CAN_RED,
                          (third * 2, h - WC_BAR_H, w - third * 2, WC_BAR_H))
        pygame.draw.rect(self.screen, WC_GOLD,
                          (0, WC_BAR_H, 3, h - WC_BAR_H * 2))
        pygame.draw.rect(self.screen, WC_GOLD,
                          (w - 3, WC_BAR_H, 3, h - WC_BAR_H * 2))
