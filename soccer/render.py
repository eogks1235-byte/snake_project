"""축구 매치 시각화 — 다크 톤 + 픽셀 경기장 + 점 선수 + 이벤트 피드."""
import pygame
from typing import List, Optional

from .match_engine import Match, MatchEvent
from .teams import Team
from .tournament import Tournament


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
        pygame.display.flip()

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

        pygame.display.flip()

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
