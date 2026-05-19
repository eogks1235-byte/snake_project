"""pygame 기반 시각화 - 파스텔 + 타이포그래픽 오버레이 스타일"""
import pygame
import numpy as np
from typing import List

from .simulation import Simulation
from .agents import Agent

# 파스텔 팔레트 (peach/cream — claude.com/code-with-claude 톤)
BG_COLOR = (240, 222, 205)        # peach cream
PANEL_COLOR = (228, 208, 188)     # 카드 톤 (살짝 진한 peach)
GRID_BG = (250, 236, 220)         # 빈 칸: 거의 흰 크림
GRID_LINE = (210, 188, 165)       # 격자선: 따뜻한 뮤트
TEXT_PRIMARY = (55, 38, 28)       # 진한 brown
TEXT_SECONDARY = (120, 92, 72)    # 뮤트 brown
BAR_TRACK = (215, 195, 175)
HEAD_OUTLINE = (40, 28, 20)
INVULN_OUTLINE = (210, 150, 40)
FLASH_COLOR = (55, 38, 28)
WALL_COLOR = (90, 68, 50)         # 벽: 진한 warm brown

# 영토 위에 깔리는 글리프 패턴 — 에이전트마다 다른 문자 배정.
GLYPH_POOL = ['*', '\\', '/', '+', '#', '◆', '※', 'x', '◇', '·']


class Renderer:
    def __init__(
        self,
        sim: Simulation,
        agents: List[Agent],
        cell_size: int = 16,
        header_height: int = 220,
        sidebar_height: int = None,
        margin_x: int = 60,
        layout: str = 'portrait',
    ):
        self.sim = sim
        self.agents = agents
        self.cell_size = cell_size
        self.layout = layout

        self.grid_w = sim.grid.width * cell_size
        self.grid_h = sim.grid.height * cell_size

        # 레이아웃별 기본값
        #   portrait : 9:16 (Shorts) — 헤더 + 그리드 + 큰 랭킹 패널
        #   square   : 1:1 — 그리드 위에 라벨만, 패널 없음
        if layout == 'portrait':
            if sidebar_height is None:
                # window_h = 1920 가 되도록 역산 (cell_size=16 → grid_h=960)
                sidebar_height = max(480, 1920 - self.grid_h - header_height)
        else:
            if sidebar_height is None:
                sidebar_height = 30

        self.margin_x = margin_x
        self.window_w = self.grid_w + 2 * margin_x
        self.window_h = self.grid_h + header_height + sidebar_height
        self.sidebar_height = sidebar_height

        self.grid_x = margin_x
        self.grid_y = header_height
        self.sidebar_y = self.grid_y + self.grid_h + 24

        self._label_fonts: dict = {}

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption('Territory War')
        self.clock = pygame.time.Clock()

        if layout == 'portrait':
            self.font_title = pygame.font.SysFont('arial,malgungothic', 56, bold=True)
            self.font_sub = pygame.font.SysFont('arial,malgungothic', 22)
            self.font_stat = pygame.font.SysFont('arial,malgungothic', 26, bold=True)
            self.font_pct = pygame.font.SysFont('arial,malgungothic', 34, bold=True)
            self.font_pin = pygame.font.SysFont('arial,malgungothic', 22, bold=True)
            self.font_desc = pygame.font.SysFont('arial,malgungothic', 18)
        else:
            self.font_title = pygame.font.SysFont('arial,malgungothic', 28, bold=True)
            self.font_sub = pygame.font.SysFont('arial,malgungothic', 13)
            self.font_stat = pygame.font.SysFont('arial,malgungothic', 14, bold=True)
            self.font_pct = pygame.font.SysFont('arial,malgungothic', 16, bold=True)
            self.font_pin = pygame.font.SysFont('arial,malgungothic', 12, bold=True)
            self.font_desc = pygame.font.SysFont('arial,malgungothic', 13)

        self._color_lookup = self._build_color_lookup()
        self._gridline_overlay = self._build_gridline_overlay()
        self._glyph_surfaces = self._build_glyph_surfaces()

    def _build_color_lookup(self) -> np.ndarray:
        # +2 슬롯: 0(빈 칸), 1..N(에이전트), 마지막 슬롯=벽
        # numpy 음수 인덱싱 덕분에 cells 값 -1 (WALL) 은 자동으로 마지막 슬롯을 가리킨다.
        lookup = np.zeros((len(self.agents) + 2, 3), dtype=np.uint8)
        lookup[0] = GRID_BG
        for agent in self.agents:
            lookup[agent.id] = agent.color
        lookup[-1] = WALL_COLOR
        return lookup

    def _winner_highlight_lookup(self, winner_id: int,
                                   winner_team: int = None) -> np.ndarray:
        """Winner 외 영토는 회색계로 desaturate한 lookup."""
        lookup = self._color_lookup.copy()
        for agent in self.agents:
            keep = (agent.id == winner_id
                    or (winner_team is not None
                        and getattr(self.sim, 'teams', {}).get(agent.id) == winner_team))
            if keep:
                continue
            r, g, b = agent.color
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            # 60% gray + 40% original — 흐릿하지만 정체성 유지
            lookup[agent.id] = (
                int(gray * 0.60 + r * 0.40),
                int(gray * 0.60 + g * 0.40),
                int(gray * 0.60 + b * 0.40),
            )
        return lookup

    def _build_gridline_overlay(self) -> pygame.Surface:
        """격자선 오버레이 - 한 번만 만들고 재사용"""
        surf = pygame.Surface((self.grid_w, self.grid_h), pygame.SRCALPHA)
        for i in range(self.sim.grid.width + 1):
            x = i * self.cell_size
            pygame.draw.line(surf, (*GRID_LINE, 140), (x, 0), (x, self.grid_h), 1)
        for i in range(self.sim.grid.height + 1):
            y = i * self.cell_size
            pygame.draw.line(surf, (*GRID_LINE, 140), (0, y), (self.grid_w, y), 1)
        return surf

    def _build_glyph_surfaces(self) -> dict:
        """에이전트별 cell_size 크기의 글리프 surface 캐시.

        각 에이전트에 고유 문자를 배정, 자기 색의 진한 톤으로 렌더해서
        솔리드 컬러 위에 텍스처처럼 겹쳐 보이게 한다.
        """
        cache: dict = {}
        cs = self.cell_size
        glyph_size = max(8, int(cs * 1.5))
        font = pygame.font.SysFont('couriernew,consolas,arial', glyph_size, bold=True)
        for i, agent in enumerate(self.agents):
            glyph = GLYPH_POOL[i % len(GLYPH_POOL)]
            r, g, b = agent.color
            # 같은 hue 의 더 진한 톤 (가독성용)
            dark = (max(0, r - 95), max(0, g - 95), max(0, b - 95))
            text_surf = font.render(glyph, True, dark)
            tw, th = text_surf.get_size()
            cell_surf = pygame.Surface((cs, cs), pygame.SRCALPHA)
            cell_surf.blit(text_surf, ((cs - tw) // 2, (cs - th) // 2))
            cache[agent.id] = cell_surf
        return cache

    def draw(self, fast_forward: bool = False):
        self.screen.fill(BG_COLOR)
        self._draw_header(fast_forward)
        self._draw_grid_panel()
        winner_id = getattr(self.sim, 'winner_id', None)
        if winner_id is None:
            self._draw_area_labels()
        else:
            self._draw_winner_overlay(winner_id)
        if self.layout == 'portrait':
            self._draw_sidebar()
        pygame.display.flip()

    def _get_label_font(self, size: int):
        if size not in self._label_fonts:
            self._label_fonts[size] = pygame.font.SysFont(
                'arial,malgungothic', size, bold=True)
        return self._label_fonts[size]

    def _blit_text_outline(self, text, font, pos,
                            color=(255, 255, 255), outline=(0, 0, 0)):
        rendered = font.render(text, True, color)
        shadow = font.render(text, True, outline)
        x, y = pos
        for dx, dy in ((-2, -2), (-2, 0), (-2, 2), (0, -2),
                       (0, 2), (2, -2), (2, 0), (2, 2)):
            self.screen.blit(shadow, (x + dx, y + dy))
        self.screen.blit(rendered, (x, y))

    def _draw_winner_overlay(self, winner_id: int):
        """게임 종료 후 winner 강조 — 그리드 중앙에 큰 WINNER 텍스트."""
        winner = next((a for a in self.agents if a.id == winner_id), None)
        if winner is None:
            return
        big_size = 84 if self.layout == 'portrait' else 48
        small_size = 28 if self.layout == 'portrait' else 18
        big = self._get_label_font(big_size)
        small = self._get_label_font(small_size)

        cx = self.grid_x + self.grid_w // 2
        cy = self.grid_y + self.grid_h // 2

        label_text = 'WINNER'
        name_text = winner.name
        areas = self.sim.get_areas()
        total = self.sim.grid.total_cells()
        pct = (100.0 * areas.get(winner.id, 0) / total) if total else 0
        pct_text = f'{pct:.0f}%'

        label_w, label_h = small.size(label_text)
        name_w, name_h = big.size(name_text)
        pct_w, pct_h = big.size(pct_text)

        gap = 8
        total_h = label_h + name_h + pct_h + gap * 2
        y0 = cy - total_h // 2

        self._blit_text_outline(label_text, small,
                                  (cx - label_w // 2, y0),
                                  color=(245, 230, 215), outline=(40, 25, 15))
        self._blit_text_outline(name_text, big,
                                  (cx - name_w // 2, y0 + label_h + gap),
                                  color=winner.color, outline=(20, 12, 6))
        self._blit_text_outline(pct_text, big,
                                  (cx - pct_w // 2,
                                   y0 + label_h + name_h + gap * 2),
                                  color=(255, 255, 255), outline=(20, 12, 6))

    def _draw_area_labels(self):
        """5% 이상 점유한 에이전트의 영토 무게중심에 [이름 + %] 라벨."""
        cells = self.sim.grid.cells
        total = self.sim.grid.total_cells()
        areas = self.sim.get_areas()
        cs = self.cell_size

        for agent in self.agents:
            area = areas.get(agent.id, 0)
            pct = 100.0 * area / total if total else 0
            if pct < 5.0:
                continue

            ys, xs = (cells == agent.id).nonzero()
            if len(xs) == 0:
                continue

            # centroid → 가장 가까운 자기 영토 칸 (라벨이 자기 색 위에 있도록)
            cx_g = float(xs.mean())
            cy_g = float(ys.mean())
            dists = (xs - cx_g) ** 2 + (ys - cy_g) ** 2
            i = int(dists.argmin())
            gx, gy = int(xs[i]), int(ys[i])

            px = self.grid_x + gx * cs + cs // 2
            py = self.grid_y + gy * cs + cs // 2

            # 폰트 크기: 영토 크기에 비례
            big_size = int(min(54, max(22, 16 + (area ** 0.5) * 0.9)))
            small_size = max(11, int(big_size * 0.42))
            big = self._get_label_font(big_size)
            small = self._get_label_font(small_size)

            pct_text = f'{pct:.0f}%'
            name_text = agent.name
            pct_w, pct_h = big.size(pct_text)
            name_w, name_h = small.size(name_text)

            # 위: 이름, 아래: 큰 비율
            name_pos = (px - name_w // 2, py - pct_h // 2 - name_h - 2)
            pct_pos = (px - pct_w // 2, py - pct_h // 2)

            self._blit_text_outline(name_text, small, name_pos)
            self._blit_text_outline(pct_text, big, pct_pos)

    def _draw_header(self, fast_forward: bool):
        if self.layout == 'portrait':
            title_y, sub_y = 56, 130
            tick_y, ff_y = 64, 120
        else:
            title_y, sub_y = 22, 56
            tick_y, ff_y = 28, 52

        title_text = 'TERRITORY WAR'
        if getattr(self.sim, 'team_mode', False):
            title_text = 'TEAM WAR'
        if getattr(self.sim, 'time_limit', None) is not None:
            title_text = 'BLITZ ' + title_text
        title = self.font_title.render(title_text, True, TEXT_PRIMARY)
        map_name = getattr(self.sim, 'map_name', 'open')
        n_alive = sum(1 for a in self.agents
                      if not getattr(self.sim, 'dead', {}).get(a.id, False))
        mode_label = 'team mode' if getattr(self.sim, 'team_mode', False) else 'free-for-all'
        # Blitz 모드면 남은 시간(초) 표시
        time_limit = getattr(self.sim, 'time_limit', None)
        if time_limit is not None:
            remaining = max(0, (time_limit - self.sim.tick) / 30.0)
            mode_label = f'BLITZ  {remaining:.1f}s left'
        subtitle = (f'{n_alive}/{len(self.agents)} alive  ·  {mode_label}  ·  map: {map_name}')
        sub = self.font_sub.render(subtitle, True, TEXT_SECONDARY)
        self.screen.blit(title, (self.margin_x, title_y))
        self.screen.blit(sub, (self.margin_x, sub_y))

        # 우측: tick 카운터
        tick_label = self.font_pin.render(f'TICK  {self.sim.tick:>5}',
                                          True, TEXT_SECONDARY)
        self.screen.blit(
            tick_label,
            (self.window_w - self.margin_x - tick_label.get_width(), tick_y),
        )

        # Fast forward 핀
        if fast_forward:
            label = self.font_pin.render('FAST FORWARD', True, BG_COLOR)
            pad_x, pad_y = 10, 6
            rect = pygame.Rect(
                self.window_w - self.margin_x - label.get_width() - pad_x * 2,
                ff_y, label.get_width() + pad_x * 2, label.get_height() + pad_y * 2,
            )
            pygame.draw.rect(self.screen, (255, 215, 90), rect, border_radius=10)
            self.screen.blit(label, (rect.x + pad_x, rect.y + pad_y))

    def _draw_grid_panel(self):
        # 패널 배경 (격자보다 살짝 큰 카드 느낌)
        pad = 8
        panel = pygame.Rect(
            self.grid_x - pad, self.grid_y - pad,
            self.grid_w + pad * 2, self.grid_h + pad * 2,
        )
        pygame.draw.rect(self.screen, PANEL_COLOR, panel, border_radius=10)

        # 셀 색 — winner 단계엔 desaturated lookup 으로 winner 외 영토 흐림
        cells = self.sim.grid.cells
        winner_id = getattr(self.sim, 'winner_id', None)
        if winner_id is not None:
            lookup = self._winner_highlight_lookup(
                winner_id, getattr(self.sim, 'winner_team', None))
        else:
            lookup = self._color_lookup
        color_arr = lookup[cells]
        surf = pygame.surfarray.make_surface(color_arr.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (self.grid_w, self.grid_h))
        self.screen.blit(surf, (self.grid_x, self.grid_y))

        # 글리프 오버레이: 솔리드 컬러 위에 에이전트별 문자 패턴
        cs = self.cell_size
        for agent in self.agents:
            glyph_surf = self._glyph_surfaces.get(agent.id)
            if glyph_surf is None:
                continue
            ys, xs = (cells == agent.id).nonzero()
            for y, x in zip(ys.tolist(), xs.tolist()):
                self.screen.blit(glyph_surf,
                                 (self.grid_x + x * cs, self.grid_y + y * cs))

        # 사망 마커 (해골): 죽은 에이전트의 마지막 위치에 영구 표시
        for agent in self.agents:
            if not getattr(self.sim, 'dead', {}).get(agent.id, False):
                continue
            pos = self.sim.death_positions.get(agent.id)
            if pos is None:
                continue
            dx, dy = pos
            cx = self.grid_x + dx * cs + cs // 2
            cy = self.grid_y + dy * cs + cs // 2
            self._draw_skull(cx, cy, size=int(cs * 2.2))

        # 격자선 오버레이
        self.screen.blit(self._gridline_overlay, (self.grid_x, self.grid_y))

        # 끈적이 함정 — 보라 점 패턴
        sticky_cells = getattr(self.sim, 'sticky_cells', None)
        if sticky_cells:
            r = max(2, cs // 4)
            for sx, sy in sticky_cells:
                px = self.grid_x + sx * cs + cs // 2
                py = self.grid_y + sy * cs + cs // 2
                pygame.draw.circle(self.screen, (170, 100, 180), (px, py), r)
                pygame.draw.circle(self.screen, (90, 50, 100), (px, py), r, 1)

        # 아이템 (다이아몬드 + 작은 ?표시)
        for it in self.sim.items:
            cx = self.grid_x + it.x * cs + cs // 2
            cy = self.grid_y + it.y * cs + cs // 2
            size = max(4, cs // 2 + 1)
            points = [(cx, cy - size), (cx + size, cy),
                      (cx, cy + size), (cx - size, cy)]
            pygame.draw.polygon(self.screen, (255, 235, 130), points)
            pygame.draw.polygon(self.screen, (60, 50, 20), points, 1)

        # 에이전트 헤드: 검정 테두리 + 흰 점 (모든 배경에서 잘 보임)
        # 사망한 에이전트는 헤드 대신 위에서 그린 해골만 표시
        for agent in self.agents:
            if getattr(self.sim, 'dead', {}).get(agent.id, False):
                continue
            state = self.sim.states[agent.id]
            px = self.grid_x + state.x * self.cell_size
            py = self.grid_y + state.y * self.cell_size
            cx, cy = px + cs // 2, py + cs // 2
            dot_r = max(2, cs // 3)

            # 상태별 테두리 색 우선순위: startup 디버프 > freeze > 일반
            if getattr(agent, 'is_debuffed', None) and agent.is_debuffed():
                outline = (240, 150, 40)   # 주황 — 능력 봉인 중
                pygame.draw.circle(self.screen, outline, (cx, cy), dot_r + 3, 2)
            elif self.sim.freeze_ticks.get(agent.id, 0) > 0:
                outline = (90, 160, 220)
                pygame.draw.circle(self.screen, outline, (cx, cy), dot_r + 2)
            else:
                outline = (10, 10, 12)
                pygame.draw.circle(self.screen, outline, (cx, cy), dot_r + 2)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), dot_r)

        # 효과 플래시 (텍스트가 위로 올라가면서 페이드)
        for f in self.sim.effect_flashes:
            ratio = f.ticks_left / f.initial_ticks
            alpha = max(0, min(255, int(255 * ratio)))
            text_surf = self.font_stat.render(f.text, True, f.color)
            text_surf.set_alpha(alpha)
            rise = int((1 - ratio) * 24)  # 0~24px 위로
            tx = self.grid_x + f.x * cs + cs // 2 - text_surf.get_width() // 2
            ty = self.grid_y + f.y * cs - 16 - rise
            # 가독성용 어두운 배경 외곽
            shadow = self.font_stat.render(f.text, True, (0, 0, 0))
            shadow.set_alpha(alpha)
            self.screen.blit(shadow, (tx + 1, ty + 1))
            self.screen.blit(text_surf, (tx, ty))

    def _draw_sidebar(self):
        """그리드 아래의 랭킹 패널. portrait 레이아웃에서만 호출됨.

        - 영토 % 내림차순 정렬
        - 사망한 에이전트: 행 자체를 흐리게 + 좌측에 작은 해골 + 0%
        """
        areas = self.sim.get_areas()
        total = self.sim.grid.total_cells()

        team_mode_pre = getattr(self.sim, 'team_mode', False)
        # 팀 모드: 팀별 합산 영토 큰 순으로 정렬, 같은 팀 안에선 영토 큰 순.
        # 일반: 영토 큰 순. 둘 다 사망자는 맨 아래.
        if team_mode_pre:
            team_total = {}
            for a in self.agents:
                tid = self.sim.teams[a.id]
                team_total[tid] = team_total.get(tid, 0) + areas.get(a.id, 0)

            def sort_key(a):
                dead = self.sim.dead.get(a.id, False)
                tid = self.sim.teams[a.id]
                return (1 if dead else 0, -team_total[tid], tid, -areas.get(a.id, 0))
        else:
            def sort_key(a):
                dead = self.sim.dead.get(a.id, False)
                return (1 if dead else 0, -areas.get(a.id, 0))
        ranking = sorted(self.agents, key=sort_key)

        n = len(ranking)
        avail_h = self.sidebar_height - 32  # 위/아래 패딩
        row_h = max(50, min(80, avail_h // max(1, n) - 12))
        row_gap = 12
        row_x = self.grid_x
        row_w = self.grid_w

        # column 폭 분배
        rank_col_w = 50
        text_col_w = int(row_w * 0.34)
        pct_col_w = 90
        spark_col_w = 80     # sparkline 폭
        bar_x = row_x + rank_col_w + text_col_w + 12
        bar_w = (row_w - rank_col_w - text_col_w - pct_col_w
                 - spark_col_w - 32)
        track_h = max(10, row_h // 5)

        team_mode = getattr(self.sim, 'team_mode', False)
        # 팀 모드: 팀별 합산 영토도 계산해서 옆에 표시
        team_areas: dict = {}
        if team_mode:
            for a in self.agents:
                tid = self.sim.teams[a.id]
                team_areas[tid] = team_areas.get(tid, 0) + areas.get(a.id, 0)

        for i, agent in enumerate(ranking):
            dead = self.sim.dead.get(agent.id, False)
            area = areas.get(agent.id, 0)
            pct = 100.0 * area / total if total else 0
            y = self.sidebar_y + i * (row_h + row_gap)

            color = agent.color
            text_color = TEXT_PRIMARY
            sub_color = TEXT_SECONDARY
            if dead:
                # 흐림 처리: 색은 회색계열, 텍스트도 뮤트
                color = (170, 155, 140)
                text_color = (140, 115, 95)
                sub_color = (155, 130, 110)

            # 순위 번호만 — 팀은 색으로 구별
            rank_text = f'{i + 1}'
            rank_surf = self.font_pct.render(rank_text, True, sub_color)
            self.screen.blit(
                rank_surf,
                (row_x + rank_col_w - rank_surf.get_width() - 8,
                 y + (row_h - rank_surf.get_height()) // 2),
            )

            # 색 점 (사망 시 작은 해골 대체)
            dot_x = row_x + rank_col_w + 14
            dot_y = y + row_h // 2
            if dead:
                self._draw_skull(dot_x, dot_y, size=row_h - 16,
                                 bone=(220, 205, 188), ink=(70, 50, 40))
            else:
                pygame.draw.circle(self.screen, color, (dot_x, dot_y), row_h // 4)
                pygame.draw.circle(self.screen, (60, 45, 35),
                                   (dot_x, dot_y), row_h // 4, 2)

            # 이름 + 설명. Betrayer 는 이름 뒤에 표식.
            is_betrayer = agent.__class__.__name__ == 'BetrayerAgent'
            suffix = ''
            if dead:
                suffix = '   [DEAD]'
            elif is_betrayer and team_mode:
                suffix = '   [TRAITOR]'
            # 카테고리 라벨 (이름 뒤) + 팀 역할 (팀모드)
            from .agents import ROLE_GLYPHS
            role_glyph = ROLE_GLYPHS.get(agent.name, '')
            team_role = getattr(agent, 'team_role', None)
            parts = [agent.name]
            if role_glyph:
                parts.append(f'[{role_glyph}]')
            if team_role and team_mode:
                parts.append(f'({team_role[:3].upper()})')
            name_text = '  '.join(parts) + suffix
            # 누적 승률 (있을 때만)
            stats = getattr(self.sim, 'agent_stats', {}).get(agent.id, {})
            games = stats.get('games', 0)
            if games > 0:
                wins = stats.get('wins', 0)
                desc_text = f'{agent.description}   ·   {wins}/{games}W'
            else:
                desc_text = agent.description
            name_surf = self.font_stat.render(name_text, True, text_color)
            desc_surf = self.font_desc.render(desc_text, True, sub_color)
            text_x = dot_x + row_h // 3 + 12
            self.screen.blit(name_surf,
                             (text_x, y + row_h // 2 - name_surf.get_height() - 2))
            self.screen.blit(desc_surf, (text_x, y + row_h // 2 + 2))

            # 막대 (사망 시에도 0% 트랙만 표시)
            track_y = y + (row_h - track_h) // 2
            pygame.draw.rect(
                self.screen, BAR_TRACK,
                (bar_x, track_y, bar_w, track_h), border_radius=track_h // 2,
            )
            fill_w = int(bar_w * pct / 100.0)
            if fill_w > 0 and not dead:
                pygame.draw.rect(
                    self.screen, agent.color,
                    (bar_x, track_y, fill_w, track_h), border_radius=track_h // 2,
                )

            # 퍼센트 (좌측), 그 오른쪽에 sparkline
            pct_text = '—' if dead else f'{pct:.0f}%'
            pct_surf = self.font_pct.render(pct_text, True, text_color)
            pct_right = row_x + row_w - spark_col_w - 16
            self.screen.blit(
                pct_surf,
                (pct_right - pct_surf.get_width(),
                 y + (row_h - pct_surf.get_height()) // 2),
            )

            # Sparkline — 최근 50틱 영토 % 추이
            history = list(getattr(self.sim, 'area_history', {}).get(agent.id, []))
            if len(history) >= 2:
                spark_x = row_x + row_w - spark_col_w
                spark_y = y + 8
                spark_h = row_h - 16
                hmax = max(max(history), 1.0)
                step = spark_col_w / (len(history) - 1)
                pts = []
                for k, h in enumerate(history):
                    px = int(spark_x + k * step)
                    py = int(spark_y + spark_h - (h / hmax) * spark_h)
                    pts.append((px, py))
                line_color = (170, 155, 140) if dead else agent.color
                if len(pts) >= 2:
                    pygame.draw.lines(self.screen, line_color, False, pts, 2)
                # 끝점에 작은 원
                pygame.draw.circle(self.screen, line_color, pts[-1], 3)

    def _draw_skull(self, cx: int, cy: int, size: int,
                     bone=(245, 232, 215), ink=(35, 22, 14)):
        """간단한 해골 마커: bone 색 두개골 + 진한 brown 디테일."""
        head_r = max(5, size // 2)
        # 두개골 본체
        pygame.draw.circle(self.screen, bone, (cx, cy - head_r // 6), head_r)
        pygame.draw.circle(self.screen, ink, (cx, cy - head_r // 6), head_r, 2)

        # 눈구멍
        eye_r = max(2, head_r // 3)
        eye_offset = head_r // 2
        pygame.draw.circle(self.screen, ink, (cx - eye_offset, cy - head_r // 6), eye_r)
        pygame.draw.circle(self.screen, ink, (cx + eye_offset, cy - head_r // 6), eye_r)

        # 콧구멍 (작은 삼각형)
        nose_h = max(2, head_r // 4)
        nose_w = max(1, head_r // 6)
        nose_y = cy + head_r // 6
        pygame.draw.polygon(self.screen, ink, [
            (cx, nose_y),
            (cx - nose_w, nose_y + nose_h),
            (cx + nose_w, nose_y + nose_h),
        ])

        # 턱 (이빨 같은 짧은 세로선들)
        jaw_y = cy + head_r // 2
        jaw_w = head_r
        n_teeth = 4
        for i in range(n_teeth + 1):
            tx = cx - jaw_w // 2 + (jaw_w * i) // n_teeth
            pygame.draw.line(self.screen, ink, (tx, jaw_y),
                             (tx, jaw_y + max(2, head_r // 3)), 2)
        pygame.draw.line(self.screen, ink,
                         (cx - jaw_w // 2, jaw_y),
                         (cx + jaw_w // 2, jaw_y), 2)

    def tick_clock(self, fps: int):
        self.clock.tick(fps)

    def quit(self):
        pygame.quit()
