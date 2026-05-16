"""pygame 기반 시각화 - 다크 모던 스타일 + 격자선 + 충돌 이펙트"""
import pygame
import numpy as np
from typing import List

from .simulation import Simulation
from .agents import Agent

# 다크 팔레트
BG_COLOR = (15, 17, 21)        # #0F1115
PANEL_COLOR = (22, 25, 31)     # #16191F
GRID_BG = (26, 30, 38)         # #1A1E26
GRID_LINE = (32, 36, 44)       # #20242C
TEXT_PRIMARY = (235, 235, 240)
TEXT_SECONDARY = (140, 145, 160)
BAR_TRACK = (40, 44, 54)
HEAD_OUTLINE = (255, 255, 255)
INVULN_OUTLINE = (255, 215, 90)
FLASH_COLOR = (255, 255, 255)
WALL_COLOR = (60, 65, 75)         # 벽 색 (다크 그레이)


class Renderer:
    def __init__(
        self,
        sim: Simulation,
        agents: List[Agent],
        cell_size: int = 9,
        header_height: int = 90,
        sidebar_height: int = None,
        margin_x: int = 30,
    ):
        self.sim = sim
        self.agents = agents
        self.cell_size = cell_size

        self.grid_w = sim.grid.width * cell_size
        self.grid_h = sim.grid.height * cell_size

        # 사이드바 없음 — 영토 위에 직접 라벨 표시. 격자 아래 약간의 패딩만.
        if sidebar_height is None:
            sidebar_height = 30

        self.margin_x = margin_x
        self.window_w = self.grid_w + 2 * margin_x
        self.window_h = self.grid_h + header_height + sidebar_height

        self.grid_x = margin_x
        self.grid_y = header_height
        self.sidebar_y = self.grid_y + self.grid_h + 24

        self._label_fonts: dict = {}

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption('Territory War')
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont('arial,malgungothic', 28, bold=True)
        self.font_sub = pygame.font.SysFont('arial,malgungothic', 13)
        self.font_stat = pygame.font.SysFont('arial,malgungothic', 14, bold=True)
        self.font_pct = pygame.font.SysFont('arial,malgungothic', 16, bold=True)
        self.font_pin = pygame.font.SysFont('arial,malgungothic', 12, bold=True)

        self._color_lookup = self._build_color_lookup()
        self._gridline_overlay = self._build_gridline_overlay()

    def _build_color_lookup(self) -> np.ndarray:
        # +2 슬롯: 0(빈 칸), 1..N(에이전트), 마지막 슬롯=벽
        # numpy 음수 인덱싱 덕분에 cells 값 -1 (WALL) 은 자동으로 마지막 슬롯을 가리킨다.
        lookup = np.zeros((len(self.agents) + 2, 3), dtype=np.uint8)
        lookup[0] = GRID_BG
        for agent in self.agents:
            lookup[agent.id] = agent.color
        lookup[-1] = WALL_COLOR
        return lookup

    def _build_gridline_overlay(self) -> pygame.Surface:
        """격자선 오버레이 - 한 번만 만들고 재사용"""
        surf = pygame.Surface((self.grid_w, self.grid_h), pygame.SRCALPHA)
        for i in range(self.sim.grid.width + 1):
            x = i * self.cell_size
            pygame.draw.line(surf, (*GRID_LINE, 180), (x, 0), (x, self.grid_h), 1)
        for i in range(self.sim.grid.height + 1):
            y = i * self.cell_size
            pygame.draw.line(surf, (*GRID_LINE, 180), (0, y), (self.grid_w, y), 1)
        return surf

    def draw(self, fast_forward: bool = False):
        self.screen.fill(BG_COLOR)
        self._draw_header(fast_forward)
        self._draw_grid_panel()
        self._draw_area_labels()
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
        title = self.font_title.render('TERRITORY WAR', True, TEXT_PRIMARY)
        map_name = getattr(self.sim, 'map_name', 'open')
        subtitle = f'{len(self.agents)} algorithms · map: {map_name}'
        sub = self.font_sub.render(subtitle, True, TEXT_SECONDARY)
        self.screen.blit(title, (self.margin_x, 22))
        self.screen.blit(sub, (self.margin_x, 56))

        # 우측: tick 카운터
        tick_label = self.font_pin.render(f'TICK  {self.sim.tick:>5}', True, TEXT_SECONDARY)
        self.screen.blit(
            tick_label,
            (self.window_w - self.margin_x - tick_label.get_width(), 28),
        )

        # Fast forward 핀
        if fast_forward:
            label = self.font_pin.render('FAST FORWARD', True, BG_COLOR)
            pad_x, pad_y = 10, 6
            rect = pygame.Rect(
                self.window_w - self.margin_x - label.get_width() - pad_x * 2,
                52, label.get_width() + pad_x * 2, label.get_height() + pad_y * 2,
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

        # 셀 색
        cells = self.sim.grid.cells
        color_arr = self._color_lookup[cells]
        surf = pygame.surfarray.make_surface(color_arr.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (self.grid_w, self.grid_h))
        self.screen.blit(surf, (self.grid_x, self.grid_y))

        # 격자선 오버레이
        self.screen.blit(self._gridline_overlay, (self.grid_x, self.grid_y))

        # 아이템 (다이아몬드 + 작은 ?표시)
        cs = self.cell_size
        for it in self.sim.items:
            cx = self.grid_x + it.x * cs + cs // 2
            cy = self.grid_y + it.y * cs + cs // 2
            size = max(4, cs // 2 + 1)
            points = [(cx, cy - size), (cx + size, cy),
                      (cx, cy + size), (cx - size, cy)]
            pygame.draw.polygon(self.screen, (255, 235, 130), points)
            pygame.draw.polygon(self.screen, (60, 50, 20), points, 1)

        # 에이전트 헤드: 검정 테두리 + 흰 점 (모든 배경에서 잘 보임)
        for agent in self.agents:
            state = self.sim.states[agent.id]
            px = self.grid_x + state.x * self.cell_size
            py = self.grid_y + state.y * self.cell_size
            cx, cy = px + cs // 2, py + cs // 2
            dot_r = max(2, cs // 3)
            # freeze 상태면 테두리 색 변경
            outline = (90, 160, 220) if self.sim.freeze_ticks.get(agent.id, 0) > 0 else (10, 10, 12)
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
        areas = self.sim.get_areas()
        total = self.sim.grid.total_cells()
        ranking = sorted(self.agents, key=lambda a: -areas[a.id])

        row_h = 30
        row_gap = 12
        row_x = self.grid_x
        row_w = self.grid_w

        text_area_w = int(row_w * 0.50)
        pct_area_w = 60
        bar_area_x = row_x + text_area_w + 12
        bar_area_w = row_w - text_area_w - pct_area_w - 24
        track_h = 8

        for i, agent in enumerate(ranking):
            area = areas[agent.id]
            pct = 100.0 * area / total
            y = self.sidebar_y + i * (row_h + row_gap)

            # 좌측: 색 점 + 이름 + 설명
            dot_x = row_x + 8
            dot_y = y + row_h // 2
            pygame.draw.circle(self.screen, agent.color, (dot_x, dot_y), 6)

            name_surf = self.font_stat.render(agent.name, True, TEXT_PRIMARY)
            desc_surf = self.font_sub.render(f'  {agent.description}', True, TEXT_SECONDARY)
            self.screen.blit(
                name_surf,
                (dot_x + 14, y + (row_h - name_surf.get_height()) // 2),
            )
            self.screen.blit(
                desc_surf,
                (dot_x + 14 + name_surf.get_width(),
                 y + (row_h - desc_surf.get_height()) // 2),
            )

            # 중앙: 막대
            track_y = y + (row_h - track_h) // 2
            pygame.draw.rect(
                self.screen, BAR_TRACK,
                (bar_area_x, track_y, bar_area_w, track_h), border_radius=4,
            )
            fill_w = int(bar_area_w * pct / 100.0)
            if fill_w > 0:
                pygame.draw.rect(
                    self.screen, agent.color,
                    (bar_area_x, track_y, fill_w, track_h), border_radius=4,
                )

            # 우측: 퍼센트
            pct_text = f'{pct:.0f}%'
            pct_surf = self.font_pct.render(pct_text, True, TEXT_PRIMARY)
            self.screen.blit(
                pct_surf,
                (row_x + row_w - pct_surf.get_width() - 8,
                 y + (row_h - pct_surf.get_height()) // 2),
            )

    def tick_clock(self, fps: int):
        self.clock.tick(fps)

    def quit(self):
        pygame.quit()
